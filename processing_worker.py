# --- START OF FILE processing_worker.py ---
import json
import base64
import cv2
import traceback
from queue import Empty
import time
import image_processor_optimized

def should_reject_pane(pane_json, shared_settings, pixels_per_mm):
    """Determines if a pane should be rejected based on defect size and type."""
    if pane_json['image_status'] == 'OK': return False
    max_size_thresh = shared_settings.max_defect_size_mm
    for defect in pane_json.get('defects', []):
        defect_type = defect.get('type')
        if defect_type in ['Q', 'X']: return True
        if defect_type in ['B', 'L']:
            location = defect.get('location', {})
            length_mm = location.get('length_mm', 0)
            width_mm = location.get('width_mm', 0)
            defect_size_mm = max(length_mm, width_mm)
            if defect_size_mm >= max_size_thresh: return True
    return False

def calculation_worker(process_index, task_queue, results_queue, stop_event, run_event, config, shared_settings):
    """A worker process that consumes raw images and produces analysis results."""
    print(f"[计算进程 {process_index}]: 已启动。")
    # 打印并行参数（进程内）
    try:
        sp = config.get('system_params', {})
        print(f"[计算进程 {process_index}]: opencv_threads={sp.get('opencv_threads')}, roi_threads={sp.get('roi_threads', 0) or sp.get('max_roi_workers', 0)}")
    except Exception:
        pass
    cv2.setUseOptimized(True)
    # OpenCV 线程数可控
    try:
        ocv_threads = int(config.get('system_params', {}).get('opencv_threads', 1) or 1)
        cv2.setNumThreads(max(1, ocv_threads))
    except Exception:
        pass
    PIXELS_PER_MM = config['system_params']['pixels_per_mm']
    # 合帧策略：短时间窗口内清空队列，仅保留每个相机的最新一帧
    drain_budget_ms = int(config.get('system_params', {}).get('coalesce_drain_budget_ms', 5))
    drain_max_n = int(config.get('system_params', {}).get('coalesce_max_drain', 500))
    
    # ROI 缓存：支持为每个相机指定独立的 ROI 文件
    roi_cache: dict[int, list] = {}
    camera_rois_cfg = config.get('camera_rois', {})
    def load_rois_for_cam(cam_idx: int):
        # 1) 配置优先：camera_rois 可为 dict 或 list
        roi_path = None
        if isinstance(camera_rois_cfg, dict):
            roi_path = camera_rois_cfg.get(str(cam_idx)) or camera_rois_cfg.get(cam_idx)
        elif isinstance(camera_rois_cfg, list) and cam_idx < len(camera_rois_cfg):
            roi_path = camera_rois_cfg[cam_idx]
        if not roi_path:
            roi_path = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
        try:
            with open(roi_path, 'r', encoding='utf-8') as f:
                averaged_data = json.load(f)
            best_group_key = max(averaged_data, key=lambda k: averaged_data[k].get('source_image_count', 0))
            rois = averaged_data[best_group_key]['averaged_rois']
            print(f"[计算进程 {process_index}]: Cam{cam_idx} 载入ROI: '{roi_path}', 组 '{best_group_key}'")
            return rois
        except Exception as e:
            print(f"[计算进程 {process_index}]: Cam{cam_idx} 加载ROI失败: {e}")
            return []
    
    processor_config = config.get('defect_detection_params', {})
    
    while not stop_event.is_set():
        # 如果未处于运行状态，阻塞等待 run_event 触发或 stop_event 结束
        if not run_event.is_set():
            try:
                _ = task_queue.get(timeout=1)  # 丢弃或暂存，不处理
            except Empty:
                pass
            continue
        try:
            # 先阻塞取一帧，保证不空转
            first_task = task_queue.get(timeout=1)
        except Empty:
            continue
        try:
            # 在极短时间窗口内/最大次数内，清空队列，仅保留每个相机最新的一帧
            latest_by_cam = {}
            if first_task is not None and 'cam_index' in first_task:
                latest_by_cam[first_task['cam_index']] = first_task
            t0 = time.perf_counter()
            drained = 0
            while drained < max(1, drain_max_n) and (time.perf_counter() - t0) * 1000.0 < max(0, drain_budget_ms):
                try:
                    item = task_queue.get_nowait()
                    cam_idx = item.get('cam_index')
                    if cam_idx is not None:
                        latest_by_cam[cam_idx] = item
                    drained += 1
                except Empty:
                    break

            # 依次处理每个相机的最新帧
            for cam_idx, task_data in latest_by_cam.items():
                frame_data = task_data.get('data')
                if frame_data is None:
                    continue

                if cam_idx not in roi_cache:
                    roi_cache[cam_idx] = load_rois_for_cam(cam_idx)
                pane_json, annotated_image = image_processor_optimized.process_image_from_memory_parallel(
                    frame_data, roi_cache[cam_idx], processor_config, draw_contours=True)

                should_reject_overall = should_reject_pane(pane_json, shared_settings, PIXELS_PER_MM)

                # 保存用高质量图
                jpg_q_main = int(config.get('system_params', {}).get('jpeg_quality_main', 85) or 85)
                success_original, original_buffer_encoded = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_main])
                if not success_original:
                    continue

                # WebSocket 预览小图
                PREVIEW_WIDTH = int(config.get('system_params', {}).get('preview_width', 800) or 800)
                height, width, _ = annotated_image.shape
                scale = PREVIEW_WIDTH / width
                preview_image = cv2.resize(annotated_image, (PREVIEW_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
                jpg_q_preview = int(config.get('system_params', {}).get('jpeg_quality_preview', 75) or 75)
                success_preview, preview_buffer_encoded = cv2.imencode('.jpg', preview_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_preview])

                base64_image_string = base64.b64encode(preview_buffer_encoded if success_preview else original_buffer_encoded).decode('utf-8')

                result = {
                    "camera_index": cam_idx,
                    "timestamp": time.time(),
                    "image": f"data:image/jpeg;base64,{base64_image_string}",
                    "image_status": pane_json.get('image_status', 'OK'),
                    "defects": pane_json.get('defects', []),
                    "state_code": pane_json.get('state_code', 0),
                    "should_reject": should_reject_overall,
                    "annotated_image_buffer": original_buffer_encoded.tobytes(),
                }
                results_queue.put(result)
        except Exception as e:
            print(f"[计算进程 {process_index}]: 处理错误: {e}")
            traceback.print_exc()