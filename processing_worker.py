# --- START OF FILE processing_worker.py ---
import json
import base64
import cv2
import traceback
from queue import Empty
import time
import fused_image_processor

def should_reject_pane(pane_json, shared_settings, pixels_per_mm):
    """根据缺陷类型与尺寸判定是否剔废。
    规则：
    - Q(缺角)、X(斜边) 直接判定剔废。
    - B(崩边)、L(裂纹) 若任一缺陷的 length_mm 或 width_mm >= 阈值(max_defect_size_mm) 判定剔废。
    - 其余或无缺陷 => 不剔废。
    """
    if pane_json.get('image_status') == 'OK':
        return False
    max_size_thresh = float(getattr(shared_settings, 'max_defect_size_mm', 20))
    for defect in pane_json.get('defects', []):
        defect_type = defect.get('type')
        if defect_type in ('X'):
            return True
        if defect_type in ('Q', 'B', 'L'):
            loc = defect.get('location', {})
            length_mm = float(loc.get('length_mm', 0) or 0)
            width_mm = float(loc.get('width_mm', 0) or 0)
            if max(length_mm, width_mm) >= max_size_thresh:
                return True
    return False

def calculation_worker(process_index, task_queue, results_queue, stop_event, run_event, config, shared_settings):
    """A worker process that consumes raw images and produces analysis results."""
    print(f"[计算进程 {process_index}]: 已启动。")
    try:
        sp = config.get('system_params', {})
        print(f"[计算进程 {process_index}]: opencv_threads={sp.get('opencv_threads')}, roi_threads={sp.get('roi_threads', 0) or sp.get('max_roi_workers', 0)}")
    except Exception:
        pass
    cv2.setUseOptimized(True)
    try:
        ocv_threads = int(config.get('system_params', {}).get('opencv_threads', 1) or 1)
        cv2.setNumThreads(max(1, ocv_threads))
    except Exception:
        pass
    PIXELS_PER_MM = config['system_params']['pixels_per_mm']
    drain_budget_ms = int(config.get('system_params', {}).get('coalesce_drain_budget_ms', 5))
    drain_max_n = int(config.get('system_params', {}).get('coalesce_max_drain', 500))
    
    roi_cache: dict[int, list] = {}
    camera_rois_cfg = config.get('camera_rois', {})
    def load_rois_for_cam(cam_idx: int):
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
    
    while not stop_event.is_set():
        if not run_event.is_set():
            try:
                _ = task_queue.get(timeout=1)
            except Empty:
                pass
            continue
        try:
            first_task = task_queue.get(timeout=1)
        except Empty:
            continue
        try:
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

            for cam_idx, task_data in latest_by_cam.items():
                frame_data = task_data.get('data')
                if frame_data is None:
                    continue

                if cam_idx not in roi_cache:
                    roi_cache[cam_idx] = load_rois_for_cam(cam_idx)
                
                # 根据共享模式 (1=浅色,2=深色) 选择算法
                try:
                    algo_mode = int(getattr(shared_settings, 'algorithm_mode', 1))
                except Exception:
                    algo_mode = 1
                pane_json, annotated_image = fused_image_processor.process_image(
                    frame_data, roi_cache[cam_idx], config, algo_mode)

                should_reject_overall = should_reject_pane(pane_json, shared_settings, PIXELS_PER_MM)

                jpg_q_main = int(config.get('system_params', {}).get('jpeg_quality_main', 85) or 85)
                success_original, original_buffer_encoded = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_main])
                if not success_original:
                    continue

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
# --- END OF FILE processing_worker.py ---