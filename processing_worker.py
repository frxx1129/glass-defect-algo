# --- START OF FILE processing_worker.py ---
import os
import cv2
import json
import base64
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

def calculation_worker(process_index, task_queue, results_queue, stop_event, run_event, config, shared_settings, data_sessions=None):
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

                # 如果任务中直接提供了统一 ROI（模拟模式），优先使用
                supplied_rois = task_data.get('rois')
                if supplied_rois is not None and isinstance(supplied_rois, list) and len(supplied_rois) > 0:
                    roi_cache[cam_idx] = supplied_rois
                elif cam_idx not in roi_cache:
                    roi_cache[cam_idx] = load_rois_for_cam(cam_idx)
                
                # 根据共享模式 (1=浅色,2=深色) 选择算法
                try:
                    algo_mode = int(getattr(shared_settings, 'algorithm_mode', 1))
                except Exception:
                    algo_mode = 1
                pane_json, annotated_image = fused_image_processor.process_image(
                    frame_data, roi_cache[cam_idx], config, algo_mode)

                # 会话逻辑：只有检测到玻璃(state_code>0)才开启文件夹；玻璃离开(state_code==0 且之前active)结束。
                is_collection = bool(getattr(shared_settings, 'data_collection_mode', False))
                session_info = None
                if is_collection and data_sessions is not None:
                    try:
                        # 读取状态机共享的跨相机会话状态
                        coll_active = bool(getattr(shared_settings, 'collection_pane_active', False))
                        coll_seq = int(getattr(shared_settings, 'collection_pane_seq', 0) or 0)
                        day_dir = getattr(shared_settings, 'collection_day_dir', time.strftime('%Y%m%d'))
                        state_code = int(pane_json.get('state_code', 0) or 0)
                        ts_now = int(time.time()*1000)
                        # 使用同一个 pane 序号和同一天目录
                        if coll_active and state_code > 0:
                            root = getattr(shared_settings, 'collection_output_root', 'collected_dataset')
                            pane_folder = f"pane{coll_seq}"
                            base = os.path.join(root, day_dir, pane_folder)
                            os.makedirs(base, exist_ok=True)
                            # 为每个相机缓存一次，避免重复判断
                            if str(cam_idx) not in data_sessions:
                                data_sessions[str(cam_idx)] = {'active': True, 'start_ts': ts_now, 'folder': base, 'pane_seq': coll_seq}
                            else:
                                # 更新路径和激活状态（以防日期或序号变化）
                                info = data_sessions[str(cam_idx)]
                                info.update({'active': True, 'folder': base, 'pane_seq': coll_seq})
                                data_sessions[str(cam_idx)] = info
                        else:
                            # 若状态机未激活或玻璃离开，标记为非激活
                            if str(cam_idx) in data_sessions:
                                info = data_sessions[str(cam_idx)]
                                info['active'] = False
                                data_sessions[str(cam_idx)] = info
                    except Exception:
                        pass

                # ================= ROI 裁剪保存 (采集模式) =================
                try:
                    if is_collection:
                        # 仅在状态机标记为激活时保存，并使用共享的 pane 序号目录
                        coll_active = bool(getattr(shared_settings, 'collection_pane_active', False))
                        coll_seq = int(getattr(shared_settings, 'collection_pane_seq', 0) or 0)
                        day_dir = getattr(shared_settings, 'collection_day_dir', time.strftime('%Y%m%d'))
                        if not coll_active or coll_seq <= 0:
                            raise Exception("采集模式未激活或序号无效，跳过保存")
                        base_folder = os.path.join(getattr(shared_settings, 'collection_output_root', 'collected_dataset'), day_dir, f"pane{coll_seq}")
                        os.makedirs(base_folder, exist_ok=True)
                        # 使用已建立的会话根目录，不再重复创建 pane_*；save_roi_crops 传 per_pane_folder=False 然后我们手动组织
                        # 将会话下的 original/ng 结构与先前逻辑兼容：直接把会话 folder 当 output_root 且不再创建 date 层
                        # 为复用函数，临时构造一个路径：在函数里 per_pane_folder=False 时会 base_dir=output_root/日期，需要改写: 这里复制函数逻辑较重，改简单包装
                        # 简化：复制一份核心循环保存（避免改原函数复杂度），避免局部导入以防遮蔽
                        rois = roi_cache[cam_idx]
                        millis = int(time.time()*1000)
                        # 仍使用 original / ng 子目录，便于区分
                        orig_dir = os.path.join(base_folder, 'original')
                        ng_dir = os.path.join(base_folder, 'ng')
                        os.makedirs(orig_dir, exist_ok=True)
                        if pane_json.get('image_status') == 'NG':
                            os.makedirs(ng_dir, exist_ok=True)
                        for roi_idx, roi in enumerate(rois):
                            try:
                                x=int(roi.get('x',0));y=int(roi.get('y',0));w=int(roi.get('width',0));h=int(roi.get('height',0))
                                if w<=0 or h<=0: continue
                                crop_gray = frame_data[y:y+h, x:x+w]
                                if crop_gray is None or crop_gray.size==0: continue
                                fn_png = f"cam{cam_idx}_ts{millis}_roi{roi_idx}.png"
                                fn_jpg = f"cam{cam_idx}_ts{millis}_roi{roi_idx}.jpg"
                                if pane_json.get('state_code',0)>0:
                                    # 原始 ROI 保存为 PNG
                                    cv2.imwrite(os.path.join(orig_dir, fn_png), crop_gray, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                                if pane_json.get('image_status')=='NG':
                                    crop_anno = annotated_image[y:y+h, x:x+w]
                                    if crop_anno is not None and crop_anno.size>0:
                                        # NG 标注保存为 JPG
                                        cv2.imwrite(os.path.join(ng_dir, fn_jpg), crop_anno, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            except Exception:
                                continue
                except Exception:
                    pass

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
                # 采集模式下不保存 inspection_results 目录（主逻辑已有 storage_path，但这里只控制结果入队即可）
                results_queue.put(result)
        except Exception:
            traceback.print_exc()
# --- END OF FILE processing_worker.py ---