# --- START OF FILE processing_worker.py ---
import os
import cv2
import json
import base64
import traceback
from queue import Empty
import time
import fused_image_processor
import copy

def should_reject_pane(pane_json, shared_settings, pixels_per_mm):
    """根据缺陷类型与尺寸判定是否剔废：Q/E/B/L/X 均按照尺寸阈值处理。"""
    if pane_json.get('image_status') == 'OK':
        return False
    max_size_thresh = float(getattr(shared_settings, 'max_defect_size_mm', 20))
    for defect in pane_json.get('defects', []):
        defect_type = str(defect.get('type', '')).upper()
        if defect_type == 'X':
            return True
        if defect_type in ('B', 'L', 'Q', 'E'):
            loc = defect.get('location', {}) or {}
            length_mm = float(loc.get('length_mm', 0) or 0)
            width_mm = float(loc.get('width_mm', 0) or 0)
            if max(length_mm, width_mm) >= max_size_thresh:
                return True
    return False

def _scale_rois(rois, scale):
    """将 ROI 坐标按比例缩放，用于降采样后的处理。"""
    scaled = []
    for r in rois:
        sr = dict(r)
        for key in ('x', 'y', 'width', 'height', 'w', 'h'):
            if key in sr:
                try:
                    sr[key] = int(round(float(sr[key]) * scale))
                except Exception:
                    pass
        # 处理 left/top/right/bottom 格式
        for key in ('left', 'top', 'right', 'bottom'):
            if key in sr:
                try:
                    sr[key] = int(round(float(sr[key]) * scale))
                except Exception:
                    pass
        scaled.append(sr)
    return scaled


def _unscale_results(pane_json, scale):
    """将检测结果中的像素坐标从缩放空间反映射回原始空间。
    mm 单位的尺寸无需调整（因为 pixels_per_mm 已同步缩放，mm 值本身是正确的）。"""
    inv = 1.0 / scale
    # 反缩放 ROI 报告中的像素坐标
    for roi_rpt in pane_json.get('rois', []):
        for key in ('x', 'y', 'w', 'h'):
            if key in roi_rpt:
                try:
                    roi_rpt[key] = int(round(float(roi_rpt[key]) * inv))
                except Exception:
                    pass
        # 反缩放 ROI 内缺陷的像素坐标
        for defect in roi_rpt.get('defects', []):
            loc = defect.get('location', {})
            if isinstance(loc, dict):
                for key in ('x', 'y'):
                    if key in loc:
                        try:
                            loc[key] = int(round(float(loc[key]) * inv))
                        except Exception:
                            pass
    # 反缩放顶层 defects 列表中的像素坐标
    for defect in pane_json.get('defects', []):
        loc = defect.get('location', {})
        if isinstance(loc, dict):
            for key in ('x', 'y'):
                if key in loc:
                    try:
                        loc[key] = int(round(float(loc[key]) * inv))
                    except Exception:
                        pass


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
    # 降采样比例：1.0=不缩放(原始分辨率), 0.75=缩放到75%(推荐), 0.5=缩放到50%
    try:
        processing_scale = float(config.get('system_params', {}).get('processing_scale', 1.0) or 1.0)
        if processing_scale <= 0 or processing_scale > 1.0:
            processing_scale = 1.0
    except Exception:
        processing_scale = 1.0
    if processing_scale < 1.0:
        print(f"[计算进程 {process_index}]: 启用帧降采样，比例={processing_scale:.2f}")
    drain_budget_ms = int(config.get('system_params', {}).get('coalesce_drain_budget_ms', 5))
    drain_max_n_cfg = int(config.get('system_params', {}).get('coalesce_max_drain', 50))
    try:
        expected_cams = int(config.get('camera_setup', {}).get('expected_cameras', 0) or 0)
    except Exception:
        expected_cams = 0
    if expected_cams <= 0:
        expected_cams = 5
    # 防止单次批量过大导致队列抖动和端到端延迟上升。
    drain_cap_by_topology = max(20, expected_cams * 8)
    drain_max_n = max(1, min(drain_max_n_cfg, drain_cap_by_topology))
    
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
    
    last_paused_log_ts = 0.0
    last_results_full_log_ts = 0.0
    try:
        while not stop_event.is_set():
            if not run_event.is_set():
                # 检测被暂停时，仍会持续从队列取数据以防队列堆积；这里补一条低频日志，避免“静默不处理”难定位。
                now = time.time()
                if now - last_paused_log_ts >= 30.0:
                    last_paused_log_ts = now
                    try:
                        print(f"[计算进程 {process_index}]: run_event=OFF，暂停处理（仍在清空输入队列）")
                    except Exception:
                        pass
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
                # 取消“只取最新”的合并策略：改为按入队顺序处理所有已取出的任务，避免丢弃旧帧
                tasks_to_process = []
                if first_task is not None:
                    tasks_to_process.append(first_task)
                t0 = time.perf_counter()
                drained = 0
                # 继续在时间/数量预算内尽可能多取任务，但不去重/覆盖，保持 FIFO 处理
                while drained < max(0, drain_max_n - 1) and (time.perf_counter() - t0) * 1000.0 < max(0, drain_budget_ms):
                    try:
                        item = task_queue.get_nowait()
                        tasks_to_process.append(item)
                        drained += 1
                    except Empty:
                        break

                for task_data in tasks_to_process:
                    cam_idx = task_data.get('cam_index')
                    if cam_idx is None:
                        continue
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
                # 在进入处理器前，根据"仅头尾相机启用 Q"策略，为本次调用构造局部配置副本并注入运行时开关
                # 优化：避免每帧 deepcopy 整个 config，仅浅拷贝并局部修改需要变更的嵌套字典
                if isinstance(config, dict):
                    conf_local = dict(config)  # 浅拷贝顶层
                    # 仅对需要修改的嵌套字典做浅拷贝
                    if 'hough_inspector_params' in config and isinstance(config['hough_inspector_params'], dict):
                        conf_local['hough_inspector_params'] = dict(config['hough_inspector_params'])
                        if 'DEFECT_DETECTION' in conf_local['hough_inspector_params']:
                            conf_local['hough_inspector_params']['DEFECT_DETECTION'] = dict(conf_local['hough_inspector_params']['DEFECT_DETECTION'])
                    if 'hough_inspector_dark_params' in config and isinstance(config['hough_inspector_dark_params'], dict):
                        conf_local['hough_inspector_dark_params'] = dict(config['hough_inspector_dark_params'])
                        if 'DEFECT_DETECTION' in conf_local['hough_inspector_dark_params']:
                            conf_local['hough_inspector_dark_params']['DEFECT_DETECTION'] = dict(conf_local['hough_inspector_dark_params']['DEFECT_DETECTION'])
                else:
                    conf_local = config
                try:
                    cam_setup = config.get('camera_setup', {}) or {}
                    total_cams_rt = int(cam_setup.get('expected_cameras', 0) or 0)
                    if total_cams_rt <= 0:
                        try:
                            total_cams_rt = int(len(cam_setup.get('camera_bindings', []) or []))
                        except Exception:
                            total_cams_rt = 0
                    if total_cams_rt <= 0:
                        try:
                            cro = config.get('camera_rois', {}) or {}
                            if isinstance(cro, dict):
                                total_cams_rt = len(cro.keys())
                            elif isinstance(cro, list):
                                total_cams_rt = len(cro)
                        except Exception:
                            total_cams_rt = 0
                    q_enabled_rt = True
                    if total_cams_rt >= 1:
                        first_idx_rt = 0
                        last_idx_rt = max(0, total_cams_rt - 1)
                        q_enabled_rt = (cam_idx in (first_idx_rt, last_idx_rt))
                    # 将运行时开关写入浅色与深色参数节
                    try:
                        hip = conf_local.setdefault('hough_inspector_params', {})
                        dd  = hip.setdefault('DEFECT_DETECTION', {})
                        dd['Q_ENABLED'] = bool(q_enabled_rt)
                    except Exception:
                        pass
                    try:
                        hid = conf_local.setdefault('hough_inspector_dark_params', {})
                        ddd = hid.setdefault('DEFECT_DETECTION', {})
                        ddd['Q_ENABLED'] = bool(q_enabled_rt)
                    except Exception:
                        pass
                    # 记录当前相机索引，便于下游按需使用
                    try:
                        rt = conf_local.setdefault('__runtime__', {})
                        rt['current_cam_idx'] = cam_idx
                    except Exception:
                        pass
                    # 注入 lineName 和 cam_index 到 hough_inspector_params 供不检测区域使用
                    try:
                        line_name_rt = str(getattr(shared_settings, 'lineName', '') or config.get('lineName', ''))
                        hip = conf_local.setdefault('hough_inspector_params', {})
                        hip['_RUNTIME_LINE_NAME'] = line_name_rt
                        hip['_RUNTIME_CAM_INDEX'] = cam_idx
                        # 同步到深色参数（如果使用深色模式）
                        hid = conf_local.setdefault('hough_inspector_dark_params', {})
                        hid['_RUNTIME_LINE_NAME'] = line_name_rt
                        hid['_RUNTIME_CAM_INDEX'] = cam_idx
                    except Exception:
                        pass
                except Exception:
                    conf_local = config

                # === 帧降采样处理 ===
                # 在送入检测器前缩放帧和ROI，检测后反缩放坐标并放大标注图
                if processing_scale < 1.0:
                    orig_h, orig_w = frame_data.shape[:2]
                    new_w = int(orig_w * processing_scale)
                    new_h = int(orig_h * processing_scale)
                    frame_scaled = cv2.resize(frame_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    rois_scaled = _scale_rois(roi_cache[cam_idx], processing_scale)
                    # 同步缩放 pixels_per_mm，确保 mm 级尺寸判定不受影响
                    conf_scaled = dict(conf_local)
                    sys_params_scaled = dict(conf_scaled.get('system_params', {}))
                    sys_params_scaled['pixels_per_mm'] = PIXELS_PER_MM * processing_scale
                    conf_scaled['system_params'] = sys_params_scaled
                    # 注入缩放因子供内部模块（如排除区域坐标缩放）使用
                    conf_scaled['_PROCESSING_SCALE'] = processing_scale

                    pane_json, annotated_image = fused_image_processor.process_image(
                        frame_scaled, rois_scaled, conf_scaled, algo_mode)

                    # 将像素坐标反缩放回原始空间
                    _unscale_results(pane_json, processing_scale)
                    # 将标注图放大回原始尺寸（保证上传/预览画质）
                    annotated_image = cv2.resize(annotated_image, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    pane_json, annotated_image = fused_image_processor.process_image(
                        frame_data, roi_cache[cam_idx], conf_local, algo_mode)

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

                # 新策略：仅“头尾”两个逻辑相机启用缺角(Q)检测，其余相机剔除所有 Q 缺陷；若无其他缺陷则判 OK。
                try:
                    cam_setup = config.get('camera_setup', {}) or {}
                    total_cams = int(cam_setup.get('expected_cameras', 0) or 0)
                    if total_cams <= 0:
                        # 回退：优先按 camera_bindings 数量，再按 camera_rois 键数量
                        try:
                            total_cams = int(len(cam_setup.get('camera_bindings', []) or []))
                        except Exception:
                            total_cams = 0
                    if total_cams <= 0:
                        try:
                            cro = config.get('camera_rois', {}) or {}
                            if isinstance(cro, dict):
                                total_cams = len(cro.keys())
                            elif isinstance(cro, list):
                                total_cams = len(cro)
                        except Exception:
                            total_cams = 0
                    # 仅当能确定总相机数>=2时，执行“只保留头尾相机的Q”策略
                    if total_cams >= 1:
                        first_idx = 0
                        last_idx = max(0, total_cams - 1)
                        if cam_idx not in (first_idx, last_idx):
                            defs = pane_json.get('defects')
                            if isinstance(defs, list):
                                kept = [d for d in defs if str(d.get('type', '')).upper() != 'Q']
                                pane_json['defects'] = kept
                                if not kept:
                                    pane_json['image_status'] = 'OK'
                except Exception:
                    pass

                # 判定是否剔废需要基于剔除 Q 后的结果
                should_reject_overall = should_reject_pane(pane_json, shared_settings, PIXELS_PER_MM)

                jpg_q_main = int(config.get('system_params', {}).get('jpeg_quality_main', 85) or 85)
                # 编码标注后的整帧（用于上传/预览）
                success_original, original_buffer_encoded = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_main])
                if not success_original:
                    continue
                # 额外：仅当本帧为 NG 时，编码未标注的原始整帧（灰度），用于本地保存“未标注原图”
                success_raw, raw_buffer_encoded = False, None
                if str(pane_json.get('image_status', 'OK')).upper() == 'NG':
                    try:
                        success_raw, raw_buffer_encoded = cv2.imencode('.jpg', frame_data, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_main])
                    except Exception:
                        success_raw, raw_buffer_encoded = False, None

                PREVIEW_WIDTH = int(config.get('system_params', {}).get('preview_width', 800) or 800)
                height, width, _ = annotated_image.shape
                scale = PREVIEW_WIDTH / width
                preview_image = cv2.resize(annotated_image, (PREVIEW_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
                jpg_q_preview = int(config.get('system_params', {}).get('jpeg_quality_preview', 75) or 75)
                success_preview, preview_buffer_encoded = cv2.imencode('.jpg', preview_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_preview])

                base64_image_string = base64.b64encode(preview_buffer_encoded if success_preview else original_buffer_encoded).decode('utf-8')

                # 取消基于“近竖直线数量>=2”的 Q 过滤逻辑（已由“仅头尾相机启用 Q”策略取代）

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
                # 提供 ROI 精简信息用于状态机的竖直线统计与进入判定
                if isinstance(pane_json.get('rois'), list):
                    result['rois'] = pane_json.get('rois')
                if success_raw and raw_buffer_encoded is not None:
                    # 提供未标注原图的 JPEG 字节给状态机线程做本地保存
                    result["raw_image_buffer"] = raw_buffer_encoded.tobytes()
                # 采集模式下不保存 inspection_results 目录（主逻辑已有 storage_path，但这里只控制结果入队即可）
                # 关键：避免 results_queue 满时永久阻塞，导致所有 worker 卡死 -> 系统表面存活但不再出图。
                enqueued = False
                try:
                    results_queue.put(result, timeout=0.2)
                    enqueued = True
                except TypeError:
                    # 兼容某些 QueueProxy 不支持 timeout 参数的情况
                    try:
                        results_queue.put_nowait(result)
                        enqueued = True
                    except Exception:
                        enqueued = False
                except Exception:
                    enqueued = False
                    if not enqueued:
                        now = time.time()
                        if now - last_results_full_log_ts >= 5.0:
                            last_results_full_log_ts = now
                            try:
                                # 不打印过多细节，避免刷屏；此日志用于定位“下游不消费/队列满”问题
                                print(f"[计算进程 {process_index}]: ⚠️ results_queue 写入失败(可能已满)，已丢弃一帧结果")
                            except Exception:
                                pass
            except KeyboardInterrupt:
                raise
            except Exception:
                traceback.print_exc()
    except KeyboardInterrupt:
        # Ctrl+C 触发时安静退出
        try:
            print(f"[计算进程 {process_index}]: 收到键盘中断，退出。")
        except Exception:
            pass
# --- END OF FILE processing_worker.py ---