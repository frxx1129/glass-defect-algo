# --- START OF FILE state_machine.py ---
import time
import json
import os
import asyncio
import numpy as np
from queue import Empty
from datetime import datetime
import yield_manager
from server_comms import send_report_to_server, send_reports_batch_to_server, fetch_collection_id_from_server, fetch_initial_state_from_server, broadcast_rejections, broadcast_yield_and_rejections

def results_and_state_machine_thread(num_cameras, results_queue, connection_manager, stop_event, loop, shared_settings, counters, queues, flags, machine_state_shared, stats_lock, metadata, run_event_proxy, http_client, alarm_light_controller):
    print("[状态机线程]: 已启动。")
    
    # alarm_light_controller 实例现在通过函数参数直接传入
    
    STORAGE_PATH = shared_settings.storage_path
    # 修改：移除产量计数，仅保留剔废计数
    (shared_rejection_counter, shared_yield_counter) = counters  # 0=rejections 1=yield
    (rejection_queue,) = queues
    (manual_reject_flag, shared_rejection_mode, can_late_reject) = flags
    (shared_collection_id, shared_user_id_auto, shared_user_id_manual) = metadata

    machine_state = "WAITING_FOR_PANE"
    machine_state_shared.value = 0 
    last_camera_states = np.zeros(num_cameras, dtype=np.int32)
    current_pane_ng_buffer = []  # 收集该片玻璃所有 NG 帧的原始结果（含图像缓冲）
    current_pane_reports = []    # 收集该片玻璃所有 NG 报告（即时写入 JSON）
    current_pane_folder = None   # 当前玻璃的存储文件夹
    pane_ng_frame_counter = 0    # 当前玻璃 NG 帧计数
    last_pane_data = {}
    
    with stats_lock:
        last_reset_date_str, total_yield, total_rejections = yield_manager.load_stats()
        shared_rejection_counter.value = total_rejections
        shared_yield_counter.value = total_yield
    
    max_complexity_snapshot = np.zeros(num_cameras, dtype=np.int32)
    is_current_event_rejected = False
    # 黄灯策略：每出现一帧新的 NG 图像就刷新黄灯持续时间，直到剔废(红灯)或玻璃离开
    # 不再使用单次触发标志
    rejection_details = {}
    saved_for_this_pane = False
    # 采集模式下：初始化跨相机共享的 pane 序号与激活标记（通过 shared_settings 暴露给各计算进程）
    try:
        if getattr(shared_settings, 'data_collection_mode', False):
            from datetime import datetime as _dt
            if not hasattr(shared_settings, 'collection_pane_seq'):
                shared_settings.collection_pane_seq = 0
            if not hasattr(shared_settings, 'collection_pane_active'):
                shared_settings.collection_pane_active = False
            if not hasattr(shared_settings, 'collection_day_dir'):
                shared_settings.collection_day_dir = _dt.now().strftime('%Y%m%d')
    except Exception:
        pass
    
    # 进入/离开 去抖（帧）——避免算法偶发抖动导致反复进入/离开
    ENTER_CONFIRM_FRAMES = int(getattr(shared_settings, 'enter_confirm_frames', 5))
    LEAVE_CONFIRM_FRAMES = int(getattr(shared_settings, 'leave_confirm_frames', 5))
    presence_streak = 0
    absence_streak = 0
    
    def get_defect_size(defect):
        defect_type = defect.get('type')
        if defect_type in ['B', 'L']:
            rect = defect.get('location', {})
            return max(rect.get('length_mm', 0), rect.get('width_mm', 0))
        return 0
    
    def find_best_ng_result(ng_buffer):
        if not ng_buffer: return None
        return max(ng_buffer, key=lambda r: max([get_defect_size(d) for d in r.get('defects', [])] or [-1]))

    def build_report_for_frame(result_obj, rejection_details_to_save):
        defects = result_obj.get('defects', [])
        def primary_size(d):
            loc = d.get('location', {})
            return max(loc.get('length_mm', 0), loc.get('width_mm', 0))
        max_defect_size = max([primary_size(d) for d in defects] or [0])
        has_x_defect = any(d.get('type') == 'X' for d in defects)
        size_label_parts = []
        if max_defect_size > 0:
            size_label_parts.append(f"{int(round(max_defect_size))}mm")
        if has_x_defect:
            size_label_parts.append('X')
        final_size_label = ','.join(size_label_parts) if size_label_parts else ''
        raw_cid = shared_collection_id.value
        try:
            if isinstance(raw_cid, (bytes, bytearray)):
                collection_id = int(raw_cid.decode('utf-8', errors='ignore').strip() or -1)
            else:
                collection_id = int(raw_cid)
        except Exception:
            collection_id = -1
        user_id = shared_user_id_manual.value if rejection_details_to_save.get('rejection_type') == '2' else shared_user_id_auto.value
        return {
            'collection_id': int(collection_id),
            'rejection_type': rejection_details_to_save.get('rejection_type', 'unknown'),
            'rejection_time': rejection_details_to_save.get('rejection_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            'userId': user_id,
            'size_label': final_size_label,
            'image_status': result_obj['image_status'],
            'defects': defects,
            'camera_index': result_obj['camera_index']
        }

    def upload_current_pane_if_needed():
        """在玻璃离开时上传该片期间所有已保存的 NG 报告与图像。"""
        nonlocal current_pane_ng_buffer, current_pane_reports, current_pane_folder
        if not current_pane_ng_buffer or not current_pane_reports:
            return
        # 统一使用最终剔废类型（若发生剔废则全部标记其类型；否则保持各自已有）
        final_type = None
        if is_current_event_rejected and rejection_details.get('rejection_type'):
            final_type = rejection_details.get('rejection_type')
        reports_for_upload = []
        images_for_upload = []
        for rpt, res in zip(current_pane_reports, current_pane_ng_buffer):
            if final_type and rpt.get('rejection_type') in ('0', 'pending', 'unknown'):
                rpt['rejection_type'] = final_type
                # 统一 rejection_time 为剔废时间
                if 'rejection_time' in rejection_details:
                    rpt['rejection_time'] = rejection_details['rejection_time'].strftime('%Y-%m-%d %H:%M:%S')
            reports_for_upload.append(rpt)
            images_for_upload.append(res.get('annotated_image_buffer'))
        send_reports_batch_to_server(reports_for_upload, images_for_upload, shared_settings.upload_url, http_client, upload_timeout_s=getattr(shared_settings, 'http_upload_timeout_s', 30))
        print(f"    [状态机]: 本片玻璃上传完成，共 {len(reports_for_upload)} 张 NG 图像。")

    # Initial state fetch
    fetch_collection_id_from_server(shared_settings, shared_collection_id)
    initial_state = fetch_initial_state_from_server(shared_settings)
    if initial_state:
        shared_rejection_mode.value = initial_state.get("rejectionMode", 1)
        threshold_int = initial_state.get("rejectionThreshold", 20)
        shared_settings.max_defect_size_mm = threshold_int if threshold_int else 20
        shared_user_id_auto.value = initial_state.get("algUserVO", {}).get("userId", 7)
        # 新增: 读取算法模式 (1=浅色 2=深色)
        try:
            init_algo_mode = int(initial_state.get("algorithmMode", getattr(shared_settings, 'algorithm_mode', 1)) or 1)
            if init_algo_mode in (1,2):
                shared_settings.algorithm_mode = init_algo_mode
                print(f"[状态机]: 初始算法模式设置为 {init_algo_mode}")
            else:
                print(f"[状态机]: 初始算法模式值非法 {init_algo_mode}, 使用默认 {getattr(shared_settings,'algorithm_mode',1)}")
        except Exception as e:
            print(f"[状态机]: 解析初始算法模式失败: {e}")
        if initial_state.get("enable") == 1: 
            run_event_proxy.set()
            print("[状态机]: 从服务器获取到启用状态，设置运行事件。")
    else:
        print("[状态机]: 未能从服务器获取状态，保持当前运行状态。")
    
    while not stop_event.is_set():
        # 处理滞后剔废（在等待新玻璃状态下）
        if machine_state == "WAITING_FOR_PANE" and manual_reject_flag.value and can_late_reject.value:
            manual_reject_flag.value = False
            can_late_reject.value = False
            print(f"--- [状态机]: 检测到滞后手动剔废指令 ---")
            if current_pane_ng_buffer and not is_current_event_rejected:
                # 更新所有已缓存报告的剔废类型并上传（之前未上传因为处于模式2下且未剔废）
                rejection_details = {"rejection_time": datetime.now(), "rejection_type": "2"}
                for rpt in current_pane_reports:
                    rpt['rejection_type'] = '2'
                    rpt['rejection_time'] = rejection_details['rejection_time'].strftime('%Y-%m-%d %H:%M:%S')
                send_reports_batch_to_server(current_pane_reports, [r.get('annotated_image_buffer') for r in current_pane_ng_buffer], shared_settings.upload_url, http_client, upload_timeout_s=getattr(shared_settings, 'http_upload_timeout_s', 30))
                # 入队硬件动作（滞后手动剔废）
                try:
                    route = getattr(shared_settings, 'manual_reject_route', None)
                except Exception:
                    route = None
                rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, -1, route))
                if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                    try:
                        alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                    except Exception:
                        pass
                with stats_lock:
                    total_rejections += 1
                    # 之前离开时已把该片计为产量 +1，此处回滚 -1
                    if total_yield > 0:
                        total_yield -= 1
                        shared_yield_counter.value = total_yield
                    shared_rejection_counter.value = total_rejections
                    yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                print(f"    [状态机]: 滞后剔废完成并上传。本片NG帧数量: {len(current_pane_ng_buffer)}")
                # 上传后清空，为下一片做准备
                current_pane_ng_buffer.clear()
                current_pane_reports.clear()
                current_pane_folder = None
                pane_ng_frame_counter = 0
            continue

        try:
            result = results_queue.get(timeout=0.04)
        except Empty:
            continue
        except Exception as e:
            print(f"[状态机]: 从队列获取结果出错: {e}")
            continue

        try:
            cam_index = result["camera_index"]
            ws_data = {k: v for k, v in result.items() if k != 'annotated_image_buffer'}
            try:
                asyncio.run_coroutine_threadsafe(connection_manager.broadcast(json.dumps(ws_data), cam_index), loop)
            except Exception:
                pass

            now_str = datetime.now().strftime('%Y-%m-%d')
            if now_str != last_reset_date_str:
                with stats_lock:
                    print(f"--- [状态机]: 日期变更，保存 {last_reset_date_str} 日志并重置计数 ---")
                    yield_manager.save_daily_report(last_reset_date_str, total_yield, total_rejections)
                    total_rejections = 0
                    total_yield = 0
                    shared_rejection_counter.value = 0
                    shared_yield_counter.value = 0
                    last_reset_date_str = now_str
                    yield_manager.save_stats(now_str, total_yield, total_rejections)
                # 日期变更时，采集模式下 pane 序号从 1 重新开始（此处置 0，进入时 +1）
                try:
                    if getattr(shared_settings, 'data_collection_mode', False):
                        shared_settings.collection_pane_seq = 0
                        shared_settings.collection_day_dir = datetime.now().strftime('%Y%m%d')
                except Exception:
                    pass

            if cam_index >= len(last_camera_states):
                last_camera_states = np.pad(last_camera_states, (0, cam_index - len(last_camera_states) + 1), 'constant')
                max_complexity_snapshot = np.pad(max_complexity_snapshot, (0, cam_index - len(max_complexity_snapshot) + 1), 'constant')

            last_camera_states[cam_index] = result['state_code']
            current_total_panes = int(np.sum(last_camera_states))

            if machine_state == "PANE_DETECTED":
                # 检测离开
                if current_total_panes == 0:
                    absence_streak += 1
                    if absence_streak >= LEAVE_CONFIRM_FRAMES:
                        print("--- [状态机]: 玻璃离开事件 ---")
                        machine_state = "WAITING_FOR_PANE"
                        machine_state_shared.value = 0
                        # 采集模式下：关闭跨相机 pane 激活标记
                        try:
                            if getattr(shared_settings, 'data_collection_mode', False):
                                shared_settings.collection_pane_active = False
                        except Exception:
                            pass
                        absence_streak = 0
                        presence_streak = 0

                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                if is_current_event_rejected:
                                    alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                                else:
                                    alarm_light_controller.set_normal_state()
                            except Exception:
                                pass

                        # 离开时：若为模式2且未剔废，允许滞后剔废 -> 不立即上传
                        if shared_rejection_mode.value == 2 and not is_current_event_rejected and current_pane_ng_buffer:
                            can_late_reject.value = True
                            print("    [状态机]: 等待可能的滞后剔废（模式2），暂不上传。")
                        else:
                            # 立即上传策略：该调用占位以便未来恢复批量上传
                            upload_current_pane_if_needed()
                            # 上传后清理
                            current_pane_ng_buffer.clear()
                            current_pane_reports.clear()
                            current_pane_folder = None
                            pane_ng_frame_counter = 0
                            can_late_reject.value = False
                            # 若没有被剔废（包括未触发滞后），计入产量
                            if not is_current_event_rejected:
                                with stats_lock:
                                    total_yield += 1
                                    shared_yield_counter.value = total_yield
                                    yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                                broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                            is_current_event_rejected = False
                        continue
                else:
                    absence_streak = 0

                # NG 帧处理
                if result['image_status'] == 'NG':
                    if current_pane_folder is None:
                        pane_start_ts = datetime.now()
                        raw_cid = shared_collection_id.value
                        try:
                            if isinstance(raw_cid, (bytes, bytearray)):
                                cid_int = int(raw_cid.decode('utf-8', errors='ignore').strip() or -1)
                            else:
                                cid_int = int(raw_cid)
                        except Exception:
                            cid_int = -1
                        date_dir = os.path.join(STORAGE_PATH, pane_start_ts.strftime('%Y-%m-%d'))
                        os.makedirs(date_dir, exist_ok=True)
                        current_pane_folder = os.path.join(date_dir, f"pane_{pane_start_ts.strftime('%H%M%S')}_{cid_int}")
                        os.makedirs(current_pane_folder, exist_ok=True)
                        pane_ng_frame_counter = 0
                    current_pane_ng_buffer.append(result)
                    frame_ts = datetime.now()
                    temp_rej_details = {
                        'rejection_time': frame_ts,
                        'rejection_type': (rejection_details.get('rejection_type') if is_current_event_rejected else '0')
                    }
                    rpt = build_report_for_frame(result, temp_rej_details)
                    current_pane_reports.append(rpt)
                    try:
                        cam_disp = int(result['camera_index']) + 1
                    except Exception:
                        cam_disp = result.get('camera_index', 'X')
                    pane_ng_frame_counter += 1
                    base_name = f"{pane_ng_frame_counter:04d}_Cam{cam_disp}"
                    img_buf = result.get('annotated_image_buffer')
                    if img_buf:
                        with open(os.path.join(current_pane_folder, base_name + '.jpg'), 'wb') as f:
                            f.write(img_buf)
                    with open(os.path.join(current_pane_folder, base_name + '.json'), 'w', encoding='utf-8') as f:
                        json.dump(rpt, f, ensure_ascii=False, indent=2, default=str)
                    if not is_current_event_rejected and alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                        try:
                            alarm_light_controller.set_ng_detected_state(shared_settings.ng_buzz_duration_s)
                        except Exception:
                            pass

                if np.sum(last_camera_states) > np.sum(max_complexity_snapshot):
                    max_complexity_snapshot = last_camera_states.copy()

                if not is_current_event_rejected:
                    # 自动剔废
                    if shared_rejection_mode.value == 1 and result.get('should_reject', False):
                        is_current_event_rejected = True
                        saved_for_this_pane = True
                        rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                        # 自动模式：暂不分路，route=None -> 控制器将执行 ALL。后续可在此根据缺陷位置决定路由。
                        rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, cam_index, None))
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                            except Exception:
                                pass
                        with stats_lock:
                            total_rejections += 1
                            shared_rejection_counter.value = total_rejections
                            yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                        broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                        print("    [状态机]: 自动剔废触发！")
                    # 即时手动剔废（模式2）
                    elif manual_reject_flag.value and shared_rejection_mode.value == 2:
                        is_current_event_rejected = True
                        manual_reject_flag.value = False
                        rejection_details = {"rejection_time": datetime.now(), "rejection_type": "2"}
                        try:
                            route = getattr(shared_settings, 'manual_reject_route', None)
                        except Exception:
                            route = None
                        rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, -1, route))
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                            except Exception:
                                pass
                        with stats_lock:
                            total_rejections += 1
                            shared_rejection_counter.value = total_rejections
                            yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                        broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                        print("    [状态机]: 即时手动剔废触发！")

            elif machine_state == "WAITING_FOR_PANE":
                if current_total_panes > 0:
                    presence_streak += 1
                    if presence_streak >= ENTER_CONFIRM_FRAMES:
                        # 若上一片玻璃存在 NG 但处于滞后等待且最终未被剔废，应当此时补记产量并上传（不再等待滞后剔废）。
                        if can_late_reject.value and not is_current_event_rejected and current_pane_ng_buffer:
                            # 立即上传策略：该调用占位以便未来恢复批量上传
                            upload_current_pane_if_needed()
                            current_pane_ng_buffer.clear()
                            current_pane_reports.clear()
                            current_pane_folder = None
                            pane_ng_frame_counter = 0
                            with stats_lock:
                                total_yield += 1
                                shared_yield_counter.value = total_yield
                                yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                            broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                            can_late_reject.value = False
                            print("    [状态机]: 上一片未滞后剔废，已自动计入产量并上传。")
                        machine_state = "PANE_DETECTED"
                        machine_state_shared.value = 1
                        # 采集模式下：进入时统一递增 pane 序号并置为激活
                        try:
                            if getattr(shared_settings, 'data_collection_mode', False):
                                try:
                                    cur_seq = int(getattr(shared_settings, 'collection_pane_seq', 0) or 0)
                                except Exception:
                                    cur_seq = 0
                                shared_settings.collection_pane_seq = cur_seq + 1
                                shared_settings.collection_pane_active = True
                                shared_settings.collection_day_dir = datetime.now().strftime('%Y%m%d')
                                print(f"    [状态机]: 采集会话开启 -> pane{shared_settings.collection_pane_seq}")
                        except Exception:
                            pass
                        current_pane_ng_buffer.clear()
                        current_pane_reports.clear()
                        current_pane_folder = None
                        pane_ng_frame_counter = 0
                        max_complexity_snapshot.fill(0)
                        is_current_event_rejected = False
                        rejection_details = {}
                        saved_for_this_pane = False
                        can_late_reject.value = False
                        presence_streak = 0
                        absence_streak = 0
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_normal_state()
                            except Exception:
                                pass
                        print("--- [状态机]: 玻璃进入事件 ---")
                else:
                    presence_streak = 0
        except Exception as e:
            print(f"[状态机]: 处理结果时出错: {e}")