# --- START OF FILE state_machine.py ---
import time
import json
import os
import asyncio
import numpy as np
from queue import Empty
from datetime import datetime
import yield_manager
from server_comms import send_report_to_server, fetch_collection_id_from_server, fetch_initial_state_from_server, broadcast_rejections

def results_and_state_machine_thread(num_cameras, results_queue, connection_manager, stop_event, loop, shared_settings, counters, queues, flags, machine_state_shared, stats_lock, metadata, run_event_proxy, http_client, alarm_light_controller):
    print("[状态机线程]: 已启动。")
    
    # alarm_light_controller 实例现在通过函数参数直接传入
    
    STORAGE_PATH = shared_settings.storage_path
    # 修改：移除产量计数，仅保留剔废计数
    (shared_rejection_counter,) = counters
    (rejection_queue,) = queues
    (manual_reject_flag, shared_rejection_mode, can_late_reject) = flags
    (shared_collection_id, shared_user_id_auto, shared_user_id_manual) = metadata

    machine_state = "WAITING_FOR_PANE"
    machine_state_shared.value = 0 
    last_camera_states = np.zeros(num_cameras, dtype=np.int32)
    current_pane_ng_buffer = []
    last_pane_data = {}
    
    with stats_lock:
        last_reset_date_str, _legacy_yield, total_rejections = yield_manager.load_stats()
        shared_rejection_counter.value = total_rejections
    total_yield = 0  # 产量统计已废弃
    
    max_complexity_snapshot = np.zeros(num_cameras, dtype=np.int32)
    is_current_event_rejected = False
    is_ng_alarm_triggered_for_pane = False # 确保NG报警每片玻璃只触发一次
    rejection_details = {}
    saved_for_this_pane = False
    
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

    def save_and_upload_report(result_to_save, rejection_details_to_save):
        defects = result_to_save.get('defects', [])
        max_defect_size = max([get_defect_size(d) for d in defects] or [0])
        has_x_defect = any(d.get('type') == 'X' for d in defects)
        has_q_defect = any(d.get('type') == 'Q' for d in defects)
        
        size_part = ""
        if max_defect_size > 0:
            if max_defect_size >= 100: size_part = "100mm"
            elif max_defect_size >= 50: size_part = "50mm"
            elif max_defect_size >= 20: size_part = "20mm"
            else: size_part = "<20mm"
        label_parts = [p for p in [size_part, "X" if has_x_defect else "", "Q" if has_q_defect else ""] if p]
        final_size_label = ",".join(label_parts)
        
        raw_cid = shared_collection_id.value
        try:
            if isinstance(raw_cid, (bytes, bytearray)):
                collection_id = int(raw_cid.decode('utf-8', errors='ignore').strip() or -1)
            else:
                collection_id = int(raw_cid)
        except Exception:
            collection_id = -1
        user_id = shared_user_id_manual.value if rejection_details_to_save.get("rejection_type") == "2" else shared_user_id_auto.value
        
        report = {
            "collection_id": int(collection_id),
            "rejection_type": rejection_details_to_save.get("rejection_type", "unknown"),
            "rejection_time": rejection_details_to_save.get("rejection_time", datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
            "userId": user_id,
            "size_label": final_size_label,
            "image_status": result_to_save["image_status"],
            "defects": defects,
            "camera_index": result_to_save["camera_index"]
        }

        save_dir = os.path.join(STORAGE_PATH, datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(save_dir, exist_ok=True)
        base_name = f"{datetime.strptime(report['rejection_time'], '%Y-%m-%d %H:%M:%S').strftime('%H%M%S')}_{collection_id}_Cam{report['camera_index']}"
        image_buffer = result_to_save.get('annotated_image_buffer')

        if image_buffer:
            with open(os.path.join(save_dir, f"{base_name}.jpg"), 'wb') as f: f.write(image_buffer)
            send_report_to_server(report, image_buffer, shared_settings.upload_url, http_client, upload_timeout_s=getattr(shared_settings, 'http_upload_timeout_s', 15))
        with open(os.path.join(save_dir, f"{base_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4, default=str)
        print(f"    [状态机]: 已保存报告 (ID: {collection_id}), 尺寸标签: {final_size_label}")

    # Initial state fetch
    fetch_collection_id_from_server(shared_settings, shared_collection_id)
    initial_state = fetch_initial_state_from_server(shared_settings)
    if initial_state:
        shared_rejection_mode.value = initial_state.get("rejectionMode", 1)
        threshold_str = str(initial_state.get("rejectionThreshold", 20.0)).replace("mm", "").strip()
        shared_settings.max_defect_size_mm = float(threshold_str) if threshold_str else 20.0
        shared_user_id_auto.value = initial_state.get("algUserVO", {}).get("userId", 7)
        if initial_state.get("enable") == 1: 
            run_event_proxy.set()
            print("[状态机]: 从服务器获取到启用状态，设置运行事件。")
    else:
        print("[状态机]: 未能从服务器获取状态，保持当前运行状态。")
    
    while not stop_event.is_set():
        # Handle late manual rejection
        if machine_state == "WAITING_FOR_PANE" and manual_reject_flag.value and can_late_reject.value:
            manual_reject_flag.value = False; can_late_reject.value = False
            print(f"--- [状态机]: 检测到滞后手动剔废指令 ---")
            if last_pane_data and not last_pane_data.get("was_rejected_in_view"):
                best_result = find_best_ng_result(last_pane_data["ng_buffer"])
                if best_result:
                    save_and_upload_report(best_result, {"rejection_time": datetime.now(), "rejection_type": "2"})
                    # 剃废即刻触发红灯报警（与状态无关）
                    if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                        try:
                            alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                        except Exception:
                            pass
                    with stats_lock:
                        total_rejections += 1; shared_rejection_counter.value = total_rejections
                        yield_manager.save_stats(last_reset_date_str, 0, total_rejections)
                    broadcast_rejections(shared_settings, shared_collection_id, shared_rejection_counter, http_client)
                    print(f"    [状态机]: 滞后剔废计数+1, 今日总剔废: {total_rejections}")
            last_pane_data = {}

        try:
            result = results_queue.get(timeout=0.04)
            cam_index = result["camera_index"]
            ws_data = {k:v for k,v in result.items() if k != 'annotated_image_buffer'}
            try:
                asyncio.run_coroutine_threadsafe(connection_manager.broadcast(json.dumps(ws_data), cam_index), loop)
            except Exception:
                pass
            
            now_str = datetime.now().strftime('%Y-%m-%d')
            if now_str != last_reset_date_str:
                with stats_lock:
                    print(f"--- [状态机]: 日期已变更，正在保存 {last_reset_date_str} 的最终报告... ---")
                    yield_manager.save_daily_report(last_reset_date_str, 0, total_rejections)
                    total_rejections = 0
                    shared_rejection_counter.value = 0
                    last_reset_date_str = now_str
                    yield_manager.save_stats(now_str, 0, total_rejections)
            
            if cam_index >= len(last_camera_states):
                last_camera_states = np.pad(last_camera_states, (0, cam_index - len(last_camera_states) + 1), 'constant')
                max_complexity_snapshot = np.pad(max_complexity_snapshot, (0, cam_index - len(max_complexity_snapshot) + 1), 'constant')
            
            last_camera_states[cam_index] = result['state_code']
            current_total_panes = np.sum(last_camera_states)
            
            # State machine logic with debounce and integrated alarm control
            if machine_state == "PANE_DETECTED":
                # 离开事件去抖
                if current_total_panes == 0:
                    absence_streak += 1
                    if absence_streak >= LEAVE_CONFIRM_FRAMES:
                        print(f"--- [状态机]: 玻璃离开事件 ---")
                        machine_state = "WAITING_FOR_PANE"; machine_state_shared.value = 0
                        absence_streak = 0; presence_streak = 0
                        
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                if is_current_event_rejected:
                                    alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                                else:
                                    alarm_light_controller.set_normal_state()
                            except Exception:
                                pass

                        # 产量统计移除

                        if is_current_event_rejected and not saved_for_this_pane:
                            best_result = find_best_ng_result(current_pane_ng_buffer)
                            if best_result: save_and_upload_report(best_result, rejection_details)
                        
                        last_pane_data = {}
                        can_late_reject.value = False
                        if not is_current_event_rejected and current_pane_ng_buffer and shared_rejection_mode.value == 2:
                            last_pane_data = {"ng_buffer": list(current_pane_ng_buffer), "was_rejected_in_view": False}
                            can_late_reject.value = True
                            print(f"    [状态机]: 上一片玻璃存在NG，已暂存信息，等待可能的滞后剔废指令。")
                    # 若未达到阈值，暂不处理业务逻辑
                    continue
                else:
                    absence_streak = 0
                if result['image_status'] == 'NG': 
                    current_pane_ng_buffer.append(result)
                    if not is_ng_alarm_triggered_for_pane:
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_ng_detected_state(shared_settings.ng_buzz_duration_s)
                            except Exception:
                                pass
                        is_ng_alarm_triggered_for_pane = True

                if np.sum(last_camera_states) > np.sum(max_complexity_snapshot):
                    max_complexity_snapshot = last_camera_states.copy()
                
                if not is_current_event_rejected:
                    if shared_rejection_mode.value == 1 and result.get('should_reject', False):
                        is_current_event_rejected = True; saved_for_this_pane = True
                        rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                        save_and_upload_report(result, rejection_details)
                        rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, cam_index))
                        # 剃废即刻触发红灯报警（与状态无关）
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                            except Exception:
                                pass
                        with stats_lock:
                            total_rejections += 1; shared_rejection_counter.value = total_rejections
                            yield_manager.save_stats(last_reset_date_str, 0, total_rejections)
                        broadcast_rejections(shared_settings, shared_collection_id, shared_rejection_counter, http_client)
                        print(f"    [状态机]: 自动剔废触发！")
                    elif manual_reject_flag.value and shared_rejection_mode.value == 2:
                        is_current_event_rejected = True; manual_reject_flag.value = False
                        rejection_details = {"rejection_time": datetime.now(), "rejection_type": "2"}
                        rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, -1))
                        # 剃废即刻触发红灯报警（与状态无关）
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                            except Exception:
                                pass
                        with stats_lock:
                            total_rejections += 1; shared_rejection_counter.value = total_rejections
                            yield_manager.save_stats(last_reset_date_str, 0, total_rejections)
                        broadcast_rejections(shared_settings, shared_collection_id, shared_rejection_counter, http_client)
                        print(f"    [状态机]: 即时手动剔废触发！")

            elif machine_state == "WAITING_FOR_PANE":
                # 进入事件去抖
                if current_total_panes > 0:
                    presence_streak += 1
                    if presence_streak >= ENTER_CONFIRM_FRAMES:
                        machine_state = "PANE_DETECTED"; machine_state_shared.value = 1
                        current_pane_ng_buffer.clear(); max_complexity_snapshot.fill(0)
                        is_current_event_rejected = False; rejection_details = {}; saved_for_this_pane = False
                        is_ng_alarm_triggered_for_pane = False
                        can_late_reject.value = False
                        presence_streak = 0; absence_streak = 0
                        
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_normal_state()
                            except Exception:
                                pass
                        
                        print(f"--- [状态机]: 玻璃进入事件 ---")
                else:
                    presence_streak = 0
        except Empty: 
            pass
        except Exception as e:
            print(f"[状态机]: 处理结果时出错: {e}")