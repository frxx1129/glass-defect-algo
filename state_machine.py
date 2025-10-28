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
    ENTER_CONFIRM_FRAMES = int(getattr(shared_settings, 'enter_confirm_frames', 1))
    LEAVE_CONFIRM_FRAMES = int(getattr(shared_settings, 'leave_confirm_frames', 4))
    presence_streak = 0
    absence_streak = 0
    # 新增：玻璃进入后的最大持续时间（秒），超时强制退出
    try:
        pane_max_duration_s = float(getattr(shared_settings, 'pane_max_duration_s', 5.0) or 5.0)
    except Exception:
        pane_max_duration_s = 5.0
    pane_enter_time_s = None

    # 自动分路：每片玻璃内的 NG 汇聚与一次性触发
    auto_ng_cams = set()           # 出现NG的相机集合（逻辑索引）
    last_result_by_cam = {}        # 最近一帧NG结果（含缺陷坐标）的引用
    auto_first_ng_ts_ms = None     # 第一次检测到NG的时间（毫秒）

    def _decide_route_for_auto(expected_cams: int, ng_cams: set) -> str | None:
        """依据规则决定路由：返回 'left' | 'mid' | 'right' | 'all' 或 None(尚未能决定)。
        - 公共：len(ng_cams) >= 3 -> 'all'
        - 4相机：{1,2} -> 'mid'；包含{2,3} -> 'left'；包含{0,1} -> 'right'；单相机2/3->left，0/1->right
        - 5相机：{1,2}或{2,3} -> 'mid'；{3,4} -> 'left'；{0,1} -> 'right'；仅{2} -> None（坐标判定）；单相机3/4->left，0/1->right
        """
        cams = set(ng_cams)
        if len(cams) >= 3:
            return 'all'
        if expected_cams == 4:
            if {1, 2}.issubset(cams):
                return 'mid'
            if {2, 3}.issubset(cams):
                return 'left'
            if {0, 1}.issubset(cams):
                return 'right'
            if cams.issubset({2, 3}) and len(cams) >= 1:
                return 'left'
            if cams.issubset({0, 1}) and len(cams) >= 1:
                return 'right'
            return None
        if expected_cams == 5:
            if {1, 2}.issubset(cams) or {2, 3}.issubset(cams):
                return 'mid'
            if {3, 4}.issubset(cams):
                return 'left'
            if {0, 1}.issubset(cams):
                return 'right'
            if cams == {2}:
                return None
            if cams.issubset({3, 4}) and len(cams) >= 1:
                return 'left'
            if cams.issubset({0, 1}) and len(cams) >= 1:
                return 'right'
            return None
        return None

    def _decide_left_right_by_coord_for_cam2(result_obj: dict) -> str | None:
        """仅在5相机且只有cam2为NG时调用。根据缺陷的x坐标决定左右：小->right，大->left。"""
        try:
            W = int(getattr(shared_settings, 'cam_width', 0) or 0)
            if W <= 0:
                return None
            cx_values = []
            for d in result_obj.get('defects', []):
                loc = d.get('location', {})
                if isinstance(loc, dict):
                    if 'x' in loc and 'width' in loc:
                        try:
                            cx_values.append(float(loc.get('x', 0)) + float(loc.get('width', 0)) / 2.0)
                        except Exception:
                            pass
                c = d.get('center')
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    try:
                        cx_values.append(float(c[0]))
                    except Exception:
                        pass
            if not cx_values:
                return None
            avg_cx = sum(cx_values) / len(cx_values)
            return 'right' if avg_cx < (W / 2.0) else 'left'
        except Exception:
            return None
    
    def get_defect_size(defect):
        defect_type = defect.get('type')
        if defect_type in ['B', 'L']:
            rect = defect.get('location', {})
            return max(rect.get('length_mm', 0), rect.get('width_mm', 0))
        return 0
    
    def find_best_ng_result(ng_buffer):
        if not ng_buffer: return None
        return max(ng_buffer, key=lambda r: max([get_defect_size(d) for d in r.get('defects', [])] or [-1]))

    def _count_near_vertical_per_cam(result_obj: dict) -> int:
        """统计该相机返回的各 ROI 中近竖直主边的数量总和（按 roi_report['near_vertical_line_count'] 汇总）。"""
        try:
            rois = result_obj.get('rois', [])
            if isinstance(rois, list) and rois:
                return int(sum(int(r.get('near_vertical_line_count', 0) or 0) for r in rois))
        except Exception:
            pass
        return 0

    def _decide_reject_marks_by_vertical_counts(line_name: str, cam_vertical_counts: dict, expected_cams: int) -> list[int] | None:
        """根据每相机近竖直主边数量估计切割（0=一整块，2=一切二，3=一切三），并返回要触发的剔废mark列表；
        不同产线以配置中的 lineName 区分（"Line1" / "Line2" / "Line3"）。
        - 若所有相机竖直到1条或更少：认为一整块（0），若需要剔废则触发 mark 0（通道映射由配置决定）。
        - 若恰有单个相机竖直=2条：认为一切二，按该相机索引划分左右，触发左右两侧的 mark（line1/3：左1右3；line2：左1右3 或 左2右4 视现场接线习惯，这里提供两组供配置映射）。
        - 若恰有两个相机竖直=2条：认为一切三，按位置取中间相机为中线，两边划分三路；
          line1/3 -> 触发 1 或 2 或 3； line2 -> 触发 1 或 2 或 3 或 4（根据映射组成列表）。
        具体 mark->通道由 shared_settings.rejection_mark_to_channel 决定。
        """
        # 收集两条竖直的相机索引
        two_line_cams = sorted([ci for ci, cnt in cam_vertical_counts.items() if cnt >= 2])
        max_cnt = max(cam_vertical_counts.values()) if cam_vertical_counts else 0

        # 一整块：所有相机 0或1条
        if max_cnt <= 1:
            return [0]

        # 一切二：恰好一个相机两条
        if len(two_line_cams) == 1:
            mid_cam = two_line_cams[0]
            # 左右划分：以中间索引为分界（小于mid为右，大于mid为左，与相机坐标系一致）
            # 最终触发由服务器配置映射到具体通道
            if str(line_name).strip() == "Line2":
                # 产线2支持双侧两通道，使用 1/2 表示左侧组合，3/4 表示右侧组合（实际映射由 config 决定）
                return [1, 3]  # 由调用处按左右拆分具体触发
            else:
                return [1, 3]

        # 一切三：恰好两个相机两条（取较小/较大为两次切割位置，形成三段）
        if len(two_line_cams) == 2:
            left_idx, right_idx = two_line_cams[0], two_line_cams[1]
            # 触发三路，具体通道映射交由配置
            if str(line_name).strip() == "Line2":
                return [1, 2, 3, 4]  # line2 可用四路，实际触发时按位置选择其中三路
            else:
                return [1, 2, 3]

        # 其它复杂情况：兜底全部
        return [0]

    def build_report_for_frame(result_obj, rejection_details_to_save):
        defects = result_obj.get('defects', [])
        def primary_size(d):
            # 修改：按缺陷的“最短边”来确定尺寸标签
            loc = d.get('location', {})
            length = float(loc.get('length_mm', 0) or 0)
            width = float(loc.get('width_mm', 0) or 0)
            if length <= 0 and width <= 0:
                return 0.0
            return min(length, width)
        # 汇总：取所有缺陷“最短边”的最大值，作为 size_label 的数值
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
        shared_user_id_auto.value = initial_state.get("algUserVO", {}).get("id", 7)
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
        # 手动剔废（等待新玻璃状态下也可随时触发）：始终进行剔废控制；仅在既可滞后且确有NG时才上传
        if machine_state == "WAITING_FOR_PANE" and manual_reject_flag.value:
            manual_reject_flag.value = False
            # 硬件剔废始终执行
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
            # 计入一次剔废
            with stats_lock:
                total_rejections += 1
                shared_rejection_counter.value = total_rejections
                yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
            broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
            # 仅当处于可滞后窗口且确有上一片NG缓存时执行上传与回滚，其余情况不上传
            if can_late_reject.value and current_pane_ng_buffer and not is_current_event_rejected:
                can_late_reject.value = False
                rejection_details = {"rejection_time": datetime.now(), "rejection_type": "2"}
                for rpt in current_pane_reports:
                    rpt['rejection_type'] = '2'
                    rpt['rejection_time'] = rejection_details['rejection_time'].strftime('%Y-%m-%d %H:%M:%S')
                send_reports_batch_to_server(current_pane_reports, [r.get('annotated_image_buffer') for r in current_pane_ng_buffer], shared_settings.upload_url, http_client, upload_timeout_s=getattr(shared_settings, 'http_upload_timeout_s', 30))
                # 回滚上一片产量（若已计）
                with stats_lock:
                    if total_yield > 0:
                        total_yield -= 1
                        shared_yield_counter.value = total_yield
                        yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                print(f"    [状态机]: 手动剔废完成并上传。本片NG帧数量: {len(current_pane_ng_buffer)}")
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
                    # 先刷新 collection_id，再进行当天产量重置
                    try:
                        fetch_collection_id_from_server(shared_settings, shared_collection_id)
                    except Exception as e:
                        print(f"[状态机]: 刷新 collection_id 失败: {e}")
                    yield_manager.save_daily_report(last_reset_date_str, total_yield, total_rejections)
                    total_rejections = 0
                    total_yield = 0
                    shared_rejection_counter.value = 0
                    shared_yield_counter.value = 0
                    last_reset_date_str = now_str
                    yield_manager.save_stats(now_str, total_yield, total_rejections)
                # （可选）立即广播一次新的统计（包含新的 collection_id）
                try:
                    broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                except Exception as e:
                    print(f"[状态机]: 日期变更后广播统计失败: {e}")
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
                            # 重置自动分路聚合状态
                            auto_ng_cams.clear(); last_result_by_cam.clear(); auto_first_ng_ts_ms = None
                        pane_enter_time_s = None
                        continue
                else:
                    absence_streak = 0

                # 超时强制退出：玻璃处于检测状态超过 pane_max_duration_s
                try:
                    if pane_enter_time_s is not None and (time.time() - pane_enter_time_s) >= pane_max_duration_s:
                        print(f"--- [状态机]: 玻璃检测超时（>{pane_max_duration_s:.1f}s），强制退出 ---")
                        machine_state = "WAITING_FOR_PANE"
                        machine_state_shared.value = 0
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
                        # 超时强退时，与“离开事件”一致的上传/计数策略
                        if shared_rejection_mode.value == 2 and not is_current_event_rejected and current_pane_ng_buffer:
                            can_late_reject.value = True
                            print("    [状态机]: 等待可能的滞后剔废（模式2），暂不上传。")
                        else:
                            upload_current_pane_if_needed()
                            current_pane_ng_buffer.clear()
                            current_pane_reports.clear()
                            current_pane_folder = None
                            pane_ng_frame_counter = 0
                            can_late_reject.value = False
                            if not is_current_event_rejected:
                                with stats_lock:
                                    total_yield += 1
                                    shared_yield_counter.value = total_yield
                                    yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                                broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                            is_current_event_rejected = False
                            auto_ng_cams.clear(); last_result_by_cam.clear(); auto_first_ng_ts_ms = None
                        pane_enter_time_s = None
                        continue
                except Exception:
                    pass

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
                    # 记录自动分路用的NG集合、结果引用以及第一帧NG时间
                    try:
                        cam_i = int(result['camera_index'])
                        auto_ng_cams.add(cam_i)
                        last_result_by_cam[cam_i] = result
                        if auto_first_ng_ts_ms is None:
                            auto_first_ng_ts_ms = int(time.time() * 1000)
                    except Exception:
                        pass

                if np.sum(last_camera_states) > np.sum(max_complexity_snapshot):
                    max_complexity_snapshot = last_camera_states.copy()

                if not is_current_event_rejected:
                    # 自动剔废：汇聚后一次性分路触发
                    if shared_rejection_mode.value == 1 and result.get('should_reject', False):
                        try:
                            hold_ms = int(getattr(shared_settings, 'auto_route_decision_hold_ms', 120) or 120)
                        except Exception:
                            hold_ms = 120
                        now_ms = int(time.time() * 1000)
                        can_decide_time = (auto_first_ng_ts_ms is not None) and (now_ms - auto_first_ng_ts_ms >= hold_ms)
                        expected = int(getattr(shared_settings, 'expected_cameras', num_cameras) or num_cameras)
                        route = _decide_route_for_auto(expected, auto_ng_cams)
                        # 新增：基于竖直线统计的多路剔废逻辑
                        try:
                            # 汇总每个相机的“近竖直主边”数量
                            vertical_counts = {}
                            for ci, res in last_result_by_cam.items():
                                vertical_counts[ci] = _count_near_vertical_per_cam(res)
                            if vertical_counts:
                                line_name = str(getattr(shared_settings, 'lineName', 'UNKNOWN')).strip()
                                marks = _decide_reject_marks_by_vertical_counts(line_name, vertical_counts, expected)
                                if marks:
                                    is_current_event_rejected = True
                                    saved_for_this_pane = True
                                    rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                                    # 将 marks 列表直接作为 route 传入，由 rejection_control 逐一触发
                                    rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, cam_index, marks))
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
                                    print(f"    [状态机]: 基于竖直线统计剔废触发，marks={marks}，counts={vertical_counts}")
                                    continue
                        except Exception as e:
                            print(f"[状态机]: 竖直线分路逻辑异常: {e}")
                        if route is None and can_decide_time:
                            # 特例：5路且仅cam2，根据坐标判定
                            if expected == 5 and auto_ng_cams == {2}:
                                base_res = last_result_by_cam.get(2)
                                if base_res:
                                    route = _decide_left_right_by_coord_for_cam2(base_res)
                        # 仍未能决定，但已到达hold窗口，避免漏剔的兜底
                        if route is None and can_decide_time and len(auto_ng_cams) >= 1:
                            route = 'all' if len(auto_ng_cams) >= 3 else None
                        if route is not None:
                            is_current_event_rejected = True
                            saved_for_this_pane = True
                            rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                            rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, cam_index, route))
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
                            print(f"    [状态机]: 自动剔废触发，路由={route}，NG相机={sorted(list(auto_ng_cams))}")
                    # 即时手动剔废：任何时候都可触发。始终执行硬件动作；仅在有NG缓存时才追加上传
                    elif manual_reject_flag.value:
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
                        # 可选上传：只有当当前片有NG缓存时才上传
                        if current_pane_ng_buffer and current_pane_reports:
                            send_reports_batch_to_server(current_pane_reports, [r.get('annotated_image_buffer') for r in current_pane_ng_buffer], shared_settings.upload_url, http_client, upload_timeout_s=getattr(shared_settings, 'http_upload_timeout_s', 30))

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
                        # 重置自动分路聚合状态
                        auto_ng_cams.clear(); last_result_by_cam.clear(); auto_first_ng_ts_ms = None
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_normal_state()
                            except Exception:
                                pass
                        pane_enter_time_s = time.time()
                        print("--- [状态机]: 玻璃进入事件 ---")
                else:
                    presence_streak = 0
        except Exception as e:
            print(f"[状态机]: 处理结果时出错: {e}")