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
    # 手动剔废“模式标记”开关：
    # - 当在等待或检测过程中触发手动剔废时置为 True；
    # - 该标记对“当前这片玻璃”的所有 NG 报告生效（报告中标明剔废模式=手动）；
    # - 若在等待阶段触发手动剔废，则对“下一片玻璃”生效，直到该片离开为止；
    # - 在玻璃离开/超时强退时重置为 False。
    manual_reject_active_mode = False
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
    # 默认帧数略降到3，在引入“进入最短时间”门槛后整体更稳且响应更快
    ENTER_CONFIRM_FRAMES = int(getattr(shared_settings, 'enter_confirm_frames', 3))
    LEAVE_CONFIRM_FRAMES = int(getattr(shared_settings, 'leave_confirm_frames', 3))
    presence_streak = 0
    absence_streak = 0
    presence_start_time_s = None  # 不再用于进入判定，仅用于可能的扩展/调试
    # 新增：玻璃进入后的最大持续时间（秒），超时强制退出
    try:
        pane_max_duration_s = float(getattr(shared_settings, 'pane_max_duration_s', 10.0) or 10.0)
    except Exception:
        pane_max_duration_s = 10.0
    # 进入后的最短停留时间（秒）：进入判定不等待此时长；仅用于在离开与自动剔废前校验最小驻留
    try:
        enter_min_time_s = float(getattr(shared_settings, 'enter_min_time_s', 1.5) or 1.5)
    except Exception:
        enter_min_time_s = 1.5
    pane_enter_time_s = None
    # 记录触发进入的首个相机与ROI（仅用于打印）
    first_presence_cam_idx = None
    first_presence_roi_idx = None

    # 自动分路：每片玻璃内的 NG 汇聚与一次性触发
    auto_ng_cams = set()           # 出现NG的相机集合（逻辑索引）
    last_result_by_cam = {}        # 最近一帧NG结果（含缺陷坐标）的引用
    auto_first_ng_ts_ms = None     # 第一次检测到NG的时间（毫秒）
    # 新增：按整片玻璃周期统计的“竖直边总数”的最大值（跨相机求和，跨时刻取最大）
    pane_max_total_vertical_count = 0

    def _get_line_mark_info(shared_settings):
        """获取当前产线的标记映射信息：
        - return (max_mark, line_map) 其中 max_mark 为该 line 下的最大标记（int），line_map 为 {str(mark): channel}
        - 若未配置，则回退为 max_mark=3 且 line_map={}（实际触发层将按 mark+1 映射）
        """
        try:
            line_name = str(getattr(shared_settings, 'lineName', 'Line1'))
        except Exception:
            line_name = 'Line1'
        try:
            all_map = getattr(shared_settings, 'rejection_mark_to_channel', {})
        except Exception:
            all_map = {}
        line_map = {}
        if isinstance(all_map, dict):
            line_map = all_map.get(line_name) or {}
        max_mark = 3
        try:
            keys = [int(k) for k in line_map.keys()] if isinstance(line_map, dict) else []
            if keys:
                max_mark = max(keys)
        except Exception:
            pass
        return max_mark, line_map

    def _decide_route_marks_for_auto(expected_cams: int, ng_cams: set, shared_settings) -> list[int] | None:
        """依据规则决定'标记'(0..max_mark) 列表，用于剔废触发；不再返回语义字符串。
        约定：
        - 0 表示“整片撤清”；
        - 1 表示偏左端；2 表示中间；最大标记(max_mark，通常3或4)表示偏右端；
        - 实际通道由 rejection_control 中按 lineName 映射决定。
        """
        cams = set(ng_cams)
        max_mark, _ = _get_line_mark_info(shared_settings)
        center = (expected_cams - 1) / 2.0 if expected_cams > 0 else 1.5
        if len(cams) >= 3:
            return [0]
        if expected_cams == 4:
            # 0,1,2,3 索引；对称两切常见于 cam1 或 cam2 出现双竖直
            # 由外部“竖直线统计”逻辑优先判断，这里仅按 NG 分布兜底
            if {1, 2}.issubset(cams):
                return [2]
            if {2, 3}.issubset(cams):
                return [1]
            if {0, 1}.issubset(cams):
                return [max_mark]
            if cams.issubset({2, 3}) and len(cams) >= 1:
                return [1]
            if cams.issubset({0, 1}) and len(cams) >= 1:
                return [max_mark]
            return None
        if expected_cams == 5:
            if {1, 2}.issubset(cams) or {2, 3}.issubset(cams):
                return [2]
            if {3, 4}.issubset(cams):
                return [1]
            if {0, 1}.issubset(cams):
                return [max_mark]
            if cams == {2}:
                # 中间单相机，需用坐标另判左右，外层已处理；此处返回 None 等待坐标判定
                return None
            if cams.issubset({3, 4}) and len(cams) >= 1:
                return [1]
            if cams.issubset({0, 1}) and len(cams) >= 1:
                return [max_mark]
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
        """用于挑选代表帧：按缺陷的尺寸（最短边或已提供的长/短边）估算大小。
        将 E/Q/B/L 一并纳入（E 与 Q 与 B/L 一样，依据 location 中的 mm 尺寸）。"""
        try:
            defect_type = str(defect.get('type', '')).upper()
            if defect_type in ['B', 'L', 'E', 'Q']:
                loc = defect.get('location', {}) if isinstance(defect.get('location'), dict) else {}
                length = float(loc.get('length_mm', 0) or 0)
                width  = float(loc.get('width_mm', 0) or 0)
                # 代表帧挑选上，倾向用较大的度量；若两者皆有，取 max
                if length > 0 or width > 0:
                    return max(length, width)
        except Exception:
            pass
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

    def _estimate_piece_count(cam_vertical_counts: dict) -> int:
        """根据每相机的近竖直线数量估计切割片数（1~4）。
        粗略规则：cut_count = min( len([ci for cnt>=2]), 3 ); piece_count = cut_count + 1。
        """
        try:
            indicators = [ci for ci, cnt in cam_vertical_counts.items() if int(cnt or 0) >= 2]
            cut_count = min(len(indicators), 3)
            return cut_count + 1 if cut_count >= 1 else 1
        except Exception:
            return 1

    def _gather_defect_centers_for_cam(result_obj: dict) -> list[float]:
        """提取单相机内缺陷的 x 中心（像素）。优先 location(x,width)，否则 center[0]；无法得到返回空。"""
        xs = []
        try:
            for d in result_obj.get('defects', []):
                loc = d.get('location', {}) if isinstance(d.get('location'), dict) else {}
                if 'x' in loc and 'width' in loc:
                    try:
                        xs.append(float(loc.get('x', 0) or 0) + float(loc.get('width', 0) or 0) / 2.0)
                        continue
                    except Exception:
                        pass
                c = d.get('center')
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    try:
                        xs.append(float(c[0]))
                    except Exception:
                        pass
        except Exception:
            pass
        return xs

    def _decide_marks_by_pieces_and_positions(cam_vertical_counts: dict, last_result_by_cam: dict, expected_cams: int, shared_settings, piece_count_override: int | None = None) -> list[int] | None:
        """
        依据估计的切片数（1~4）以及缺陷位置，返回需要触发的'标记'集合：
        - 1 切 2 块 -> 允许 {0,1,2}，优先给出属于缺陷所在块的标记（1 或 2）；
          中间相机(5路的cam2)用相机内 X 坐标区分左右（小->1，大->2）。
        - 2 切 3 块 -> 允许 {0,1,2,3}，按全局位置映射至 1/2/3；
        - 3 切 4 块 -> 允许 {0,1,2,3,4}，按全局位置映射至 1..4；
        - 若无法判定，兜底 [0]。
        """
        piece_count = piece_count_override if isinstance(piece_count_override, int) and piece_count_override >= 1 else _estimate_piece_count(cam_vertical_counts)
        if piece_count <= 1:
            return [0]

        # 计算全局中心索引（相机序号）
        n = int(expected_cams or 0)
        center = (n - 1) / 2.0 if n > 0 else 1.5

        # 收集“每个缺陷”的全局位置参数，简化为 (cam_index, x_center_px or None)
        samples: list[tuple[int, float | None]] = []
        for ci, res in last_result_by_cam.items():
            try:
                xs = _gather_defect_centers_for_cam(res)
                if xs:
                    for x in xs:
                        samples.append((int(ci), float(x)))
            except Exception:
                continue

        if not samples:
            return [0]

        # 获取相机宽度（用于中间相机的细分；若无则仅用相机序号粗分）
        try:
            cam_w = float(getattr(shared_settings, 'cam_width', 0) or 0)
        except Exception:
            cam_w = 0.0

        marks: set[int] = set()
        max_mark, _ = _get_line_mark_info(shared_settings)

        # 将全局范围 [0,1] 均分为 piece_count 份；边界位于 i/piece_count
        def _global_piece_index(ci: int, x_px: float | None) -> int:
            # 计算全局位置 g ∈ [0,1]：g ≈ (ci + local_u)/n
            if n <= 0:
                return 1
            if x_px is not None and cam_w > 1e-6:
                local_u = min(1.0, max(0.0, x_px / cam_w))
            else:
                # 无 x 或无 cam_width：只按相机索引粗分
                local_u = 0.5
            g = (float(ci) + local_u) / float(n)
            idx = int(np.floor(g * piece_count)) + 1
            idx = min(max(1, idx), piece_count)
            return idx

        for ci, x in samples:
            if piece_count == 2:
                if ci < center:
                    marks.add(1)
                elif ci > center:
                    marks.add(2)
                else:
                    # 中间相机：按 x 分左右
                    if x is not None and cam_w > 1e-6:
                        marks.add(1 if x < (cam_w / 2.0) else 2)
                    else:
                        marks.add(1)
            elif piece_count in (3, 4):
                marks.add(_global_piece_index(ci, x))
            else:
                marks.add(1)

        # 将标记限制在 [1..max_mark] 范围内（0 保留为整片撤清，非此处产生）
        final_marks = sorted({m for m in marks if 1 <= m <= max_mark})
        return final_marks if final_marks else [0]

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
        # 仅在存在 X 型时追加 'X' 标签；E 不追加字母，仅以尺寸参与 size_label
        size_label_parts = []
        if max_defect_size > 0:
            size_label_parts.append(f"{int(round(max_defect_size))}mm")
        if any(d.get('type') == 'X' for d in defects):
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
        # 仅调整 rejection_type 的标记方法与位置：
        # - 若当前处于“手动剔废模式窗口”（manual_reject_active_mode=True），则本帧报告的 rejection_type 直接标记为 '2'
        # - 否则沿用传入的临时值（若无则为 '0'）
        try:
            final_rej_type = '2' if manual_reject_active_mode else str(rejection_details_to_save.get('rejection_type', '0'))
        except Exception:
            final_rej_type = '0'
        # userId 仍按最终 rejection_type 决定
        user_id = shared_user_id_manual.value if str(final_rej_type) == '2' else shared_user_id_auto.value
        return {
            'collection_id': int(collection_id),
            'rejection_type': final_rej_type,
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
            # 同步：在本地保存未标注的原始整帧（仅在NG上传时保存）
            try:
                raw_buf = res.get('raw_image_buffer')
                if raw_buf:
                    ts = float(res.get('timestamp', time.time()) or time.time())
                    cam_idx = int(res.get('camera_index', -1) or -1)
                    day_dir = time.strftime('%Y-%m-%d', time.localtime(ts))
                    out_dir = os.path.join(STORAGE_PATH, day_dir, 'original')
                    os.makedirs(out_dir, exist_ok=True)
                    ms = int(ts * 1000)
                    out_name = f"cam{cam_idx}_ts{ms}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    # 直接写入 JPEG 字节
                    with open(out_path, 'wb') as f:
                        f.write(raw_buf)
            except Exception as e:
                print(f"[状态机]: 保存未标注原图失败: {e}")
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
    
    def _soft_reset_state_machine():
        nonlocal machine_state, last_camera_states, current_pane_ng_buffer, current_pane_reports, current_pane_folder
        nonlocal pane_ng_frame_counter, max_complexity_snapshot, is_current_event_rejected, manual_reject_active_mode
        nonlocal rejection_details, saved_for_this_pane, presence_streak, absence_streak, pane_enter_time_s
        nonlocal auto_ng_cams, last_result_by_cam, auto_first_ng_ts_ms, pane_max_total_vertical_count
        try:
            machine_state = "WAITING_FOR_PANE"
            machine_state_shared.value = 0
        except Exception:
            pass
        try:
            last_camera_states[:] = 0
        except Exception:
            pass
        try:
            current_pane_ng_buffer.clear(); current_pane_reports.clear()
        except Exception:
            pass
        current_pane_folder = None
        pane_ng_frame_counter = 0
        try:
            max_complexity_snapshot.fill(0)
        except Exception:
            pass
        is_current_event_rejected = False
        manual_reject_active_mode = False
        rejection_details = {}
        saved_for_this_pane = False
        presence_streak = 0
        absence_streak = 0
        pane_enter_time_s = None
        first_presence_cam_idx = None
        first_presence_roi_idx = None
        try:
            auto_ng_cams.clear(); last_result_by_cam.clear()
        except Exception:
            pass
        auto_first_ng_ts_ms = None
        pane_max_total_vertical_count = 0
        # 灯光恢复为正常状态
        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
            try:
                alarm_light_controller.set_normal_state()
            except Exception:
                pass
        # 采集模式：重置跨相机 pane 激活标记
        try:
            if getattr(shared_settings, 'data_collection_mode', False):
                shared_settings.collection_pane_active = False
        except Exception:
            pass

    while not stop_event.is_set():
        # 主进程请求软重置：清空状态、回到等待状态
        try:
            if bool(getattr(shared_settings, 'request_state_machine_reset', False)):
                _soft_reset_state_machine()
                setattr(shared_settings, 'request_state_machine_reset', False)
        except Exception:
            pass
        # 手动剔废（等待新玻璃状态下也可随时触发）：始终进行剔废控制；不再支持滞后剔废
        if machine_state == "WAITING_FOR_PANE" and manual_reject_flag.value:
            manual_reject_flag.value = False
            # 手动剔废“模式标记”自本次触发起生效，直到下一次玻璃离开为止
            manual_reject_active_mode = True
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
            # 避免广播二进制缓冲区
            ws_data = {k: v for k, v in result.items() if k not in ('annotated_image_buffer', 'raw_image_buffer')}
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
                    # 在达到“进入后的最短持续时间”之前，禁止离开（强制保持检测状态）
                    try:
                        dwell_s = (time.time() - pane_enter_time_s) if pane_enter_time_s is not None else 0.0
                    except Exception:
                        dwell_s = 0.0
                    if dwell_s < enter_min_time_s:
                        # 仍在最短持续时间窗口内：不允许离开，重置离开去抖
                        absence_streak = 0
                        continue

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
                        # 玻璃离开时，重置手动剔废模式标记
                        manual_reject_active_mode = False

                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                if is_current_event_rejected:
                                    alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                                else:
                                    alarm_light_controller.set_normal_state()
                            except Exception:
                                pass

                        # 立即上传并结算；不支持滞后剔废
                        upload_current_pane_if_needed()
                        # 上传后清理
                        current_pane_ng_buffer.clear()
                        current_pane_reports.clear()
                        current_pane_folder = None
                        pane_ng_frame_counter = 0
                        can_late_reject.value = False
                        # 若没有被剔废，计入产量
                        if not is_current_event_rejected:
                            with stats_lock:
                                total_yield += 1
                                shared_yield_counter.value = total_yield
                                yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                            broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                        is_current_event_rejected = False
                        # 重置自动分路聚合状态
                        auto_ng_cams.clear(); last_result_by_cam.clear(); auto_first_ng_ts_ms = None
                        pane_max_total_vertical_count = 0
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
                        # 超时强退也视作一次“离开”——重置手动模式标记
                        manual_reject_active_mode = False
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                if is_current_event_rejected:
                                    alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                                else:
                                    alarm_light_controller.set_normal_state()
                            except Exception:
                                pass
                        # 超时强退时也立即上传并结算；不支持滞后剔废
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
                        pane_max_total_vertical_count = 0
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
                    # 仅当进入后持续时间达到 enter_min_time_s 才允许触发自动剔废，避免瞬时误触发
                    dwell_ok = (pane_enter_time_s is not None) and ((time.time() - pane_enter_time_s) >= enter_min_time_s)
                    if shared_rejection_mode.value == 1 and result.get('should_reject', False) and dwell_ok:
                        try:
                            hold_ms = int(getattr(shared_settings, 'auto_route_decision_hold_ms', 120) or 120)
                        except Exception:
                            hold_ms = 120
                        now_ms = int(time.time() * 1000)
                        can_decide_time = (auto_first_ng_ts_ms is not None) and (now_ms - auto_first_ng_ts_ms >= hold_ms)
                        expected = int(getattr(shared_settings, 'expected_cameras', num_cameras) or num_cameras)
                        # 新：基于竖直线统计的'标记'决策（0 起始）
                        # 新增：基于竖直线统计的多路剔废逻辑
                        try:
                            # 汇总每个相机的“近竖直主边”数量
                            vertical_counts = {}
                            for ci, res in last_result_by_cam.items():
                                vertical_counts[ci] = _count_near_vertical_per_cam(res)
                            if vertical_counts:
                                # 新规则：跨相机求和，并在整个玻璃周期内取最大值
                                current_total_vertical = int(sum(int(v or 0) for v in vertical_counts.values()))
                                if current_total_vertical > pane_max_total_vertical_count:
                                    pane_max_total_vertical_count = current_total_vertical
                                # 将“最大竖直边总数”换算为切片数：2->1片，4->2片，6->3片，最大4片
                                try:
                                    piece_count_override = max(1, min(4, int(round(pane_max_total_vertical_count / 2.0))))
                                except Exception:
                                    piece_count_override = 1
                                # 结合缺陷位置计算应触发的标记集合
                                marks = _decide_marks_by_pieces_and_positions(
                                    vertical_counts, last_result_by_cam, expected, shared_settings,
                                    piece_count_override=piece_count_override
                                )
                                if marks:
                                    is_current_event_rejected = True
                                    saved_for_this_pane = True
                                    rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                                    # 将 '标记' 列表直接作为 route 传入，由剔废线程按 lineName 映射后“同步触发”
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
                        # 若未得到 marks，则按 NG 分布兜底生成 '标记' 路由
                        marks_fallback = None
                        if can_decide_time:
                            if expected == 5 and auto_ng_cams == {2}:
                                # 用坐标决定左右 -> 1 或 max_mark
                                base_res = last_result_by_cam.get(2)
                                if base_res:
                                    side = _decide_left_right_by_coord_for_cam2(base_res)
                                    max_mark, _ = _get_line_mark_info(shared_settings)
                                    if side == 'right':
                                        marks_fallback = [1]
                                    elif side == 'left':
                                        marks_fallback = [max_mark]
                            if marks_fallback is None and len(auto_ng_cams) >= 1:
                                marks_fallback = _decide_route_marks_for_auto(expected, auto_ng_cams, shared_settings)
                                if marks_fallback is None and len(auto_ng_cams) >= 3:
                                    marks_fallback = [0]
                        if marks_fallback is not None:
                            is_current_event_rejected = True
                            saved_for_this_pane = True
                            rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                            rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, cam_index, marks_fallback))
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
                            print(f"    [状态机]: 自动剔废触发，marks={marks_fallback}，NG相机={sorted(list(auto_ng_cams))}")
                    # 即时手动剔废：任何时候都可触发。始终执行硬件动作；仅在有NG缓存时才追加上传
                    elif manual_reject_flag.value:
                        is_current_event_rejected = True
                        manual_reject_flag.value = False
                        # 触发后，自当前时刻起，直到该片离开，均标记为“手动剔废模式”
                        manual_reject_active_mode = True
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
                            # 在即时上传前，保存当前片中所有 NG 帧的未标注原图
                            try:
                                for res_item in current_pane_ng_buffer:
                                    raw_buf = res_item.get('raw_image_buffer')
                                    if not raw_buf:
                                        continue
                                    ts = float(res_item.get('timestamp', time.time()) or time.time())
                                    cam_idx = int(res_item.get('camera_index', -1) or -1)
                                    day_dir = time.strftime('%Y-%m-%d', time.localtime(ts))
                                    out_dir = os.path.join(STORAGE_PATH, day_dir, 'original')
                                    os.makedirs(out_dir, exist_ok=True)
                                    ms = int(ts * 1000)
                                    out_name = f"cam{cam_idx}_ts{ms}.jpg"
                                    with open(os.path.join(out_dir, out_name), 'wb') as f:
                                        f.write(raw_buf)
                            except Exception as e:
                                print(f"[状态机]: 手动剔废即时上传前保存未标注原图失败: {e}")
                            send_reports_batch_to_server(current_pane_reports, [r.get('annotated_image_buffer') for r in current_pane_ng_buffer], shared_settings.upload_url, http_client, upload_timeout_s=getattr(shared_settings, 'http_upload_timeout_s', 30))

            elif machine_state == "WAITING_FOR_PANE":
                if current_total_panes > 0:
                    presence_streak += 1
                    # 记录首次出现 state_code>0 的相机与其触发 ROI（只记录一次）
                    try:
                        if first_presence_cam_idx is None and int(result.get('state_code', 0) or 0) > 0:
                            first_presence_cam_idx = int(cam_index)
                            # 选择触发ROI：优先 near_vertical_line_count>0，其次 defects 非空，否则取第一个
                            trig_roi = 0
                            rois_list = result.get('rois', []) or []
                            for i_roi, r in enumerate(rois_list):
                                try:
                                    if int(r.get('near_vertical_line_count', 0) or 0) > 0:
                                        trig_roi = i_roi
                                        break
                                except Exception:
                                    continue
                            else:
                                for i_roi, r in enumerate(rois_list):
                                    if r.get('defects'):
                                        trig_roi = i_roi
                                        break
                            first_presence_roi_idx = int(trig_roi)
                    except Exception:
                        pass
                    # 仅使用帧数去抖判定进入；最短持续时间在离开时及剔废触发时验证
                    if presence_streak >= ENTER_CONFIRM_FRAMES:
                        # 不再支持滞后剔废：无需处理上一片的延迟上传
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
                        pane_max_total_vertical_count = 0
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_normal_state()
                            except Exception:
                                pass
                        pane_enter_time_s = time.time()
                        if first_presence_cam_idx is not None:
                            print(f"--- [状态机]: 玻璃进入事件 (触发: 相机{first_presence_cam_idx + 1} ROI{first_presence_roi_idx}) ---")
                        else:
                            print("--- [状态机]: 玻璃进入事件 (触发: 未捕获) ---")
                else:
                    # 连续无玻璃：重置帧计数
                    presence_streak = 0
        except Exception as e:
            print(f"[状态机]: 处理结果时出错: {e}")