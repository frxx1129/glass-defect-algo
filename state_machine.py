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

# --- 内存状态获取辅助函数 ---
def _get_process_tree_memory_gb() -> float | None:
    """返回当前进程 + 所有子进程的 RSS 内存占用 (GB)。优先 psutil，回退 Windows API。"""
    # 1) 优先 psutil - 可以准确统计进程树
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        total_rss = 0
        try:
            total_rss += int(proc.memory_info().rss)
        except Exception:
            pass
        try:
            for ch in proc.children(recursive=True):
                try:
                    total_rss += int(ch.memory_info().rss)
                except Exception:
                    pass
        except Exception:
            pass
        if total_rss > 0:
            return float(total_rss) / float(1024 ** 3)
    except Exception:
        pass
    
    # 2) 回退: Windows API 获取当前进程内存（不含子进程）
    try:
        import ctypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        h_proc = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize) / float(1024 ** 3)
    except Exception:
        pass
    
    return None

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
    last_camera_vertical_flags = np.zeros(num_cameras, dtype=bool)
    last_camera_horizontal_flags = np.zeros(num_cameras, dtype=bool)
    current_pane_ng_buffer = []  # 收集该片玻璃所有 NG 帧的原始结果（含图像缓冲）
    current_pane_reports = []    # 收集该片玻璃所有 NG 报告（即时写入 JSON）
    current_pane_folder = None   # 当前玻璃的存储文件夹
    pane_ng_frame_counter = 0    # 当前玻璃 NG 帧计数
    last_pane_data = {}

    # 防止单片玻璃长期/高频 NG 导致缓存无限增长：限制每片玻璃在内存中保留的 NG 帧数量
    try:
        max_ng_frames_buffered_per_pane = int(getattr(shared_settings, 'max_ng_frames_buffered_per_pane', 200) or 200)
        if max_ng_frames_buffered_per_pane < 1:
            max_ng_frames_buffered_per_pane = 200
    except Exception:
        max_ng_frames_buffered_per_pane = 200
    
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
    
    # 进入/离开 去抖（帧）——用户要求“进入事件不加延迟，直接进入”。
    # 因此进入确认帧固定为 1；离开仍保留去抖避免抖动。
    ENTER_CONFIRM_FRAMES = 1
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
    # 新增：按整片玻璃周期统计的“竖直边总数”的最大值（跨相机求和，跨时刻取最大）
    pane_max_total_vertical_count = 0
    # 新增：每个相机在当前玻璃周期内看到的最大竖直边数量（key: cam_idx int, value: int）
    last_vertical_counts_by_cam = {}
    # 新增：记录每个相机看到竖直边的具体位置信息（key: cam_idx int, value: set of strings）
    last_vertical_details_by_cam = {}
    # 新增：已打印的切片数（用于避免重复打印）
    last_printed_piece_count = 0

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

    def _suppress_q_for_multi_vertical_cam(result_obj: dict) -> None:
        """若该相机任一 ROI 的 near_vertical_line_count>=2，则剔除所有 Q 缺陷；
        若剔除后已无缺陷，则将 image_status 设置为 'OK'。
        """
        try:
            rois = result_obj.get('rois', []) or []
            if not isinstance(rois, list) or not rois:
                return
            has_multi_vert = any(int(r.get('near_vertical_line_count', 0) or 0) >= 2 for r in rois)
            if not has_multi_vert:
                return
            defs = result_obj.get('defects')
            if isinstance(defs, list):
                kept = [d for d in defs if str(d.get('type', '')).upper() != 'Q']
                result_obj['defects'] = kept
                if not kept:
                    # 无其他缺陷，视为 OK 帧
                    result_obj['image_status'] = 'OK'
        except Exception:
            pass

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

    def _estimate_piece_count(cam_vertical_counts: dict, expected_cams: int) -> int:
        """
        依据中间相机是否看到竖直边来推断玻璃切片数。
        
        简化逻辑：
        - 排除边缘两个相机（第一个和最后一个）
        - 中间有几个相机看到了竖直边，就认为玻璃切了几次
        - 切片数 = 看到竖直边的中间相机数 + 1
        - 5相机时最大4片，4相机时最大3片
        
        示例（5相机配置，边缘为cam0和cam4，中间为cam1,cam2,cam3）：
        - 0个中间相机看到竖直边 -> 1片（不切）
        - 1个中间相机看到竖直边 -> 2片（切1刀）
        - 2个中间相机看到竖直边 -> 3片（切2刀）
        - 3个中间相机看到竖直边 -> 4片（切3刀，最大）
        
        示例（4相机配置，边缘为cam0和cam3，中间为cam1,cam2）：
        - 0个中间相机看到竖直边 -> 1片（不切）
        - 1个中间相机看到竖直边 -> 2片（切1刀）
        - 2个中间相机看到竖直边 -> 3片（切2刀，最大）
        
        相机配置（内部索引为0-based，websocket URL为1-based但内部转换为0-based）：
        - Line2/Line3: 5台相机 (index 0,1,2,3,4)，边缘相机为 0 和 4
        - Line1: 4台相机 (index 0,1,2,3)，边缘相机为 0 和 3
        """
        try:
            n = int(expected_cams or 0)
            if n <= 0:
                return 1
            
            # 定义边缘相机索引（0-based，内部表示）
            first_cam = 0
            last_cam = max(0, n - 1)
            
            # 统计有竖直边的中间相机数量
            middle_cams_with_vertical = 0
            
            for ci_str, cnt in cam_vertical_counts.items():
                try:
                    ci = int(ci_str)
                    line_count = max(0, int(cnt or 0))
                    
                    # 跳过边缘相机
                    if ci == first_cam or ci == last_cam:
                        continue
                    
                    # 只要该中间相机看到了竖直边（>=1条），就计数
                    if line_count >= 1:
                        middle_cams_with_vertical += 1
                except Exception:
                    continue
            
            # 切片数 = 看到竖直边的中间相机数 + 1
            piece_count = middle_cams_with_vertical + 1

            # 特殊情况处理：若配置为 5 相机但只有 Cam3 (index 2) 看到且 expected_cams=5，
            # 逻辑上 middle_cams_with_vertical=1 -> quantity=2. Correct.
            # 但若用户测试时仅用单一相机 Cam3，且 num_cameras=1, 则 expected 可能为 1
            # 此时 first=0, last=0. ci=2. ci!=first, ci!=last. middle+=1. quantity=2. Correct.
            
            # 兜底：如果检测到 vertical edge 的数量 > 0 但 middle_cams 为 0 (都被过滤了)
            # 检查是否有中间相机看到 vertical edge（0-based索引）
            if middle_cams_with_vertical == 0:
                # 检查是否存在 index=2 的相机看到了竖直边（针对 Line3 Cam3 测试，0-based: cam3=index2）
                # 即使 expected_cams 配置偏差，Cam3（index 2）物理上是中间，应算切分
                if cam_vertical_counts.get('2', 0) >= 1 or cam_vertical_counts.get('3', 0) >= 1:
                     # 再次确认不是边缘 (针对 expected_cams 极小的情况)
                     # 在 Line3 (5 cam, 0-based) 中, 0和4 是边缘. 2 是中间.
                     # 只要 index 2 有竖直边，且 expected_cams >= 3，它就一定是中间
                     if n >= 3:
                         if cam_vertical_counts.get('2', 0) >= 1:
                             middle_cams_with_vertical += 1
            
            piece_count = middle_cams_with_vertical + 1
            
            # 根据相机数量限制最大切片数
            # 5相机: 最大4片; 4相机: 最大3片
            max_pieces = 4 if n >= 5 else 3
            piece_count = max(1, min(max_pieces, piece_count))

            return piece_count
            
        except Exception:
            pass
        return 1

    def _estimate_piece_count_with_validation(cam_vertical_counts: dict, expected_cams: int, 
                                               last_result_by_cam: dict) -> int:
        """
        分片估计：直接调用 _estimate_piece_count。
        
        简化后的逻辑已经足够直接（中间相机有几个看到竖直边就切几刀），
        不再需要额外的验证逻辑。
        """
        return _estimate_piece_count(cam_vertical_counts, expected_cams)


    def _gather_defect_centers_for_cam(result_obj: dict) -> list[float]:
        """提取单相机内缺陷的 x 中心（像素）。
        
        优先级：
        1. location['x'] - 这是全局坐标，直接表示缺陷位置
        2. raw_defect['center'][0] - 备选方案，从原始缺陷对象获取
        3. defect['center'][0] - 进一步备选
        """
        xs = []
        try:
            for d in result_obj.get('defects', []):
                loc = d.get('location', {}) if isinstance(d.get('location'), dict) else {}
                # 优先使用 location['x'] - 这是已经转换到全局坐标的 x 位置
                if 'x' in loc:
                    try:
                        x_val = float(loc.get('x', 0) or 0)
                        if x_val > 0:  # 有效的 x 坐标
                            xs.append(x_val)
                            continue
                    except Exception:
                        pass
                # 备选：从 raw_defect 中获取 center
                raw = d.get('raw_defect', {}) if isinstance(d.get('raw_defect'), dict) else {}
                c = raw.get('center')
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    try:
                        xs.append(float(c[0]))
                        continue
                    except Exception:
                        pass
                # 进一步备选：直接从 defect 顶层获取 center
                c2 = d.get('center')
                if isinstance(c2, (list, tuple)) and len(c2) >= 2:
                    try:
                        xs.append(float(c2[0]))
                    except Exception:
                        pass
        except Exception:
            pass
        return xs

    def _decide_marks_by_pieces_and_positions(cam_vertical_counts: dict, last_result_by_cam: dict, expected_cams: int, shared_settings, piece_count_override: int | None = None) -> list[int] | None:
        """
        依据估计的切片数（1~3）以及缺陷位置，返回需要触发的'标记'集合：
        - 1片（不切）-> [0] 整片撤清
        - 2片（切1刀）-> 左片标记2，右片标记1
        - 3片（切2刀）-> 左片标记3，中片标记2，右片标记1
        
        标记约定（max_mark=3）：
        - 0: 整片撤清
        - 1: 最右边的片
        - 2: 中间的片（3片时）或左边的片（2片时）
        - 3: 最左边的片（仅3片时）
        """
        piece_count = piece_count_override if isinstance(piece_count_override, int) and piece_count_override >= 1 else _estimate_piece_count(cam_vertical_counts, expected_cams)
        if piece_count <= 1:
            return [0]

        # 计算全局中心索引（相机序号）
        n = int(expected_cams or 0)
        center = (n - 1) / 2.0 if n > 0 else 1.5

        # 收集"每个缺陷"的全局位置参数，简化为 (cam_index, x_center_px or None)
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

        for ci, x in samples:
            if piece_count == 2:
                # 切2片：左片(标记2) | 右片(标记1)
                if ci < center:
                    marks.add(2)  # 左片
                elif ci > center:
                    marks.add(1)  # 右片
                else:
                    # 中间相机：按 x 分左右
                    if x is not None and cam_w > 1e-6:
                        marks.add(2 if x < (cam_w / 2.0) else 1)
                    else:
                        marks.add(1)
            elif piece_count == 3:
                # 切3片：左片(标记3) | 中片(标记2) | 右片(标记1)
                # 根据相机位置分配到三个区域
                # 5相机: cam0,cam1->左片, cam2->中片, cam3,cam4->右片
                # 4相机: cam0->左片, cam1,cam2->中片（需x坐标细分）, cam3->右片
                if n == 5:
                    if ci <= 1:
                        marks.add(3)  # 左片
                    elif ci == 2:
                        marks.add(2)  # 中片
                    else:  # ci >= 3
                        marks.add(1)  # 右片
                elif n == 4:
                    if ci == 0:
                        marks.add(3)  # 左片
                    elif ci == 3:
                        marks.add(1)  # 右片
                    else:  # ci in (1, 2)
                        # 中间两个相机按位置细分
                        if ci == 1:
                            # cam1: 偏左 -> 左片或中片
                            if x is not None and cam_w > 1e-6 and x < cam_w / 2:
                                marks.add(3)  # 左片
                            else:
                                marks.add(2)  # 中片
                        else:  # ci == 2
                            # cam2: 偏右 -> 中片或右片
                            if x is not None and cam_w > 1e-6 and x > cam_w / 2:
                                marks.add(1)  # 右片
                            else:
                                marks.add(2)  # 中片
                else:
                    marks.add(2)  # 默认中片
            elif piece_count == 4 and n == 5:
                # 切4片（仅5相机）：片1(标记4) | 片2(标记3) | 片3(标记2) | 片4(标记1)
                # cam0->片1, cam1->片2, cam2->片2或片3(按x), cam3->片3, cam4->片4
                if ci == 0:
                    marks.add(min(4, max_mark))  # 片1（最左）
                elif ci == 1:
                    marks.add(3)  # 片2
                elif ci == 2:
                    # cam2按x坐标细分
                    if x is not None and cam_w > 1e-6:
                        marks.add(3 if x < cam_w / 2 else 2)
                    else:
                        marks.add(2)  # 默认片3
                elif ci == 3:
                    marks.add(2)  # 片3
                else:  # ci == 4
                    marks.add(1)  # 片4（最右）
            else:
                marks.add(0)  # 兜底整片

        # 将标记限制在 [1..max_mark] 范围内（0 保留为整片撤清，非此处产生）
        final_marks = sorted({m for m in marks if 1 <= m <= max_mark})
        return final_marks if final_marks else [0]

    def build_report_for_frame(result_obj, rejection_details_to_save):
        defects = result_obj.get('defects', [])
        def primary_size(d):
            """缺陷尺寸标签：E 用最长边，其余维持最短边。"""
            loc = d.get('location', {})
            length = float(loc.get('length_mm', 0) or 0)
            width = float(loc.get('width_mm', 0) or 0)
            if length <= 0 and width <= 0:
                return 0.0
            defect_type = str(d.get('type', '')).upper()
            return max(length, width) if defect_type == 'E' else min(length, width)
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
        nonlocal auto_ng_cams, last_result_by_cam, pane_max_total_vertical_count, last_vertical_counts_by_cam, last_vertical_details_by_cam
        nonlocal last_camera_vertical_flags, last_camera_horizontal_flags
        try:
            machine_state = "WAITING_FOR_PANE"
            machine_state_shared.value = 0
        except Exception:
            pass
        try:
            last_camera_states[:] = 0
            last_camera_vertical_flags[:] = False
            last_camera_horizontal_flags[:] = False
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
            auto_ng_cams.clear(); last_result_by_cam.clear(); last_vertical_counts_by_cam.clear(); last_vertical_details_by_cam.clear()
        except Exception:
            pass
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
                last_camera_vertical_flags = np.pad(last_camera_vertical_flags, (0, cam_index - len(last_camera_vertical_flags) + 1), 'constant')
                last_camera_horizontal_flags = np.pad(last_camera_horizontal_flags, (0, cam_index - len(last_camera_horizontal_flags) + 1), 'constant')

            last_camera_states[cam_index] = result['state_code']
            
            # --- 更新当前相机的（本帧）竖直/水平 状态位 ---
            _frame_v_cnt = 0
            _frame_h_cnt = 0
            for _r in result.get('rois', []) or []:
                try:
                    _ef = int(_r.get('edges_found', 0) or 0)
                    _vl = int(_r.get('near_vertical_line_count', 0) or 0)
                    _frame_v_cnt += _vl
                    _frame_h_cnt += max(0, _ef - _vl)
                except Exception:
                    pass
            last_camera_vertical_flags[cam_index] = (_frame_v_cnt > 0)
            last_camera_horizontal_flags[cam_index] = (_frame_h_cnt > 0)

            current_total_panes = int(np.sum(last_camera_states))

            # --- 计算新的进入触发条件 ---
            # 条件：(两侧任一相机看到竖直边) AND (所有任一相机看到水平边)
            _n_current_cams = len(last_camera_states)
            _exp_cams = int(getattr(shared_settings, 'expected_cameras', _n_current_cams) or _n_current_cams)
            # 边缘相机索引：0-based，内部表示 (cam0=0, camN-1=N-1)
            _idx_first = 0
            _idx_last = max(0, _exp_cams - 1)
            
            _has_side_vert = False
            if _idx_first < len(last_camera_vertical_flags) and last_camera_vertical_flags[_idx_first]:
                _has_side_vert = True
            if _idx_last < len(last_camera_vertical_flags) and last_camera_vertical_flags[_idx_last]:
                _has_side_vert = True
                
            _has_any_horz = np.any(last_camera_horizontal_flags)
            
            # 新的进入触发标记
            entry_trigger = (_has_side_vert and _has_any_horz)

            # 对外广播：仅在“玻璃进入事件(检测窗口)”内允许出现 NG。
            # 说明：广播发生在进入判定逻辑之前，因此这里需要“预测”本帧是否将触发进入。
            # - WAITING 且本帧将触发进入：允许按真实 NG/OK 广播（视为进入事件内的首帧）。
            # - WAITING 且尚未触发进入：强制汇报 OK。
            # - PANE_DETECTED 但 current_total_panes==0（离开去抖/空场）：强制汇报 OK。
            ws_data = {k: v for k, v in result.items() if k not in ('annotated_image_buffer', 'raw_image_buffer')}
            try:
                will_enter_now = False
                if machine_state == "WAITING_FOR_PANE" and entry_trigger:
                    next_presence = presence_streak + 1
                    will_enter_now = (next_presence >= ENTER_CONFIRM_FRAMES)

                in_detection_window = (machine_state == "PANE_DETECTED" and current_total_panes > 0) or will_enter_now

                if not in_detection_window:
                    if str(ws_data.get('image_status', 'OK')).upper() == 'NG':
                        ws_data['image_status'] = 'OK'
                        ws_data['defects'] = []
                        ws_data['should_reject'] = False

                asyncio.run_coroutine_threadsafe(connection_manager.broadcast(json.dumps(ws_data), cam_index), loop)
            except Exception:
                pass

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
                        _mem_gb = _get_process_tree_memory_gb()
                        _mem_str = f" | 内存: {_mem_gb:.2f}GB" if _mem_gb is not None else ""
                        # 计算并打印本片玻璃的切片数
                        try:
                            # 离开时优先使用累计的最大竖直边计数
                            vertical_counts_str = {str(ci): cnt for ci, cnt in last_vertical_counts_by_cam.items()}
                            expected = int(num_cameras or 5)
                            final_piece_count = _estimate_piece_count(vertical_counts_str, expected)
                            print(f"--- [状态机]: 玻璃离开事件 (切{final_piece_count}片) ---{_mem_str}")
                        except Exception:
                            print(f"--- [状态机]: 玻璃离开事件 ---{_mem_str}")
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
                        auto_ng_cams.clear(); last_result_by_cam.clear(); last_vertical_counts_by_cam.clear()
                        pane_max_total_vertical_count = 0
                        pane_enter_time_s = None
                        continue
                else:
                    absence_streak = 0

                # 超时强制退出：玻璃处于检测状态超过 pane_max_duration_s
                try:
                    if pane_enter_time_s is not None and (time.time() - pane_enter_time_s) >= pane_max_duration_s:
                        _mem_gb = _get_process_tree_memory_gb()
                        _mem_str = f" | 内存: {_mem_gb:.2f}GB" if _mem_gb is not None else ""
                        # 计算并打印本片玻璃的切片数
                        try:
                            vertical_counts_str = {str(ci): cnt for ci, cnt in last_vertical_counts_by_cam.items()}
                            expected = int(num_cameras or 5)
                            final_piece_count = _estimate_piece_count(vertical_counts_str, expected)
                            print(f"--- [状态机]: 玻璃检测超时（>{pane_max_duration_s:.1f}s），强制退出 (切{final_piece_count}片) ---{_mem_str}")
                        except Exception:
                            print(f"--- [状态机]: 玻璃检测超时（>{pane_max_duration_s:.1f}s），强制退出 ---{_mem_str}")
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
                        auto_ng_cams.clear(); last_result_by_cam.clear(); last_vertical_counts_by_cam.clear()
                        pane_max_total_vertical_count = 0
                        pane_enter_time_s = None
                        continue
                except Exception:
                    pass

                # NG 帧处理：仅在“确有玻璃存在”(current_total_panes>0)时才允许记录NG。
                # 这可避免离开去抖阶段(无玻璃但尚未退出状态)产生的空场误报被计入本片。
                if current_total_panes > 0 and result['image_status'] == 'NG':
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
                    # 先尽早将原始整帧(若存在)落盘并从内存中移除，避免单片玻璃缓存占用过大
                    try:
                        raw_buf = result.get('raw_image_buffer')
                        if raw_buf:
                            ts = float(result.get('timestamp', time.time()) or time.time())
                            cam_idx = int(result.get('camera_index', -1) or -1)
                            day_dir = time.strftime('%Y-%m-%d', time.localtime(ts))
                            out_dir = os.path.join(STORAGE_PATH, day_dir, 'original')
                            os.makedirs(out_dir, exist_ok=True)
                            ms = int(ts * 1000)
                            out_name = f"cam{cam_idx}_ts{ms}.jpg"
                            with open(os.path.join(out_dir, out_name), 'wb') as f:
                                f.write(raw_buf)
                            # 释放内存：后续不再依赖 raw_image_buffer
                            try:
                                result['raw_image_buffer'] = None
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"[状态机]: NG帧原图落盘失败: {e}")

                    current_pane_ng_buffer.append(result)
                    frame_ts = datetime.now()
                    temp_rej_details = {
                        'rejection_time': frame_ts,
                        'rejection_type': (rejection_details.get('rejection_type') if is_current_event_rejected else '0')
                    }
                    rpt = build_report_for_frame(result, temp_rej_details)
                    current_pane_reports.append(rpt)

                    # 缓存上限：超过上限时丢弃最早帧，避免内存持续增长
                    try:
                        while len(current_pane_ng_buffer) > max_ng_frames_buffered_per_pane:
                            dropped = current_pane_ng_buffer.pop(0)
                            try:
                                if isinstance(dropped, dict):
                                    dropped.pop('annotated_image_buffer', None)
                                    dropped.pop('raw_image_buffer', None)
                            except Exception:
                                pass
                            if len(current_pane_reports) > 0:
                                current_pane_reports.pop(0)
                    except Exception:
                        pass
                    try:
                        # 文件名使用 1-based 索引，与前端一致（内部0对应前端cam1）
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
                        # 自动分路仅需要缺陷/ROI信息，不应长期持有大图像buffer
                        slim = dict(result)
                        slim.pop('annotated_image_buffer', None)
                        slim.pop('raw_image_buffer', None)
                        last_result_by_cam[cam_i] = slim
                    except Exception:
                        pass

                if np.sum(last_camera_states) > np.sum(max_complexity_snapshot):
                    max_complexity_snapshot = last_camera_states.copy()

                # --- 实时更新竖直边计数（此时机覆盖所有帧，包括 OK 帧）---
                if current_total_panes > 0:
                    try:
                        c_idx = int(result.get('camera_index', -1))
                        if c_idx >= 0:
                            v_cnt = _count_near_vertical_per_cam(result)
                            # 使用 max 累积该相机在本片玻璃内的最大竖直边数
                            last_vertical_counts_by_cam[c_idx] = max(last_vertical_counts_by_cam.get(c_idx, 0), v_cnt)
                            
                            if v_cnt > 0:
                                try:
                                    _rois = result.get('rois', [])
                                    _found_details = set()
                                    _cam_w = float(getattr(shared_settings, 'cam_width', 2448) or 2448)
                                    _cam_h = float(getattr(shared_settings, 'cam_height', 2048) or 2048)
                                    for _r in _rois:
                                        if int(_r.get('near_vertical_line_count', 0) or 0) > 0:
                                            _rx, _ry = int(_r.get('x',0)), int(_r.get('y',0))
                                            _rw, _rh = int(_r.get('w',0)), int(_r.get('h',0))
                                            _cx, _cy = _rx + _rw/2.0, _ry + _rh/2.0
                                            _h_pos = "左" if _cx < (_cam_w/2.0) else "右"
                                            _v_pos = "上" if _cy < (_cam_h/2.0) else "下"
                                            _ridx = _r.get('roi_idx', '?')
                                            _found_details.add(f"ROI{_ridx}{_h_pos}{_v_pos}侧({_rx},{_ry})")
                                    if _found_details:
                                        _existing = last_vertical_details_by_cam.get(c_idx, set())
                                        _existing.update(_found_details)
                                        last_vertical_details_by_cam[c_idx] = _existing
                                except Exception:
                                    pass

                            # 实时计算并打印切片数
                            try:
                                expected = int(getattr(shared_settings, 'expected_cameras', num_cameras) or num_cameras)
                                vert_counts_str = {str(ci): cnt for ci, cnt in last_vertical_counts_by_cam.items()}
                                current_piece_count = _estimate_piece_count(vert_counts_str, expected)
                                if current_piece_count != last_printed_piece_count:
                                    last_printed_piece_count = current_piece_count
                                    # 0-based索引：边缘相机为 0 和 expected-1
                                    first_cam, last_cam = 0, max(0, expected - 1)
                                    cams_with_vert = [ci for ci, cnt in last_vertical_counts_by_cam.items() if int(ci) != first_cam and int(ci) != last_cam and int(cnt) >= 1]
                                    if cams_with_vert:
                                        _info_parts = []
                                        for _c in sorted(int(c) for c in cams_with_vert):
                                            _d_set = last_vertical_details_by_cam.get(_c, set())
                                            _d_str = f"[{','.join(sorted(list(_d_set)))}]" if _d_set else ""
                                            # 显示时使用 1-based 索引，与前端一致（内部0对应前端cam1）
                                            _info_parts.append(f"cam{_c + 1}{_d_str}")
                                        print(f"    [状态机]: 检测到切{current_piece_count}片 ({','.join(_info_parts)}看到切片)")
                            except Exception:
                                pass
                    except Exception:
                        pass

                auto_reject_ready = (current_total_panes > 0 and shared_rejection_mode.value == 1 and result.get('should_reject', False))
                if auto_reject_ready:
                    expected = int(getattr(shared_settings, 'expected_cameras', num_cameras) or num_cameras)
                    marks = None
                    # 使用累积的 last_vertical_counts_by_cam 构建 vertical_counts_str
                    vertical_counts_str = {str(ci): cnt for ci, cnt in last_vertical_counts_by_cam.items()}
                    try:
                        piece_count_override = None
                        if vertical_counts_str:
                            # 分片估计直接使用累积的计数值
                            piece_count_override = _estimate_piece_count_with_validation(
                                vertical_counts_str, expected, last_result_by_cam
                            )
                            marks = _decide_marks_by_pieces_and_positions(
                                vertical_counts_str, last_result_by_cam, expected, shared_settings,
                                piece_count_override=piece_count_override
                            )
                    except Exception as e:
                        print(f"[状态机]: 竖直线分路逻辑异常: {e}")

                    if not marks:
                        marks = None
                        if expected == 5 and auto_ng_cams == {2}:
                            base_res = last_result_by_cam.get(2)
                            if base_res:
                                side = _decide_left_right_by_coord_for_cam2(base_res)
                                max_mark, _ = _get_line_mark_info(shared_settings)
                                if side == 'right':
                                    marks = [1]
                                elif side == 'left':
                                    marks = [max_mark]
                        if marks is None and len(auto_ng_cams) >= 1:
                            marks = _decide_route_marks_for_auto(expected, auto_ng_cams, shared_settings)
                            if marks is None and len(auto_ng_cams) >= 3:
                                marks = [0]

                    # 最外层兜底：只要有 NG 相机且 should_reject，本帧仍算不出 marks，则整片剔 [0]
                    if not marks and len(auto_ng_cams) >= 1:
                        marks = [0]

                    if marks:
                        # 每一帧 should_reject 的 NG 图像都入队执行剔废脉冲，不管当前玻璃是否已判定过剔除
                        rejection_queue.put((time.time() + shared_settings.REJECTION_DELAY_S, cam_index, marks))

                        # 统计与上传仍按"每片玻璃只计一次"的原则：
                        # 第一次触发自动剔废时更新 rejection_details 和全局计数，其余帧只做硬件剔废，不再重复计数。
                        log_counts = vertical_counts_str if vertical_counts_str else 'N/A'
                        if not is_current_event_rejected:
                            is_current_event_rejected = True
                            saved_for_this_pane = True
                            rejection_details = {"rejection_time": datetime.now(), "rejection_type": "1"}
                            with stats_lock:
                                total_rejections += 1
                                shared_rejection_counter.value = total_rejections
                                yield_manager.save_stats(last_reset_date_str, total_yield, total_rejections)
                            broadcast_yield_and_rejections(shared_settings, shared_collection_id, shared_yield_counter, shared_rejection_counter, http_client)
                            if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                                try:
                                    alarm_light_controller.set_rejection_state(shared_settings.rejection_buzz_duration_s)
                                except Exception:
                                    pass
                            # 简化打印：cam{x}检测到缺陷，触发撤清{y}
                            ng_cam_str = ','.join([f'cam{c}' for c in sorted(auto_ng_cams)])
                            mark_str = ','.join([str(m) for m in marks])
                            print(f"    [状态机]: {ng_cam_str}检测到缺陷，触发撤清{mark_str} (切{piece_count_override}片)")
                        else:
                            # 已经标记过剔除的玻璃，后续帧继续入队但不再增加计数
                            # 后续帧只做硬件剔废，简化打印
                            mark_str = ','.join([str(m) for m in marks])
                            print(f"    [状态机]: 再次触发撤清{mark_str}")

                if manual_reject_flag.value:
                    is_current_event_rejected = True
                    manual_reject_flag.value = False
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
                    # 手动剔废简化打印
                    route_str = str(route) if route else '0'
                    print(f"    [状态机]: 手动触发撤清{route_str}")
                    if current_pane_ng_buffer and current_pane_reports:
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
                if entry_trigger:
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
                        auto_ng_cams.clear(); last_result_by_cam.clear(); last_vertical_counts_by_cam.clear()
                        pane_max_total_vertical_count = 0
                        last_printed_piece_count = 0
                        if alarm_light_controller and getattr(alarm_light_controller, 'is_active', False):
                            try:
                                alarm_light_controller.set_normal_state()
                            except Exception:
                                pass
                        pane_enter_time_s = time.time()
                        # 获取内存状态用于显示
                        _mem_gb = _get_process_tree_memory_gb()
                        _mem_str = f" | 内存: {_mem_gb:.2f}GB" if _mem_gb is not None else ""
                        if first_presence_cam_idx is not None:
                            print(f"--- [状态机]: 玻璃进入事件 (触发: 相机{first_presence_cam_idx + 1} ROI{first_presence_roi_idx}) ---{_mem_str}")
                        else:
                            print(f"--- [状态机]: 玻璃进入事件 (触发: 未捕获) ---{_mem_str}")
                else:
                    # 连续无玻璃：重置帧计数
                    presence_streak = 0
        except Exception as e:
            try:
                print(f"[状态机]: 处理结果时出错: {e}，已跳过该帧。")
            except Exception:
                pass
            # 避免异常风暴导致主进程无法及时响应
            time.sleep(0.2)
            continue