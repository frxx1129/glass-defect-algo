# --- START OF FILE rejection_control.py ---
import time
from queue import Empty

def _resolve_marks_to_channels(marks, shared_settings) -> list[int]:
    """将'标记'(0..N)映射为实际通道号，依据配置 rejection_mark_to_channel 和 lineName。
    - marks: int 或 可迭代的 int；值从 0 开始，最大到 3/4（随产线配置而变）。
    - 若未配置映射，则按通用规则 channel = mark + 1。
    """
    # 统一为列表
    if marks is None:
        return []
    if not isinstance(marks, (list, tuple, set)):
        marks_list = [marks]
    else:
        marks_list = list(marks)

    # 读取映射
    try:
        line_name = str(getattr(shared_settings, 'lineName', 'Line1'))
    except Exception:
        line_name = 'Line1'
    try:
        mark_map_all = getattr(shared_settings, 'rejection_mark_to_channel', {})
    except Exception:
        mark_map_all = {}
    line_map = {}
    if isinstance(mark_map_all, dict):
        line_map = mark_map_all.get(line_name) or {}

    channels: set[int] = set()
    for m in marks_list:
        try:
            key = str(int(m))
        except Exception:
            continue
        ch = None
        if isinstance(line_map, dict) and key in line_map:
            ch = line_map[key]
        # 回退：未配置时按 mark+1
        if ch is None:
            try:
                ch = int(m) + 1
            except Exception:
                continue
        try:
            ch_int = int(ch)
            if 1 <= ch_int <= 8:
                channels.add(ch_int)
        except Exception:
            continue
    return sorted(channels)


def rejection_handler_thread(rejection_queue, rejection_controller, stop_event, shared_settings):
    """A thread that waits for rejection signals and triggers the hardware.
    变更：
    - route 支持 0 起始的 '标记'(int/list[int])，将按 lineName 的映射转换为通道号；
    - 多通道改为“同步触发”：先对所有通道发送 ON，再延时，再对所有通道发送 OFF。
    """
    print("[剔除处理器线程]: 已启动。")
    while not stop_event.is_set():
        try:
            payload = rejection_queue.get(timeout=0.1)
            # 兼容旧格式: (fire_time, cam_index)
            if isinstance(payload, (list, tuple)):
                if len(payload) == 2:
                    fire_time, cam_index = payload
                    route = None
                elif len(payload) == 3:
                    fire_time, cam_index, route = payload
                else:
                    # 不支持的格式，跳过
                    continue
            else:
                # 不支持的消息格式
                continue

            # 等到触发时刻
            sleep_duration = fire_time - time.time()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

            # 解析 route -> marks -> channels
            try:
                pulse_ms = int(getattr(shared_settings, 'REJECTION_PULSE_MS', 100) or 100)
            except Exception:
                pulse_ms = 100

            channels = _resolve_marks_to_channels(route, shared_settings)
            if not channels:
                # 若未给出 route/marks，则不触发
                continue

            # 同步触发：ON 所有 -> 延时 -> OFF 所有
            try:
                for ch in channels:
                    rejection_controller.send_command(ch, 0x01)
                time.sleep(max(0.0, pulse_ms / 1000.0))
                for ch in channels:
                    rejection_controller.send_command(ch, 0x00)
                print(f"[剔除处理器线程]: 同步触发 - marks={route}, channels={channels}, duration={pulse_ms}ms")
            except Exception as e:
                print(f"[剔除处理器线程]: 触发失败: {e}")
        except Empty:
            continue
        except Exception as e:
            print(f"[剔除处理器线程]: 发生错误: {e}")
    print("[剔除处理器线程]: 已停止。")