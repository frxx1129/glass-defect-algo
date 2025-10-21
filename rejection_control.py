# --- START OF FILE rejection_control.py ---
import time
from queue import Empty

def rejection_handler_thread(rejection_queue, rejection_controller, stop_event, shared_settings):
    """A thread that waits for rejection signals and triggers the hardware."""
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
            if (sleep_duration := fire_time - time.time()) > 0:
                time.sleep(sleep_duration)
            try:
                pulse_ms = int(getattr(shared_settings, 'REJECTION_PULSE_MS', 100) or 100)
            except Exception:
                pulse_ms = 100
            # 支持一次触发多个路由（列表或元组）
            routes = route if isinstance(route, (list, tuple)) else [route]
            for r in routes:
                rejection_controller.trigger_rejection_signal(cam_index, pulse_ms, r)
        except Empty:
            continue
        except Exception as e:
            print(f"[剔除处理器线程]: 发生错误: {e}")
    print("[剔除处理器线程]: 已停止。")