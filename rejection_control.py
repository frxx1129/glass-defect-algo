# --- START OF FILE rejection_control.py ---
import time
from queue import Empty

def rejection_handler_thread(rejection_queue, rejection_controller, stop_event, shared_settings):
    """A thread that waits for rejection signals and triggers the hardware."""
    print("[剔除处理器线程]: 已启动。")
    while not stop_event.is_set():
        try:
            fire_time, cam_index = rejection_queue.get(timeout=0.1)
            if (sleep_duration := fire_time - time.time()) > 0:
                time.sleep(sleep_duration)
            rejection_controller.trigger_rejection_signal(cam_index, shared_settings.REJECTION_PULSE_MS)
        except Empty:
            continue
        except Exception as e:
            print(f"[剔除处理器线程]: 发生错误: {e}")
    print("[剔除处理器线程]: 已停止。")