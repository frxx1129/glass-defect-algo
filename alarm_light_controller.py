# --- START OF FILE alarm_light_controller.py ---
import serial
import time
import threading
from queue import Queue, Empty
import os
import threading
import json
import urllib.request
import urllib.error

# -------------------------------------------------------------------
#  核心控制器（保持私有，在主进程中运行）
# -------------------------------------------------------------------
class _AlarmLightService:
    """
    实际管理串口和硬件通信的服务。
    这个类的实例应该只在主进程中存在。
    """
    _CMD_HEADER = 0xFF
    _CMD_TAIL = 0xAA
    _WORK_MODE = { "off": 0x01, "green": 0x02, "yellow": 0x03, "red": 0x04 }
    _BUZZER_MODE = { "off": 0x01, "on": 0x02 }
    _FLICKER_RATE = { "off": 0x01, "slow": 0x04, "medium": 0x03, "fast": 0x02 }

    def __init__(self, port, baud_rate, command_queue):
        self.ser = None
        self.command_queue = command_queue
        self.stop_event = threading.Event()
        
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=0.5)
            print(f"✅ [声光报警器服务]: 串口 {port} 连接成功。")
        except serial.SerialException as e:
            print(f"❌ [声光报警器服务]: 严重错误 - 无法打开串口 '{port}'.")
            print(f"   错误详情: {e}")
            self.ser = None

    def _send_command(self, light="off", buzzer="off", flicker="off"):
        if not self.ser: return
        try:
            work_mode = self._WORK_MODE.get(light, 0x01)
            buzzer_mode = self._BUZZER_MODE.get(buzzer, 0x01)
            flicker_rate = self._FLICKER_RATE.get(flicker, 0x01)
            command = bytes([self._CMD_HEADER, work_mode, buzzer_mode, flicker_rate, self._CMD_TAIL])
            self.ser.flushInput()
            self.ser.write(command)
            time.sleep(0.05)
            self.ser.read(5)
        except Exception as e:
            print(f"❌ [声光报警器服务]: 发送指令时发生错误: {e}")

    def run(self):
        """服务的主循环，在单独的线程中运行。"""
        print("[声光报警器服务]: 后台服务线程已启动。")
        while not self.stop_event.is_set():
            try:
                command, args = self.command_queue.get(timeout=1)
                
                if command == "SET_STATE":
                    light, buzzer_on, duration = args
                    current_light_state = light
                    self._send_command(light=light, buzzer="on" if buzzer_on else "off")
                    if duration > 0:
                        time.sleep(duration)
                        # 持续时间结束：先关闭蜂鸣
                        self._send_command(light=current_light_state, buzzer="off")
                        # 若为红色剔废模式，结束后自动恢复为绿色常亮
                        if str(current_light_state).lower() == "red":
                            self._send_command(light="green", buzzer="off")
                
                elif command == "SHUTDOWN":
                    self._send_command(light="off", buzzer="off")
                    break

            except Empty:
                continue
        
        if self.ser:
            self.ser.close()
        print("[声光报警器服务]: 后台服务线程已停止。")

    def stop(self):
        self.stop_event.set()

# -------------------------------------------------------------------
#  代理对象（可安全传递）
# -------------------------------------------------------------------
class AlarmLightController:
    """
    一个进程安全的代理控制器。
    它的方法只是将命令放入一个共享队列中，可以安全地被任何进程/线程调用。
    """
    def __init__(self, command_queue):
        if command_queue is None:
            # 如果队列是None (例如，因为串口初始化失败)，创建一个虚拟队列
            # 这样调用其方法时不会出错，只是没有任何效果。
            self.command_queue = Queue()
            self.is_active = False
        else:
            self.command_queue = command_queue
            self.is_active = True
        # 远端推送目标，可通过环境变量或后续赋值配置
        self.remote_alarm_host = os.environ.get("REMOTE_ALARM_HOST")  # 例如 192.168.1.50
        self.remote_alarm_port = os.environ.get("REMOTE_ALARM_PORT")  # 例如 9000

    # ---------------- 远端推送辅助 ----------------
    def _push_remote(self, light: str, buzzer: bool, duration_s: float):
        host = (self.remote_alarm_host or '').strip()
        port = (self.remote_alarm_port or '').strip()
        if not host or not port:
            return
        def _send():
            try:
                url = f"http://{host}:{port}/alarm/update"
                data = json.dumps({"light": light, "buzzer": bool(buzzer), "duration_s": float(duration_s or 0.0)}).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    _ = resp.read()
            except Exception:
                pass
        threading.Thread(target=_send, daemon=True).start()

    def set_startup_state(self):
        if self.is_active: self.command_queue.put(("SET_STATE", ("green", True, 0.25)))
        self._push_remote("green", True, 0.25)

    def set_normal_state(self):
        if self.is_active: self.command_queue.put(("SET_STATE", ("green", False, 0)))
        self._push_remote("green", False, 0)
        
    def set_ng_detected_state(self, duration_s):
        if self.is_active: self.command_queue.put(("SET_STATE", ("yellow", True, duration_s)))
        self._push_remote("yellow", True, duration_s)

    def set_rejection_state(self, duration_s):
        if self.is_active: self.command_queue.put(("SET_STATE", ("red", True, duration_s)))
        self._push_remote("red", True, duration_s)
        # 在控制器层也安排一次延时恢复绿色，以保证远端推送同步恢复
        def _delayed_green():
            try:
                import time as _t
                _t.sleep(max(0.0, float(duration_s or 0.0)))
                self.set_normal_state()
            except Exception:
                pass
        threading.Thread(target=_delayed_green, daemon=True).start()

    # 通用接口：便于远端服务复用
    def set_state(self, light: str, buzzer: bool, duration_s: float = 0.0):
        light = (light or 'off').lower()
        if light == 'off':
            # 关闭：用绿色无蜂鸣代表 idle/off
            if self.is_active: self.command_queue.put(("SET_STATE", ("off", False, 0)))
            self._push_remote("off", False, 0)
        elif light == 'green':
            self.set_normal_state()
        elif light == 'yellow':
            if self.is_active: self.command_queue.put(("SET_STATE", ("yellow", bool(buzzer), float(duration_s or 0.0))))
            self._push_remote("yellow", bool(buzzer), float(duration_s or 0.0))
        elif light == 'red':
            if self.is_active: self.command_queue.put(("SET_STATE", ("red", bool(buzzer), float(duration_s or 0.0))))
            self._push_remote("red", bool(buzzer), float(duration_s or 0.0))
        else:
            self.set_normal_state()
    
    def close(self):
        """注意：代理对象的 close 只是发送关闭信号，并不真正关闭串口。"""
        if self.is_active: self.command_queue.put(("SHUTDOWN", None))