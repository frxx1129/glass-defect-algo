# --- START OF FILE alarm_light_controller.py ---
import serial
import time
import threading
from queue import Queue, Empty

class AlarmLightController:
    """
    管理声光报警器的状态，通过串口发送指令。
    使用后台线程处理带延时的操作，避免阻塞调用者。
    """
    # --- 根据协议定义指令常量 ---
    _CMD_HEADER = 0xFF
    _CMD_TAIL = 0xAA

    _WORK_MODE = { "off": 0x01, "green": 0x02, "yellow": 0x03, "red": 0x04 }
    _BUZZER_MODE = { "off": 0x01, "on": 0x02 }
    _FLICKER_RATE = { "off": 0x01, "slow": 0x04, "medium": 0x03, "fast": 0x02 }

    def __init__(self, port, baud_rate=9600):
        """
        初始化控制器并尝试连接到指定的串口。
        :param port: 串口号 (例如 'COM5')。
        :param baud_rate: 波特率 (固定为9600)。
        """
        self.ser = None
        self.command_queue = Queue()
        self.stop_event = threading.Event()
        self.current_light_state = "off"

        try:
            print(f"✅ [声光报警器]: 正在尝试连接串口 {port}...")
            self.ser = serial.Serial(port, baud_rate, timeout=0.5)
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            print(f"✅ [声光报警器]: 串口 {port} 连接成功，后台线程已启动。")
        except serial.SerialException as e:
            print(f"❌ [声光报警器]: 严重错误 - 无法打开串口 '{port}'. 请检查连接和端口号。")
            print(f"   错误详情: {e}")
            # 即使串口失败，系统其他部分也应能继续运行
            self.ser = None

    def _send_command(self, light="off", buzzer="off", flicker="off"):
        """底层函数，构建并发送指令。"""
        if not self.ser: return

        try:
            work_mode = self._WORK_MODE.get(light, 0x01)
            buzzer_mode = self._BUZZER_MODE.get(buzzer, 0x01)
            flicker_rate = self._FLICKER_RATE.get(flicker, 0x01)
            self.current_light_state = light # 记录当前灯的状态

            command = bytes([self._CMD_HEADER, work_mode, buzzer_mode, flicker_rate, self._CMD_TAIL])
            self.ser.flushInput()
            self.ser.write(command)
            # 短暂等待以确保命令发送
            time.sleep(0.05)
            # 读取并忽略返回值，以保持缓冲区干净
            self.ser.read(5)
        except Exception as e:
            print(f"❌ [声光报警器]: 发送指令时发生错误: {e}")

    def _worker(self):
        """后台工作线程，处理指令队列。"""
        while not self.stop_event.is_set():
            try:
                # 阻塞等待新指令，超时1秒
                command, args = self.command_queue.get(timeout=1)

                if command == "SET_STATE":
                    light, buzzer, duration = args
                    self._send_command(light=light, buzzer="on" if duration > 0 else "off")
                    if duration > 0:
                        # 等待指定时间
                        time.sleep(duration)
                        # 等待后，关闭蜂鸣器，保持灯的状态不变
                        self._send_command(light=light, buzzer="off")
                
                elif command == "SHUTDOWN":
                    self._send_command(light="off", buzzer="off")
                    break

            except Empty:
                continue
    
    def set_startup_state(self):
        """设置启动状态：绿灯亮，蜂鸣0.25秒。"""
        self.command_queue.put(("SET_STATE", ("green", "buzz", 0.25)))

    def set_normal_state(self):
        """设置正常/待机状态：绿灯常亮。"""
        self.command_queue.put(("SET_STATE", ("green", "no_buzz", 0)))
        
    def set_ng_detected_state(self, duration_s):
        """设置检测到NG状态：黄灯亮，蜂鸣指定时间。"""
        self.command_queue.put(("SET_STATE", ("yellow", "buzz", duration_s)))

    def set_rejection_state(self, duration_s):
        """设置剔废后状态：红灯亮，蜂鸣指定时间。"""
        self.command_queue.put(("SET_STATE", ("red", "buzz", duration_s)))

    def close(self):
        """关闭控制器，释放资源。"""
        if self.ser:
            print("[声光报警器]: 正在关闭...")
            self.command_queue.put(("SHUTDOWN", None))
            self.stop_event.set()
            self.worker_thread.join(timeout=2)
            self.ser.close()
            print("[声光报警器]: 已安全关闭。")