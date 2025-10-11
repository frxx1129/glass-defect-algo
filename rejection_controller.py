'''
 @Author: LI Zhaoyang
 @Date: 2025-08-21 11:16:07
 @Last Modified by:   LI Zhaoyang
 @Last Modified time: 2025-08-21 11:16:07
'''
# rejection_controller.py

import time
from typing import Optional, Union

try:
    import serial  # pyserial
except Exception:
    serial = None


class RejectionController:
    """
    多路剔废控制器：支持串口协议 A0 xx yy zz（zz=前三字节求和&0xFF）。

    - 默认通道映射支持 >=4 路，扩展至最多 8 路（根据硬件与配置决定）。
    - 命令: 0x01=ON, 0x00=OFF, 0x02=QUERY
    - 脉冲: 发送ON, 延时, 发送OFF
    - 若未配置串口或 pyserial 不可用，则自动降级为模拟模式（仅打印日志）。
    """

    def __init__(self, port: Optional[str] = None, baud: int = 115200):
        self.port = port
        self.baud = int(baud or 115200)
        self.ser = None
        if port and serial is not None:
            try:
                self.ser = serial.Serial(port, self.baud, timeout=0.2)
                print(f"✅ [剔除控制器]: 串口已连接 {port} @ {self.baud}")
            except Exception as e:
                self.ser = None
                print(f"❌ [剔除控制器]: 无法打开串口 {port}: {e}，将使用模拟模式。")
        else:
            if port and serial is None:
                print("⚠️ [剔除控制器]: 未安装 pyserial，使用模拟模式。")
            else:
                print("✅ [剔除控制器]: 未配置串口，使用模拟模式。")

    @staticmethod
    def _as_channel(route: Optional[Union[str, int]], cam_index: int) -> int:
        """根据 API 指令 route 解析通道号(1..8)。不使用 cam_index 作为映射依据。"""
        if isinstance(route, str):
            r = route.lower()
            if r in ("l", "left"): return 1
            if r in ("m", "mid", "middle", "center", "centre"): return 2
            if r in ("r", "right"): return 3
            if r in ("a", "all", "any"): return 4
        if isinstance(route, int) and 1 <= route <= 8:
            return route
        # 默认：未指定路由时，使用 ALL
        return 4

    @staticmethod
    def _packet(channel: int, cmd: int) -> bytes:
        head = 0xA0
        # 允许最多 8 路，若硬件不支持，多余通道不会生效
        channel = max(1, min(8, int(channel)))
        cmd = int(cmd) & 0xFF
        chk = (head + channel + cmd) & 0xFF
        return bytes([head, channel, cmd, chk])

    def send_command(self, channel: int, cmd: int):
        pkt = self._packet(channel, cmd)
        if self.ser:
            try:
                self.ser.write(pkt)
            except Exception as e:
                print(f"❌ [剔除控制器]: 串口发送失败: {e}")
        else:
            print(f"[剔除控制器][模拟] -> {list(pkt)} (ch={channel}, cmd={cmd})")

    def pulse(self, channel: int, duration_ms: int):
        duration_s = max(0.0, (int(duration_ms or 0)) / 1000.0)
        self.send_command(channel, 0x01)  # ON
        time.sleep(duration_s)
        self.send_command(channel, 0x00)  # OFF

    def query(self, channel: int):
        self.send_command(channel, 0x02)

    def trigger_rejection_signal(self, cam_index: int, pulse_duration_ms: int = 100, route: Optional[Union[str, int]] = None):
        channel = self._as_channel(route, cam_index)
        # 记录日志
        cam_desc = "ALL" if cam_index == -1 else f"Cam{cam_index+1}"
        print(f"🔥🔥🔥 [剔除控制器]: {cam_desc} -> 通道 {channel} 脉冲 {pulse_duration_ms}ms")
        self.pulse(channel, pulse_duration_ms)
        print("🔥🔥🔥 [剔除控制器]: 脉冲完成。")

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        print("✅ [剔除控制器]: 控制器已关闭。")