''' 
 @Author: LI Zhaoyang  
 @Date: 2025-08-21 11:16:07  
 @Last Modified by:   LI Zhaoyang  
 @Last Modified time: 2025-08-21 11:16:07  
''' 
# rejection_controller.py

import time

class RejectionController:
    """
    管理剔废信号的发送。
    这是一个抽象接口，未来可以填充具体的硬件控制代码（例如USB继电器、DIO卡、以太网模块等）。
    """
    def __init__(self):
        """
        初始化与剔废硬件的连接。
        """
        print("✅ [剔除控制器]: 已初始化。当前为模拟模式。")
        # 示例：未来您可以在此添加硬件初始化代码
        # import serial
        # try:
        #     self.ser = serial.Serial('COM3', 9600, timeout=1)
        #     print("✅ [剔除控制器]: 成功连接到COM3端口。")
        # except Exception as e:
        #     self.ser = None
        #     print(f"❌ [剔除控制器]: 无法连接到硬件: {e}")

    def trigger_rejection_signal(self, cam_index, pulse_duration_ms=100):
        """
        发送一个剔废脉冲信号。
        :param cam_index: 触发剔除的相机索引，可用于日志或多通道控制。
        :param pulse_duration_ms: 信号脉冲的持续时间（毫秒）。
        """
        
        # --- 未来在此处填充您的真实硬件控制代码 ---
        
        # 目前，我们只打印一条日志信息来模拟信号发送
        if cam_index == -1:
            print(f"🔥🔥🔥 [剔除控制器]: 正在发送剔除信号 (持续 {pulse_duration_ms}ms)...")
        else:
            print(f"🔥🔥🔥 [剔除控制器]: 相机 {cam_index} 正在发送剔除信号 (持续 {pulse_duration_ms}ms)...")

        # 示例（使用pyserial库）:
        # if self.ser:
        #     self.ser.write(b'RELAY_ON_COMMAND')
        #     time.sleep(pulse_duration_ms / 1000.0)
        #     self.ser.write(b'RELAY_OFF_COMMAND')
        
        time.sleep(pulse_duration_ms / 1000.0) # 模拟硬件操作耗时
        
        print(f"🔥🔥🔥 [剔除控制器]: 信号发送完毕。")

    def close(self):
        """关闭硬件连接，释放资源。"""
        # 在此添加硬件资源释放代码
        # e.g., if self.ser: self.ser.close()
        print("✅ [剔除控制器]: 控制器已关闭。")