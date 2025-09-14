''' 
 @Author: LI Zhaoyang  
 @Date: 2025-08-21 11:16:18  
 @Last Modified by: GitHub Copilot
 @Last Modified time: 2025-09-14 15:30:00
''' 
# camera_manager.py (重构版 - 基于MAC地址的相机绑定机制)

import time
import json
import sys
import re
import subprocess
import os
import traceback
from MVGigE import *
from GigECamera_Types import *

# ======================================================================
# --- 单个相机控制器 ---
# ======================================================================
class CameraManager:
    """管理单个相机实例的生命周期和参数。此类应在独立的进程中实例化。"""
    _lib_initialized = False

    @classmethod
    def ensure_lib_initialized(cls):
        if not cls._lib_initialized:
            MVInitLib()
            cls._lib_initialized = True

    @classmethod
    def terminate_lib(cls):
        if cls._lib_initialized:
            MVTerminateLib()
            cls._lib_initialized = False

    def __init__(self, cam_index, mac=None):
        """
        初始化相机管理器
        :param cam_index: 配置中的相机索引
        :param mac: 相机的MAC地址（优先使用）
        """
        self.cam_index = cam_index
        self.mac = mac
        self.ip = None  # 将在open过程中根据MAC查找或设置
        self.handle = 0
        self.opened_physical_index = None  # SDK看到的实际索引
        self.max_open_attempts = 3
        self.open_backoff_base_s = 2
        self.heartbeat_timeout_ms = 30000
        self._ip_bindings = {}  # 用于首次运行时

    def set_ip_bindings(self, ip_bindings):
        """设置IP绑定映射（用于首次运行时）"""
        self._ip_bindings = ip_bindings or {}

    def open(self):
        """
        打开相机，首先尝试通过MAC地址查找IP，然后进行打开
        如果MAC地址为空，则尝试直接通过配置的IP打开
        """
        CameraManager.ensure_lib_initialized()
        
        # 首先基于MAC地址查找相机的IP
        if self.mac:
            self._find_camera_by_mac()
        
        # 如果没有MAC或找不到对应IP，尝试使用IP绑定
        if not self.ip and str(self.cam_index) in self._ip_bindings:
            self.ip = self._ip_bindings.get(str(self.cam_index))
            print(f"[相机进程 {self.cam_index}]: 使用配置的IP绑定: {self.ip}")
        
        # 如果仍然没有IP，退出
        if not self.ip:
            print(f"[相机进程 {self.cam_index}]: 无法确定相机IP，无法打开")
            return False

        for attempt in range(1, self.max_open_attempts + 1):
            print(f"[相机进程 {self.cam_index}]: 尝试打开相机 MAC={self.mac or '未知'} IP={self.ip} (第 {attempt}/{self.max_open_attempts} 次)...")
            
            try:
                # 主要方法：通过配置的IP打开
                res, self.handle = MVOpenCamByIP(self.ip)
                
                if res == MVST_SUCCESS and self.handle != 0:
                    try:
                        # 打开后验证MAC地址
                        if self.mac:
                            res_list, num_cams = MVEnumerateAllDevices()
                            found_match = False
                            if res_list == MVST_SUCCESS:
                                for i in range(num_cams):
                                    res_info, info = MVGetDevInfo(i)
                                    if res_info == MVST_SUCCESS:
                                        mac_str = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                                        ip_str = ".".join(map(str, info.mIpAddr))
                                        if ip_str == self.ip and mac_str.upper() == self.mac.upper():
                                            self.opened_physical_index = i
                                            found_match = True
                                            break
                            if not found_match:
                                print(f"  [相机进程 {self.cam_index}]: 警告 - IP {self.ip} 成功打开，但未验证到匹配的MAC {self.mac}。")
                    except Exception as e:
                        print(f"  [相机进程 {self.cam_index}]: 验证MAC时发生异常: {e}")

                    try:
                        MVSetHeartbeatTimeout(self.handle, int(self.heartbeat_timeout_ms))
                    except Exception:
                        pass
                    
                    print(f"✅ [相机进程 {self.cam_index}]: 相机已成功打开。句柄: {self.handle}")
                    return True
            except Exception as e:
                print(f"  [相机进程 {self.cam_index}]: 打开异常: {e}")
                self.handle = 0
            
            print(f"  [相机进程 {self.cam_index}]: 打开失败。")
            self.handle = 0
            
            if attempt < self.max_open_attempts:
                wait_time = max(0.5, attempt * self.open_backoff_base_s)
                print(f"  [相机进程 {self.cam_index}]: 将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        print(f"❌ [相机进程 {self.cam_index}]: 经过 {self.max_open_attempts} 次尝试后，仍无法打开相机。")
        return False

    def _find_camera_by_mac(self):
        """基于MAC地址查找相机的IP"""
        try:
            res_list, num_cams = MVEnumerateAllDevices()
            if res_list == MVST_SUCCESS:
                for i in range(num_cams):
                    res_info, info = MVGetDevInfo(i)
                    if res_info == MVST_SUCCESS:
                        mac_str = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                        if mac_str.upper() == self.mac.upper():
                            self.ip = ".".join(map(str, info.mIpAddr))
                            print(f"[相机进程 {self.cam_index}]: 找到匹配MAC {self.mac} 的相机，IP: {self.ip}")
                            return
            print(f"[相机进程 {self.cam_index}]: 未找到MAC为 {self.mac} 的相机")
        except Exception as e:
            print(f"[相机进程 {self.cam_index}]: 查找MAC时发生异常: {e}")

    def close(self):
        """关闭相机并释放库资源。"""
        try:
            if self.handle != 0:
                MVStopGrab(self.handle)
                MVCloseCam(self.handle)
                print(f"[相机进程 {self.cam_index}]: 相机已关闭。")
                self.handle = 0
        except Exception as e:
            print(f"[相机进程 {self.cam_index}]: 关闭相机时发生异常: {e}")

    def get_all_params(self):
        """获取当前相机的主要参数，并以结构化字典形式返回。"""
        if self.handle == 0: return None
        try:
            _, width = MVGetWidth(self.handle); _, height = MVGetHeight(self.handle)
            _, fps = MVGetFrameRate(self.handle); _, pixel_format = MVGetPixelFormat(self.handle)
            _, exposure = MVGetExposureTime(self.handle); _, exp_auto = MVGetExposureAuto(self.handle)
            _, gain = MVGetGain(self.handle); _, gain_auto = MVGetGainAuto(self.handle)
            _, gamma = MVGetGamma(self.handle)
            _, packet_size = MVGetPacketSize(self.handle); _, packet_delay = MVGetPacketDelay(self.handle)
            _, trigger_mode = MVGetTriggerMode(self.handle); _, trigger_source = MVGetTriggerSource(self.handle)
            _, trigger_activation = MVGetTriggerActivation(self.handle)

            return {
                "acquisition": {"width": width, "height": height, "frame_rate": fps, "pixel_format": hex(pixel_format)},
                "exposure": {"auto_mode": exp_auto, "value_us": exposure},
                "gain": {"auto_mode": gain_auto, "value_db": gain},
                "gamma": {"value": gamma},
                "trigger": {"mode": trigger_mode, "source": trigger_source, "activation": trigger_activation},
                "network": {"packet_size_bytes": packet_size, "packet_delay_us": packet_delay}
            }
        except Exception as e:
            print(f"[相机 {self.cam_index}]: 获取参数时出错: {e}")
            return None

    def get_full_status(self):
        """获取相机完整状态信息"""
        if self.handle == 0:
            return {"status": "Not Connected"}

        try:
            idx = self.opened_physical_index if self.opened_physical_index is not None else self.cam_index
            res, cam_info = MVGetDevInfo(idx)
            if res != MVST_SUCCESS:
                hardware_info = {"error": "Failed to get device info"}
            else:
                try:
                    model = cam_info.mModelName.decode('ascii', errors='ignore').strip('\x00')
                except Exception:
                    model = ''
                hardware_info = {
                    'model_name': model,
                    'mac_address': ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr]),
                    'ip_address': ".".join(map(str, cam_info.mIpAddr))
                }
            
            parameters = self.get_all_params()

            return {
                "status": "Connected",
                "index": self.cam_index,
                "handle": self.handle,
                "hardware": hardware_info,
                "parameters": parameters if parameters else "Failed to get parameters"
            }
        except Exception as e:
            return {
                "status": "Error", 
                "index": self.cam_index,
                "details": str(e)
            }

    def set_params(self, params_to_set):
        """根据结构化字典设置参数。"""
        if self.handle == 0 or not params_to_set: return None
        print(f"[相机进程 {self.cam_index}]: 正在应用配置参数...")
        results = {}
        
        trig_cfg = params_to_set.get("trigger")
        if trig_cfg:
            mode = trig_cfg.get("mode", TriggerMode_Off)
            res = MVSetTriggerMode(self.handle, int(mode))
            if res == MVST_SUCCESS:
                print(f"  [相机 {self.cam_index}]: 触发模式设置为 {mode}")
            else:
                print(f"  [相机 {self.cam_index}]: 警告 - 设置触发模式失败，错误码: {res}")
            if mode == TriggerMode_On:
                if "source" in trig_cfg: MVSetTriggerSource(self.handle, int(trig_cfg["source"]))
                if "activation" in trig_cfg: MVSetTriggerActivation(self.handle, int(trig_cfg["activation"]))
        else:
            MVSetTriggerMode(self.handle, TriggerMode_Off)
        
        if "network" in params_to_set:
            net_params = params_to_set["network"]
            if "packet_size_bytes" in net_params: MVSetPacketSize(self.handle, int(net_params["packet_size_bytes"]))
            if "packet_delay_us" in net_params: MVSetPacketDelay(self.handle, int(net_params["packet_delay_us"]))
        if "acquisition" in params_to_set:
            acq_params = params_to_set["acquisition"]
            if "width" in acq_params: MVSetWidth(self.handle, int(acq_params["width"]))
            if "height" in acq_params: MVSetHeight(self.handle, int(acq_params["height"]))
            if "frame_rate" in acq_params: MVSetFrameRate(self.handle, float(acq_params["frame_rate"]))
            if "pixel_format" in acq_params: MVSetPixelFormat(self.handle, int(acq_params["pixel_format"]))
        if "exposure" in params_to_set:
            exp_params = params_to_set["exposure"]
            if "auto_mode" in exp_params: MVSetExposureAuto(self.handle, exp_params["auto_mode"])
            if "value_us" in exp_params:
                MVSetExposureAuto(self.handle, ExposureAuto_Off)
                MVSetExposureTime(self.handle, float(exp_params["value_us"]))
        if "gain" in params_to_set:
            gain_params = params_to_set["gain"]
            if "auto_mode" in gain_params: MVSetGainAuto(self.handle, gain_params["auto_mode"])
            if "value_db" in gain_params:
                MVSetGainAuto(self.handle, GainAuto_Off)
                MVSetGain(self.handle, float(gain_params["value_db"]))
        if "gamma" in params_to_set:
            gamma_params = params_to_set["gamma"]
            if "value" in gamma_params:
                MVSetGamma(self.handle, float(gamma_params["value"]))
        
        print(f"[相机进程 {self.cam_index}]: 参数应用完成。")
        return results

# ======================================================================
# --- 多相机设置工具 ---
# ======================================================================
class MultiCameraSetup:
    """工具类：管理相机发现、MAC绑定和IP分配。支持首次运行自动配置。"""
    DEFAULT_IP = "192.168.200.1"  # 默认的相机初始IP（相机板载程序设置）
    
    def __init__(self, config):
        """初始化多相机设置工具"""
        self.config = config
        self.camera_setup = self.config.get('camera_setup', {})
        if not self.camera_setup:
            print("[SetupTool]: 警告 - 在 config.json 中未找到 'camera_setup' 部分。")
        
        # 读取配置参数
        self.bootstrap_default_ip = self.camera_setup.get('bootstrap_default_ip', self.DEFAULT_IP)
        self.bootstrap_assign_ips = self.camera_setup.get('bootstrap_assign_ips', True)
        
        # 初始化相机SDK
        CameraManager.ensure_lib_initialized()
        print("[SetupTool]: 相机库已初始化。")

    def bootstrap_mac_bindings(self):
        """
        引导式MAC绑定：检查现有绑定，如果没有则执行自动发现
        返回值：成功则返回True，并且会在必要时更新config
        """
        # 检查是否有现有的相机绑定
        bindings = self.camera_setup.get('camera_bindings', [])
        if bindings:
            print(f"[SetupTool]: 已从配置中加载 {len(bindings)} 个相机绑定")
            return True
        
        print("[SetupTool]: 未找到相机绑定，执行首次运行引导...")
        
        # 扫描所有相机
        res, num_devices = MVEnumerateAllDevices()
        if res != MVST_SUCCESS or num_devices == 0:
            print("❌ [SetupTool]: 未发现任何相机设备。引导程序失败。")
            return False
        
        print(f"[SetupTool]: 发现 {num_devices} 台相机设备")
        
        # 获取本机网卡IP列表
        nic_ips = self._get_physical_nic_ipv4()
        if not nic_ips:
            print("❌ [SetupTool]: 未找到可用的物理以太网卡IPv4地址。无法分配IP。")
            return False
        
        print(f"[SetupTool]: 发现可用网卡IP: {nic_ips}")
        
        # 收集所有相机的MAC地址和当前IP
        cameras = []
        for i in range(num_devices):
            res_info, info = MVGetDevInfo(i)
            if res_info == MVST_SUCCESS:
                mac = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                ip = ".".join(map(str, info.mIpAddr))
                cameras.append({
                    "physical_index": i,
                    "mac": mac.upper(),
                    "current_ip": ip
                })
        
        if not cameras:
            print("❌ [SetupTool]: 未能获取相机信息。")
            return False
        
        print(f"[SetupTool]: 成功获取 {len(cameras)} 台相机的信息")
        
        # 如果开启自动分配IP且相机当前IP是默认IP
        if self.bootstrap_assign_ips:
            needs_ip_assignment = any(cam["current_ip"] == self.bootstrap_default_ip for cam in cameras)
            if needs_ip_assignment:
                print("[SetupTool]: 检测到相机使用默认IP，将自动分配新IP...")
                self._assign_ips_to_cameras(cameras, nic_ips)
        
        # 创建相机绑定并保存到配置
        new_bindings = []
        for i, cam in enumerate(cameras):
            new_bindings.append({
                "index": i,
                "mac": cam["mac"],
                "ip": cam["current_ip"]
            })
        
        if new_bindings:
            # 更新配置
            self.config["camera_setup"]["camera_bindings"] = new_bindings
            self.config["camera_setup"]["expected_cameras"] = len(new_bindings)
            
            try:
                with open('config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                print(f"✅ [SetupTool]: 已将 {len(new_bindings)} 台相机的绑定信息保存到config.json")
            except Exception as e:
                print(f"❌ [SetupTool]: 写入配置文件失败: {e}")
                return False
        
        return True

    def _assign_ips_to_cameras(self, cameras, nic_ips):
        """
        为处于默认IP的相机分配新IP
        根据网卡IP分配相机IP（网卡IP+1，+2，...）
        """
        # 按网卡分组，为每个网卡连接的相机分配IP
        nic_cameras = {}
        
        # 先对使用默认IP的相机进行分配
        default_ip_cameras = [cam for cam in cameras if cam["current_ip"] == self.bootstrap_default_ip]
        if not default_ip_cameras:
            return
        
        print(f"[SetupTool]: 发现 {len(default_ip_cameras)} 台相机使用默认IP {self.bootstrap_default_ip}")
        
        # 简单分配策略：平均分配到各网卡
        cameras_per_nic = len(default_ip_cameras) // len(nic_ips) + 1
        
        current_nic_idx = 0
        current_nic_count = 0
        
        for cam in default_ip_cameras:
            # 切换网卡
            if current_nic_count >= cameras_per_nic:
                current_nic_idx = (current_nic_idx + 1) % len(nic_ips)
                current_nic_count = 0
            
            nic_ip = nic_ips[current_nic_idx]
            nic_prefix = ".".join(nic_ip.split('.')[:3])
            
            # 分配IP：网卡IP+1+当前计数
            host_part = int(nic_ip.split('.')[-1]) + 1 + current_nic_count
            if host_part >= 254:  # 避免广播地址
                host_part = 254
            
            target_ip = f"{nic_prefix}.{host_part}"
            
            # 设置IP
            print(f"  -> 正在为 MAC {cam['mac']} 分配 IP {target_ip} (通过网卡 {nic_ip})...")
            
            # 从MAC字符串转换为字节数组
            mac_bytes = bytes([int(x, 16) for x in cam["mac"].split(":")])
            
            res_force = MVForceIp(
                mac_bytes, 
                target_ip.encode('ascii'), 
                "255.255.255.0".encode('ascii'), 
                "0.0.0.0".encode('ascii')
            )
            
            if res_force == MVST_SUCCESS:
                print(f"  ✅ IP分配成功")
                cam["current_ip"] = target_ip  # 更新内存中的IP
            else:
                print(f"  ❌ IP分配失败，错误码 {res_force}")
            
            current_nic_count += 1
            time.sleep(1)  # 等待相机应用设置
    
    def _get_physical_nic_ipv4(self):
        """收集本机物理以太网IPv4地址列表，过滤虚拟网卡。"""
        ips = []
        try:
            output = subprocess.check_output(['ipconfig', '/all'], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
            text = output.decode(sys.getdefaultencoding(), errors='ignore')
            
            sections = re.split(r"\r?\n\r?\n", text)
            for sec in sections:
                header_match = re.search(r"^(.*?):\s*$", sec, flags=re.MULTILINE)
                if not header_match: continue
                
                header = header_match.group(1).lower()
                is_ethernet = 'ethernet' in header or '以太网' in header
                is_virtual = any(x in header for x in ['wlan', 'wi-fi', '无线', 'bluetooth', '蓝牙', 'vmware', 'virtual', 'hyper-v', 'vethernet', 'loopback'])
                
                if is_ethernet and not is_virtual:
                    m = re.search(r"IPv4 (?:Address|地址)[^:]*: ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", sec)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith('127.'):
                            ips.append(ip)
        except Exception as e:
            print(f"[SetupTool]: 解析网卡IPv4时出错: {e}")
        return ips

    def cleanup(self):
        """释放资源"""
        try:
            CameraManager.terminate_lib()
            print("[SetupTool]: 相机库资源已释放。")
        except Exception as e:
            print(f"[SetupTool]: 清理资源时出错: {e}")