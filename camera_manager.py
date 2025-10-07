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
import ctypes
from MVGigE import *
from GigECamera_Types import *

def is_admin():
    """检查当前进程是否具有管理员权限"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False

def run_as_admin(script_path=None, *argv):
    """在 Windows 上尝试以管理员权限重新启动当前程序。
    - 若已是管理员，直接返回 True。
    - 若非管理员：通过 ShellExecuteW('runas') 启动提升权限的新进程，成功返回 True，失败返回 False。
    - 参数兼容调用方 (script_path, *sys.argv[1:]) 的用法。
    """
    try:
        # 非 Windows 平台不尝试提权
        if os.name != 'nt':
            print("[相机初始化]: 非 Windows 平台，跳过管理员提权。")
            return False
        if is_admin():
            print("[相机初始化]: 已以管理员权限运行")
            return True
        print("[相机初始化]: 尝试管理员提权启动...")
        # 组装参数
        if script_path is None:
            script_path = sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]
        # 若为打包的可执行文件，直接以自身提权；否则以 python.exe 运行脚本
        if getattr(sys, 'frozen', False):
            executable = script_path
            parameters = " ".join(argv)
        else:
            executable = sys.executable
            # 将脚本路径作为第一个参数，其余参数拼接
            quoted_script = f'"{script_path}"' if ' ' in str(script_path) else str(script_path)
            quoted_args = " ".join([f'"{a}"' if isinstance(a, str) and (' ' in a) else str(a) for a in argv])
            parameters = (quoted_script + (" " + quoted_args if quoted_args else "")).strip()
        # 调用 ShellExecuteW 以 runas 方式启动
        ret = ctypes.windll.shell32.ShellExecuteW(None, "runas", executable, parameters, None, 1)
        if int(ret) <= 32:
            print(f"[相机初始化]: ShellExecute 提权失败，返回码: {ret}")
            return False
        return True
    except Exception as e:
        print(f"[相机初始化]: 提权过程中发生异常: {e}")
        return False

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

    def open(self):
        """
        打开相机，优先尝试通过物理索引(相机ID)打开，然后尝试通过MAC地址查找IP，最后尝试通过IP打开
        """
        CameraManager.ensure_lib_initialized()
        
        # 步骤1: 使用MVEnumerateAllDevices确保我们能看到所有相机，包括不在同一网段的
        res_enum, num_all_cams = MVEnumerateAllDevices()
        
        # 步骤2: 更新相机列表并获取数量（这只会获取网络可达的相机）
        MVUpdateCameraList()
        res_list, num_cams = MVGetNumOfCameras()
        
        # 如果未找到任何相机，直接返回失败
        if num_all_cams == 0:
            print(f"[相机进程 {self.cam_index}]: 未找到任何相机设备")
            return False
            
        # 步骤3: 尝试通过MAC地址查找物理索引(SDK看到的索引)
        physical_index = None
        if self.mac:
            for i in range(num_all_cams):
                res_info, info = MVGetDevInfo(i)
                if res_info == MVST_SUCCESS:
                    mac_str = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                    if mac_str.upper() == self.mac.upper():
                        physical_index = i
                        self.ip = ".".join(map(str, info.mIpAddr))
                        print(f"[相机进程 {self.cam_index}]: 相机MAC={self.mac}，当前运行时物理索引={physical_index}，IP={self.ip}")
                        break
        
        # 步骤4: 优先尝试通过物理索引(相机ID)打开
        if physical_index is not None:
            for attempt in range(1, self.max_open_attempts + 1):
                if attempt > 1:
                    print(f"[相机进程 {self.cam_index}]: 重试通过物理索引打开，尝试 {attempt}/{self.max_open_attempts}")
                
                try:
                    res, self.handle = MVOpenCamByIndex(physical_index)
                    
                    if res == MVST_SUCCESS and self.handle != 0:
                        self.opened_physical_index = physical_index
                        try:
                            MVSetHeartbeatTimeout(self.handle, int(self.heartbeat_timeout_ms))
                        except Exception:
                            pass
                        
                        print(f"✅ [相机进程 {self.cam_index}]: 相机已通过运行时物理索引 {physical_index} 成功打开（逻辑索引={self.cam_index}）")
                        return True
                except Exception as e:
                    print(f"[相机进程 {self.cam_index}]: 通过物理索引打开失败: {str(e)[:50]}")
                    self.handle = 0
                
                if attempt < self.max_open_attempts:
                    wait_time = max(0.5, attempt * self.open_backoff_base_s)
                    time.sleep(wait_time)
        
        # 步骤5: 如果通过物理索引打开失败，尝试通过IP打开
        # 首先基于MAC地址查找相机的IP（如果还没有）
        if not self.ip and self.mac:
            self._find_camera_by_mac()
        
        # 如果仍然没有IP，从camera_bindings中获取
        if not self.ip:
            try:
                import json
                with open('config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                camera_bindings = config.get('camera_setup', {}).get('camera_bindings', [])
                for binding in camera_bindings:
                    if binding.get('index') == self.cam_index:
                        self.ip = binding.get('ip')
                        if self.ip:
                            print(f"[相机进程 {self.cam_index}]: 从配置文件获取IP: {self.ip}")
                            break
            except Exception:
                pass
        
        # 如果仍然没有IP，退出
        if not self.ip:
            print(f"[相机进程 {self.cam_index}]: 无法确定相机IP，无法打开")
            return False

        for attempt in range(1, self.max_open_attempts + 1):
            if attempt > 1:
                print(f"[相机进程 {self.cam_index}]: 重试通过IP打开，尝试 {attempt}/{self.max_open_attempts}")
            else:
                print(f"[相机进程 {self.cam_index}]: 尝试通过IP打开相机: {self.ip}")
            
            try:
                # 通过配置的IP打开
                res, self.handle = MVOpenCamByIP(self.ip)
                
                if res == MVST_SUCCESS and self.handle != 0:
                    try:
                        # 设置心跳超时
                        MVSetHeartbeatTimeout(self.handle, int(self.heartbeat_timeout_ms))
                    except Exception:
                        pass
                    
                    print(f"✅ [相机进程 {self.cam_index}]: 相机已通过IP成功打开")
                    return True
            except Exception as e:
                print(f"[相机进程 {self.cam_index}]: 通过IP打开失败: {str(e)[:50]}")
                self.handle = 0
            
            if attempt < self.max_open_attempts:
                wait_time = max(0.5, attempt * self.open_backoff_base_s)
                time.sleep(wait_time)
        
        print(f"❌ [相机进程 {self.cam_index}]: 经过多次尝试后，仍无法打开相机")
        return False

    def _find_camera_by_mac(self):
        """基于MAC地址查找相机的IP和物理索引"""
        try:
            # 使用 MVEnumerateAllDevices 获取所有相机，包括不在同一网段的
            res_list, num_cams = MVEnumerateAllDevices()
            if res_list == MVST_SUCCESS:
                for i in range(num_cams):
                    res_info, info = MVGetDevInfo(i)
                    if res_info == MVST_SUCCESS:
                        mac_str = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                        if mac_str.upper() == self.mac.upper():
                            self.ip = ".".join(map(str, info.mIpAddr))
                            self.opened_physical_index = i
                            print(f"[相机进程 {self.cam_index}]: 找到MAC对应相机，当前运行时物理索引={i}, IP={self.ip}")
                            return
        except Exception as e:
            print(f"[相机进程 {self.cam_index}]: 查找MAC时出错: {str(e)[:50]}")

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
    DEFAULT_IP = "192.168.1.200"  # 默认的相机初始IP（相机板载程序设置）
    
    def __init__(self, config):
        """初始化多相机设置工具"""
        self.config = config
        self.camera_setup = self.config.get('camera_setup', {})
        if not self.camera_setup:
            print("[SetupTool]: 警告 - 在 config.json 中未找到 'camera_setup' 部分。")
        
        # 读取配置参数
        self.bootstrap_default_ip = self.camera_setup.get('bootstrap_default_ip', self.DEFAULT_IP)
        self.bootstrap_assign_ips = self.camera_setup.get('bootstrap_assign_ips', True)
        
        # 检查管理员权限
        self.has_admin = is_admin()
        if not self.has_admin:
            print("[SetupTool]: ⚠️ 未获得管理员权限，网络配置操作可能受限")
        
        # 初始化相机SDK
        CameraManager.ensure_lib_initialized()
        print("[SetupTool]: 相机库已初始化。")
        
        # 优先从配置中读取网卡信息
        nic_config = self.camera_setup.get('network_interfaces', [])
        if nic_config:
            self.nic_info = self._load_nic_info_from_config(nic_config)
            print(f"[SetupTool]: 从配置文件加载了 {len(self.nic_info)} 个网卡")
        else:
            # 如果配置中没有网卡信息，自动获取
            self.nic_info = self._get_physical_nic_info()
            print(f"[SetupTool]: 自动检测到 {len(self.nic_info)} 个物理网卡")
        
        if self.nic_info:
            for nic in self.nic_info:
                nic_index = nic.get('index', -1)
                print(f"  网卡 {nic_index}: IP={nic['ip']}")
        else:
            print("[SetupTool]: ❌ 未发现有效网卡")

    def _load_nic_info_from_config(self, nic_config):
        """从配置文件加载网卡信息"""
        nic_info = []
        
        for nic in nic_config:
            if not isinstance(nic, dict):
                continue
                
            # 必须包含index和ip
            if 'index' not in nic or 'ip' not in nic:
                print(f"[SetupTool]: 警告 - 网卡配置缺少必要字段 (index 或 ip): {nic}")
                continue
                
            # 创建网卡信息对象（只需要index和ip）
            nic_obj = {
                'index': nic['index'],
                'ip': nic['ip'],
                'mac': '00:00:00:00:00:00',  # 使用默认值
                'name': f"网卡 {nic['index']}"  # 使用默认名称
            }
            
            nic_info.append(nic_obj)
            
        return nic_info

    def _get_physical_nic_info(self):
        """获取物理网卡的详细信息，包括MAC地址、IP地址和名称"""
        nic_info = []
        try:
            output = subprocess.check_output(['ipconfig', '/all'], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
            text = output.decode(sys.getdefaultencoding(), errors='ignore')
            
            sections = re.split(r"\r?\n\r?\n", text)
            for sec in sections:
                header_match = re.search(r"^(.*?):\s*$", sec, flags=re.MULTILINE)
                if not header_match: continue
                
                header = header_match.group(1)
                header_lower = header.lower()
                is_ethernet = 'ethernet' in header_lower or '以太网' in header_lower
                is_virtual = any(x in header_lower for x in ['wlan', 'wi-fi', '无线', 'bluetooth', '蓝牙', 
                                                         'vmware', 'virtual', 'hyper-v', 'vethernet', 'loopback'])
                
                if is_ethernet and not is_virtual:
                    ip_match = re.search(r"IPv4 (?:Address|地址)[^:]*: ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", sec)
                    mac_match = re.search(r"(?:Physical Address|物理地址)[^:]*: ([0-9A-F-]+)", sec)
                    
                    if ip_match and mac_match:
                        ip = ip_match.group(1)
                        mac = mac_match.group(1).replace('-', ':')
                        
                        if not ip.startswith('127.'):
                            nic_info.append({
                                'name': header,
                                'ip': ip,
                                'mac': mac
                            })
        except Exception as e:
            print(f"[SetupTool]: 解析网卡信息时出错: {e}")
        return nic_info

    def _get_physical_nic_ipv4(self):
        """返回所有物理网卡的IP地址列表"""
        return [nic['ip'] for nic in self.nic_info] if self.nic_info else []

    def scan_all_cameras(self):
        """扫描所有可用的相机，返回相机信息列表"""
        print("[SetupTool]: 扫描所有可用相机...")
        cameras = []
        
        res, num_devices = MVEnumerateAllDevices()
        if res != MVST_SUCCESS or num_devices == 0:
            print("[SetupTool]: 未发现任何相机设备")
            return cameras
        
        print(f"[SetupTool]: 发现 {num_devices} 台相机")
        
        # 获取相机网卡连接关系
        camera_nic_map = self._map_cameras_to_nics()
        
        for i in range(num_devices):
            res_info, info = MVGetDevInfo(i)
            if res_info == MVST_SUCCESS:
                mac = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                ip = ".".join(map(str, info.mIpAddr))
                
                # 查找该相机连接的网卡
                connected_nic = None
                for mac_addr, nic_idx in camera_nic_map.items():
                    if mac.upper() == mac_addr.upper():
                        if nic_idx < len(self.nic_info):
                            connected_nic = self.nic_info[nic_idx]
                            break
                
                try:
                    model = info.mModelName.decode('ascii', errors='ignore').strip('\x00')
                except Exception:
                    model = 'Unknown'
                    
                camera_info = {
                    "physical_index": i,
                    "mac": mac.upper(),
                    "current_ip": ip,
                    "model": model,
                    "connected_nic": connected_nic
                }
                
                cameras.append(camera_info)
                
                # 打印相机信息
                nic_info = f", 网卡IP: {connected_nic['ip']}" if connected_nic else ""
                print(f"  相机 {i}: MAC={mac}, IP={ip}{nic_info}")
        
        return cameras

    def _map_cameras_to_nics(self):
        """尝试映射相机到它们连接的物理网卡"""
        camera_nic_map = {}  # 相机MAC -> 网卡索引
        
        if not self.nic_info:
            return camera_nic_map
        
        # 查看配置文件中是否有camera_nic_bindings
        camera_nic_bindings = self.camera_setup.get('camera_nic_bindings', {})
        if camera_nic_bindings:
            print("[SetupTool]: 从配置文件加载相机-网卡绑定关系")
            # 直接从配置中加载绑定关系
            for cam_idx, nic_idx in camera_nic_bindings.items():
                try:
                    cam_idx_str = str(cam_idx)
                    # 查找该相机的MAC地址
                    for binding in self.camera_setup.get('camera_bindings', []):
                        if str(binding.get('index')) == cam_idx_str:
                            cam_mac = binding.get('mac')
                            if cam_mac:
                                camera_nic_map[cam_mac.upper()] = int(nic_idx)
                                print(f"  相机 {cam_idx} (MAC: {cam_mac}) -> 网卡 {nic_idx}")
                except Exception as e:
                    print(f"[SetupTool]: 处理相机-网卡绑定时出错: {e}")
            
            if camera_nic_map:
                return camera_nic_map
        
        # 如果没有配置或配置无效，我们需要确定相机与网卡的对应关系
        # 此处不再自动检测网络，而是基于配置的网卡索引进行分配
        print("[SetupTool]: 使用配置的网卡索引分配相机-网卡连接关系")
        
        try:
            # 扫描所有相机
            res, num_devices = MVEnumerateAllDevices()
            if res != MVST_SUCCESS or num_devices == 0:
                return camera_nic_map
                
            # 获取所有相机MAC和IP
            cameras = []
            for i in range(num_devices):
                res_info, info = MVGetDevInfo(i)
                if res_info == MVST_SUCCESS:
                    mac = ":".join([f"{b:02X}" for b in info.mEthernetAddr]).upper()
                    ip = ".".join(map(str, info.mIpAddr))
                    cameras.append({"mac": mac, "ip": ip, "physical_index": i})
            
            # 为每台相机分配网卡，优先考虑物理索引对应网卡索引
            for camera in cameras:
                camera_mac = camera["mac"]
                physical_idx = camera["physical_index"]
                
                # 查找该相机对应的逻辑索引（仅用于日志显示）
                cam_logical_idx = None
                for binding in self.camera_setup.get('camera_bindings', []):
                    if binding.get('mac', '').upper() == camera_mac:
                        cam_logical_idx = binding.get('index')
                        break
                
                # 检查是否有匹配物理索引的网卡
                nic_idx_found = False
                for nic_idx, nic in enumerate(self.nic_info):
                    if nic.get('index') == physical_idx:
                        camera_nic_map[camera_mac] = physical_idx
                        print(f"  相机 MAC={camera_mac} (物理索引={physical_idx}, 逻辑索引={cam_logical_idx}) -> 网卡 {physical_idx} (物理索引匹配)")
                        nic_idx_found = True
                        break
                
                # 如果没有匹配物理索引的网卡，尝试使用逻辑索引
                if not nic_idx_found and cam_logical_idx is not None:
                    for nic in self.nic_info:
                        if nic.get('index') == cam_logical_idx:
                            camera_nic_map[camera_mac] = cam_logical_idx
                            print(f"  相机 MAC={camera_mac} (物理索引={physical_idx}, 逻辑索引={cam_logical_idx}) -> 网卡 {cam_logical_idx} (逻辑索引匹配)")
                            nic_idx_found = True
                            break
                
                # 如果仍未找到，根据物理索引在可用网卡中的位置分配
                if not nic_idx_found and self.nic_info:
                    try:
                        # 使用物理索引作为优先依据
                        nic_index_in_list = physical_idx % len(self.nic_info)
                        nic_idx = self.nic_info[nic_index_in_list].get('index', nic_index_in_list)
                        camera_nic_map[camera_mac] = nic_idx
                        print(f"  相机 MAC={camera_mac} (物理索引={physical_idx}) -> 网卡 {nic_idx} (基于物理索引分配)")
                    except Exception as e:
                        print(f"  相机 MAC={camera_mac} -> 分配网卡失败: {str(e)}")
                        # 如果有可用网卡，默认使用第一个
                        if self.nic_info:
                            nic_idx = self.nic_info[0].get('index', 0)
                            camera_nic_map[camera_mac] = nic_idx
                            print(f"  相机 MAC={camera_mac} -> 默认使用网卡 {nic_idx}")
            
        except Exception as e:
            print(f"[SetupTool]: 映射相机到网卡时出错: {e}")
            
        return camera_nic_map

    def assign_camera_ips(self, cameras):
        """为每个相机分配IP地址，根据其连接的网卡"""
        if not cameras:
            print("[SetupTool]: 没有发现相机，无法分配IP")
            return cameras
            
        print(f"[SetupTool]: 为 {len(cameras)} 台相机分配IP地址")
        
        # 检查网卡信息
        if not self.nic_info:
            print("[SetupTool]: ❌ 没有可用的网卡信息，无法分配相机IP")
            return cameras
            
        # 创建网卡索引映射，可能是基于index字段，而不是列表索引
        indexed_nics = {}
        for nic in self.nic_info:
            nic_idx = nic.get('index', -1)
            if nic_idx >= 0:
                indexed_nics[nic_idx] = nic
        
        if not indexed_nics:
            print("[SetupTool]: ❌ 未找到有效的网卡索引，使用自动索引")
            # 回退到使用列表索引
            indexed_nics = {i: nic for i, nic in enumerate(self.nic_info)}
            
        print(f"[SetupTool]: 找到 {len(indexed_nics)} 个有索引的网卡")
            
        # 获取相机与网卡的绑定关系，优先使用硬件特性建立
        camera_nic_map = self._map_cameras_to_nics()
        
        # 为每个相机分配IP
        for cam in cameras:
            # 使用MAC地址作为稳定标识
            cam_mac = cam["mac"].upper()
            physical_idx = cam["physical_index"]
            binding_found = False
            skip_ip_calculation = False  # 初始化标志变量
            target_ip = None  # 初始化目标IP
            nic_idx = None
            
            # 查找该相机对应的逻辑索引（仅用于日志显示）
            cam_logical_idx = None
            for binding in self.camera_setup.get('camera_bindings', []):
                if binding.get('mac', '').upper() == cam_mac:
                    cam_logical_idx = binding.get('index')
                    break
            
            # 优先使用从_map_cameras_to_nics获取的映射
            if cam_mac in camera_nic_map:
                nic_idx = camera_nic_map[cam_mac]
                if nic_idx in indexed_nics:
                    binding_found = True
                    print(f"[SetupTool]: 相机 MAC={cam_mac} (物理索引={physical_idx}, 逻辑索引={cam_logical_idx}) 使用映射的网卡 {nic_idx}")
            
            # 如果没有找到绑定关系，使用连接的网卡
            if not binding_found:
                # 查找相机连接的网卡
                connected_nic = cam.get("connected_nic")
                if connected_nic:
                    # 根据IP找到网卡索引
                    for idx, nic in indexed_nics.items():
                        if nic["ip"] == connected_nic["ip"]:
                            binding_found = True
                            nic_idx = idx
                            print(f"[SetupTool]: 相机 MAC={cam_mac} 使用连接的网卡 {nic_idx}")
                            break
            
            # 如果仍然没有找到，使用物理索引对应的网卡或第一个可用的网卡
            if not binding_found:
                # 尝试使用物理索引对应的网卡
                if physical_idx in indexed_nics:
                    nic_idx = physical_idx
                    binding_found = True
                    print(f"[SetupTool]: 相机 MAC={cam_mac} 使用物理索引 {physical_idx} 对应的网卡")
                else:
                    # 仍然没有找到，使用第一个可用的网卡
                    nic_idx = next(iter(indexed_nics.keys()))
                    print(f"[SetupTool]: 相机 MAC={cam_mac} 无法确定网卡，使用默认网卡 {nic_idx}")
            
            # 获取网卡信息并设置IP
            nic = indexed_nics[nic_idx]
            nic_ip = nic["ip"]
            nic_prefix = ".".join(nic_ip.split('.')[:3])
            nic_host_part = int(nic_ip.split('.')[-1])
            
            # 检查现有的IP配置，如果存在则优先使用
            existing_ip = None
            for binding in self.camera_setup.get('camera_bindings', []):
                if binding.get('mac', '').upper() == cam_mac:
                    existing_ip = binding.get('ip')
                    break
            
            # 验证IP前缀是否与网卡在同一网段
            if existing_ip:
                ip_prefix = ".".join(existing_ip.split('.')[:3])
                if ip_prefix == nic_prefix:
                    # IP在同一网段，保持原有IP
                    target_ip = existing_ip
                    # 设置跳过后续计算的标志
                    skip_ip_calculation = True
                    print(f"  -> 保持相机 MAC={cam_mac} (逻辑索引={cam_logical_idx}) 的现有IP: {existing_ip}")
            
            # 默认初始化host_part变量，避免未定义错误
            host_part = 100  # 默认值
            
            # 如果没有找到有效的现有IP，则计算新IP
            if not skip_ip_calculation:
                # 从MAC地址生成一个稳定的主机部分，而不依赖于逻辑索引
                try:
                    # 解析MAC地址的最后一个字节作为基础
                    mac_parts = cam_mac.split(':')
                    mac_last_byte = int(mac_parts[-1], 16)
                    # 确保在有效范围内（避免0、1和255）
                    host_part = 100 + (mac_last_byte % 100)
                    print(f"  -> 使用MAC地址 {cam_mac} 的最后部分 {mac_last_byte} 计算IP")
                except Exception:
                    # 如果解析MAC失败，回退到简单的递增方式
                    host_part = nic_host_part + 10  # 从网卡IP+10开始
                    print(f"  -> 无法解析MAC地址，使用默认偏移 {host_part}")
                
                # 确保地址不超过254（避免广播地址）
                if host_part >= 254:
                    host_part = 100 + (host_part % 100)  # 回环到100-253范围
                
                target_ip = f"{nic_prefix}.{host_part}"
            
            # 如果IP未变，则不需要重新分配
            if cam["current_ip"] == target_ip:
                print(f"  -> 相机 MAC={cam_mac} 已经使用正确的IP: {target_ip}")
                continue
                
            # 设置IP
            print(f"  -> 为相机 MAC={cam_mac} 分配IP: {target_ip} (原IP: {cam['current_ip']})")
            
            # 转换MAC地址为字节数组
            mac_bytes = bytes([int(x, 16) for x in cam_mac.split(":")])
            
            res_force = MVForceIp(
                mac_bytes, 
                target_ip.encode('ascii'), 
                "255.255.255.0".encode('ascii'), 
                "0.0.0.0".encode('ascii')
            )
            
            if res_force == MVST_SUCCESS:
                print(f"  ✅ IP分配成功: {cam['current_ip']} -> {target_ip}")
                cam["current_ip"] = target_ip  # 更新内存中的IP
            else:
                print(f"  ❌ IP分配失败，错误码: {res_force}")
                
            # 等待相机应用新IP
            time.sleep(1)
        
        return cameras

    def create_camera_mappings(self, cameras):
        """
        创建配置中的相机映射，包括:
        1. 物理索引到逻辑索引的映射
        2. MAC地址到逻辑索引的映射
        3. IP地址到逻辑索引的映射
        """
        if not cameras:
            print("[SetupTool]: 没有相机信息，无法创建映射")
            return False
            
        # 检查配置中是否有现有的相机映射
        existing_bindings = self.camera_setup.get('camera_bindings', [])
        existing_indices = {}
        
        # 如果存在现有绑定，提取每个MAC的逻辑索引
        if existing_bindings:
            for binding in existing_bindings:
                mac = binding.get('mac', '').upper()
                idx = binding.get('index')
                if mac and idx is not None:
                    existing_indices[mac] = idx
        
        # 创建新的相机绑定列表
        new_bindings = []
        logical_indices = set()
        mac_index_map = {}
        
        # 为每个相机分配逻辑索引
        for cam in cameras:
            mac = cam["mac"].upper()
            
            # 如果MAC地址已有映射，使用现有索引
            if mac in existing_indices:
                logical_idx = existing_indices[mac]
            else:
                # 否则分配一个新的唯一索引
                logical_idx = len(new_bindings)
                while logical_idx in logical_indices:
                    logical_idx += 1
            
            logical_indices.add(logical_idx)
            mac_index_map[mac] = logical_idx
            
            new_bindings.append({
                "index": logical_idx,  # 逻辑索引：用户定义的相机顺序，重要且应该保持稳定
                "physical_index": cam["physical_index"],  # 物理索引：仅当前运行时有效，每次启动可能变化
                "mac": cam["mac"],  # MAC地址：相机的唯一硬件标识
                "ip": cam["current_ip"],  # 当前IP地址
                "model": cam.get("model", "Unknown")  # 相机型号
            })
        
        # 按逻辑索引排序
        new_bindings.sort(key=lambda x: x["index"])
        
        # 更新配置
        if "camera_setup" not in self.config:
            self.config["camera_setup"] = {}
            
        self.config["camera_setup"]["camera_bindings"] = new_bindings
        self.config["camera_setup"]["expected_cameras"] = len(new_bindings)
        
        # 保存配置
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"[SetupTool]: ✅ 已将 {len(new_bindings)} 台相机的映射信息保存到config.json")
            
            # 打印映射信息
            print("[SetupTool]: 相机映射信息:")
            for binding in new_bindings:
                print(f"  逻辑索引 {binding['index']} (用户配置): 运行时物理索引={binding['physical_index']}, MAC={binding['mac']}, IP={binding['ip']}, 型号={binding['model']}")
                
            return True
        except Exception as e:
            print(f"[SetupTool]: ❌ 写入配置文件失败: {e}")
            return False

    def bootstrap_mac_bindings(self):
        """
        引导式MAC绑定：检查现有绑定，如果没有则执行自动发现和配置
        返回值：成功则返回True，并且会在必要时更新config
        """
        # 检查是否有现有的相机绑定
        bindings = self.camera_setup.get('camera_bindings', [])
        if bindings:
            print(f"[SetupTool]: 已加载 {len(bindings)} 个相机绑定")
            return True
        
        print("[SetupTool]: 未找到相机绑定，执行首次运行引导...")
        
        # 扫描所有相机
        cameras = self.scan_all_cameras()
        if not cameras:
            print("[SetupTool]: 未发现任何相机设备，引导失败")
            return False
        
        # 分配IP地址
        if self.bootstrap_assign_ips:
            cameras = self.assign_camera_ips(cameras)
        
        # 创建相机映射
        return self.create_camera_mappings(cameras)

    def auto_assign_ips(self):
        """自动扫描并为所有相机分配IP（无论初始IP是什么）"""
        print("[SetupTool]: 开始执行相机IP自动分配...")
        
        # 扫描所有相机
        cameras = self.scan_all_cameras()
        if not cameras:
            print("[SetupTool]: ❌ 未发现任何相机设备，IP分配失败")
            return False
        
        # 分配IP地址
        cameras = self.assign_camera_ips(cameras)
        
        # 创建相机映射
        return self.create_camera_mappings(cameras)

    def auto_configure_nics(self):
        """
        自动配置网卡IP以确保能与相机通信（需要管理员权限）
        此功能用于确保网卡IP配置正确，避免IP冲突
        """
        if not self.has_admin:
            print("[SetupTool]: ❌ 配置网卡需要管理员权限，请以管理员身份运行程序")
            return False
            
        print("[SetupTool]: 开始配置网卡IP...")
        
        # 扫描相机
        cameras = self.scan_all_cameras()
        if not cameras:
            print("[SetupTool]: 未发现相机，无法配置网卡")
            return False
            
        # 查找哪些网卡需要配置
        nics_to_configure = {}
        camera_count_per_nic = {}
        
        for cam in cameras:
            connected_nic = cam.get("connected_nic")
            if not connected_nic:
                continue
                
            nic_ip = connected_nic["ip"]
            nic_name = connected_nic["name"]
            
            # 记录每个网卡连接的相机数量
            if nic_name not in camera_count_per_nic:
                camera_count_per_nic[nic_name] = 0
            camera_count_per_nic[nic_name] += 1
            
            # 检查相机IP与网卡IP是否在同一网段
            cam_ip = cam["current_ip"]
            cam_prefix = ".".join(cam_ip.split('.')[:3])
            nic_prefix = ".".join(nic_ip.split('.')[:3])
            
            if cam_prefix != nic_prefix:
                if nic_name not in nics_to_configure:
                    nics_to_configure[nic_name] = {
                        "current_ip": nic_ip,
                        "target_prefix": cam_prefix,
                        "cameras": []
                    }
                nics_to_configure[nic_name]["cameras"].append(cam)
        
        if not nics_to_configure:
            print("[SetupTool]: 所有网卡IP配置正确，无需修改")
            return True
            
        # 为每个需要配置的网卡设置新IP
        success_count = 0
        for nic_name, config_info in nics_to_configure.items():
            current_ip = config_info["current_ip"]
            target_prefix = config_info["target_prefix"]
            cameras = config_info["cameras"]
            
            # 计算新IP：使用网段前缀，保留最后一位
            current_host = current_ip.split('.')[-1]
            new_ip = f"{target_prefix}.{current_host}"
            
            print(f"[SetupTool]: 正在配置网卡 '{nic_name}': {current_ip} -> {new_ip}")
            
            try:
                # 使用netsh命令配置IP
                nic_name_escaped = nic_name.replace('"', '\\"')
                cmd = f'netsh interface ip set address name="{nic_name_escaped}" static {new_ip} 255.255.255.0'
                
                print(f"[SetupTool]: 执行命令: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                
                if result.returncode == 0:
                    print(f"[SetupTool]: ✅ 网卡 '{nic_name}' IP已修改为 {new_ip}")
                    success_count += 1
                else:
                    print(f"[SetupTool]: ❌ 网卡 '{nic_name}' IP修改失败: {result.stderr}")
            except Exception as e:
                print(f"[SetupTool]: ❌ 配置网卡 '{nic_name}' 时出错: {e}")
        
        if success_count > 0:
            print(f"[SetupTool]: 已成功配置 {success_count}/{len(nics_to_configure)} 个网卡")
            # 配置网卡后等待网络恢复
            time.sleep(2)
            return True
        else:
            print("[SetupTool]: 未能成功配置任何网卡")
            return False

    def cleanup(self):
        """释放资源"""
        try:
            CameraManager.terminate_lib()
            print("[SetupTool]: 相机库资源已释放。")
        except Exception as e:
            print(f"[SetupTool]: 清理资源时出错: {e}")