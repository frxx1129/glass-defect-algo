''' 
 @Author: LI Zhaoyang  
 @Date: 2025-08-21 11:16:18  
 @Last Modified by:   Gemini AI  
 @Last Modified time: 2025-08-22 03:45:00
''' 
# camera_manager.py (V30 - Process-Safe Architecture)

import time
import json
import sys
from MVGigE import *
from GigECamera_Types import *

# ======================================================================
# --- Individual Camera Controller (No changes) ---
# This class is now instantiated inside each child process.
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

    def __init__(self, cam_index, ip: str | None = None, mac: str | None = None, serial: str | None = None):
        self.cam_index = cam_index
        self.ip = ip
        self.mac = mac
        self.serial = serial
        self.handle = 0
        self.opened_index = None
        # 可配置：重试次数、退避基数（秒）、心跳超时（毫秒）
        self.max_open_attempts = 3
        self.open_backoff_base_s = 2
        self.heartbeat_timeout_ms = 30000

    def open(self):
        """
        打开此进程专属的相机，并在失败时进行重试。
        """
        CameraManager.ensure_lib_initialized()

        # 对齐示例：先刷新相机列表，校验索引范围，避免 -1007 (INVALID_ID)
        try:
            MVUpdateCameraList()
            _, cam_count = MVGetNumOfCameras()
            # 优先使用序列号匹配到索引
            if self.serial:
                target_sn = str(self.serial).strip().upper()
                found_idx = None
                for i in range(cam_count):
                    res, cam_info = MVGetCameraInfo(i)
                    if res == MVST_SUCCESS:
                        try:
                            sn = cam_info.mSerialNumber.decode('ascii', errors='ignore').strip('\x00').upper()
                        except Exception:
                            sn = ''
                        if sn and sn == target_sn:
                            found_idx = i
                            break
                if found_idx is None:
                    print(f"[相机进程 {self.cam_index}]: 未在设备列表中找到 序列号={self.serial} 的相机。")
                    return False
                print(f"[相机进程 {self.cam_index}]: 通过序列号打开相机 {self.serial} (索引 {found_idx})...")
                res, self.handle = MVOpenCamByIndex(found_idx)
                if res == MVST_SUCCESS and self.handle != 0:
                    self.opened_index = found_idx
            elif self.mac:
                # 通过 MAC 查找设备索引
                target = self.mac.upper().replace('-', ':')
                found_idx = None
                for i in range(cam_count):
                    res, cam_info = MVGetDevInfo(i)
                    if res == MVST_SUCCESS:
                        mac_str = ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr])
                        if mac_str.upper() == target:
                            found_idx = i
                            break
                if found_idx is None:
                    print(f"[相机进程 {self.cam_index}]: 未在设备列表中找到 MAC={self.mac} 的相机。")
                    return False
                # 使用索引打开
                print(f"[相机进程 {self.cam_index}]: 通过MAC打开相机 {self.mac} (索引 {found_idx})...")
                res, self.handle = MVOpenCamByIndex(found_idx)
                if res == MVST_SUCCESS and self.handle != 0:
                    self.opened_index = found_idx
            else:
                if self.cam_index < 0 or self.cam_index >= cam_count:
                    print(f"[相机进程 {self.cam_index}]: 索引越界，当前相机数量: {cam_count}。")
                    return False
        except Exception:
            pass

        for attempt in range(1, self.max_open_attempts + 1):
            if self.handle:
                # 已通过 MAC 成功打开
                res = MVST_SUCCESS
            elif self.ip:
                print(f"[相机进程 {self.cam_index}]: 尝试按 IP 打开相机 {self.ip} (第 {attempt}/{self.max_open_attempts} 次)...")
                res, self.handle = MVOpenCamByIP(self.ip)
            else:
                print(f"[相机进程 {self.cam_index}]: 正在尝试按索引打开相机 (第 {attempt}/{self.max_open_attempts} 次)...")
                res, self.handle = MVOpenCamByIndex(self.cam_index)
            
            if res == MVST_SUCCESS and self.handle != 0:
                # 成功！
                try:
                    MVSetHeartbeatTimeout(self.handle, int(self.heartbeat_timeout_ms))
                except Exception:
                    # 旧版SDK可能不支持，忽略
                    pass
                print(f"[相机进程 {self.cam_index}]: 相机已成功打开。句柄: {self.handle}")
                return True
            
            # 如果失败了
            print(f"[相机进程 {self.cam_index}]: 打开失败，错误码: {res}。")
            self.handle = 0 # 确保句柄被重置
            
            if attempt < self.max_open_attempts:
                # 如果不是最后一次尝试，则等待一段时间再重试
                try:
                    backoff = float(self.open_backoff_base_s)
                except Exception:
                    backoff = 2.0
                wait_time = max(0.5, attempt * backoff)  # 线性退避
                print(f"[相机进程 {self.cam_index}]: 将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        # 如果所有尝试都失败了
        print(f"❌ [相机进程 {self.cam_index}]: 经过 {self.max_open_attempts} 次尝试后，仍无法打开相机。")
        return False

    def close(self):
        """关闭相机并释放库资源。"""
        if self.handle != 0:
            MVStopGrab(self.handle)
            MVCloseCam(self.handle)
            print(f"[相机进程 {self.cam_index}]: 相机已关闭。")
            self.handle = 0
        # 不在这里终止库，交由进程退出时统一处理

    def get_frame_buffer(self):
        """根据相机当前设置，创建一个用于接收图像的NumPy缓冲区。"""
        if self.handle == 0: return None, None
        res, img_buffer = MVGetImgBuf(self.handle)
        return res, img_buffer

    def grab_single_frame_to_buffer(self, frame_buffer, timeout_ms=1000):
        """使用 MVGetSampleGrabBuf 将一帧图像采集到指定的缓冲区。"""
        if self.handle == 0: return MVST_ERROR, -1
        res, frame_id = MVGetSampleGrabBuf(self.handle, frame_buffer, timeout_ms)
        return res, frame_id
    
    # Add this new method to the CameraManager class in camera_manager.py
    
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
        """Gathers and returns a complete status dictionary for this camera."""
        if self.handle == 0:
            return {"status": "Not Connected"}

        try:
            # 1. Get static hardware info
            # 优先使用实际打开的索引获取硬件信息
            idx = self.opened_index if self.opened_index is not None else self.cam_index
            res, cam_info = MVGetCameraInfo(idx)
            if res != MVST_SUCCESS:
                hardware_info = {"error": "Failed to get hardware info"}
            else:
                hardware_info = {
                    'model_name': cam_info.mModelName.decode('ascii', errors='ignore').strip('\x00'),
                    'serial_number': cam_info.mSerialNumber.decode('ascii', errors='ignore').strip('\x00'),
                    'mac_address': ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr]),
                    'ip_address': ".".join(map(str, cam_info.mIpAddr))
                }
            
            # 2. Get dynamic parameters
            parameters = self.get_all_params()

            return {
                "status": "Connected",
                "handle": self.handle,
                "hardware": hardware_info,
                "parameters": parameters if parameters else "Failed to get parameters"
            }
        except Exception as e:
            return {"status": "Error", "details": str(e)}

    def set_params(self, params_to_set):
        """根据结构化字典设置参数，包括新的网络参数。"""
        if self.handle == 0 or not params_to_set: return None
        print(f"[相机进程 {self.cam_index}]: 正在应用配置参数...")
        results = {}
        
        # --- 触发模式设置：如果提供了 trigger 配置，则按配置设置；否则默认连续采集 ---
        trig_cfg = params_to_set.get("trigger")
        if trig_cfg:
            # mode: TriggerMode_Off / TriggerMode_On
            mode = trig_cfg.get("mode", TriggerMode_Off)
            res = MVSetTriggerMode(self.handle, int(mode))
            if res == MVST_SUCCESS:
                print(f"  [相机 {self.cam_index}]: 触发模式设置为 {mode}")
            else:
                print(f"  [相机 {self.cam_index}]: 警告 - 设置触发模式失败，错误码: {res}")

            # 当触发开启时，可选设置触发源与极性
            if mode == TriggerMode_On:
                if "source" in trig_cfg:
                    res = MVSetTriggerSource(self.handle, int(trig_cfg["source"]))
                    if res != MVST_SUCCESS:
                        print(f"  [相机 {self.cam_index}]: 警告 - 设置触发源失败，错误码: {res}")
                if "activation" in trig_cfg:
                    res = MVSetTriggerActivation(self.handle, int(trig_cfg["activation"]))
                    if res != MVST_SUCCESS:
                        print(f"  [相机 {self.cam_index}]: 警告 - 设置触发极性失败，错误码: {res}")
        else:
            # 默认连续采集
            res = MVSetTriggerMode(self.handle, TriggerMode_Off)
            if res == MVST_SUCCESS:
                print(f"  [相机 {self.cam_index}]: 已设置为连续采集模式")
            else:
                print(f"  [相机 {self.cam_index}]: 警告 - 设置连续采集模式失败，错误码: {res}")
        
        # --- 修改：优先设置网络参数，因为它们影响数据流基础 ---
        if "network" in params_to_set:
            net_params = params_to_set["network"]
            results["network"] = {}
            if "packet_size_bytes" in net_params:
                res = MVSetPacketSize(self.handle, int(net_params["packet_size_bytes"]))
                results["network"]["packet_size_bytes"] = (res == MVST_SUCCESS)
                if res != MVST_SUCCESS: print(f"  [相机 {self.cam_index}]: 警告 - 设置包大小失败，错误码: {res}")
            if "packet_delay_us" in net_params:
                res = MVSetPacketDelay(self.handle, int(net_params["packet_delay_us"]))
                results["network"]["packet_delay_us"] = (res == MVST_SUCCESS)
                if res != MVST_SUCCESS: print(f"  [相机 {self.cam_index}]: 警告 - 设置包延迟失败，错误码: {res}")

        if "acquisition" in params_to_set:
            acq_params = params_to_set["acquisition"]
            results["acquisition"] = {}
            if "width" in acq_params: results["acquisition"]["width"] = (MVSetWidth(self.handle, int(acq_params["width"])) == MVST_SUCCESS)
            if "height" in acq_params: results["acquisition"]["height"] = (MVSetHeight(self.handle, int(acq_params["height"])) == MVST_SUCCESS)
            if "frame_rate" in acq_params: results["acquisition"]["frame_rate"] = (MVSetFrameRate(self.handle, float(acq_params["frame_rate"])) == MVST_SUCCESS)
            # 新增：像素格式
            if "pixel_format" in acq_params:
                pf = acq_params["pixel_format"]
                # 兼容字符串传参，如 "mono8"
                if isinstance(pf, str):
                    pf_lower = pf.strip().lower()
                    if pf_lower in ("mono8", "monochrome8", "gray8", "greyscale8", "grayscale8"):
                        pf_val = PixelFormat_Mono8
                    elif pf_lower in ("mono16", "monochrome16", "gray16", "greyscale16", "grayscale16"):
                        pf_val = PixelFormat_Mono16
                    else:
                        # 未识别则不处理
                        pf_val = None
                else:
                    pf_val = int(pf)

                if pf_val is not None:
                    results["acquisition"]["pixel_format"] = (MVSetPixelFormat(self.handle, pf_val) == MVST_SUCCESS)
        
        if "exposure" in params_to_set:
            exp_params = params_to_set["exposure"]
            results["exposure"] = {}
            if "auto_mode" in exp_params: MVSetExposureAuto(self.handle, exp_params["auto_mode"])
            if "value_us" in exp_params:
                MVSetExposureAuto(self.handle, ExposureAuto_Off)
                results["exposure"]["value_us"] = (MVSetExposureTime(self.handle, float(exp_params["value_us"])) == MVST_SUCCESS)

        if "gain" in params_to_set:
            gain_params = params_to_set["gain"]
            results["gain"] = {}
            if "auto_mode" in gain_params: MVSetGainAuto(self.handle, gain_params["auto_mode"])
            if "value_db" in gain_params:
                MVSetGainAuto(self.handle, GainAuto_Off)
                results["gain"]["value_db"] = (MVSetGain(self.handle, float(gain_params["value_db"])) == MVST_SUCCESS)

        if "gamma" in params_to_set:
            gamma_params = params_to_set["gamma"]
            results["gamma"] = {}
            if "value" in gamma_params:
                results["gamma"]["value"] = (MVSetGamma(self.handle, float(gamma_params["value"])) == MVST_SUCCESS)
                print(f"  [相机 {self.cam_index}]: 伽马值设置为 {gamma_params['value']}")
        
        print(f"[相机进程 {self.cam_index}]: 参数应用完成。")
        return results

# ======================================================================
# --- Multi-Camera Setup Utility (For main process only) ---
# ======================================================================
class MultiCameraSetup:
    """一个工具类，仅在主进程中用于启动时发现并配置所有相机的IP地址。"""
    def __init__(self, config):
        self.config = config
        self.camera_setup = self.config.get('camera_setup', {})
        if not self.camera_setup:
            print("[SetupTool]: 警告 - 在 config.json 中未找到 'camera_setup' 部分。")
            
        MVInitLib()
        print("[SetupTool]: 相机库已初始化。")

    def bootstrap_id_bindings(self):
        """
        不限网段扫描相机，按序列号(id)建立绑定。
        若 config 中缺少 id_bindings 或与当前检测到的设备不一致，则用检测结果覆盖并保存 config.json。
        """
        print("[SetupTool]: 不限网段扫描相机(按序列号)...")
        res, num_devices = MVEnumerateAllDevices()
        if res != MVST_SUCCESS or num_devices <= 0:
            print("[SetupTool]: 未发现相机设备。")
            return False
        discovered_ids = []
        print("[SetupTool]: 发现的设备列表 (索引 -> 序列号):")
        for i in range(num_devices):
            # 用 CameraInfo 获取序列号
            r1, info = MVGetCameraInfo(i)
            if r1 == MVST_SUCCESS:
                try:
                    sn = info.mSerialNumber.decode('ascii', errors='ignore').strip('\x00')
                except Exception:
                    sn = ''
            else:
                sn = ''
            discovered_ids.append(sn)
            print(f"  - [{i}] -> SN='{sn}'")

        ib = self.camera_setup.get('id_bindings')
        need_write = False
        # 兼容老配置: 若仅存在 mac_bindings，则迁移为 id_bindings
        if not ib:
            need_write = True
        else:
            # 校验现有绑定是否与当前发现一致（数量/内容）
            try:
                existing = [ib.get(str(i)) for i in range(len(ib))]
                if len(existing) != len(discovered_ids) or any((existing[i] or '') != discovered_ids[i] for i in range(min(len(existing), len(discovered_ids)))):
                    need_write = True
            except Exception:
                need_write = True

        if need_write:
            new_map = {str(i): discovered_ids[i] for i in range(len(discovered_ids))}
            self.camera_setup['id_bindings'] = new_map
            self.camera_setup.pop('mac_bindings', None)
            self.camera_setup.setdefault('expected_cameras', len(discovered_ids))
            try:
                with open('config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                print(f"[SetupTool]: 已写入/覆盖 id_bindings 到 config.json: {new_map}")
            except Exception as e:
                print(f"[SetupTool]: 写入 config.json 失败: {e}")
        else:
            print("[SetupTool]: id_bindings 与当前设备一致，跳过写入。")
        MVUpdateCameraList()
        return True

    def auto_assign_ips(self):
        """尝试基于本机以太网网卡地址为相机分配IP：按网卡地址+偏移设置。需要管理员权限。
        注意：忽略 config 中的全局 subnet_mask/default_gateway，因多网卡下不具普适性；统一按 /24 段处理。
        """
        try:
            import subprocess
        except Exception:
            print("[SetupTool]: 无法导入 subprocess，跳过自动分配IP。")
            return False
        # 读取本机 IPv4 列表（仅以太网适配器）
        try:
            output = subprocess.check_output(['ipconfig', '/all'], shell=True)
            try:
                text = output.decode('utf-8', errors='ignore')
            except Exception:
                text = output.decode('gbk', errors='ignore')
            import re
            # 按节解析适配器，筛选标题含“以太网/Ethernet”且不含 Virtual/Hyper-V/VMware/WLAN/Wi-Fi
            sections = re.split(r"\r?\n\r?\n", text)
            nic_bases = []
            for sec in sections:
                header_match = re.search(r"^(.*?):\s*$", sec, flags=re.MULTILINE)
                if not header_match:
                    continue
                header = header_match.group(1)
                header_l = header.lower()
                if (('ethernet' in header_l or '以太网' in header_l)
                    and not any(x in header_l for x in ['wlan', 'wi-fi', '无线', 'bluetooth', '蓝牙', 'vmware', 'virtual', 'hyper-v', 'vethernet'])):
                    # 找 IPv4
                    m = re.search(r"IPv4 (?:地址|Address)[^:]*: ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", sec)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith('127.'):
                            nic_bases.append(ip)
        except Exception as e:
            print(f"[SetupTool]: 解析网卡IPv4失败: {e}")
            nic_bases = []

        res, num_devices = MVEnumerateAllDevices()
        if res != MVST_SUCCESS or num_devices == 0:
            print("[SetupTool]: 无相机可分配IP。")
            return False
        # 生成目标IP列表（简单轮询分配到各网卡，主机位=本机最后一段+1, +2...）
        cams = []
        for i in range(num_devices):
            rr, cam_info = MVGetDevInfo(i)
            if rr == MVST_SUCCESS:
                cams.append(cam_info)
        assigned = 0
        # 统一使用 /24 掩码与无默认网关
        subnet_mask = '255.255.255.0'
        default_gw = '0.0.0.0'
        for idx, cam_info in enumerate(cams):
            nic_ip = nic_bases[idx % max(1, len(nic_bases))] if nic_bases else '192.168.10.1'
            parts = nic_ip.split('.')
            host = min(254, int(parts[3]) + 1 + (idx // max(1, len(nic_bases))))
            target_ip = '.'.join(parts[:3] + [str(host)])
            mac_str = ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr])
            print(f"  - 为相机(MAC {mac_str}) 分配IP: {target_ip}")
            res2 = MVForceIp(cam_info.mEthernetAddr, target_ip.encode('ascii'), subnet_mask.encode('ascii'), default_gw.encode('ascii'))
            if res2 == MVST_SUCCESS:
                assigned += 1
            else:
                print(f"    ...失败。错误码: {res2}")
        if assigned:
            print("[SetupTool]: IP配置指令已发送。等待3秒...")
            time.sleep(3)
            MVUpdateCameraList()
        return assigned > 0

    def auto_configure_nics(self):
        """仅为“以太网/Ethernet”物理网卡设置不同网段的静态IPv4(示例：192.168.10.1/24、192.168.11.1/24...)。需要管理员权限。"""
        try:
            import subprocess, json as _json
            # 使用 PowerShell 获取启用的以太网适配器列表（名称、当前IPv4）
            ps = (
                "Get-NetAdapter | Where-Object {$_.Status -eq 'Up' -and $_.HardwareInterface -eq $true} | "
                # 仅保留以太网（排除Wi-Fi/WLAN/虚拟/蓝牙等）
                "Where-Object {($_.Name -match 'Ethernet|以太网') -and ($_.Name -notmatch 'Wi-?Fi|WLAN|无线|Bluetooth|蓝牙|VMware|Hyper-V|vEthernet|Virtual')} | "
                "Select-Object -Property Name | ConvertTo-Json"
            )
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps])
            txt = out.decode('utf-8', errors='ignore').strip()
            names = []
            if txt:
                try:
                    data = _json.loads(txt)
                    if isinstance(data, list):
                        names = [d.get('Name') for d in data if isinstance(d, dict) and d.get('Name')]
                    elif isinstance(data, dict) and data.get('Name'):
                        names = [data.get('Name')]
                except Exception:
                    pass
            if not names:
                print("[SetupTool]: 未发现需要配置的启用网卡。")
                return False
            base_net = 10  # 从 10 段开始：192.168.10.0/24
            configured = 0
            for i, name in enumerate(names):
                third = base_net + i
                ip = f"192.168.{third}.1"
                prefix = 24
                # 先尝试移除现有IPv4（忽略错误），再添加新地址
                ps_set = (
                    f"$if='{name}'; "
                    f"$existing=(Get-NetIPAddress -InterfaceAlias $if -AddressFamily IPv4 -ErrorAction SilentlyContinue); "
                    f"if ($existing) {{ foreach($e in $existing) {{ try {{ Remove-NetIPAddress -InputObject $e -Confirm:$false -ErrorAction SilentlyContinue }} catch {{}} }} }}; "
                    f"New-NetIPAddress -InterfaceAlias $if -IPAddress '{ip}' -PrefixLength {prefix} -DefaultGateway '0.0.0.0' -ErrorAction SilentlyContinue"
                )
                try:
                    subprocess.check_call(["powershell", "-NoProfile", "-Command", ps_set])
                    print(f"[SetupTool]: 网卡 '{name}' 已设置为 {ip}/{prefix}")
                    configured += 1
                except Exception as e:
                    print(f"[SetupTool]: 配置网卡 '{name}' 失败: {e}")
            return configured > 0
        except Exception as e:
            print(f"[SetupTool]: 自动配置网卡异常: {e}")
            return False

    def cleanup(self):
        """释放主进程中使用的库资源。"""
        MVTerminateLib()
        print("[SetupTool]: 相机库已释放。")