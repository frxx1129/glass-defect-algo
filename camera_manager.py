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

    def __init__(self, cam_index, ip: str | None = None, mac: str | None = None):
            # 基本属性
            self.cam_index = cam_index
            self.ip = ip
            self.mac = mac
            self.handle = 0
            self.opened_index = None
            # 绑定映射（由上层注入）
            self._ip_bindings = None  # { index(str): ip }
            # 可配置：重试次数、退避基数（秒）、心跳超时（毫秒）
            self.max_open_attempts = 3
            self.open_backoff_base_s = 2
            self.heartbeat_timeout_ms = 30000
            # 记录是否已经尝试过自动强制分配临时IP，避免死循环
            self._force_ip_attempted = False
            self._last_forced_ip = None  # type: ignore

    def set_ip_bindings(self, ip_bindings: dict | None):
        """允许外部注入 index->IP 映射。"""
        if isinstance(ip_bindings, dict):
            self._ip_bindings = ip_bindings
        else:
            self._ip_bindings = None

    def open(self):
        """
        打开此进程专属的相机，并在失败时进行重试。
        """
        CameraManager.ensure_lib_initialized()

        # 对齐示例：先刷新相机列表，校验索引范围，避免 -1007 (INVALID_ID)
        try:
            MVUpdateCameraList()
            _, cam_count = MVGetNumOfCameras()
            print(f"[相机进程 {self.cam_index}]: 本地网段可见相机数量={cam_count}")
            if self.mac:
                target = self.mac.upper().replace('-', ':')
                found_idx = None
                available = []
                for i in range(cam_count):
                    res, cam_info = MVGetDevInfo(i)
                    if res == MVST_SUCCESS:
                        mac_str = ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr])
                        available.append(f"{i}:{mac_str}")
                        if mac_str.upper() == target:
                            found_idx = i
                            break
                print(f"[相机进程 {self.cam_index}]: 当前MVGetDevInfo列表={available}")
                if found_idx is None:
                    # 可能原因：MVEnumerateAllDevices 与 MVUpdateCameraList 列表不一致 / 不同子网
                    print(f"[相机进程 {self.cam_index}]: 列表未直接匹配 MAC={self.mac}，尝试暴力匹配打开...")
                    # 先尝试跨网段枚举（可能包含不在同一子网的设备）
                    try:
                        res_all, all_cnt = MVEnumerateAllDevices()
                        print(f"[相机进程 {self.cam_index}]: MVEnumerateAllDevices 返回数量={all_cnt} res={res_all}")
                        if res_all == MVST_SUCCESS and all_cnt > 0:
                            for j in range(all_cnt):
                                r2, info2 = MVGetDevInfo(j)
                                if r2 == MVST_SUCCESS:
                                    mac2 = ":".join([f"{b:02X}" for b in info2.mEthernetAddr])
                                    ip2 = ".".join(str(x) for x in info2.mIpAddr)
                                    if mac2.upper() == target:
                                        # 直接尝试按 IP 打开（跨网段情况下按 index 可能失败）
                                        if ip2.count('.') == 3:
                                            print(f"[相机进程 {self.cam_index}]: 通过跨网段扫描匹配 MAC，尝试IP {ip2} 打开...")
                                            r_ip2, h_ip2 = MVOpenCamByIP(ip2)
                                            if r_ip2 == MVST_SUCCESS and h_ip2 != 0:
                                                self.handle = h_ip2
                                                self.opened_index = j
                                                print(f"[相机进程 {self.cam_index}]: 通过跨网段 IP 打开成功。")
                                                return True
                    except Exception as _e_enum:
                        print(f"[相机进程 {self.cam_index}]: 跨网段枚举异常: {_e_enum}")
                    # 先尝试使用提供的 ip_bindings 中同 index 的 IP 打开
                    if not self.handle and self._ip_bindings:
                        ip_try = self._ip_bindings.get(str(self.cam_index)) if isinstance(self._ip_bindings, dict) else None
                        if ip_try:
                            print(f"[相机进程 {self.cam_index}]: 尝试通过绑定IP {ip_try} 打开...")
                            r_ip, h_ip = MVOpenCamByIP(ip_try)
                            if r_ip == MVST_SUCCESS and h_ip != 0:
                                # 验证MAC（如果能拿到）
                                try:
                                    res_info, cam_info = MVGetDevInfo(self.cam_index)
                                    if res_info == MVST_SUCCESS:
                                        mac_str = ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr])
                                        print(f"[相机进程 {self.cam_index}]: 通过IP打开(未验证MAC或MAC={mac_str})")
                                except Exception:
                                    pass
                                self.handle = h_ip
                                self.opened_index = self.cam_index
                                # 直接返回成功
                                return True
                    # 暴力尝试逐个打开并比对MAC
                    for i in range(cam_count):
                        res_open, h = MVOpenCamByIndex(i)
                        if res_open == MVST_SUCCESS and h != 0:
                            try:
                                res_info, cam_info = MVGetDevInfo(i)
                                if res_info == MVST_SUCCESS:
                                    mac_str = ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr])
                                    if mac_str.upper() == target:
                                        self.handle = h
                                        self.opened_index = i
                                        print(f"[相机进程 {self.cam_index}]: 通过暴力方式匹配到 MAC={self.mac} 位于索引 {i}")
                                        break
                            except Exception:
                                pass
                            # 若不是目标，关闭临时句柄
                            if self.handle != h:
                                try:
                                    MVCloseCam(h)
                                except Exception:
                                    pass
                    if not self.handle:
                        print(f"[相机进程 {self.cam_index}]: 未匹配到 MAC={self.mac}，准备尝试临时分配IP (MVForceIp)...")
                        if self._attempt_force_temp_ip():
                            # 如果成功分配了新的IP，则尝试直接按新IP打开
                            if self._last_forced_ip:
                                print(f"[相机进程 {self.cam_index}]: 尝试使用临时IP {self._last_forced_ip} 打开...")
                                r_forced, h_forced = MVOpenCamByIP(self._last_forced_ip)
                                if r_forced == MVST_SUCCESS and h_forced != 0:
                                    self.handle = h_forced
                                    self.ip = self._last_forced_ip
                                    print(f"[相机进程 {self.cam_index}]: 通过临时IP 打开成功。")
                                    return True
                        # 若仍未成功
                        print(f"[相机进程 {self.cam_index}]: 临时IP 分配流程未成功，放弃。")
                        return False
                else:
                    print(f"[相机进程 {self.cam_index}]: 通过MAC打开相机 {self.mac} (索引 {found_idx})...")
                    res, self.handle = MVOpenCamByIndex(found_idx)
                    if res == MVST_SUCCESS and self.handle != 0:
                        self.opened_index = found_idx
                    else:
                        print(f"[相机进程 {self.cam_index}]: 按索引 {found_idx} 打开失败，错误码 {res}。尝试暴力匹配...")
                        # 退化到暴力方式
                        for i in range(cam_count):
                            res_open, h = MVOpenCamByIndex(i)
                            if res_open == MVST_SUCCESS and h != 0:
                                try:
                                    res_info, cam_info = MVGetDevInfo(i)
                                    if res_info == MVST_SUCCESS:
                                        mac_str = ":".join([f"{b:02X}" for b in cam_info.mEthernetAddr])
                                        if mac_str.upper() == target:
                                            self.handle = h
                                            self.opened_index = i
                                            print(f"[相机进程 {self.cam_index}]: 通过暴力方式匹配到 MAC={self.mac} 位于索引 {i}")
                                            break
                                except Exception:
                                    pass
                                if self.handle != h:
                                    try:
                                        MVCloseCam(h)
                                    except Exception:
                                        pass
                        if not self.handle:
                            return False
            else:
                if self.cam_index < 0 or self.cam_index >= cam_count:
                    print(f"[相机进程 {self.cam_index}]: 索引越界，当前相机数量: {cam_count}。")
                    return False
        except Exception as e:
            print(f"[相机进程 {self.cam_index}]: 打开前准备阶段异常: {e}")

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
        # 在完全失败后最后再尝试一次自动分配临时IP（若未尝试过且有MAC）
        if self.mac and not self._force_ip_attempted:
            print(f"[相机进程 {self.cam_index}]: 正常打开失败，尝试最后的临时IP分配...")
            if self._attempt_force_temp_ip():
                if self._last_forced_ip:
                    print(f"[相机进程 {self.cam_index}]: 使用最后分配的临时IP {self._last_forced_ip} 重试打开...")
                    r2, h2 = MVOpenCamByIP(self._last_forced_ip)
                    if r2 == MVST_SUCCESS and h2 != 0:
                        self.handle = h2
                        self.ip = self._last_forced_ip
                        print(f"[相机进程 {self.cam_index}]: 通过临时IP 打开成功。")
                        return True
        print(f"❌ [相机进程 {self.cam_index}]: 经过 {self.max_open_attempts} 次尝试后，仍无法打开相机。")
        return False

    # ---------------- 内部辅助：强制分配临时IP -----------------
    def _attempt_force_temp_ip(self) -> bool:
        """当相机无法按既定方式打开时：
        1. 枚举全部设备，找到匹配MAC的相机结构(即使跨网段)。
        2. 采集本机以太网IPv4（排除虚拟/WLAN），选一个基地址；如果有配置的 self.ip 且前三段与某网卡一致，优先该网卡。
        3. 构造临时IP：基地址最后一段+1（若冲突则+2, +3 直到 <254）。
        4. 使用 MVForceIp 设置，等待并刷新列表。
        返回是否成功发送 Force IP 指令。
        """
        if self._force_ip_attempted:
            return False
        self._force_ip_attempted = True
        if not self.mac:
            return False
        target_mac_norm = self.mac.upper().replace('-', ':')
        try:
            res_all, all_cnt = MVEnumerateAllDevices()
            if res_all != MVST_SUCCESS or all_cnt <= 0:
                print(f"[相机进程 {self.cam_index}]: ForceIp 前枚举失败。")
                return False
            matched_cam_info = None
            for j in range(all_cnt):
                r_info, info = MVGetDevInfo(j)
                if r_info == MVST_SUCCESS:
                    mac_j = ':'.join([f"{b:02X}" for b in info.mEthernetAddr])
                    if mac_j.upper() == target_mac_norm:
                        matched_cam_info = info
                        break
            if not matched_cam_info:
                print(f"[相机进程 {self.cam_index}]: 未在全量枚举中找到用于 ForceIp 的目标MAC。")
                return False
            # 收集本机网卡IPv4
            nic_ips = self._collect_nic_ipv4()
            base_ip = None
            if self.ip:
                ip_prefix = '.'.join(self.ip.split('.')[:3]) + '.'
                for nic in nic_ips:
                    if nic.startswith(ip_prefix):
                        base_ip = nic
                        break
            if not base_ip:
                base_ip = nic_ips[0] if nic_ips else '192.168.10.1'
            parts = base_ip.split('.')
            try:
                base_host = int(parts[3])
            except Exception:
                base_host = 1
            # 生成候选 host (base+1, +2, +3 ...)
            candidate_ip = None
            for offset in range(1, 10):
                host_val = base_host + offset
                if host_val >= 254:
                    break
                candidate_ip = '.'.join(parts[:3] + [str(host_val)])
                # 可以添加：避免与已知网卡地址或已发现相机IP冲突（简单跳过与 base 相同）
                if candidate_ip != base_ip:
                    break
            if not candidate_ip:
                print(f"[相机进程 {self.cam_index}]: 无法生成候选临时IP。")
                return False
            subnet_mask = '255.255.255.0'
            default_gw = '0.0.0.0'
            print(f"[相机进程 {self.cam_index}]: 发送 ForceIp -> {candidate_ip}")
            res_force = MVForceIp(matched_cam_info.mEthernetAddr, candidate_ip.encode('ascii'), subnet_mask.encode('ascii'), default_gw.encode('ascii'))
            if res_force == MVST_SUCCESS:
                self._last_forced_ip = candidate_ip
                print(f"[相机进程 {self.cam_index}]: ForceIp 指令成功，下次尝试将使用 {candidate_ip}")
                time.sleep(2.0)
                MVUpdateCameraList()
                return True
            else:
                print(f"[相机进程 {self.cam_index}]: ForceIp 失败，错误码 {res_force}")
                return False
        except Exception as e:
            print(f"[相机进程 {self.cam_index}]: ForceIp 流程异常: {e}")
            return False

    @staticmethod
    def _collect_nic_ipv4():
        """收集本机物理以太网IPv4地址列表。"""
        ips = []
        try:
            import subprocess, re
            output = subprocess.check_output(['ipconfig', '/all'], shell=True)
            try:
                text = output.decode('utf-8', errors='ignore')
            except Exception:
                text = output.decode('gbk', errors='ignore')
            sections = re.split(r"\r?\n\r?\n", text)
            for sec in sections:
                header_match = re.search(r"^(.*?):\s*$", sec, flags=re.MULTILINE)
                if not header_match:
                    continue
                header = header_match.group(1)
                hl = header.lower()
                if (('ethernet' in hl or '以太网' in hl) and not any(x in hl for x in ['wlan', 'wi-fi', '无线', 'bluetooth', '蓝牙', 'vmware', 'virtual', 'hyper-v', 'vethernet'])):
                    m = re.search(r"IPv4 (?:地址|Address)[^:]*: ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)", sec)
                    if m:
                        ip = m.group(1)
                        if not ip.startswith('127.'):
                            ips.append(ip)
        except Exception:
            pass
        return ips

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
            # 使用 MVGetDevInfo 获取网络与型号信息；不再暴露序列号
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
    """一个工具类：启动时发现相机并建立 MAC 绑定（替换原序列号绑定）。"""
    def __init__(self, config):
        self.config = config
        self.camera_setup = self.config.get('camera_setup', {})
        if not self.camera_setup:
            print("[SetupTool]: 警告 - 在 config.json 中未找到 'camera_setup' 部分。")
            
        MVInitLib()
        print("[SetupTool]: 相机库已初始化。")

    def bootstrap_mac_bindings(self):
        """扫描相机并建立 MAC -> 索引 绑定，写入 config.json 中的 mac_bindings。"""
        print("[SetupTool]: 扫描相机(按MAC绑定)...")
        res, num_devices = MVEnumerateAllDevices()
        if res != MVST_SUCCESS or num_devices <= 0:
            print("[SetupTool]: 未发现相机设备。")
            return False
        discovered_macs = []
        discovered_ips = []
        print("[SetupTool]: 发现的设备列表 (索引 -> MAC / IP):")
        for i in range(num_devices):
            r1, info = MVGetDevInfo(i)
            if r1 == MVST_SUCCESS:
                mac_str = ":".join([f"{b:02X}" for b in info.mEthernetAddr])
                ip_str = ".".join(str(x) for x in info.mIpAddr)
            else:
                mac_str = ''
                ip_str = ''
            discovered_macs.append(mac_str)
            discovered_ips.append(ip_str)
            print(f"  - [{i}] -> MAC='{mac_str}' IP='{ip_str}'")

        mb = self.camera_setup.get('mac_bindings')
        ib = self.camera_setup.get('ip_bindings')
        need_write = False
        if not mb or not ib:
            need_write = True
        else:
            try:
                existing_mac = [mb.get(str(i)) for i in range(len(mb))]
                existing_ip = [ib.get(str(i)) for i in range(len(ib))]
                if (len(existing_mac) != len(discovered_macs)
                    or any((existing_mac[i] or '') != discovered_macs[i] for i in range(min(len(existing_mac), len(discovered_macs))))
                    or len(existing_ip) != len(discovered_ips)
                    or any((existing_ip[i] or '') != discovered_ips[i] for i in range(min(len(existing_ip), len(discovered_ips))))):
                    need_write = True
            except Exception:
                need_write = True

        if need_write:
            new_map = {str(i): discovered_macs[i] for i in range(len(discovered_macs))}
            new_ip_map = {str(i): discovered_ips[i] for i in range(len(discovered_ips))}
            self.camera_setup['mac_bindings'] = new_map
            self.camera_setup['ip_bindings'] = new_ip_map
            # 清理旧字段
            self.camera_setup.pop('id_bindings', None)
            self.camera_setup.setdefault('expected_cameras', len(discovered_macs))
            try:
                with open('config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, ensure_ascii=False, indent=2)
                print(f"[SetupTool]: 已写入/覆盖 mac_bindings 与 ip_bindings 到 config.json")
            except Exception as e:
                print(f"[SetupTool]: 写入 config.json 失败: {e}")
        else:
            print("[SetupTool]: mac/ip 绑定与当前设备一致，跳过写入。")
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