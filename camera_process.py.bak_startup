import time
import queue
import traceback
import numpy as np
import multiprocessing as mp
import json
import yaml
from camera_manager import CameraManager, MultiCameraSetup
from GigECamera_Types import TriggerMode_On, TriggerMode_Off, TriggerSource_Software, TriggerActivation_RisingEdge, MVStreamCB  # noqa
from MVGigE import *
from generate_test_image import generate_test_image
import platform
import ctypes


def _boost_process_priority_windows():
    """在 Windows 上提升当前进程优先级，减少被其他进程/线程抢占的概率。"""
    try:
        if platform.system() != 'Windows':
            return
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        kernel32 = ctypes.windll.kernel32
        hProc = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(hProc, ABOVE_NORMAL_PRIORITY_CLASS)
        print("[相机池]: 进程优先级已提升为 Above Normal")
    except Exception:
        pass


def _compute_trigger_period_ms(config: dict) -> float:
    """计算软触发周期（毫秒）。默认10fps=100ms；保证周期>=曝光(μs)/1000+2ms。"""
    cs = (config.get('camera_setup') or {}) if isinstance(config, dict) else {}
    runtime = cs.get('runtime', {}) if isinstance(cs, dict) else {}
    cam_cfg = cs.get('unified_params', {}) if isinstance(cs, dict) else {}
    # 曝光（默认800us）
    try:
        exp_us = float(((cam_cfg or {}).get('exposure') or {}).get('value_us', 800))
    except Exception:
        exp_us = 800.0
    # 目标帧率（默认10fps）
    try:
        fps = float(((cam_cfg.get('acquisition') or {}).get('frame_rate')
                     or runtime.get('frame_rate')
                     or runtime.get('target_fps')
                     or (config.get('system_params') or {}).get('target_fps', 10)))
        if fps <= 0:
            fps = 10.0
    except Exception:
        fps = 10.0
    period_ms = 1000.0 / fps
    min_ms = (exp_us / 1000.0) + 2.0
    return max(period_ms, min_ms)


def _privacy_enabled(config: dict) -> bool:
    """默认启用隐私日志模式，可通过 system_params.hide_sensitive_logs 显式关闭。"""
    try:
        sp = (config.get('system_params') or {}) if isinstance(config, dict) else {}
        return bool(sp.get('hide_sensitive_logs', True))
    except Exception:
        return True


def _mask_mac(mac: str) -> str:
    try:
        m = str(mac or '').upper()
        parts = m.split(':')
        if len(parts) == 6:
            return ':'.join(parts[:2] + ['**', '**', '**', parts[-1]])
        return '***'
    except Exception:
        return '***'


def _mask_ip(ip: str) -> str:
    try:
        s = str(ip or '')
        seg = s.split('.')
        if len(seg) == 4:
            return f"{seg[0]}.{seg[1]}.{seg[2]}.***"
        return '***'
    except Exception:
        return '***'


def _camera_worker_process(cam_index: int, task_queue, stop_event, run_event, config: dict,
                           shared_states, capture_interval_s: float, use_soft_trigger: bool):
        """每个相机一个独立进程：打开相机→应用参数→注册回调→按周期软件触发→写入共享队列。"""
        from camera_manager import CameraManager  # 进程内导入，避免在spawn时重复初始化
        from MVGigE import MVStartGrab, MVStopGrab, MVTriggerSoftware, MVST_SUCCESS
        import copy

        # 读取配置中的相机映射信息
        cs = (config.get('camera_setup', {}) or {})
        camera_bindings = cs.get('camera_bindings', [])
        
        # 查找该逻辑索引对应的相机信息
        cam_binding = None
        for binding in camera_bindings:
            if binding.get('index') == cam_index:
                cam_binding = binding
                break
        
        # 根据配置信息选择相机打开方式
        if cam_binding:
            # 如果找到绑定信息，优先使用MAC地址
            desired_mac = cam_binding.get('mac')
            desired_ip = cam_binding.get('ip')
            physical_index = cam_binding.get('physical_index')
            
            if _privacy_enabled(config):
                print(f"[相机进程 {cam_index}]: 配置 逻辑索引={cam_index}, MAC={_mask_mac(desired_mac)}, IP={_mask_ip(desired_ip)}, 配置中的物理索引={physical_index}（仅参考）")
            else:
                print(f"[相机进程 {cam_index}]: 配置 逻辑索引={cam_index}, MAC={desired_mac}, IP={desired_ip}, 配置中的物理索引={physical_index}（仅参考）")
            
            # 确保更新相机列表获取最新状态
            MVUpdateCameraList()
            
            # 使用MAC地址和IP打开相机，IP可触发快速连接
            cam = CameraManager(cam_index, mac=desired_mac, ip=desired_ip)
        else:
            # 如果没有绑定信息，使用索引直接打开
            print(f"[相机进程 {cam_index}]: 未找到相机绑定信息，使用索引打开")
            cam = CameraManager(cam_index)
        
        # 从配置注入可调的打开参数
        cm_setup = (config.get('camera_setup') or {}) if isinstance(config, dict) else {}
        try:
            cam.max_open_attempts = int(cm_setup.get('open_retry_count', 3) or 3)
        except Exception:
            cam.max_open_attempts = 3
        try:
            cam.open_backoff_base_s = float(cm_setup.get('open_retry_backoff_base_s', 2) or 2)
        except Exception:
            cam.open_backoff_base_s = 2.0
        try:
            cam.heartbeat_timeout_ms = int(cm_setup.get('heartbeat_timeout_ms', 30000) or 30000)
        except Exception:
            cam.heartbeat_timeout_ms = 30000
        try:
            open_result = cam.open()
            if not open_result:
                shared_states[cam_index] = {"status": "Open Failed", "error": "打开相机失败"}
                print(f"[相机进程 {cam_index}]: 打开相机失败")
                return

            # 相机已成功打开，更新状态
            shared_states[cam_index] = {"status": "Opened", "mac": cam.mac, "ip": cam.ip}
            if _privacy_enabled(config):
                print(f"[相机进程 {cam_index}]: 相机就绪，MAC: {_mask_mac(cam.mac)}, IP: {_mask_ip(cam.ip)}")
            else:
                print(f"[相机进程 {cam_index}]: 相机就绪，MAC: {cam.mac}, IP: {cam.ip}")

            # 应用相机参数
            unified_params = copy.deepcopy(config.get('camera_setup', {}).get('unified_params', {}) or {})
            if 'network' not in unified_params:
                unified_params['network'] = {}
            if 'packet_size_bytes' not in unified_params['network']:
                default_pkt = int(config.get('system_params', {}).get('default_packet_size_bytes', 1500))
                unified_params['network']['packet_size_bytes'] = default_pkt
            # 提供可选默认包延迟
            if 'packet_delay_us' not in unified_params['network']:
                default_delay = int(config.get('system_params', {}).get('default_packet_delay_us', 0))
                if default_delay > 0:
                    unified_params['network']['packet_delay_us'] = default_delay
            if 'acquisition' not in unified_params:
                unified_params['acquisition'] = {}
            # 是否强制 Mono8
            runtime_cfg = (config.get('camera_setup', {}) or {}).get('runtime', {})
            enforce_mono8 = bool(runtime_cfg.get('enforce_mono8', (config.get('system_params', {}) or {}).get('enforce_mono8', True)))
            if enforce_mono8:
                unified_params['acquisition']['pixel_format'] = int(0x01080001)

            # 触发配置：根据是否启用软触发
            if use_soft_trigger:
                unified_params['trigger'] = {
                    'mode': TriggerMode_On,
                    'source': TriggerSource_Software,
                    'activation': TriggerActivation_RisingEdge,
                }
            else:
                unified_params['trigger'] = {
                    'mode': TriggerMode_Off
                }

            cam.set_params(unified_params)
            shared_states[cam_index] = cam.get_full_status()

            # 回调函数：转图并复制缓冲，投入队列（回调内尽量短）
            def _on_frame(info_ptr, user_val_ptr):
                try:
                    img, fid = MV_info_to_image(cam.handle, info_ptr)
                    try:
                        # 始终投递帧（临时屏蔽运行开关）
                        task_queue.put_nowait({"data": img, "cam_index": cam_index, "frame_id": fid})
                    except queue.Full:
                        pass
                except Exception:
                    return 0
                return 0

            # 启动回调式抓取
            cb = MVStreamCB(_on_frame)
            res = MVStartGrab(cam.handle, cb, cam.handle)
            if res != MVST_SUCCESS:
                print(f"[相机进程 {cam_index}]: 启动图像采集失败: {res}")
                shared_states[cam_index] = {
                    "status": "Grab Failed", 
                    "error": f"相机启动采集失败: {res}",
                    "mac": cam.mac, 
                    "ip": cam.ip
                }
                return

            # 更新相机状态为正在采集
            shared_states[cam_index] = cam.get_full_status()
            shared_states[cam_index]["status"] = "Grabbing"
            
            # 计算触发周期
            period_ms = _compute_trigger_period_ms(config)
            period_s = max(0.001, period_ms / 1000.0)
            if use_soft_trigger:
                print(f"[相机进程 {cam_index}]: 软件触发模式，周期 {period_ms:.1f} ms")
            else:
                print(f"[相机进程 {cam_index}]: 连续采集模式（Free Run）")

            # 触发主循环（若软触发启用则发触发；否则仅维持运行）
            while not stop_event.is_set():
                if use_soft_trigger:
                    try:
                        MVTriggerSoftware(cam.handle)
                    except Exception:
                        time.sleep(min(period_s, 0.01))
                        continue
                    time.sleep(period_s)
                else:
                    # free-run：无须触发，仅小睡避免空转
                    time.sleep(0.005)

        except KeyboardInterrupt:
            # Ctrl+C 触发时安静退出，交给 finally 做资源释放
            try:
                shared_states[cam_index] = {"status": "Stopping"}
            except Exception:
                pass
        except Exception as e:
            error_msg = traceback.format_exc()
            print(f"[相机进程 {cam_index}]: 出现异常: {e}")
            shared_states[cam_index] = {"status": "Error", "error": str(e)}
        finally:
            try:
                print(f"[相机进程 {cam_index}]: 正在关闭相机...")
                if cam and hasattr(cam, 'handle') and cam.handle:
                    try:
                        MVStopGrab(cam.handle)
                    except Exception:
                        pass
                
                if cam:
                    cam.close()
                    print(f"[相机进程 {cam_index}]: 相机已关闭")
            except Exception:
                pass


def _test_camera_worker_process(cam_index: int, task_queue, stop_event, run_event, capture_interval_s: float,
                                shared_states):
    from generate_test_image import generate_test_image
    shared_states[cam_index] = {"status": "Connected (Test Mode)"}
    last_emit_ts = 0.0
    while not stop_event.is_set():
        if capture_interval_s and capture_interval_s > 0:
            now = time.perf_counter()
            if now - last_emit_ts < capture_interval_s:
                time.sleep(min(capture_interval_s - (now - last_emit_ts), 0.01))
                continue
        frame = generate_test_image()
        try:
            task_queue.put_nowait({"data": frame, "cam_index": cam_index})
            if capture_interval_s and capture_interval_s > 0:
                last_emit_ts = time.perf_counter()
        except queue.Full:
            pass
        if not (capture_interval_s and capture_interval_s > 0):
            time.sleep(0.003)


def camera_pool_process(task_queue, stop_event, run_event, cameras_ready_event, config, shared_states):
    """
    多相机采集进程：
    - 初始化与打开相机
    - 每台相机独立线程采集，抓到就立即投递（非阻塞），保持"只取最新"由处理端合帧
    - 与相机帧率解耦，不再使用处理节拍节流
    - 使用 cameras_ready_event 通知主进程相机已就绪
    """
    print("[相机池]: 进程已启动。")
    # 提升进程优先级，减少被其他进程抢占的概率
    _boost_process_priority_windows()

    # 取消处理节拍节流，采集线程抓到即投递；如需限速请改在处理端控制

    # 1) 初始化与配置 IP
    # 读取可选的采集间隔（秒）：优先 camera_capture_interval，其次 capture_interval；未设置或<=0 表示不限制
    try:
        cs = (config.get('camera_setup') or {}) if isinstance(config, dict) else {}
        runtime = cs.get('runtime', {}) if isinstance(cs, dict) else {}
        legacy_sys = (config.get('system_params') or {}) if isinstance(config, dict) else {}
        capture_interval_s = float(runtime.get('capture_interval_s', legacy_sys.get('camera_capture_interval', legacy_sys.get('capture_interval', 0))) or 0)
        if capture_interval_s < 0:
            capture_interval_s = 0.0
        use_soft_trigger = bool(runtime.get('use_software_trigger', legacy_sys.get('use_software_trigger', False)))
    except Exception:
        capture_interval_s = 0.0
        use_soft_trigger = False
        
    # 初始化相机设置工具
    setup_tool = MultiCameraSetup(config)
    
    # 检查是否有管理员权限，如果没有且需要则尝试获取
    if not setup_tool.has_admin and (config.get('camera_setup', {}) or {}).get('require_admin', True):
        print("[相机池]: 相机设置需要管理员权限，正在尝试提升权限...")
        from camera_manager import run_as_admin
        import sys
        import os
        
        # 当前脚本路径
        if getattr(sys, 'frozen', False):
            # PyInstaller 打包的应用
            script_path = sys.executable
        else:
            # 正常 Python 脚本
            script_path = sys.argv[0]
            
        # 尝试以管理员身份重新运行
        if run_as_admin(script_path, *sys.argv[1:]):
            print("[相机池]: 已以管理员身份重新启动程序，当前进程将退出")
            os._exit(0)
        else:
            print("[相机池]: ⚠️ 无法获取管理员权限，某些相机操作可能受限")
    
    # 1) 启动阶段：扫描并创建相机MAC映射，如果没有则执行自动发现
    # 引导绑定：只在配置缺少绑定时执行
    try:
        has_bindings = bool(((config.get('camera_setup') or {}).get('camera_bindings') or []))
    except Exception:
        has_bindings = False
    setup_tool.bootstrap_mac_bindings()
    
    # 2) 可选：自动配置本机网卡到正确网段（需要管理员权限）
    if (config.get('camera_setup', {}) or {}).get('auto_configure_nics', False):
        setup_tool.auto_configure_nics()
    
    # 3) 统一扫描并分配相机IP（无论初始IP是什么）
    # 仅在缺少IP或绑定不完整时尝试统一分配IP，避免重复打印与不必要的变更
    try:
        do_assign_ips = (config.get('camera_setup', {}) or {}).get('bootstrap_assign_ips', True)
        bindings = (config.get('camera_setup', {}) or {}).get('camera_bindings', [])
        need_ip_assign = True
        if isinstance(bindings, list) and bindings:
            # 如果每条绑定都已有 ip 字段，则无需再次分配
            need_ip_assign = any((not isinstance(b, dict)) or (not b.get('ip')) for b in bindings)
        if do_assign_ips and need_ip_assign:
            setup_tool.auto_assign_ips()
        elif do_assign_ips:
            print("[相机池]: 已存在完整的IP绑定，跳过自动分配IP。")
    except Exception:
        # 出错则保守执行一次，确保连通
        setup_tool.auto_assign_ips()

    # 优先使用 MVEnumerateAllDevices() 检测所有相机，包括不在同一网段的
    res_enum, num_devices = MVEnumerateAllDevices()
    
    # 然后检查 MVGetNumOfCameras() 结果，看有多少相机可以直接访问
    res, num_cameras = MVGetNumOfCameras()
    
    if num_devices > 0 and num_cameras < num_devices:
        print(f"[相机池]: 检测到 {num_devices} 台相机，直接可访问 {num_cameras} 台")
        num_cameras = num_devices  # 使用 MVEnumerateAllDevices 的结果继续
    
    if num_cameras == 0:
        # ---------------- 新的模拟模式实现 ----------------
        # 要求：5路相机分别播放 4,5,4,5,4.avi；所有相机统一使用 cam5_roi_averaged_by_group.json 的 ROI 组
        print("[相机池]: 未检测到物理相机，进入'视频回放+统一ROI'模拟模式...")
        setup_tool.cleanup()
        import cv2, os

        # 读取期望的模拟相机数量（默认 5 以满足需求；向下兼容用户配置）
        try:
            test_cameras = int((config.get('camera_setup', {}) or {}).get('expected_cameras', 5) or 5)
        except Exception:
            test_cameras = 5
        if test_cameras <= 0:
            test_cameras = 5
        test_cameras = min(test_cameras, 5)  # 映射只定义到 5 路

        acq = (config.get('camera_setup', {}) or {}).get('unified_params', {}).get('acquisition', {})
        cam_w = int(acq.get('width', 1280) or 1280)
        cam_h = int(acq.get('height', 960) or 960)

        # 1) 载入统一 ROI 文件
        roi_override_file = 'cam5_roi_averaged_by_group.json'
        def _load_averaged_rois(path: str):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # 选 source_image_count 最大的 averaged_rois
                    best_key = None; best_cnt = -1
                    for k, v in data.items():
                        if isinstance(v, dict) and 'averaged_rois' in v:
                            cnt = v.get('source_image_count', len(v['averaged_rois']))
                            if cnt > best_cnt:
                                best_cnt = cnt; best_key = k
                    if best_key is not None:
                        return data[best_key]['averaged_rois']
                    # 兼容直接 rois 列表
                    if 'rois' in data and isinstance(data['rois'], list):
                        return data['rois']
                elif isinstance(data, list):
                    return data
            except Exception as e:
                print(f"[相机池][测试模式]: 载入统一ROI失败: {e}")
            return []
        unified_rois = _load_averaged_rois(roi_override_file)
        print(f"[相机池][测试模式]: 统一ROI载入完成, 数量={len(unified_rois)} 来自 {roi_override_file}")

        # 2) 建立视频映射
        video_map = ['4.avi', '5.avi', '4.avi', '5.avi', '4.avi']
        caps = []              # 每路 VideoCapture
        frame_intervals = []    # 每路帧间隔 (秒)
        next_times = []         # 下一帧时间戳
        for idx in range(test_cameras):
            vf = video_map[idx] if idx < len(video_map) else video_map[-1]
            cap = cv2.VideoCapture(vf)
            if not cap.isOpened():
                print(f"[相机池][测试模式]: ⚠️ 相机{idx} 无法打开视频 {vf}，使用黑帧占位。")
                cap = None
                interval = 0.1
            else:
                fps_v = cap.get(cv2.CAP_PROP_FPS) or 10.0
                if fps_v <= 0: fps_v = 10.0
                interval = 1.0 / fps_v
                print(f"[相机池][测试模式]: 相机{idx} 使用 {vf} (fps={fps_v:.2f})")
            caps.append(cap)
            frame_intervals.append(interval)
            next_times.append(time.perf_counter())

        # 3) 初始化共享状态
        for idx in range(test_cameras):
            shared_states[idx] = {
                "status": "Connected (Test Mode)",
                "type": "VideoSim",
                "camera_index": idx,
                "frame_size": f"{cam_w}x{cam_h}",
                "rois_count": len(unified_rois),
                "video_file": video_map[idx] if idx < len(video_map) else video_map[-1]
            }

        print(f"[相机池][测试模式]: 视频相机={test_cameras} 路, 分辨率={cam_w}x{cam_h}")
        print("[相机池][测试模式]: 所有模拟相机就绪，发送就绪事件...")
        cameras_ready_event.set()

        # 4) 主循环：定时为每路相机取帧
        while not stop_event.is_set():
            now = time.perf_counter()
            for idx in range(test_cameras):
                if now < next_times[idx]:
                    continue
                cap = caps[idx]
                frame_gray = None
                if cap is not None:
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        # 循环回放
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame_bgr = cap.read()
                    if ret and frame_bgr is not None:
                        try:
                            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                        except Exception:
                            frame_gray = None
                if frame_gray is None:
                    frame_gray = np.zeros((cam_h, cam_w), dtype=np.uint8)
                else:
                    h_in, w_in = frame_gray.shape
                    if h_in != cam_h or w_in != cam_w:
                        frame_gray = cv2.resize(frame_gray, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
                try:
                    # 将统一 ROI 一并投递，供处理进程首次载入时直接使用
                    task_queue.put_nowait({"data": frame_gray, "cam_index": idx, "rois": unified_rois})
                except queue.Full:
                    pass
                next_times[idx] += frame_intervals[idx]
            time.sleep(0.001)
        # 退出清理
        for cap in caps:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
        return

    # ---------------------------- 实机模式 ----------------------------
    print(f"[相机池]: 检测到 {num_cameras} 台相机")
    setup_tool.cleanup()

    # 优先使用配置的逻辑绑定顺序启动 worker；若无绑定则按物理枚举顺序
    camera_bindings = (config.get('camera_setup', {}) or {}).get('camera_bindings', [])
    procs = []
    if isinstance(camera_bindings, list) and camera_bindings:
        try:
            sorted_bindings = sorted(camera_bindings, key=lambda b: int(b.get('index', 0)))
        except Exception:
            sorted_bindings = camera_bindings
        logical_indices = [int(b.get('index')) for b in sorted_bindings if b.get('index') is not None]
        print(f"[相机池]: 按逻辑索引启动相机进程: {logical_indices}")
        for idx in logical_indices:
            p = mp.Process(target=_camera_worker_process,
                           args=(idx, task_queue, stop_event, run_event, config, shared_states, capture_interval_s, use_soft_trigger),
                           name=f"cam_proc_{idx}", daemon=False)
            p.start()
            procs.append(p)
    else:
        # 无绑定信息时，退回到物理枚举顺序
        # 确认预期的相机数量与实际发现的相机数量
        expected_cameras = int((config.get('camera_setup', {}) or {}).get('expected_cameras', num_cameras) or num_cameras)
        if num_cameras < expected_cameras:
            print(f"[相机池]: ⚠️ 预期 {expected_cameras} 台相机，实际只检测到 {num_cameras} 台")
        for i in range(num_cameras):
            p = mp.Process(target=_camera_worker_process,
                           args=(i, task_queue, stop_event, run_event, config, shared_states, capture_interval_s, use_soft_trigger),
                           name=f"cam_proc_{i}", daemon=False)
            p.start()
            procs.append(p)

    # 等待所有相机状态更新完成
    start_time = time.time()
    timeout = 30  # 30秒超时
    all_ready = False
    # 采用实际启动的逻辑索引集合用于就绪判定
    try:
        started_indices = set(logical_indices) if (isinstance(camera_bindings, list) and camera_bindings) else set(range(num_cameras))
    except Exception:
        started_indices = set(range(num_cameras))

    while not stop_event.is_set() and not all_ready and (time.time() - start_time) < timeout:
        time.sleep(0.5)  # 每0.5秒检查一次

        # 检查所有相机是否已真正进入可采集状态
        initialized_cameras = 0
        not_ready_detail = []
        for i in started_indices:
            if i in shared_states and shared_states[i]:
                st = str((shared_states[i] or {}).get('status', ''))
                if st in ('Grabbing', 'Connected (Test Mode)'):
                    initialized_cameras += 1
                else:
                    err = str((shared_states[i] or {}).get('error', '') or '')
                    if err:
                        not_ready_detail.append(f"cam{i}:{st}({err})")
                    else:
                        not_ready_detail.append(f"cam{i}:{st}")
            else:
                not_ready_detail.append(f"cam{i}:NoState")

        if initialized_cameras == len(started_indices):
            all_ready = True
            print(f"[相机池]: 所有 {len(started_indices)} 台相机已就绪")
            
            # 检查是否需要更新camera_nic_bindings和camera_rois
            try:
                update_empty_camera_configs(config, shared_states, num_cameras)
            except Exception as e:
                print(f"[相机池]: 更新空相机配置时出错: {e}")
                
            cameras_ready_event.set()
    
    if not all_ready and not stop_event.is_set():
        print(f"[相机池]: 警告 - 超时未能初始化所有相机，继续运行 (已就绪 {initialized_cameras}/{len(started_indices)})")
        try:
            if not_ready_detail:
                print(f"[相机池]: 未就绪详情: {', '.join(not_ready_detail[:10])}")
        except Exception:
            pass
        # 即使有相机未准备好，也发送就绪事件以避免主进程无限等待
        cameras_ready_event.set()

    # 守护等待退出
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        # Ctrl+C 触发时安静退出
        pass
    finally:
        # 结束所有子进程
        print("[相机池]: 正在关闭所有相机进程...")
        for p in procs:
            try:
                p.join(timeout=2.0)
                if p.is_alive():
                    print(f"[相机池]: 进程 {p.name} 未能在超时时间内退出，执行 terminate()")
                    try:
                        p.terminate()
                        p.join(timeout=2.0)
                        if p.is_alive():
                            print(f"[相机池]: 进程 {p.name} 仍未退出，尝试 kill()")
                            try:
                                p.kill()
                                p.join(timeout=1.0)
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
        
        # 清理相机设置工具的资源
        try:
            setup_tool.cleanup()
        except Exception:
            pass
            
        print("[相机池]: 所有相机资源已释放")


def update_empty_camera_configs(config, shared_states, num_cameras):
    """
    自动更新空的camera_nic_bindings和camera_rois配置
    当相机初始化成功后调用
    """
    if not isinstance(config, dict):
        print("[相机池]: 配置非字典类型，无法更新")
        return
    
    config_updated = False
    camera_setup = config.get('camera_setup', {})
    
    if not isinstance(camera_setup, dict):
        print("[相机池]: camera_setup配置不是字典类型，无法更新")
        return
    
    # 获取相机绑定信息，查找MAC地址到逻辑索引的映射
    camera_bindings = camera_setup.get('camera_bindings', [])
    mac_to_logical = {}
    for binding in camera_bindings:
        if binding.get('mac') and binding.get('index') is not None:
            mac_to_logical[binding.get('mac').upper()] = binding.get('index')
    
    # 1. 检查camera_nic_bindings是否为空
    camera_nic_bindings = camera_setup.get('camera_nic_bindings', {})
    if not camera_nic_bindings:
        print("[相机池]: camera_nic_bindings为空，将自动更新")
        
        # 获取网卡信息
        network_interfaces = camera_setup.get('network_interfaces', [])
        if not network_interfaces:
            print("[相机池]: 未找到网卡信息，无法更新camera_nic_bindings")
        else:
            # 创建网卡索引映射
            indexed_nics = {}
            for nic in network_interfaces:
                nic_idx = nic.get('index', -1)
                if nic_idx >= 0:
                    indexed_nics[nic_idx] = nic
            
            if not indexed_nics:
                print("[相机池]: 未找到有效的网卡索引，无法更新camera_nic_bindings")
            else:
                # 为每个相机分配网卡
                new_bindings = {}
                
                # 遍历所有已初始化的相机
                for i in range(num_cameras):
                    if i in shared_states and shared_states[i]:
                        cam_info = shared_states[i]
                        cam_mac = cam_info.get('mac', '').upper()
                        cam_ip = cam_info.get('ip')

                        if not cam_mac:
                            print(f"[相机池]: 相机 {i} 缺少MAC信息，跳过自动网卡绑定推导")
                            continue
                        
                        # 找到对应的逻辑索引
                        logical_idx = None
                        if cam_mac in mac_to_logical:
                            logical_idx = mac_to_logical[cam_mac]
                        else:
                            # 如果没有找到MAC映射，使用物理索引作为逻辑索引（回退方案）
                            logical_idx = i
                            if _privacy_enabled(config):
                                print(f"[相机池]: 相机 MAC={_mask_mac(cam_mac)} 未找到逻辑索引映射，使用物理索引 {i} 作为回退")
                            else:
                                print(f"[相机池]: 相机 MAC={cam_mac} 未找到逻辑索引映射，使用物理索引 {i} 作为回退")
                        
                        if cam_ip:
                            # 根据相机IP前缀找到对应的网卡
                            cam_prefix = ".".join(cam_ip.split('.')[:3])
                            matching_nic_idx = None
                            
                            for nic_idx, nic in indexed_nics.items():
                                nic_ip = nic.get('ip', '')
                                nic_prefix = ".".join(nic_ip.split('.')[:3])
                                if cam_prefix == nic_prefix:
                                    matching_nic_idx = nic_idx
                                    break
                            
                            # 如果找到匹配的网卡，建立绑定
                            if matching_nic_idx is not None:
                                new_bindings[str(logical_idx)] = matching_nic_idx
                                if _privacy_enabled(config):
                                    print(f"[相机池]: 相机 MAC={_mask_mac(cam_mac)}, 逻辑索引={logical_idx} -> 网卡 {matching_nic_idx} (IP前缀匹配)")
                                else:
                                    print(f"[相机池]: 相机 MAC={cam_mac}, 逻辑索引={logical_idx} -> 网卡 {matching_nic_idx} (IP前缀匹配)")
                            # 如果没有找到匹配的网卡，尝试使用物理索引匹配网卡
                            elif logical_idx in indexed_nics:
                                new_bindings[str(logical_idx)] = logical_idx
                                if _privacy_enabled(config):
                                    print(f"[相机池]: 相机 MAC={_mask_mac(cam_mac)}, 逻辑索引={logical_idx} -> 网卡 {logical_idx} (索引匹配)")
                                else:
                                    print(f"[相机池]: 相机 MAC={cam_mac}, 逻辑索引={logical_idx} -> 网卡 {logical_idx} (索引匹配)")
                
                if new_bindings:
                    camera_setup['camera_nic_bindings'] = new_bindings
                    config_updated = True
                    print(f"[相机池]: 已更新camera_nic_bindings: {new_bindings}")
    
    # 2. 检查camera_rois是否为空
    camera_rois = config.get('camera_rois', {})
    if not camera_rois:
        print("[相机池]: camera_rois为空，将自动更新")
        
        # 获取ROI模板文件
        roi_template = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
        if not roi_template:
            print("[相机池]: 未找到ROI模板文件，无法更新camera_rois")
        else:
            # 为每个相机分配相同的ROI模板
            new_rois = {}
            
            # 优先使用逻辑索引关联ROI
            for mac, logical_idx in mac_to_logical.items():
                new_rois[str(logical_idx)] = roi_template
            
            # 如果没有相机绑定信息，则使用物理索引
            if not mac_to_logical:
                for i in range(num_cameras):
                    if i in shared_states and shared_states[i]:
                        new_rois[str(i)] = roi_template
            
            if new_rois:
                config['camera_rois'] = new_rois
                config_updated = True
                print(f"[相机池]: 已更新camera_rois: {new_rois}")
    
    # 如果配置有更新，保存到文件
    if config_updated:
        try:
            # 优先回写当前加载的配置文件；若未知则回写 config.yaml。
            cfg_path = str(config.get('__config_path', '') or '')
            if not cfg_path:
                cfg_path = 'config.yaml'
            cfg_to_save = dict(config)
            cfg_to_save.pop('__config_path', None)
            if cfg_path.endswith(('.yml', '.yaml')):
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(cfg_to_save, f, allow_unicode=True, sort_keys=False)
            else:
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg_to_save, f, ensure_ascii=False, indent=2)
            print(f"[相机池]: 配置文件已更新: {cfg_path}")
        except Exception as e:
            print(f"[相机池]: 写入配置文件失败: {e}")
