import time
import queue
import traceback
import numpy as np
import multiprocessing as mp
import json
from camera_manager import CameraManager, MultiCameraSetup
from GigECamera_Types import TriggerMode_On, TriggerMode_Off, TriggerSource_Software, TriggerActivation_RisingEdge, MVStreamCB  # noqa
from MVGigE import *
from genenrate_test_image import generate_test_image
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


def _camera_worker_process(cam_index: int, task_queue, stop_event, run_event, config: dict,
                           shared_states, capture_interval_s: float, use_soft_trigger: bool):
    """每个相机一个独立进程：打开相机→应用参数→注册回调→按周期软件触发→写入共享队列。"""
    from camera_manager import CameraManager  # 进程内导入，避免在spawn时重复初始化
    from MVGigE import MVStartGrab, MVStopGrab, MVTriggerSoftware, MVST_SUCCESS
    import copy

    # 读取 MAC 绑定：camera_setup.mac_bindings { index: "AA:BB:CC:DD:EE:FF" }
    cs = (config.get('camera_setup', {}) or {})
    mac_bindings = cs.get('mac_bindings', {})
    ip_bindings = cs.get('ip_bindings', {})
    desired_mac = None
    if isinstance(mac_bindings, dict):
        desired_mac = mac_bindings.get(str(cam_index)) or mac_bindings.get(cam_index)
    elif isinstance(mac_bindings, list) and cam_index < len(mac_bindings):
        desired_mac = mac_bindings[cam_index]
    cam = CameraManager(cam_index, mac=desired_mac)
    # 注入 ip_bindings 以便在 MAC 列表为空时尝试按 IP 打开
    if isinstance(ip_bindings, dict):
        cam.set_ip_bindings(ip_bindings)
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
            print(f"[相机进程 {cam_index}]: 打开相机失败，MAC: {desired_mac or '未指定'}")
            return

        # 相机已成功打开，更新状态
        shared_states[cam_index] = {"status": "Opened", "mac": cam.mac, "ip": cam.ip}
        print(f"[相机进程 {cam_index}]: 成功打开相机，MAC: {cam.mac}, IP: {cam.ip}")

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
            print(f"[相机进程 {cam_index}]: MVStartGrab 失败: {res}")
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
        
        # 计算并输出触发周期
        period_ms = _compute_trigger_period_ms(config)
        period_s = max(0.001, period_ms / 1000.0)
        if use_soft_trigger:
            print(f"[相机进程 {cam_index}]: 软件触发启用，周期 {period_ms:.2f} ms")
        else:
            print(f"[相机进程 {cam_index}]: 连续采集模式（未使用软触发）；回调即收帧。")

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

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[相机进程 {cam_index}]: 出现异常: {e}\n{error_msg}")
        shared_states[cam_index] = {"status": "Error", "error": str(e)}
    finally:
        try:
            print(f"[相机进程 {cam_index}]: 正在关闭相机...")
            if cam and hasattr(cam, 'handle') and cam.handle:
                try:
                    MVStopGrab(cam.handle)
                    print(f"[相机进程 {cam_index}]: 已停止图像采集")
                except Exception as e:
                    print(f"[相机进程 {cam_index}]: 停止采集时出错: {e}")
            
            if cam:
                cam.close()
                print(f"[相机进程 {cam_index}]: 相机已关闭")
        except Exception as e:
            print(f"[相机进程 {cam_index}]: 清理资源时出错: {e}")


def _test_camera_worker_process(cam_index: int, task_queue, stop_event, run_event, capture_interval_s: float,
                                shared_states):
    from genenrate_test_image import generate_test_image
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
    setup_tool = MultiCameraSetup(config)
    # 1) 启动阶段：不限网段发现相机MAC，并在缺失时写入 config
    setup_tool.bootstrap_mac_bindings()
    # 3) 可选：自动配置本机网卡到不同网段（谨慎使用，需管理员权限）
    if (config.get('camera_setup', {}) or {}).get('auto_configure_nics', False):
        setup_tool.auto_configure_nics()
    # 2) 可选：自动分配相机IP（需管理员权限，可能修改网络环境，默认关闭）
    if (config.get('camera_setup', {}) or {}).get('auto_assign_ips', False):
        setup_tool.auto_assign_ips()

    res, num_cameras = MVGetNumOfCameras()
    if num_cameras == 0:
        print("[相机池]: 未检测到物理相机，进入统一全景测试(模拟)模式...")
        setup_tool.cleanup()
        from synthetic_test_scene import SyntheticGlassScene
        try:
            test_cameras = int((config.get('camera_setup', {}) or {}).get('expected_cameras', 1) or 1)
        except Exception:
            test_cameras = 1
        if test_cameras <= 0:
            test_cameras = 1
        acq = (config.get('camera_setup', {}) or {}).get('unified_params', {}).get('acquisition', {})
        cam_w = int(acq.get('width', 1280) or 1280)
        cam_h = int(acq.get('height', 960) or 960)
        runtime = (config.get('camera_setup') or {}).get('runtime', {})
        legacy_sys = (config.get('system_params') or {})
        frame_rate = float(acq.get('frame_rate', runtime.get('frame_rate', legacy_sys.get('target_fps', 15))) or runtime.get('target_fps', legacy_sys.get('target_fps', 15)) or 15)
        # 设定每"行"间隔 = 1/frame_rate 秒 -> 每个相机每秒也近似输出 frame_rate 帧
        row_interval = 1.0 / max(1.0, frame_rate)
        # 垂直步进像素：使得完整穿过高度大约需要 cam_h / step_px 行；设置为高度 / (frame_rate * 6) 约 6 秒穿过
        step_px = max(1, int(cam_h / (frame_rate * 6)))
        # 读取可选的玻璃宽度控制参数
        pane_width_factor = float(runtime.get('test_scene_width_factor', legacy_sys.get('test_scene_width_factor', 0.95)) or 0.95)
        pane_margin_px = runtime.get('test_scene_margin_px', legacy_sys.get('test_scene_margin_px'))
        try:
            pane_margin_px = int(pane_margin_px) if pane_margin_px is not None else None
        except Exception:
            pane_margin_px = None
        # 加载各相机 ROI （复用与处理端类似逻辑）
        def _load_rois(path: str):
            rois = []
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # averaged_rois 结构
                    best_key = None; best_cnt = -1
                    for k,v in data.items():
                        if isinstance(v, dict) and 'averaged_rois' in v:
                            cnt = v.get('source_image_count', len(v['averaged_rois']))
                            if cnt > best_cnt:
                                best_cnt = cnt; best_key = k
                    if best_key is not None:
                        rois = data[best_key]['averaged_rois']
                    elif 'rois' in data and isinstance(data['rois'], list):
                        rois = data['rois']
                elif isinstance(data, list):
                    rois = data
            except Exception:
                rois = []
            return rois
        cam_rois_cfg = config.get('camera_rois', {})
        rois_per_cam = []
        for ci in range(test_cameras):
            roi_path = None
            if isinstance(cam_rois_cfg, dict):
                roi_path = cam_rois_cfg.get(str(ci)) or cam_rois_cfg.get(ci)
            elif isinstance(cam_rois_cfg, list) and ci < len(cam_rois_cfg):
                roi_path = cam_rois_cfg[ci]
            if not roi_path:
                roi_path = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
            rois_per_cam.append(_load_rois(roi_path))

        scene = SyntheticGlassScene(test_cameras, cam_w, cam_h, step_px=step_px,
                                    pane_width_factor=pane_width_factor, pane_margin_px=pane_margin_px,
                                    rois_per_cam=rois_per_cam, roi_only=False)
        print(f"[相机池][测试模式]: 玻璃+ROI 模式 | cameras={test_cameras}, size={cam_w}x{cam_h}, roi_loaded={[len(r) for r in rois_per_cam]}")
        
        # 初始化所有模拟相机的状态
        for idx in range(test_cameras):
            shared_states[idx] = {
                "status": "Connected (Test Mode)",
                "type": "Synthetic",
                "camera_index": idx,
                "frame_size": f"{cam_w}x{cam_h}",
                "rois_count": len(rois_per_cam[idx]) if idx < len(rois_per_cam) else 0
            }
        
        # 通知主进程相机已准备就绪
        print(f"[相机池]: 所有{test_cameras}台模拟相机已就绪，发送就绪事件...")
        cameras_ready_event.set()
        
        next_row_t = time.perf_counter()
        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_row_t:
                time.sleep(min(next_row_t - now, 0.002))
                continue
            frames = scene.next_row_frames()
            for idx, frame in enumerate(frames):
                shared_states[idx] = {"status": "Connected (Test Mode)"}
                try:
                    task_queue.put_nowait({"data": frame, "cam_index": idx})
                except queue.Full:
                    pass
            next_row_t += row_interval
        return

    # ---------------------------- 实机模式 ----------------------------
    print(f"[相机池]: 检测到 {num_cameras} 台相机，正在打开...")
    setup_tool.cleanup()
    # 使用多进程为每个相机启动一个采集进程
    procs = []
    for i in range(num_cameras):
        p = mp.Process(target=_camera_worker_process,
                       args=(i, task_queue, stop_event, run_event, config, shared_states, capture_interval_s, use_soft_trigger),
                       name=f"cam_proc_{i}", daemon=True)
        p.start()
        procs.append(p)

    # 等待所有相机状态更新完成
    start_time = time.time()
    timeout = 30  # 30秒超时
    all_ready = False
    
    while not stop_event.is_set() and not all_ready and (time.time() - start_time) < timeout:
        time.sleep(0.5)  # 每0.5秒检查一次
        
        # 检查所有相机是否已更新状态
        initialized_cameras = 0
        for i in range(num_cameras):
            if i in shared_states and shared_states[i]:
                initialized_cameras += 1
        
        if initialized_cameras == num_cameras:
            all_ready = True
            print(f"[相机池]: 所有{num_cameras}台相机已初始化完成，发送就绪事件...")
            cameras_ready_event.set()
    
    if not all_ready and not stop_event.is_set():
        print(f"[相机池]: 警告 - 在{timeout}秒内未能初始化所有相机，但仍继续运行...")
        # 即使有相机未准备好，也发送就绪事件以避免主进程无限等待
        cameras_ready_event.set()

    # 守护等待退出
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        # 结束所有子进程
        print("[相机池]: 正在关闭所有相机进程...")
        for p in procs:
            try:
                p.join(timeout=2.0)
                if p.is_alive():
                    print(f"[相机池]: 警告 - 进程 {p.name} 未能在超时时间内退出，将被强制终止")
            except Exception as e:
                print(f"[相机池]: 关闭进程 {p.name} 时出错: {e}")
        
        # 清理相机设置工具的资源
        try:
            setup_tool.cleanup()
            print("[相机池]: 相机设置工具资源已释放")
        except Exception as e:
            print(f"[相机池]: 清理相机设置工具资源时出错: {e}")
            
        print("[相机池]: 所有相机资源已释放，采集进程退出。")
