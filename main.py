# --- START OF FILE main.py ---
import multiprocessing
import threading
import json
import sys
import os
import uvicorn
import msvcrt  # Windows 按键检测
import cv2
import signal
import time
import ctypes

# Import process and thread functions from their respective modules
from http_client import NonBlockingHttpClient
from camera_process import camera_pool_process
from processing_worker import calculation_worker
from api_server import create_app
from rejection_controller import RejectionController
from alarm_light_controller import AlarmLightController, _AlarmLightService 

# 全局变量，用于在信号处理器中访问
stop_event = None
processes = []
rejection_controller = None
http_client = None
alarm_light_controller = None  # 这将是代理对象
alarm_light_service_thread = None # 新增：服务线程
alarm_light_service_instance = None # 新增：服务实例

# --- Windows 单实例与 Job 对象，确保主进程退出时自动清理子进程 ---
_global_job_handle = None
_global_mutex_handle = None

def _ensure_single_instance_and_job():
    """在 Windows 上：
    1) 使用命名互斥量保证单实例运行；
    2) 创建 Job 对象并设置 KillOnJobClose，确保主进程被任务管理器结束时，所有子进程自动被终止。
    非 Windows 平台则跳过。
    """
    global _global_job_handle, _global_mutex_handle
    try:
        if os.name != 'nt':
            return
        kernel32 = ctypes.windll.kernel32
        # 1) 单实例：命名互斥量（全局命名空间）
        mutex_name = "Global\\GlassAlgo_v4_MainMutex"
        kernel32.SetLastError(0)
        _global_mutex_handle = kernel32.CreateMutexW(None, False, ctypes.c_wchar_p(mutex_name))
        last_err = ctypes.GetLastError()
        ERROR_ALREADY_EXISTS = 183
        if last_err == ERROR_ALREADY_EXISTS or _global_mutex_handle == 0:
            # 已有实例在运行
            sys.exit("检测到已有实例在运行，已退出。")

        # 2) Job 对象：Kill on close
        job_name = None  # 匿名即可
        _global_job_handle = kernel32.CreateJobObjectW(None, job_name)
        if not _global_job_handle:
            return
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint),
                ("SchedulingClass", ctypes.c_uint)
            ]
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)
            ]
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)
            ]
        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        res = kernel32.SetInformationJobObject(_global_job_handle, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info))
        if not res:
            return
        # 将当前进程加入 Job
        hProcess = kernel32.GetCurrentProcess()
        kernel32.AssignProcessToJobObject(_global_job_handle, hProcess)
    except Exception:
        # 任一失败不影响主流程
        pass

def main():
    """Main function to initialize shared resources, start child processes, and run the API server."""
    multiprocessing.freeze_support()
    cv2.setUseOptimized(True)
    print("[主进程]: 应用程序启动...")
    # Windows: 确保单实例并启用 Job 清理
    _ensure_single_instance_and_job()
    
    # 全局变量，用于信号处理
    global stop_event, processes, rejection_controller, http_client, alarm_light_controller, alarm_light_service_thread, alarm_light_service_instance
    
    # 设置信号处理器，优雅处理CTRL+C和Windows关闭事件
    def signal_handler(sig, frame):
        print(f"\n[主进程]: 收到信号 {sig}，开始优雅关闭...")
        if stop_event:
            stop_event.set()
        # 给2秒钟让子进程开始清理
        time.sleep(2)
    
    # 注册SIGINT(Ctrl+C)和SIGTERM(终止)信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 在Windows上特殊处理CTRL+BREAK信号
    if hasattr(signal, 'SIGBREAK'):  # Windows特有
        signal.signal(signal.SIGBREAK, signal_handler)

    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        sys.exit(f"错误: 无法加载 config.json: {e}")

    # --- 启动前校验：ROI 不得越界于相机分辨率 ---
    try:
        roi_file = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
        with open(roi_file, 'r', encoding='utf-8') as f:
            averaged_data = json.load(f)
        if not averaged_data:
            sys.exit("错误: ROI 模板文件为空，无法启动。")

        best_group_key = max(averaged_data, key=lambda k: averaged_data[k].get('source_image_count', 0))
        template_rois = averaged_data[best_group_key]['averaged_rois']

        acq = config.get('camera_setup', {}).get('unified_params', {}).get('acquisition', {})
        cam_w = int(acq.get('width', 0))
        cam_h = int(acq.get('height', 0))
        if cam_w <= 0 or cam_h <= 0:
            sys.exit("错误: 未在 config.camera_setup.unified_params.acquisition 中找到有效的相机分辨率（width/height）。")

        errors = []
        for idx, r in enumerate(template_rois):
            x, y = int(r.get('x', -1)), int(r.get('y', -1))
            w, h = int(r.get('width', -1)), int(r.get('height', -1))
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                errors.append(f"ROI[{idx}] 的坐标或尺寸非法: x={x}, y={y}, w={w}, h={h}")
                continue
            if x + w > cam_w or y + h > cam_h:
                errors.append(f"ROI[{idx}] 越界: (x={x}, y={y}, w={w}, h={h}) 超出相机分辨率 ({cam_w}x{cam_h})")

        if errors:
            print("\n❌ ROI 越界校验失败：")
            for e in errors: print(" - "+e)
            sys.exit("程序已退出，请修正 ROI 或相机分辨率配置后重试。")
        else:
            print(f"✅ ROI 越界校验通过，共 {len(template_rois)} 个 ROI，分辨率 {cam_w}x{cam_h}。")
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"错误: ROI 越界校验时出现异常: {e}")

    # --- Setup Multiprocessing Manager and Shared State ---
    cs = config.get('camera_setup', {})
    try:
        num_bindings = len(cs.get('camera_bindings', []))
        NUM_CAMERAS = num_bindings if num_bindings > 0 else int(cs.get('expected_cameras', 1))
    except Exception:
        NUM_CAMERAS = 1
    if NUM_CAMERAS <= 0: NUM_CAMERAS = 1
    
    sys_params = config.get('system_params', {})
    logical_cores = os.cpu_count() or 4
    try:
        configured_proc_workers = int(sys_params.get('process_workers', 0) or 0)
    except Exception:
        configured_proc_workers = 0
    
    if configured_proc_workers > 0:
        NUM_WORKERS = max(1, configured_proc_workers)
    else:
        NUM_WORKERS = max(1, min(logical_cores - 2, NUM_CAMERAS * 2))

    print(f"[主进程]: 预期相机数={NUM_CAMERAS} | 计算进程数={NUM_WORKERS} (CPU核心={logical_cores})")

    manager = multiprocessing.Manager()

    # 确保存储目录存在
    storage_path = config.get('storage_path', 'inspection_results')
    if not os.path.exists(storage_path):
        try:
            os.makedirs(storage_path)
            print(f"[主进程]: 创建存储目录: {storage_path}")
        except Exception as e:
            print(f"[主进程]: 警告 - 无法创建存储目录 {storage_path}: {e}")

    # Shared settings object
    shared_settings = manager.Namespace()
    rejection_params = config.get('rejection_params', {})
    system_params = config.get('system_params', {})
    server_config = config.get('server_config', {})
    shared_settings.max_defect_size_mm = rejection_params.get('max_defect_size_mm', 20.0)
    shared_settings.REJECTION_PULSE_MS = rejection_params.get('REJECTION_PULSE_MS', 100)
    shared_settings.REJECTION_DELAY_S = rejection_params.get('REJECTION_DELAY_S', 1.5)
    shared_settings.storage_path = config.get('storage_path', 'inspection_results')
    shared_settings.pixels_per_mm = system_params.get('pixels_per_mm', 2.4)
    # 新增: 算法模式 (1=浅色 2=深色) 默认1
    shared_settings.algorithm_mode = 1
    shared_settings.lineName = config.get('lineName', 'UNKNOWN_LINE')
    shared_settings.server = server_config.get('server', '127.0.0.1')
    shared_settings.upload_url = server_config.get('upload_url', f'http://{shared_settings.server}:5000/upload')
    # 修改：仅剔废上报端点
    shared_settings.stats_push_url = server_config.get('stats_push_url', f'http://{shared_settings.server}:8085/fastapi/glass/updateRejections')
    shared_settings.heartbeat_url = server_config.get('heartbeat_url', f'http://{shared_settings.server}:8085/fastapi/system/heartbeat')
    
    # 读取报警器配置
    alarm_params = config.get('alarm_light_params', {})
    shared_settings.alarm_port = alarm_params.get('port')
    shared_settings.ng_buzz_duration_s = float(alarm_params.get('ng_buzz_duration_s', 1.0))
    shared_settings.rejection_buzz_duration_s = float(alarm_params.get('rejection_buzz_duration_s', 3.0))

    # Queues for data flow
    queue_size_factor = int(system_params.get('queue_size_factor', 2) or 2)
    task_queue = manager.Queue(maxsize=NUM_WORKERS * NUM_CAMERAS * queue_size_factor)
    results_queue = manager.Queue(maxsize=NUM_WORKERS * NUM_CAMERAS * queue_size_factor)
    rejection_queue = manager.Queue()
    alarm_command_queue = manager.Queue() # <-- 为报警器创建跨进程队列
    
    # Shared state
    shared_camera_states = manager.dict()
    # 恢复产量 + 剔废统计
    shared_yield_counter = manager.Value('i', 0)
    shared_rejection_counter = manager.Value('i', 0)
    manual_reject_flag = manager.Value('b', False)
    machine_state_shared = manager.Value('i', 0)
    shared_rejection_mode = manager.Value('i', 2)
    can_late_reject = manager.Value('b', False)
    shared_collection_id = manager.Value('i', -1)
    shared_user_id_auto = manager.Value('i', config.get('user_id_auto', 7))
    shared_user_id_manual = manager.Value('i', config.get('user_id_manual', 9999))

    # Process control events
    stop_event = multiprocessing.Event()
    run_event = multiprocessing.Event()
    cameras_ready_event = multiprocessing.Event()

    # Threading lock
    stats_lock = threading.Lock()

    # Hardware controllers and services
    rejection_controller = RejectionController()
    
    if shared_settings.alarm_port:
        alarm_light_service_instance = _AlarmLightService(port=shared_settings.alarm_port, baud_rate=9600, command_queue=alarm_command_queue)
        if alarm_light_service_instance.ser:
            alarm_light_service_thread = threading.Thread(target=alarm_light_service_instance.run, daemon=True)
            alarm_light_service_thread.start()
            alarm_light_controller = AlarmLightController(command_queue=alarm_command_queue)
            alarm_light_controller.set_startup_state()
        else:
            print("[主进程]: 声光报警器串口初始化失败，该功能将被禁用。")
            alarm_light_controller = AlarmLightController(command_queue=None)
    else:
        print("[主进程]: 警告 - 未在config.json中配置声光报警器端口，将不启用该功能。")
        alarm_light_controller = AlarmLightController(command_queue=None)

    http_client = NonBlockingHttpClient(max_workers=int(system_params.get('http_client_max_workers', 4) or 4))

    # --- Create and Start Child Processes ---
    processes = []
    
    pool_proc = multiprocessing.Process(
        target=camera_pool_process,
        args=(task_queue, stop_event, run_event, cameras_ready_event, config, shared_camera_states),
        daemon=False
    )
    processes.append(pool_proc)
    
    for i in range(NUM_WORKERS):
        worker_proc = multiprocessing.Process(
            target=calculation_worker, 
            args=(i, task_queue, results_queue, stop_event, run_event, config, shared_settings),
            daemon=True
        )
        processes.append(worker_proc)
        
    for p in processes:
        p.start()

    # --- Prepare Shared Objects for FastAPI ---
    shared_objects = {
        'stop_event': stop_event, 'run_event': run_event,
        'camera_states': shared_camera_states,
        'settings': shared_settings,
        'counters': (shared_rejection_counter, shared_yield_counter),  # 顺序: 0=rejections 1=yield
        'queues': {'task': task_queue, 'results': results_queue, 'rejection': rejection_queue},
        'flags': (manual_reject_flag, shared_rejection_mode, can_late_reject),
        'machine_state': machine_state_shared,
        'stats_lock': stats_lock,
        'metadata': (shared_collection_id, shared_user_id_auto, shared_user_id_manual),
        'rejection_controller': rejection_controller,
        'alarm_light_controller': alarm_light_controller, # <-- 传递安全的代理对象
    'http_client': http_client
    }

    app = create_app(num_cameras=NUM_CAMERAS, shared_objects=shared_objects)

    shared_settings.stats_push_interval_s = int(system_params.get('stats_push_interval_s', 601) or 601)
    shared_settings.http_get_timeout_s = float(system_params.get('http_get_timeout_s', 5) or 5)
    shared_settings.http_post_timeout_s = float(system_params.get('http_post_timeout_s', 5) or 5)
    shared_settings.http_upload_timeout_s = float(system_params.get('http_upload_timeout_s', 15) or 15)
    cors_list = config.get('server_config', {}).get('cors_origins', ["*"])
    try:
        shared_settings.cors_origins = list(cors_list)
    except Exception:
        shared_settings.cors_origins = ["*"]

    # --- Wait for cameras ---
    print("[主进程]: 等待相机初始化...")
    camera_ready = cameras_ready_event.wait(timeout=60)
    
    if camera_ready:
        print("✅ [主进程]: 相机就绪")
    else:
        print("⚠️ [主进程]: 等待相机初始化超时")
    
    # 根据DEBUG_MODE_ON控制前台/后台行为
    debug_mode = bool(config.get('DEBUG_MODE_ON', True))
    if not debug_mode:
        print("相机启动完成，按任意键继续...")
        try:
            # 等待任意键
            msvcrt.getch()
        except Exception:
            pass
        # 尝试将控制台隐藏（在nuitka打包成exe时生效最佳）
        try:
            import ctypes
            whnd = ctypes.windll.kernel32.GetConsoleWindow()
            if whnd:
                SW_HIDE = 0
                ctypes.windll.user32.ShowWindow(whnd, SW_HIDE)
        except Exception:
            pass
    else:
        print("[主进程]: 启动API服务器...")

    # --- Run Server ---
    auto_start = config.get('system_params', {}).get('auto_start', False)
    if auto_start:
        print("[主进程]: 系统配置为自动启动模式，设置运行事件...")
        run_event.set()
    
    try:
        listen_host = config.get('server_config', {}).get('listen_host', '0.0.0.0')
        listen_port = int(config.get('server_config', {}).get('listen_port', 12450) or 12450)
        # 在非调试模式下，避免uvicorn的冗余日志输出
        if not debug_mode:
            uvicorn.run(app, host=listen_host, port=listen_port, log_level="warning")
        else:
            uvicorn.run(app, host=listen_host, port=listen_port)
    except KeyboardInterrupt:
        print("\n[主进程]: 接收到键盘中断信号，正在进行优雅关闭...")
    except Exception as e:
        print(f"\n[主进程]: 运行时发生异常: {e}")
    finally:
        print("\n[主进程]: 正在终止所有子进程...")
        stop_event.set()
        
        for i, p in enumerate(processes):
            try:
                p.join(timeout=5.0)
            except Exception as e:
                print(f"[主进程]: 等待进程终止时出错: {e}")
        
        # --- 关闭硬件控制器和服务 ---
        if rejection_controller:
            try:
                rejection_controller.close()
                print("[主进程]: 排废硬件控制器已关闭")
            except Exception as e:
                print(f"[主进程]: 关闭排废控制器时出错: {e}")
        
        if alarm_light_controller:
            try:
                alarm_light_controller.close()
                print("[主进程]: 声光报警器关闭信号已发送。")
            except Exception as e:
                print(f"[主进程]: 发送报警器关闭信号时出错: {e}")
        
        if alarm_light_service_instance:
             try:
                alarm_light_service_instance.stop()
             except Exception as e:
                print(f"[主进程]: 停止报警器服务时出错: {e}")
        
        if alarm_light_service_thread and alarm_light_service_thread.is_alive():
            try:
                alarm_light_service_thread.join(timeout=2.0)
                print("[主进程]: 声光报警器服务已停止。")
            except Exception as e:
                print(f"[主进程]: 等待报警器服务线程退出时出错: {e}")
            
        if http_client:
            try:
                http_client.shutdown()
                print("[主进程]: HTTP客户端已关闭")
            except Exception as e:
                print(f"[主进程]: 关闭HTTP客户端时出错: {e}")
            
        print("[主进程]: 所有资源已释放，程序退出。")

if __name__ == '__main__':
    main()