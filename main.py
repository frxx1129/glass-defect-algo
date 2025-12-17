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
import yaml

# Import process and thread functions from their respective modules
from http_client import NonBlockingHttpClient
from camera_process import camera_pool_process
from processing_worker import calculation_worker
from api_server import create_app
from rejection_controller import RejectionController
from alarm_light_controller import AlarmLightController, _AlarmLightService 

# 全局变量，用于在信号处理器中访问
stop_event = None
# 子进程控制事件，与最终退出事件分离：分别控制相机与计算
cam_stop_event = None
worker_stop_event = None
processes = []
# 显式持有相机与计算进程的引用，便于分步关闭
camera_proc_ref = None
worker_procs_ref = []
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
    # --- 日志重定向：将控制台输出同时写入 error.log（每次启动覆盖重建） ---
    log_file = None
    try:
        # 在打包场景使用可执行文件目录，否则使用脚本所在目录
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(base_dir, 'error.log')
        log_file = open(log_path, mode='w', encoding='utf-8')

        class _Tee:
            def __init__(self, *streams):
                self._streams = [s for s in streams if s]
            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass
            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass
            def isatty(self):
                return False

        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(_orig_stdout, log_file)
        sys.stderr = _Tee(_orig_stderr, log_file)
        print(f"--- Log start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    except Exception:
        # 若日志初始化失败，不影响主体流程
        pass

    multiprocessing.freeze_support()
    cv2.setUseOptimized(True)
    print("[主进程]: 应用程序启动...")
    # Windows: 确保单实例并启用 Job 清理
    _ensure_single_instance_and_job()
    
    # 全局变量，用于信号处理
    global stop_event, cam_stop_event, worker_stop_event, processes, rejection_controller, http_client, alarm_light_controller, alarm_light_service_thread, alarm_light_service_instance, camera_proc_ref, worker_procs_ref
    
    # 设置信号处理器，优雅处理CTRL+C和Windows关闭事件
    def signal_handler(sig, frame):
        print(f"\n[主进程]: 收到信号 {sig}，开始优雅关闭...")
        if stop_event:
            stop_event.set()
        # 立即通知各子系统退出，避免 Ctrl+C 卡住太久
        try:
            if cam_stop_event:
                cam_stop_event.set()
        except Exception:
            pass
        try:
            if worker_stop_event:
                worker_stop_event.set()
        except Exception:
            pass
        try:
            if 'run_event' in globals() and run_event:
                run_event.clear()
        except Exception:
            pass
    
    # 注册SIGINT(Ctrl+C)和SIGTERM(终止)信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 在Windows上特殊处理CTRL+BREAK信号
    if hasattr(signal, 'SIGBREAK'):  # Windows特有
        signal.signal(signal.SIGBREAK, signal_handler)

    # 优先读取 YAML 配置；若不存在则回退到 JSON（并尝试用 YAML 解析器解析 JSON，失败再用 json.load）
    try:
        cfg_path = 'config.yaml' if os.path.exists('config.yaml') else 'config.json'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            if cfg_path.endswith(('.yml', '.yaml')):
                config = yaml.safe_load(f)
            else:
                try:
                    config = yaml.safe_load(f)  # YAML 能解析 JSON 子集
                except Exception:
                    f.seek(0)
                    config = json.load(f)
        try:
            print(f"[主进程]: 加载配置文件: {cfg_path}")
        except Exception:
            pass
    except Exception as e:
        sys.exit(f"错误: 无法加载 {cfg_path}: {e}")

    # ================= 运行模式选择移除 =================
    # 直接使用正常检测模式，不再交互选择；若需采集模式可通过配置文件显式设置 data_collection_mode=true。
    data_collection_mode = bool(config.get('data_collection_mode', False)) and False  # 强制为 False
    config['data_collection_mode'] = False
    print("[主进程]: 已开启正常检测模式。")

    # --- 启动前校验：ROI 不得越界于相机分辨率 ---
    try:
        # 如果存在 camera_rois 优先使用其中第一个文件做越界校验
        roi_file = None
        cam_rois_cfg = config.get('camera_rois')
        if isinstance(cam_rois_cfg, dict) and cam_rois_cfg:
            # 取第一个 value
            try:
                roi_file = next(iter(cam_rois_cfg.values()))
                print(f"[主进程]: 使用 camera_rois 中的 '{roi_file}' 进行ROI越界校验")
            except Exception:
                roi_file = None
        if not roi_file:
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
    # 新增: 预期相机数与相机分辨率，供自动分路策略使用
    try:
        shared_settings.expected_cameras = int(NUM_CAMERAS)
    except Exception:
        shared_settings.expected_cameras = 1
    try:
        acq = config.get('camera_setup', {}).get('unified_params', {}).get('acquisition', {})
        shared_settings.cam_width = int(acq.get('width', 0) or 0)
        shared_settings.cam_height = int(acq.get('height', 0) or 0)
    except Exception:
        shared_settings.cam_width = 0
        shared_settings.cam_height = 0
    # 新增: 自动分路汇聚等待时间(ms)
    try:
        shared_settings.auto_route_decision_hold_ms = int(system_params.get('auto_route_decision_hold_ms', 120) or 120)
    except Exception:
        shared_settings.auto_route_decision_hold_ms = 120
    # 新增: 算法模式 (1=浅色 2=深色) 默认1
    shared_settings.algorithm_mode = 1
    # 数据采集模式标记与输出根目录
    try:
        shared_settings.data_collection_mode = bool(config.get('data_collection_mode', False))
    except Exception:
        shared_settings.data_collection_mode = False
    # 可通过 config.collection_output_root 自定义目录
    shared_settings.collection_output_root = config.get('collection_output_root', 'collected_dataset')
    if getattr(shared_settings, 'data_collection_mode', False):
        try:
            day_dir = time.strftime('%Y%m%d')
            base_dir = os.path.join(shared_settings.collection_output_root, day_dir)
            for sub in ['original', 'ng']:
                os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
            print(f"[主进程]: 采集模式启用，ROI裁剪保存目录: {base_dir}")
        except Exception as e:
            print(f"[主进程]: 创建采集输出目录失败: {e}")
    shared_settings.lineName = config.get('lineName', 'UNKNOWN_LINE')
    shared_settings.server = server_config.get('server', '127.0.0.1')
    shared_settings.upload_url = server_config.get('upload_url', f'http://{shared_settings.server}:5000/upload')
    # 修改：仅剔废上报端点
    shared_settings.stats_push_url = server_config.get('stats_push_url', f'http://{shared_settings.server}:8085/fastapi/glass/updateRejections')
    shared_settings.heartbeat_url = server_config.get('heartbeat_url', f'http://{shared_settings.server}:8085/fastapi/system/heartbeat')
    # 周期推送控制参数
    sys_params_root = config.get('system_params', {}) or {}
    shared_settings.enable_periodic_stats = bool(sys_params_root.get('enable_periodic_stats', True))
    try:
        shared_settings.stats_push_interval_s = float(sys_params_root.get('stats_push_interval_s', 30) or 30)
    except Exception:
        shared_settings.stats_push_interval_s = 30.0
    
    # 读取报警器配置
    alarm_params = config.get('alarm_light_params', {})
    shared_settings.alarm_port = alarm_params.get('port')
    shared_settings.ng_buzz_duration_s = float(alarm_params.get('ng_buzz_duration_s', 1.0))
    shared_settings.rejection_buzz_duration_s = float(alarm_params.get('rejection_buzz_duration_s', 3.0))

    # 按产线配置的“rejectionMark(参数) -> 通道号(1..n)”映射，支持每条产线独立配置
    try:
        mark_map = config.get('rejection_mark_to_channel', {}) or {}
        # 期望结构示例：{"Line1": {"0":1, "1":2, "2":3, "3":4}, "Line2": {"0":1, "1":2, "2":3, "3":4, "4":5}}
        # 统一键为字符串，值为整型
        normalized = {}
        for line_name, mapping in mark_map.items():
            if not isinstance(mapping, dict):
                continue
            nm = {}
            for k, v in mapping.items():
                try:
                    nm[str(int(k))] = int(v)
                except Exception:
                    # 保底：若键不可转为int，按原样存字符串键
                    try:
                        nm[str(k)] = int(v)
                    except Exception:
                        pass
            if nm:
                normalized[str(line_name)] = nm
        shared_settings.rejection_mark_to_channel = normalized
        # 轻量校验：若存在 <1 的通道或明显超出 1..8 的值，给出一次性警告
        try:
            bad = []
            for ln, mp in normalized.items():
                for mk, ch in mp.items():
                    if not isinstance(ch, int) or ch < 1 or ch > 8:
                        bad.append((ln, mk, ch))
            if bad:
                print(f"[主进程]: 警告 - rejection_mark_to_channel 中存在越界通道: {bad}，请检查配置与硬件接线。")
        except Exception:
            pass
    except Exception:
        shared_settings.rejection_mark_to_channel = {}

    # Queues for data flow
    queue_size_factor = int(system_params.get('queue_size_factor', 2) or 2)
    # 关键：高吞吐队列使用 multiprocessing.Queue，避免 Manager.Queue 在退出/高负载下卡在 managers IPC。
    task_queue = multiprocessing.Queue(maxsize=NUM_WORKERS * NUM_CAMERAS * queue_size_factor)
    results_queue = multiprocessing.Queue(maxsize=NUM_WORKERS * NUM_CAMERAS * queue_size_factor)
    rejection_queue = multiprocessing.Queue()
    # 报警器命令队列吞吐较低，保持简单同样用 multiprocessing.Queue
    alarm_command_queue = multiprocessing.Queue()
    # 采集模式下的玻璃会话状态（跨进程） cam_idx -> {active:bool, start_ts:int, folder:str}
    data_sessions = manager.dict()
    
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
    stop_event = multiprocessing.Event()          # 最终退出事件
    cam_stop_event = multiprocessing.Event()      # 相机子系统退出事件（用于先关相机）
    worker_stop_event = multiprocessing.Event()   # 计算子系统退出事件（用于后关计算）
    run_event = multiprocessing.Event()
    cameras_ready_event = multiprocessing.Event()

    # Threading lock
    stats_lock = threading.Lock()

    # Hardware controllers and services
    # 剔废控制器：从配置读取串口与通道映射
    rej_cfg = config.get('rejection_controller', {}) or {}
    rej_port = rej_cfg.get('port') or rej_cfg.get('serial_port') or None
    rej_baud = int(rej_cfg.get('baud', rej_cfg.get('baud_rate', 9600)) or 9600)
    try:
        rejection_controller = RejectionController(port=rej_port, baud=rej_baud)
    except Exception as e:
        # 若显式配置了端口，但初始化抛错，则直接跳过该设备
        print(f"[主进程]: 初始化剔废控制器异常，已跳过: {e}")
        rejection_controller = None

    # 健康检测：若配置了端口但 5 秒内无法建立稳定通信，则跳过该设备
    if rej_port:
        if rejection_controller is None:
            print("[主进程]: 剔除控制器未就绪（初始化失败），剔除功能将被禁用。")
        else:
            try:
                healthy = (not getattr(rejection_controller, 'is_simulation', False)) and \
                          rejection_controller.health_check(timeout_s=5.0, tries=5)
            except Exception as _:
                healthy = False
            if not healthy:
                print("[主进程]: 5 秒内无法与剔除控制器建立稳定通信，已跳过连接并禁用剔除功能。")
                rejection_controller = None
    
    if shared_settings.alarm_port:
        alarm_light_service_instance = _AlarmLightService(port=shared_settings.alarm_port, baud_rate=9600, command_queue=alarm_command_queue)
        if alarm_light_service_instance.ser:
            alarm_light_service_thread = threading.Thread(target=alarm_light_service_instance.run, daemon=True)
            alarm_light_service_thread.start()
            alarm_light_controller = AlarmLightController(command_queue=alarm_command_queue)
            # 应用远程报警推送配置（若存在）
            try:
                remote_cfg = config.get('remote_alarm', {}) or {}
                host = str(remote_cfg.get('host') or '').strip()
                port = str(remote_cfg.get('port') or '').strip()
                if host and port:
                    alarm_light_controller.remote_alarm_host = host
                    alarm_light_controller.remote_alarm_port = port
            except Exception:
                pass
            alarm_light_controller.set_startup_state()
        else:
            print("[主进程]: 声光报警器串口初始化失败，该功能将被禁用。")
            alarm_light_controller = AlarmLightController(command_queue=None)
            # 即使本地禁用，也可设置远程推送目标用于镜像
            try:
                remote_cfg = config.get('remote_alarm', {}) or {}
                host = str(remote_cfg.get('host') or '').strip()
                port = str(remote_cfg.get('port') or '').strip()
                if host and port:
                    alarm_light_controller.remote_alarm_host = host
                    alarm_light_controller.remote_alarm_port = port
            except Exception:
                pass
    else:
        print("[主进程]: 警告 - 配置中未设置声光报警器端口，将不启用该功能。")
        alarm_light_controller = AlarmLightController(command_queue=None)
        # 仅远程推送场景：允许没有本地串口，仅推送远端
        try:
            remote_cfg = config.get('remote_alarm', {}) or {}
            host = str(remote_cfg.get('host') or '').strip()
            port = str(remote_cfg.get('port') or '').strip()
            if host and port:
                alarm_light_controller.remote_alarm_host = host
                alarm_light_controller.remote_alarm_port = port
        except Exception:
            pass

    http_client = NonBlockingHttpClient(max_workers=int(system_params.get('http_client_max_workers', 4) or 4))

    # --- Create and Start Child Processes (封装为函数，便于重启) ---
    def _start_children():
        """启动相机池进程与计算进程，使用 child_stop_event 控制其生命周期。"""
        global processes, cam_stop_event, worker_stop_event, camera_proc_ref, worker_procs_ref
        nonlocal cameras_ready_event
        # 重置相机就绪事件
        # 启动前，重置相机就绪事件
        try:
            cameras_ready_event = multiprocessing.Event()
        except Exception:
            pass
        processes = []
        camera_proc_ref = multiprocessing.Process(
            target=camera_pool_process,
            args=(task_queue, cam_stop_event, run_event, cameras_ready_event, config, shared_camera_states),
            daemon=False
        )
        processes.append(camera_proc_ref)
        worker_procs_ref = []
        for i in range(NUM_WORKERS):
            worker_proc = multiprocessing.Process(
                target=calculation_worker,
                args=(i, task_queue, results_queue, worker_stop_event, run_event, config, shared_settings, data_sessions),
                daemon=True
            )
            worker_procs_ref.append(worker_proc)
            processes.append(worker_proc)
        for p in processes:
            p.start()
        print(f"[主进程]: 已启动子进程：相机池1 + 计算{NUM_WORKERS}。")

    def _restart_children(grace_seconds: float = 10.0):
        """优雅重启子进程：
        - 先关闭相机，再等待计算队列处理完成，最后关闭计算；
        - 等待 grace_seconds；
        - 清理并重建事件 -> 重新启动子进程。
        """
        global processes, cam_stop_event, worker_stop_event, camera_proc_ref, worker_procs_ref
        try:
            try:
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts = ''
            # 触发原因由调用方打印；这里统一输出重启前状态快照
            try:
                try:
                    tq = task_queue.qsize()
                except Exception:
                    tq = None
                try:
                    rq = results_queue.qsize()
                except Exception:
                    rq = None
                try:
                    jq = rejection_queue.qsize()
                except Exception:
                    jq = None
                try:
                    cam_alive = bool(camera_proc_ref is not None and camera_proc_ref.is_alive())
                except Exception:
                    cam_alive = None
                try:
                    workers_alive = sum(1 for wp in (worker_procs_ref or []) if wp is not None and wp.is_alive())
                except Exception:
                    workers_alive = None
                try:
                    run_on = bool(run_event.is_set())
                except Exception:
                    run_on = None
                print(f"\n[主进程]: 子进程重启：开始（{ts}） run_event={run_on} qsize(task/results/rej)={tq}/{rq}/{jq} alive(cam/workers)={cam_alive}/{workers_alive}")
            except Exception:
                print("\n[主进程]: 子进程重启：开始重启相机与计算进程（执行冷启动式清理）...")
            try:
                run_event.clear()  # 暂停检测，避免新任务进队
            except Exception:
                pass
            try:
                cam_stop_event.set()  # 先通知相机停产，避免继续灌队列
            except Exception:
                pass
            cleared = {}
            def _drain_queue(q, name):
                c = 0
                try:
                    while True:
                        q.get_nowait()
                        c += 1
                except Exception:
                    pass
                cleared[name] = c
            _drain_queue(task_queue, 'task')
            _drain_queue(results_queue, 'results')
            _drain_queue(rejection_queue, 'rejection')
            # 清理共享状态以模拟首启
            try:
                for k in list(shared_camera_states.keys()):
                    shared_camera_states.pop(k, None)
            except Exception:
                pass
            try:
                for k in list(data_sessions.keys()):
                    data_sessions.pop(k, None)
            except Exception:
                pass
            try:
                print(f"[主进程]: 子进程重启：已清空队列与缓存 {cleared}")
            except Exception:
                pass
            # 1) 通知状态机执行软重置（避免跨代粘连）
            try:
                setattr(shared_settings, 'request_state_machine_reset', True)
            except Exception:
                pass
            # 2) 先关闭相机（已发停产信号，再保证 join/terminate）
            # 等待相机池进程退出
            try:
                if camera_proc_ref is not None:
                    camera_proc_ref.join(timeout=10.0)
                    if camera_proc_ref.is_alive():
                        camera_proc_ref.terminate()
            except Exception:
                pass
            # 3) 等待计算队列处理完成（已暂停 run_event，如仍有残留则等待耗尽）
            def _wait_queue_drained(q, timeout_s: float = 30.0):
                start = time.monotonic()
                consecutive_empty = 0
                while time.monotonic() - start < timeout_s:
                    try:
                        # 优先用 qsize；不可用则退化到 empty()
                        try:
                            sz = q.qsize()
                            is_empty = (sz == 0)
                        except Exception:
                            is_empty = q.empty()
                    except Exception:
                        is_empty = True
                    if is_empty:
                        consecutive_empty += 1
                        if consecutive_empty >= 5:
                            return True
                    else:
                        consecutive_empty = 0
                    time.sleep(0.2)
                return False
            drained = _wait_queue_drained(task_queue, timeout_s=60.0)
            if not drained:
                print("[主进程]: 子进程重启：在超时时间内队列未完全清空，继续关停计算进程。")
            # 4) 关闭计算进程
            try:
                worker_stop_event.set()
            except Exception:
                pass
            for wp in list(worker_procs_ref or []):
                try:
                    wp.join(timeout=5.0)
                    if wp.is_alive():
                        wp.terminate()
                except Exception:
                    pass
            # 等待硬件资源释放
            time.sleep(max(0.0, float(grace_seconds)))
            # 新一代事件
            cam_stop_event = multiprocessing.Event()
            worker_stop_event = multiprocessing.Event()
            # 重新启动子进程
            _start_children()
            # 等待相机再次就绪后，才恢复检测
            try:
                ready = cameras_ready_event.wait(timeout=60)
            except Exception:
                ready = False
            if ready:
                try:
                    run_event.set()
                except Exception:
                    pass
                try:
                    ts2 = time.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    ts2 = ''
                print(f"[主进程]: 子进程重启：相机就绪，已恢复检测（{ts2}）。\n")
            else:
                print("[主进程]: 子进程重启：等待相机就绪超时（暂时暂停，待就绪自动恢复）。\n")
                # 若超过超时后相机才就绪，自动恢复 run_event，避免长时间无处理又无日志。
                def _resume_when_ready():
                    try:
                        cameras_ready_event.wait()
                        if stop_event.is_set():
                            return
                        run_event.set()
                        print("[主进程]: 相机延迟就绪，已自动恢复检测。")
                    except Exception:
                        pass
                threading.Thread(target=_resume_when_ready, daemon=True).start()
        except Exception as e:
            print(f"[主进程]: 子进程重启失败: {e}")

    # 启动首代子进程
    _start_children()

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
    'http_client': http_client,
        'data_sessions': data_sessions
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

    # --- Wait for cameras (首启动) ---
    print("[主进程]: 等待相机初始化...")
    camera_ready = cameras_ready_event.wait(timeout=60)
    if camera_ready:
        print("✅ [主进程]: 相机就绪")
    else:
        print("⚠️ [主进程]: 等待相机初始化超时")

    # ============= 内存监控线程：超阈值(默认14GB)时暂停、清缓存并重启子进程 =============
    def _get_memory_usage() -> tuple[float|None, float|None]:
        """返回 (used_gb, total_gb)。优先 psutil，回退 Windows GlobalMemoryStatusEx。

        说明：这里是“系统内存”占用，仅作为最后的回退/兼容旧配置。
        主监控以进程树(main.exe及其子进程)占用为准。
        """
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            used = float(vm.total - vm.available) / float(1024 ** 3)
            total = float(vm.total) / float(1024 ** 3)
            return used, total
        except Exception:
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total = float(stat.ullTotalPhys) / float(1024 ** 3)
                    used = float(stat.ullTotalPhys - stat.ullAvailPhys) / float(1024 ** 3)
                    return used, total
            except Exception:
                return (None, None)
        return (None, None)

    def _get_process_tree_memory_gb() -> float | None:
        """返回 main.exe(当前进程) + 所有子进程的RSS内存占用(GB)。

        - 优先使用 psutil 统计进程树，准确覆盖 multiprocessing 子进程；
        - 若 psutil 不可用，则回退为仅统计当前进程（Windows API GetProcessMemoryInfo）。
        """
        # 1) 优先 psutil：统计当前进程 + 所有子进程（递归）
        try:
            import psutil  # type: ignore

            proc = psutil.Process(os.getpid())
            total_rss = 0
            try:
                total_rss += int(proc.memory_info().rss)
            except Exception:
                pass
            try:
                for ch in proc.children(recursive=True):
                    try:
                        total_rss += int(ch.memory_info().rss)
                    except Exception:
                        pass
            except Exception:
                pass
            return float(total_rss) / float(1024 ** 3)
        except Exception:
            pass

        # 2) 回退：Windows API 获取当前进程工作集（不含子进程）
        try:
            if os.name != 'nt':
                return None
            try:
                psapi = ctypes.windll.psapi
                kernel32 = ctypes.windll.kernel32
            except Exception:
                return None

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            h_proc = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(h_proc, ctypes.byref(counters), counters.cb)
            if not ok:
                return None
            return float(counters.WorkingSetSize) / float(1024 ** 3)
        except Exception:
            return None

    # 进程树内存阈值：达到该值即重启子进程（默认 9GB，满足“main.exe 总内存占用达到9G就重启”）
    try:
        mem_restart_threshold_gb = float(system_params.get('memory_restart_threshold_gb', 9.0) or 9.0)
    except Exception:
        mem_restart_threshold_gb = 9.0

    # 兼容旧配置：系统内存百分比阈值（仅作为无法获取进程占用时的最后回退）
    try:
        mem_threshold_pct = float(system_params.get('memory_pause_threshold_percent', 85.0) or 85.0)
    except Exception:
        mem_threshold_pct = 85.0
    try:
        mem_check_interval_s = float(system_params.get('memory_check_interval_s', 5.0) or 5.0)
    except Exception:
        mem_check_interval_s = 5.0

    mem_recovering_flag = threading.Event()

    try:
        print(
            f"[主进程]: 内存监控已启用 main_process_tree_threshold={mem_restart_threshold_gb:.2f}GB "
            f"(fallback_system_threshold={mem_threshold_pct:.1f}%) interval={mem_check_interval_s:.1f}s"
        )
    except Exception:
        pass

    def _clear_caches_and_temp():
        cleared = {}
        def _drain_queue(q, name):
            c = 0
            try:
                while True:
                    q.get_nowait()
                    c += 1
            except Exception:
                pass
            cleared[name] = c
        _drain_queue(task_queue, 'task')
        _drain_queue(results_queue, 'results')
        _drain_queue(rejection_queue, 'rejection')
        try:
            for k in list(shared_camera_states.keys()):
                shared_camera_states.pop(k, None)
        except Exception:
            pass
        try:
            for k in list(data_sessions.keys()):
                data_sessions.pop(k, None)
        except Exception:
            pass
        # 垃圾回收
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass
        try:
            print(f"[主进程]: 内存超阈值清理完成，队列清空计数={cleared}")
        except Exception:
            pass

    def _memory_watch_loop():
        while not stop_event.is_set():
            time.sleep(max(1.0, mem_check_interval_s))

            # 1) 主判定：进程树内存（main.exe + 子进程）
            proc_gb = _get_process_tree_memory_gb()
            trigger = False
            trigger_reason = None
            sys_used_gb, sys_total_gb = (None, None)
            sys_pct = None

            if proc_gb is not None and proc_gb >= float(mem_restart_threshold_gb):
                trigger = True
                trigger_reason = f"process_tree_memory_gb={proc_gb:.2f} >= {float(mem_restart_threshold_gb):.2f}"
            else:
                # 2) 回退：无法获得进程树内存时，兼容旧的系统内存百分比阈值
                sys_used_gb, sys_total_gb = _get_memory_usage()
                if sys_used_gb is not None and sys_total_gb is not None and sys_total_gb > 0:
                    sys_pct = (sys_used_gb / sys_total_gb) * 100.0
                    if sys_pct >= float(mem_threshold_pct):
                        trigger = True
                        trigger_reason = f"system_memory_pct={sys_pct:.1f}% >= {float(mem_threshold_pct):.1f}%"

            if trigger and not mem_recovering_flag.is_set():
                mem_recovering_flag.set()
                try:
                    try:
                        ts = time.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        ts = ''
                    try:
                        run_on = bool(run_event.is_set())
                    except Exception:
                        run_on = None
                    try:
                        tq = task_queue.qsize()
                    except Exception:
                        tq = None
                    try:
                        rq = results_queue.qsize()
                    except Exception:
                        rq = None
                    try:
                        jq = rejection_queue.qsize()
                    except Exception:
                        jq = None
                    # 兼顾打印：优先打印进程树内存，若回退则打印系统内存
                    extra = ""
                    if proc_gb is not None:
                        extra += f" process_tree={proc_gb:.2f}GB"
                    if sys_used_gb is not None and sys_total_gb is not None and sys_pct is not None:
                        extra += f" system_used={sys_used_gb:.2f}/{sys_total_gb:.2f}GB ({sys_pct:.1f}%)"
                    print(
                        f"\n[主进程]: [内存看门狗] 触发 @ {ts} reason={trigger_reason}{extra} "
                        f"run_event={run_on} qsize(task/results/rej)={tq}/{rq}/{jq}"
                    )
                except Exception:
                    pass
                try:
                    run_event.clear()
                except Exception:
                    pass
                _clear_caches_and_temp()
                try:
                    _restart_children(grace_seconds=5.0)
                except Exception as e:
                    try:
                        print(f"[主进程]: [内存看门狗] 重启子进程失败: {e}")
                    except Exception:
                        pass
                try:
                    ts3 = time.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    ts3 = ''
                try:
                    run_on2 = bool(run_event.is_set())
                except Exception:
                    run_on2 = None
                try:
                    print(f"[主进程]: [内存看门狗] 处理结束 @ {ts3} run_event={run_on2}")
                except Exception:
                    pass
                mem_recovering_flag.clear()

    threading.Thread(target=_memory_watch_loop, daemon=True).start()
    
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
        # 确保当前代子进程收到退出信号（先相机后计算）
        try:
            if cam_stop_event:
                cam_stop_event.set()
        except Exception:
            pass
        # 等相机进程退出后再通知计算退出（尽力而为）
        try:
            if camera_proc_ref is not None:
                camera_proc_ref.join(timeout=5.0)
        except Exception:
            pass
        try:
            if worker_stop_event:
                worker_stop_event.set()
        except Exception:
            pass
        
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

        # 最后关闭 Manager（在子进程尽量退出之后），避免退出时卡在 managers IPC。
        try:
            if 'manager' in locals() and manager is not None:
                manager.shutdown()
        except Exception:
            pass
            
        print("[主进程]: 所有资源已释放，程序退出。")
        # 关闭日志文件（放在所有打印之后）
        try:
            if log_file:
                log_file.flush()
                log_file.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()