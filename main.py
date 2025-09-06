# --- START OF FILE main.py ---
import multiprocessing
import threading
import json
import sys
import os
import uvicorn
import cv2

# Import process and thread functions from their respective modules
from http_client import NonBlockingHttpClient
from camera_process import camera_pool_process
from processing_worker import calculation_worker
from api_server import create_app
from rejection_controller import RejectionController

def main():
    """Main function to initialize shared resources, start child processes, and run the API server."""
    multiprocessing.freeze_support()
    cv2.setUseOptimized(True)
    print("[主进程]: 应用程序启动...")

    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        sys.exit(f"错误: 无法加载 config.json: {e}")

    # --- 启动前校验：ROI 不得越界于相机分辨率 ---
    try:
        # 默认使用修正后的 ROI 模板文件（CORRECTED 版本）
        roi_file = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
        with open(roi_file, 'r', encoding='utf-8') as f:
            averaged_data = json.load(f)
        if not averaged_data:
            sys.exit("错误: ROI 模板文件为空，无法启动。")

        # 以 source_image_count 最大的组作为使用的ROI集合（与计算进程一致）
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
    # 直接使用 expected_cameras 作为“期望的逻辑相机数量”（在无物理相机时用于模拟）
    try:
        cfg_cam_count = int(cs.get('expected_cameras', 1) or 1)
    except Exception:
        cfg_cam_count = 1
    if cfg_cam_count <= 0:
        cfg_cam_count = 1

    # 再尝试通过 SDK 实际枚举相机数量（不修改任何网络/IP 设置）
    actual_cam_count = 0
    try:
        from MVGigE import MVInitLib, MVTerminateLib, MVEnumerateAllDevices, MVUpdateCameraList, MVGetNumOfCameras, MVST_SUCCESS
        MVInitLib()
        try:
            res, n = MVEnumerateAllDevices()
            if res == MVST_SUCCESS and n is not None and int(n) >= 0:
                actual_cam_count = int(n)
            else:
                MVUpdateCameraList()
                _, n2 = MVGetNumOfCameras()
                if n2 is not None:
                    actual_cam_count = int(n2)
        finally:
            MVTerminateLib()
    except Exception:
        actual_cam_count = 0

    # 取“实际枚举值”为主，失败则退回配置估计
    NUM_CAMERAS = actual_cam_count if actual_cam_count > 0 else cfg_cam_count

    # --- 计算处理进程数量 NUM_WORKERS ---
    # 新参数: system_params.process_workers (显式进程数)
    # 旧参数: system_params.max_roi_workers (兼容, 若新参数缺省时仍可作为进程数指定)
    sys_params = config.get('system_params', {})
    logical_cores = os.cpu_count() or 4
    try:
        configured_proc_workers = int(sys_params.get('process_workers', 0) or 0)
    except Exception:
        configured_proc_workers = 0
    if configured_proc_workers <= 0:
        # 兼容旧字段作为进程数（若设置且 >0）
        try:
            legacy = int(sys_params.get('max_roi_workers', 0) or 0)
        except Exception:
            legacy = 0
        configured_proc_workers = legacy
    if configured_proc_workers > 0:
        NUM_WORKERS = max(1, configured_proc_workers)
    else:
        usable_cores = max(1, logical_cores - 2)
        baseline = min(NUM_CAMERAS, usable_cores)  # 每台相机 1 进程为基准
        NUM_WORKERS = min(baseline, NUM_CAMERAS * 2)

    print(f"[主进程]: 模式={'实机' if actual_cam_count>0 else '测试'} | Cameras={NUM_CAMERAS} | ProcWorkers={NUM_WORKERS} (CPU={logical_cores}, expected={cfg_cam_count}, actual={actual_cam_count})")

    manager = multiprocessing.Manager()

    # Shared settings object, populated directly from config.json
    shared_settings = manager.Namespace()
    rejection_params = config.get('rejection_params', {})
    system_params = config.get('system_params', {})
    server_config = config.get('server_config', {})
    shared_settings.max_defect_size_mm = rejection_params.get('max_defect_size_mm', 20.0)
    shared_settings.REJECTION_PULSE_MS = rejection_params.get('REJECTION_PULSE_MS', 100)
    shared_settings.REJECTION_DELAY_S = rejection_params.get('REJECTION_DELAY_S', 1.5)
    shared_settings.storage_path = config.get('storage_path', 'inspection_results')
    shared_settings.pixels_per_mm = system_params.get('pixels_per_mm', 2.4)
    shared_settings.lineName = config.get('lineName', 'UNKNOWN_LINE')
    shared_settings.server = server_config.get('server', '127.0.0.1')
    shared_settings.upload_url = server_config.get('upload_url', f'http://{shared_settings.server}:5000/upload')
    shared_settings.stats_push_url = server_config.get('stats_push_url', f'http://{shared_settings.server}:8085/fastapi/glass/updateYieldAndRejections')
    shared_settings.heartbeat_url = server_config.get('heartbeat_url', f'http://{shared_settings.server}:8085/fastapi/system/heartbeat')

    # Queues for data flow between processes
    queue_size_factor = int(system_params.get('queue_size_factor', 2) or 2)
    task_queue = manager.Queue(maxsize=NUM_WORKERS * NUM_CAMERAS * queue_size_factor)
    results_queue = manager.Queue(maxsize=NUM_WORKERS * NUM_CAMERAS * queue_size_factor)
    rejection_queue = manager.Queue()
    
    # Shared values, flags, and metadata
    shared_camera_states = manager.dict()
    shared_yield_counter = manager.Value('i', 0)
    shared_rejection_counter = manager.Value('i', 0)
    manual_reject_flag = manager.Value('b', False)
    machine_state_shared = manager.Value('i', 0)
    shared_rejection_mode = manager.Value('i', 2) # Default to manual
    can_late_reject = manager.Value('b', False)
    shared_collection_id = manager.Value('c', b'N/A')
    shared_user_id_auto = manager.Value('i', config.get('user_id_auto', 7))
    shared_user_id_manual = manager.Value('i', config.get('user_id_manual', 9999))

    # Process control events
    stop_event = multiprocessing.Event()
    run_event = multiprocessing.Event()

    # Threading lock for stats file access
    stats_lock = threading.Lock()

    # Hardware controller (instantiated in the main process)
    rejection_controller = RejectionController()
    
    # --- (新增) 创建非阻塞HTTP客户端实例 ---
    http_workers = int(system_params.get('http_client_max_workers', 4) or 4)
    http_client = NonBlockingHttpClient(max_workers=http_workers)

    # --- Create and Start Child Processes ---
    processes = []
    
    # 1. Camera Pool Process (1 process)
    pool_proc = multiprocessing.Process(
        target=camera_pool_process,
        args=(task_queue, stop_event, run_event, config, shared_camera_states)
    )
    processes.append(pool_proc)
    
    # 2. Calculation Worker Processes (N processes)
    for i in range(NUM_WORKERS):
        worker_proc = multiprocessing.Process(
            target=calculation_worker, 
            args=(i, task_queue, results_queue, stop_event, run_event, config, shared_settings)
        )
        processes.append(worker_proc)
        
    for p in processes:
        p.start()

    # --- Prepare Shared Objects for FastAPI and Background Threads ---
    # This dictionary bundles all shared objects to cleanly pass them to the API server module.
    shared_objects = {
        'stop_event': stop_event, 'run_event': run_event,
        'camera_states': shared_camera_states,
    'settings': shared_settings,
        'counters': (shared_yield_counter, shared_rejection_counter),
        'queues': {'task': task_queue, 'results': results_queue, 'rejection': rejection_queue},
        'flags': (manual_reject_flag, shared_rejection_mode, can_late_reject),
        'machine_state': machine_state_shared,
        'stats_lock': stats_lock,
        'metadata': (shared_collection_id, shared_user_id_auto, shared_user_id_manual),
        'rejection_controller': rejection_controller,
        'http_client': http_client
    }

    app = create_app(num_cameras=NUM_CAMERAS, shared_objects=shared_objects)

    # 将常用服务/HTTP/统计配置注入 shared_settings，供后台线程/模块读取
    shared_settings.stats_push_interval_s = int(system_params.get('stats_push_interval_s', 601) or 601)
    shared_settings.http_get_timeout_s = float(system_params.get('http_get_timeout_s', 5) or 5)
    shared_settings.http_post_timeout_s = float(system_params.get('http_post_timeout_s', 5) or 5)
    shared_settings.http_upload_timeout_s = float(system_params.get('http_upload_timeout_s', 15) or 15)
    # CORS 与服务监听
    cors_list = config.get('server_config', {}).get('cors_origins', ["*"])
    try:
        shared_settings.cors_origins = list(cors_list)
    except Exception:
        shared_settings.cors_origins = ["*"]

    # --- Run Server and Handle Graceful Shutdown ---
    try:
        listen_host = config.get('server_config', {}).get('listen_host', '0.0.0.0')
        listen_port = int(config.get('server_config', {}).get('listen_port', 12450) or 12450)
        uvicorn.run(app, host=listen_host, port=listen_port)
    finally:
        print("\n[主进程]: 正在终止所有子进程...")
        stop_event.set()
        run_event.set() # Ensure processes don't get stuck waiting on the event
        for p in processes:
            p.join(timeout=5) 
            if p.is_alive():
                print(f"警告: 进程 {p.pid} 未能在5秒内正常退出，将被强制终止。")
                p.terminate()
        
        rejection_controller.close()
        print("[主进程]: 所有资源已释放。")

if __name__ == '__main__':
    main()