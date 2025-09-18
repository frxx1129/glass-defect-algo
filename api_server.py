# --- START OF FILE api_server.py ---
import asyncio
import threading
import time
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils import ConnectionManager
from rejection_control import rejection_handler_thread
from state_machine import results_and_state_machine_thread

class CurrentUser(BaseModel):
    sessionId: str
    expiredTime: int
    userId: int
    roleCode: str

def create_app(num_cameras, shared_objects):
    """Creates and configures the FastAPI application instance."""
    
    # Unpack shared objects
    stop_event_mp, run_event_mp = shared_objects['stop_event'], shared_objects['run_event']
    shared_camera_states = shared_objects['camera_states']
    shared_settings = shared_objects['settings']
    counters = shared_objects['counters']  # (rejection_counter, yield_counter)
    queues = shared_objects['queues']
    flags = shared_objects['flags']
    machine_state_shared = shared_objects['machine_state']
    stats_lock = shared_objects['stats_lock']
    metadata = shared_objects['metadata']
    rejection_controller = shared_objects['rejection_controller']
    alarm_light_controller = shared_objects['alarm_light_controller'] # <-- 解包代理对象
    http_client = shared_objects['http_client']

    connection_manager = ConnectionManager(num_cameras)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("[主进程]: FastAPI 应用启动...")
        loop = asyncio.get_running_loop()
        thread_stop_event = threading.Event()
        
        # This proxy event bridges the FastAPI app state with the multiprocessing event
        app.state.run_event = threading.Event()
        if run_event_mp.is_set(): app.state.run_event.set()

        def event_proxy():
            while not thread_stop_event.is_set():
                if app.state.run_event.is_set(): run_event_mp.set()
                else: run_event_mp.clear()
                time.sleep(0.1)

        # Start all background threads
        threading.Thread(target=event_proxy, daemon=True).start()
        threading.Thread(target=rejection_handler_thread, args=(queues['rejection'], rejection_controller, thread_stop_event, shared_settings), daemon=True).start()
        
        # --- 修改: 将 alarm_light_controller 作为新参数传递给状态机线程 ---
        app.state.state_machine_thread = threading.Thread(target=results_and_state_machine_thread, args=(
            num_cameras, queues['results'], connection_manager, thread_stop_event, loop, shared_settings,
            counters, (queues['rejection'],), flags, machine_state_shared, stats_lock, metadata, app.state.run_event,
            http_client,
            alarm_light_controller # <-- 将代理对象作为新参数传递
        ), daemon=True)
        app.state.state_machine_thread.start()

    # 删除产量推送线程：仅保留剔废统计，不再周期推送产量
        
        yield
        
        print("[主进程]: FastAPI 应用关闭...")
        thread_stop_event.set()
        app.state.state_machine_thread.join(timeout=2)

    app = FastAPI(title="Glass Detection System", lifespan=lifespan)
    try:
        cors = list(shared_settings.cors_origins)
    except Exception:
        cors = ["*"]
    app.add_middleware(CORSMiddleware, allow_origins=cors, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    # --- API Endpoints ---
    @app.get("/status/system")
    def get_system_status():
        return {
            "code": 200, "message": "Success",
            "data": {
                "lineName": shared_settings.lineName,
                "detectionStatus": "running" if app.state.run_event.is_set() else "stopped",
                "rejectionMode": "auto" if flags[1].value == 1 else "manual",
                "rejectionThresholdMm": shared_settings.max_defect_size_mm
            }
        }

    @app.post("/actions/report_status")
    def trigger_status_report():
        report_system_status_to_server()
        return {"code": 200, "message": "状态上报任务已触发"}
        
    @app.get("/status/cameras")
    def get_camera_states():
        return {"code": 200, "message": "获取相机状态成功", "data": dict(shared_camera_states)}
        
    @app.post("/control/start")
    def start_detection(): 
        app.state.run_event.set()
        return {"code": 200, "message": "检测已启动", "data": {"status": "running"}}

    @app.post("/control/stop")
    def stop_detection(): 
        app.state.run_event.clear()
        return {"code": 200, "message": "检测已暂停", "data": {"status": "stopped"}}

    @app.get("/rejections")
    def get_rejections(): 
        return {"code": 200, "message": "获取剔废数量成功", "data": {"rejections": counters[0].value, "yield": counters[1].value}}
            
    @app.websocket("/ws/stream/{cam_index}")
    async def websocket_endpoint(websocket: WebSocket, cam_index: int):
        if not (1 <= cam_index <= num_cameras): await websocket.close(); return
        await connection_manager.connect(websocket, cam_index - 1)
        try:
            while True: await websocket.receive_text()
        except WebSocketDisconnect: connection_manager.disconnect(websocket, cam_index - 1)

    @app.post("/control/rejection_mode")
    async def set_rejection_mode(rejectionMode: int = Body(...)):
        if rejectionMode not in [1, 2]:
            return {"code": 400, "message": "无效模式，请提供 1 或 2"}
        flags[1].value = rejectionMode
        msg = f"剔废模式已切换为: {'自动' if rejectionMode == 1 else '手动'}"
        print(f"[API]: {msg}")
        return {"code": 200, "message": msg, "data": {"rejection_mode": rejectionMode}}

    @app.post("/control/thresholds")
    async def set_rejection_thresholds(rejectionThreshold: int = Body(...)):
        try:
            new_threshold = rejectionThreshold
            old_value = shared_settings.max_defect_size_mm
            shared_settings.max_defect_size_mm = new_threshold
            msg = f"剔废阈值已更新: {old_value} -> {new_threshold}"
            print(f"[API]: {msg}")
            return {"code": 200, "message": msg, "data": {"old_value": old_value, "new_value": new_threshold}}
        except (ValueError, AttributeError) as e:
            return {"code": 400, "message": f"无效的数值格式: {e}"}

    @app.post("/control/reject")
    async def manual_reject_trigger(current_user: CurrentUser):
        try:
            # 1. First, check if the system is in manual mode.
            if flags[1].value != 2: # flags[1] is shared_rejection_mode
                return {
                    "code": 403,
                    "message": "当前为自动模式，无法进行手动剔废",
                    "data": None
                }

            # 2. Proceed with the original logic only if in manual mode.
            metadata[2].value = current_user.userId # shared_user_id_manual
            is_pane_detected = (machine_state_shared.value == 1)
            is_late_rejection_possible = flags[2].value # can_late_reject

            if not is_pane_detected and not is_late_rejection_possible:
                return {
                    "code": 201,
                    "message": "当前无玻璃正在检测，且上一片玻璃已过检",
                    "data": None
                }

            if not flags[0].value: # manual_reject_flag
                flags[0].value = True
                return {
                    "code": 200,
                    "message": "手动剔废信号已发送",
                    "data": {"reject_triggered": True}
                }

            return {
                "code": 202,
                "message": "正在处理上一个剔废信号，请稍候",
                "data": {"reject_triggered": False}
            }

        except Exception as e:
            print(f"[API /control/reject] 处理手动剔废请求时出错: {e}")
            return {
                "code": 500,
                "message": f"处理请求时发生错误: {str(e)}",
                "data": None
            }
    
    def report_system_status_to_server():
        try:
            status_data = {
                "lineName": shared_settings.lineName,
                "detectionStatus": "running" if app.state.run_event.is_set() else "stopped",
                "rejectionMode": flags[1].value,
                "rejectionThresholdMm": shared_settings.max_defect_size_mm
            }
            print(f"[状态上报]: 正在提交心跳状态任务: {status_data}")
            http_client.post(shared_settings.heartbeat_url, json=status_data, timeout=shared_settings.http_post_timeout_s)
        except Exception as e:
            print(f"[状态上报]: 提交任务时发生本地错误: {e}")

    # ========== 算法模式切换 (1=浅色 2=深色) ==========
    @app.get("/algorithmMode")
    def get_algorithm_mode():
        mode = int(getattr(shared_settings, 'algorithm_mode', 1))
        return {"code": 200, "message": "获取成功", "data": {"mode": mode}}

    class AlgoModeBody(BaseModel):
        mode: int

    @app.post("/algorithmMode")
    async def set_algorithm_mode(body: AlgoModeBody):
        new_mode = int(body.mode)
        if new_mode not in (1, 2):
            return {"code": 400, "message": "mode 只能为 1(浅色) 或 2(深色)"}
        old_mode = int(getattr(shared_settings, 'algorithm_mode', 1))
        if new_mode == old_mode:
            return {"code": 200, "message": "模式未变化", "data": {"mode": new_mode}}
        setattr(shared_settings, 'algorithm_mode', new_mode)
        print(f"[API]: 算法模式切换 {old_mode} -> {new_mode}")
        # 通过所有 websocket 通道广播模式变化事件
        try:
            payload = {"event": "algorithmModeChanged", "mode": new_mode}
            # broadcast to all camera groups
            for cam_idx in range(num_cameras):
                try:
                    asyncio.create_task(connection_manager.broadcast_json(payload, cam_idx))
                except Exception:
                    pass
        except Exception as e:
            print(f"[API]: 广播算法模式变更失败: {e}")
        return {"code": 200, "message": "模式已更新", "data": {"old_mode": old_mode, "new_mode": new_mode}}

    return app