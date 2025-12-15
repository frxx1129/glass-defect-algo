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
from server_comms import periodic_stats_pusher, broadcast_yield_and_rejections

class CurrentUser(BaseModel):
    sessionId: str  # 每次登录都生成一个，用于有状态登录，判断token是否过期
    expiredTime: int  # 过期时间
    userId: int
    roleCode: str

class RejectParam(BaseModel):
    currentUser: CurrentUser
    rejectionMark: int

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

        # 手动暂停标记：只有 /control/stop 会置位；用于避免自动恢复误开启
        app.state.manual_pause = threading.Event()
        # 期望运行标记：只有当系统曾被“请求启动”（/control/start 或 初始 enable=1）时才会置位
        app.state.desired_run = threading.Event()
        if app.state.run_event.is_set():
            app.state.desired_run.set()

        # 自动恢复：run_event 连续 OFF 超过阈值则自动开启（除非 manual_pause 或 desired_run 未置位）
        def auto_resume_guard():
            try:
                sys_params = getattr(shared_settings, 'system_params', None) or {}
            except Exception:
                sys_params = {}
            try:
                auto_resume_s = float(sys_params.get('auto_resume_run_event_off_seconds', 180.0) or 180.0) if isinstance(sys_params, dict) else 180.0
            except Exception:
                auto_resume_s = 180.0
            auto_resume_s = max(10.0, float(auto_resume_s))
            off_since = None
            while not thread_stop_event.is_set():
                try:
                    desired = bool(app.state.desired_run.is_set())
                    manual = bool(app.state.manual_pause.is_set())
                    running = bool(app.state.run_event.is_set())
                except Exception:
                    desired = False; manual = False; running = True

                if desired and (not manual) and (not running):
                    if off_since is None:
                        off_since = time.monotonic()
                    else:
                        if (time.monotonic() - off_since) >= auto_resume_s:
                            try:
                                app.state.run_event.set()
                            except Exception:
                                pass
                            try:
                                run_event_mp.set()
                            except Exception:
                                pass
                            try:
                                print(f"[主进程]: run_event 已 OFF 超过 {auto_resume_s:.0f}s（非手动暂停），已自动恢复检测")
                            except Exception:
                                pass
                            off_since = None
                else:
                    off_since = None

                time.sleep(0.5)

        def event_proxy():
            while not thread_stop_event.is_set():
                if app.state.run_event.is_set():
                    run_event_mp.set()
                    # 任意来源开启检测（含服务器初始 enable），都视为“期望运行”
                    try:
                        app.state.desired_run.set()
                    except Exception:
                        pass
                else:
                    run_event_mp.clear()
                time.sleep(0.1)

        # Start all background threads
        threading.Thread(target=event_proxy, daemon=True).start()
        threading.Thread(target=auto_resume_guard, daemon=True).start()
        threading.Thread(target=rejection_handler_thread, args=(queues['rejection'], rejection_controller, thread_stop_event, shared_settings), daemon=True).start()
        
        # --- 修改: 将 alarm_light_controller 作为新参数传递给状态机线程 ---
        app.state.state_machine_thread = threading.Thread(target=results_and_state_machine_thread, args=(
            num_cameras, queues['results'], connection_manager, thread_stop_event, loop, shared_settings,
            counters, (queues['rejection'],), flags, machine_state_shared, stats_lock, metadata, app.state.run_event,
            http_client,
            alarm_light_controller # <-- 将代理对象作为新参数传递
        ), daemon=True)
        app.state.state_machine_thread.start()

        # 启动周期统计推送线程（可配置开关）
        try:
            sys_params = getattr(shared_settings, 'system_params', None) or {}
        except Exception:
            sys_params = {}
        enable_stats = True
        try:
            enable_stats = bool(getattr(shared_settings, 'enable_periodic_stats'))
        except Exception:
            # fallback to config dict if stored
            enable_stats = bool(sys_params.get('enable_periodic_stats', True)) if isinstance(sys_params, dict) else True
        interval_s = 30.0
        try:
            interval_s = float(getattr(shared_settings, 'stats_push_interval_s'))
        except Exception:
            if isinstance(sys_params, dict):
                try:
                    interval_s = float(sys_params.get('stats_push_interval_s', 30) or 30)
                except Exception:
                    interval_s = 30.0
        if enable_stats and interval_s > 0:
            app.state.stats_thread_stop = threading.Event()
            app.state.stats_thread = threading.Thread(
                target=periodic_stats_pusher,
                args=(shared_settings, metadata[0], counters[1], counters[0], http_client, app.state.stats_thread_stop, interval_s),
                daemon=True
            )
            app.state.stats_thread.start()
        else:
            app.state.stats_thread = None
            app.state.stats_thread_stop = None
        
        yield
        
        print("[主进程]: FastAPI 应用关闭...")
        thread_stop_event.set()
        app.state.state_machine_thread.join(timeout=2)
        if getattr(app.state, 'stats_thread_stop', None):
            app.state.stats_thread_stop.set()
        if getattr(app.state, 'stats_thread', None):
            app.state.stats_thread.join(timeout=2)

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
    def start_detection(request: Request):
        app.state.manual_pause.clear()
        app.state.desired_run.set()
        app.state.run_event.set()
        try:
            host = getattr(getattr(request, 'client', None), 'host', '<unknown>')
        except Exception:
            host = '<unknown>'
        try:
            print(f"[API]: 检测已启动 (/control/start) from {host} @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            pass
        return {"code": 200, "message": "检测已启动", "data": {"status": "running"}}

    @app.post("/control/stop")
    def stop_detection(request: Request):
        app.state.manual_pause.set()
        app.state.desired_run.clear()
        app.state.run_event.clear()
        try:
            host = getattr(getattr(request, 'client', None), 'host', '<unknown>')
        except Exception:
            host = '<unknown>'
        try:
            print(f"[API]: 检测已暂停 (/control/stop) from {host} @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            pass
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
    async def reject(rejectionParam: RejectParam):
        """参数化剔废：
        - 产线 Line1/Line3: 接收 4 个参数(0..3)
        - 产线 Line2: 接收 5 个参数(0..4)
        - 每个参数 -> 通道号的映射从 config.rejection_mark_to_channel 中读取（每条产线独立）
        - 仅在手动模式(flags[1]==2)下允许。
        - 触发硬件剔废并计数、广播；不涉及上传逻辑（由状态机负责）。
        """
        try:
            # 模式校验：仅手动模式有效
            if flags[1].value != 2:
                return {"code": 403, "message": "当前为自动模式，无法进行手动剔废", "data": None}

            # 提取用户、参数
            current_user = rejectionParam.currentUser
            mark = int(rejectionParam.rejectionMark)
            metadata[2].value = current_user.userId  # shared_user_id_manual

            # 校验参数范围依赖于产线名
            line_name = str(getattr(shared_settings, 'lineName', 'UNKNOWN')).strip()
            if line_name in ("Line1", "Line3"):
                valid_range = range(0, 4)  # 0..3
            elif line_name == "Line2":
                valid_range = range(0, 5)  # 0..4
            else:
                # 未知产线：默认采用 0..3（与历史保持一致）
                valid_range = range(0, 4)

            if mark not in valid_range:
                return {"code": 400, "message": f"rejectionMark 超出范围，期望 {valid_range.start}..{valid_range.stop-1}", "data": None}

            # 查询通道映射：优先 per-line 配置，否则退化到顺序映射 mark->(mark+1)
            line_map = {}
            try:
                cfg_map = getattr(shared_settings, 'rejection_mark_to_channel', {}) or {}
                line_map = cfg_map.get(line_name, {}) or {}
            except Exception:
                line_map = {}
            channel = int(line_map.get(str(mark), mark + 1))
            if channel < 1:
                channel = 1

            # 触发剔废：直接放入队列，由剔除线程按统一延迟与脉冲宽度执行
            # 这里 route 直接传递整数通道号，rejection_controller._as_channel 将按 int 处理
            fire_time = time.time() + getattr(shared_settings, 'REJECTION_DELAY_S', 1.5)
            queues['rejection'].put((fire_time, -1, channel))

            # 更新本地剔废计数与广播（保持与状态机手动一致：计数+推送看板，不做上传）
            try:
                # 计数 +1 并持久化
                from yield_manager import save_stats, load_stats
                last_date, total_yield, total_rej = load_stats()
                total_rej = (total_rej or 0) + 1
                counters[0].value = total_rej
                save_stats(last_date, total_yield, total_rej)
            except Exception as _:
                pass
            try:
                broadcast_yield_and_rejections(shared_settings, metadata[0], counters[1], counters[0], http_client)
            except Exception as _:
                pass

            print(f"[API /control/reject] user={current_user.userId}, role={current_user.roleCode}, line={line_name}, mark={mark}, channel={channel}")
            return {"code": 200, "message": "Rejected", "data": {"mark": mark, "channel": channel}}
        except Exception as e:
            print(f"[API /control/reject] 处理剔废请求错误: {e}")
            return {"code": 500, "message": f"处理请求时发生错误: {e}", "data": None}

    # 旧的 left/mid/right/all 接口已废弃，统一使用 /control/reject(rejectionMark)
    
    @app.get("/control/reject/config")
    def get_reject_config():
        """返回当前产线的剔废参数范围与通道映射，便于前端渲染与校验。"""
        try:
            line_name = str(getattr(shared_settings, 'lineName', 'UNKNOWN')).strip()
            if line_name in ("Line1", "Line3"):
                valid_range = (0, 3)
            elif line_name == "Line2":
                valid_range = (0, 4)
            else:
                valid_range = (0, 3)
            cfg_map = getattr(shared_settings, 'rejection_mark_to_channel', {}) or {}
            line_map = cfg_map.get(line_name, {}) or {}
            return {
                "code": 200,
                "message": "Success",
                "data": {
                    "lineName": line_name,
                    "allowedMarkRange": {"start": valid_range[0], "end": valid_range[1]},
                    "mapping": line_map
                }
            }
        except Exception as e:
            return {"code": 500, "message": f"获取配置失败: {e}", "data": None}
    
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

    @app.post("/control/algorithmMode")
    async def set_algorithm_mode(algorithmMode: int = Body(...)):
        new_mode = algorithmMode
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