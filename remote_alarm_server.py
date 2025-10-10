"""
远程报警接收服务：在另一台局域网电脑上运行，用于接收主机推送的声光报警器状态并控制本机报警灯。

参数来源（优先级：命令行 > 环境变量 > 配置文件 > 默认）：
    - 命令行：
            --config <path>             指定 JSON 配置文件路径（默认：程序目录/remote_alarm_server.json）
            --serial-port <port>        例如 COM3
            --baud <int>                串口波特率，默认 9600
            --host <ip>                 监听地址，默认 0.0.0.0
            --port <int>                监听端口，默认 9000
    - 环境变量：
            ALARM_SERIAL_PORT, ALARM_BAUD, LISTEN_HOST, LISTEN_PORT
            REMOTE_ALARM_CONFIG         配置文件路径（可选，低于命令行优先级）
    - 配置文件（示例 remote_alarm_server.json）：
            {
                "serial_port": "COM3",
                "baud": 9600,
                "listen_host": "0.0.0.0",
                "listen_port": 9000
            }

接口：
    POST /alarm/update
    Body(JSON): {"light": "off|green|yellow|red", "buzzer": true/false, "duration_s": 0.0}
"""
import os
import sys
import json
import argparse
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from alarm_light_controller import AlarmLightController, _AlarmLightService


class AlarmUpdate(BaseModel):
    light: str = Field(..., description="off|green|yellow|red")
    buzzer: bool = Field(default=False)
    duration_s: float = Field(default=0.0)


def create_alarm_runtime(serial_port: Optional[str], baud: int):
    """创建本机报警服务线程与代理控制器。"""
    alarm_service = _AlarmLightService(port=serial_port, baud_rate=baud, command_queue=None)
    # _AlarmLightService 需要一个 Queue，由 AlarmLightController 持有；这里由控制器创建并传入
    from queue import Queue
    q = Queue()
    alarm_service.command_queue = q

    t = threading.Thread(target=alarm_service.run, daemon=True)
    t.start()

    controller = AlarmLightController(command_queue=q)
    # 启动时显示“已就绪”状态
    controller.set_startup_state()
    return alarm_service, t, controller

def _default_paths() -> Dict[str, Any]:
    base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return {
        "base_dir": base_dir,
        "default_config": os.path.join(base_dir, "remote_alarm_server.json"),
    }


def load_config(cli_args: argparse.Namespace) -> Dict[str, Any]:
    paths = _default_paths()
    # 默认值
    cfg: Dict[str, Any] = {
        "serial_port": None,
        "baud": 9600,
        "listen_host": "0.0.0.0",
        "listen_port": 9000,
    }
    # 配置文件（环境变量可覆盖默认路径）
    cfg_path = cli_args.config or os.environ.get("REMOTE_ALARM_CONFIG") or paths["default_config"]
    try:
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            if isinstance(file_cfg, dict):
                cfg.update({k: v for k, v in file_cfg.items() if k in cfg})
    except Exception:
        pass
    # 环境变量
    if os.environ.get("ALARM_SERIAL_PORT"):
        cfg["serial_port"] = os.environ.get("ALARM_SERIAL_PORT")
    if os.environ.get("ALARM_BAUD"):
        try: cfg["baud"] = int(os.environ.get("ALARM_BAUD"))
        except Exception: pass
    if os.environ.get("LISTEN_HOST"):
        cfg["listen_host"] = os.environ.get("LISTEN_HOST")
    if os.environ.get("LISTEN_PORT"):
        try: cfg["listen_port"] = int(os.environ.get("LISTEN_PORT"))
        except Exception: pass
    # 命令行参数
    if cli_args.serial_port is not None:
        cfg["serial_port"] = cli_args.serial_port
    if cli_args.baud is not None:
        cfg["baud"] = cli_args.baud
    if cli_args.host is not None:
        cfg["listen_host"] = cli_args.host
    if cli_args.port is not None:
        cfg["listen_port"] = cli_args.port
    return cfg


def build_app(config: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Remote Alarm Receiver")
    _alarm_service, _service_thread, _controller = create_alarm_runtime(config.get("serial_port"), int(config.get("baud", 9600)))

    @app.post("/alarm/update")
    def alarm_update(update: AlarmUpdate):
        # 统一处理：直接调用通用 set_state
        try:
            light = update.light.lower().strip()
            buz = bool(update.buzzer)
            dur = float(update.duration_s or 0.0)
            # 优先使用通用接口
            if hasattr(_controller, 'set_state'):
                _controller.set_state(light, buz, dur)
            else:
                # 兼容旧接口的简单映射
                if light == 'green' and not buz:
                    _controller.set_normal_state()
                elif light == 'yellow':
                    _controller.set_ng_detected_state(dur if dur > 0 else 0.5)
                elif light == 'red':
                    _controller.set_rejection_state(dur if dur > 0 else 2.0)
                elif light == 'off':
                    _controller.set_state('off', False, 0)
                else:
                    _controller.set_normal_state()
            return {"code": 200, "message": "ok"}
        except Exception as e:
            return {"code": 500, "message": f"error: {e}"}

    return app

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Remote Alarm Receiver Server")
    paths = _default_paths()
    p.add_argument("--config", type=str, default=None, help=f"配置文件路径(默认: {paths['default_config']})")
    p.add_argument("--serial-port", type=str, default=None, help="串口号，例如 COM3")
    p.add_argument("--baud", type=int, default=None, help="串口波特率，默认 9600")
    p.add_argument("--host", type=str, default=None, help="监听地址，默认 0.0.0.0")
    p.add_argument("--port", type=int, default=None, help="监听端口，默认 9000")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args)
    app = build_app(cfg)
    uvicorn.run(app, host=cfg.get("listen_host", "0.0.0.0"), port=int(cfg.get("listen_port", 9000)))


if __name__ == "__main__":
    main()
