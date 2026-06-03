# glass-defect-algo

玻璃缺陷检测算法与产线运行程序。项目面向多相机玻璃外观检测场景，包含相机采集、ROI 内 Hough/亮度缺陷检测、结果状态机、API 服务、剔废控制和远程声光报警联动。

## 交接范围

本仓库保留代码、依赖声明、主配置模板、产线配置模板、已有 ROI 模板和硬件 SDK 包装文件。产线机器上已经存在的现场资产，例如各机台实际使用的 ROI 模板、相机 MAC/IP 绑定、串口号、服务器地址和回放数据，以现场机器为准；部署或替换代码前，需要让 `config*.yaml` 中的路径和现场文件保持一致。

不纳入仓库的内容包括运行日志、检测输出、回放压缩包、Nuitka 打包目录、历史备份文件、profile 文件和一次性统计报表。这些文件通常体积大、随运行变化，适合保留在产线机器或单独归档。

## 运行环境

- Windows 产线主机
- Python 3.12
- 工业相机 SDK：`MVGigE.dll`、`MVGigE.py`、`GigECamera_Types.py`
- 串口剔废/声光报警硬件按现场 COM 口配置

依赖管理使用 `pyproject.toml` 和 `uv.lock`。没有 uv 时，也可以用 pip 安装 `pyproject.toml` 中列出的依赖。

```powershell
uv sync
```

或：

```powershell
python -m pip install -e .
```

## 主要入口

- `main.py`：产线主程序，默认读取 `config.yaml`，启动相机采集、计算进程、API 服务、剔废和报警控制。
- `api_server.py`：FastAPI 服务，提供启停、状态、阈值、剔废模式、算法模式和 WebSocket 预览接口。
- `camera_process.py`：相机池与采集进程，按配置绑定相机、设置采集参数并投递图像帧。
- `processing_worker.py`：多进程图像计算入口，加载 ROI、调用算法、过滤飞虫、生成结果。
- `fused_image_processor.py`：明场/深色算法分发。
- `image_processor_hough.py`：核心缺陷检测算法。
- `state_machine.py`：玻璃进入/离开状态、产量统计、结果保存和上报。
- `rejection_controller.py`、`rejection_control.py`：剔废硬件控制。
- `alarm_light_controller.py`、`remote_alarm_server.py`：本机或远程声光报警控制。

## 配置文件

`main.py` 默认加载根目录的 `config.yaml`。仓库中同时保留了几份产线配置模板：

- `config.yaml`：当前默认配置。
- `config1.yaml`：Line1 配置模板。
- `config2.yaml`：Line2 配置模板。
- `config3.yaml`：Line3 配置模板。

切换产线时，建议先备份当前 `config.yaml`，再把目标配置复制为 `config.yaml`，或按现场部署流程同步配置。重点核对：

- `camera_setup.camera_bindings`：相机逻辑编号、MAC、IP、网卡绑定。
- `camera_rois`：每路相机对应的 ROI 模板文件。
- `server_config`：API 监听端口、上报地址、心跳地址。
- `rejection_controller` 和 `alarm_light_params`：串口号、波特率和报警参数。
- `system_params`：进程数、帧率、降采样、内存维护和 JPEG 质量。
- `hough_inspector_params` / `hough_inspector_dark_params`：明场与深色玻璃算法参数。

## 启动

确认配置和现场 ROI 文件就位后，在项目根目录运行：

```powershell
python main.py
```

启动后可通过配置中的 `server_config.listen_port` 访问接口。常用接口包括：

- `GET /status/system`
- `GET /status/cameras`
- `POST /control/start`
- `POST /control/stop`
- `POST /control/rejection_mode`
- `POST /control/thresholds`
- `POST /control/algorithmMode`
- `GET /algorithmMode`
- `WS /ws/stream/{cam_index}`

远程声光报警接收端可在报警灯所在机器运行：

```powershell
python remote_alarm_server.py --config remote_alarm_server.json
```

## 调试与辅助工具

- `ROI_annotator.py`：ROI 标注工具。
- `exclusion_zone_annotator.py`、`visualize_exclusion_zones.py`：排除区域标注和可视化。
- `test_reject.py`：剔废通道测试，支持 dry-run。
- `main_sim.py`：有本地样例图像时用于算法模拟。
- `read_profile_hough.py`、`add_profile_decorators.py`：性能分析辅助。
- `tools/line2_validation/`：Line2 回放数据离线评估和统计工具。

剔废 dry-run 示例：

```powershell
python test_reject.py --dry --marks 0 1 --line-name Line2
```

## 验证

代码交接前建议至少执行：

```powershell
python -m compileall *.py tools
```

产线验证按现场流程执行：

1. 校验 `config.yaml` 能加载。
2. 确认 ROI 模板路径可读且坐标不越界。
3. 启动 `python main.py`，检查相机状态、实时预览和 API 启停。
4. 使用 dry-run 或现场安全流程验证剔废通道映射。
5. 确认缺陷图片、JSON 报告、产量统计和后端上报路径正常。

## 打包说明

每次代码或算法更新后，都需要用 Nuitka 重新打包，再把打包产物发到产线电脑。打包产物不提交到仓库，生成后通常是 `main.build/`、`main.dist/` 或 `main.dist_*.zip`。

打包时要包含程序运行必需的 DLL 和静态资源，例如 `MVGigE.dll` 和 `icon.ico`；不要把本机调试用的 `config.yaml`、`config1.yaml`、`config2.yaml`、`config3.yaml` 等 YAML 配置文件打进 `main.dist`。产线配置以产线电脑 `C:\lineXXX` 目录中的现场 YAML 为准。

示例命令：

```powershell
.\.venv\Scripts\python.exe -m nuitka main.py `
  --standalone `
  --enable-plugin=multiprocessing `
  --include-package=scipy `
  --include-data-files=MVGigE.dll=MVGigE.dll `
  --include-data-files=icon.ico=icon.ico `
  --windows-icon-from-ico=icon.ico
```

打包完成后检查 `main.dist`：

- 必须有 `main.exe`、`MVGigE.dll` 和必要的 Python/第三方依赖文件。
- 不要有本地 `*.yaml` 配置文件。
- 不要有回放数据、日志、检测输出、历史压缩包或本地测试报表。

确认无误后压缩 `main.dist`：

```powershell
Compress-Archive -Path .\main.dist\* -DestinationPath .\main.dist.zip -Force
```

把 `main.dist.zip` 传到产线电脑，解压到 C 盘对应产线目录，例如 `C:\line1`、`C:\line2`、`C:\line3` 或现场约定的 `C:\lineXXX` 文件夹。解压时直接覆盖同名文件即可；不要先清空整个目录，这样可以保留产线电脑上的现场 YAML、ROI 和硬件参数文件。

## 目录维护约定

- 代码、配置模板、SDK 包装、文档进入 Git。
- 运行输出、回放数据、日志、报表、备份和打包产物不进入 Git。
- 现场 ROI 和硬件参数如果只在产线机器维护，交接时以产线机器当前文件为准。
