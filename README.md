## 系统配置 (`config.json`) 参数说明文档

本文档旨在详细解释 `config.json` 文件中各个参数的含义、作用以及建议的配置方式。

### 1. 顶层参数

这些参数定义了系统的基础路径和标识信息。

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `storage_path` | string | 用于存储检测结果（图片和JSON报告）的根目录路径。 |
| `roi_template_file`| string | 默认的感兴趣区域 (ROI) 模板文件路径。当 `camera_rois` 中未给特定相机指定文件时，会使用此文件。 |
| `lineName` | string | 当前检测线的名称，用于向服务器上报数据时标识身份。 |
| `user_id_auto` | integer | 系统自动剔废时，记录在报告中的用户ID。 |
| `user_id_manual` | integer | 操作员手动剔废时，记录在报告中的用户ID。 |

---

### 2. `system_params` (系统性能与行为参数)

此部分控制系统的核心性能、资源分配和通信行为。

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `pixels_per_mm` | float | 图像中每毫米对应的像素数量。这是将像素尺寸转换为物理尺寸（毫米）的关键校准值。 |
| `queue_size_factor` | integer | 内部数据队列大小的乘数因子。队列大小 = `process_workers` * `NUM_CAMERAS` * `queue_size_factor`，用于缓冲相机和处理进程之间的数据。 |
| `http_client_max_workers` | integer | 用于发送HTTP请求（如上传报告、心跳）的后台线程池的最大线程数。 |
| `http_get_timeout_s` | integer | 执行HTTP GET请求的超时时间（秒）。 |
| `http_post_timeout_s` | integer | 执行HTTP POST请求（如推送统计）的超时时间（秒）。 |
| `http_upload_timeout_s` | integer | 上传文件（缺陷报告和图片）的超时时间（秒）。 |
| `stats_push_interval_s` | integer | 向服务器定时推送产量和剔废统计数据的时间间隔（秒）。 |
| `opencv_threads` | integer | 每个计算进程分配给OpenCV使用的最大线程数。设为1可避免线程间竞争，适用于多进程架构。 |
| `preview_width` | integer | 推送到前端WebSocket的实时预览图的宽度（像素）。图像会按此宽度等比缩放。 |
| `jpeg_quality_main` | integer | 保存到本地的缺陷图片的JPEG压缩质量（0-100）。 |
| `jpeg_quality_preview` | integer | 推送到前端预览的图片的JPEG压缩质量（0-100）。 |
| `coalesce_drain_budget_ms`| integer | **帧合并策略**：在处理前，清空相机队列以获取最新帧的时间窗口（毫秒）。这可以有效处理相机帧率高于处理速度的情况，避免处理延迟的图像。 |
| `coalesce_max_drain` | integer | **帧合并策略**：在上述时间窗口内，最多从队列中丢弃的旧帧数量。 |
| `process_workers` | integer | 启动的图像计算进程数量。通常建议设置为CPU核心数减2。 |
| `roi_threads` | integer | **ROI并行处理**：在单个计算进程内部，用于并行处理一张图里多个ROI的线程数。 |

---

### 3. `camera_setup` (相机硬件与采集参数)

此部分定义了相机的物理连接、初始化方式和统一的采集参数。

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `expected_cameras` | integer | 系统预期的相机总数。 |
| `bootstrap_assign_ips` | boolean | 是否在系统启动时自动为所有发现的相机分配IP地址。建议在首次设置或网络环境变化时设为 `true`。 |
| `open_retry_count` | integer | 尝试打开单个相机失败时的最大重试次数。 |
| `open_retry_backoff_base_s`| float | 每次重试打开相机之间的基础等待时间（秒）。 |
| `heartbeat_timeout_ms` | integer | 相机心跳超时时间（毫秒）。如果在此时间内SDK未与相机通信，相机会自动断开连接，以防程序崩溃导致相机死锁。 |
| **`runtime` (运行时行为)** | | |
| `use_software_trigger` | boolean | 是否使用软件触发模式。`true`: 按 `frame_rate` 定时发送软触发指令采图。`false`: 相机以最高速度自由采图（Free Run模式）。 |
| `frame_rate` | integer | 目标帧率 (FPS)。在软触发模式下，决定了触发信号的频率。 |
| `enforce_mono8` | boolean | 是否强制所有相机使用 `Mono8` (8位灰度) 像素格式。 |
| **`network_interfaces` (本机网卡)** | | |
| `index` | integer | 用户定义的网卡逻辑索引，用于和相机进行绑定。 |
| `ip` | string | 该网卡的IP地址。 |
| **`camera_bindings` (相机绑定)** | | |
| `index` | integer | **逻辑索引**：用户为相机分配的稳定编号，用于ROI配置和API调用。 |
| `physical_index` | integer | **物理索引**：相机SDK运行时分配的索引，仅供参考，可能会变。 |
| `mac` | string | **核心标识**：相机的唯一MAC地址，用于将逻辑索引与物理相机稳定地绑定起来。 |
| `ip` | string | 分配给该相机的IP地址。 |
| `model` | string | 相机型号名称。 |
| **`unified_params` (统一相机参数)** | | 应用于所有相机的图像采集参数。 |
| `acquisition` | object | 图像尺寸与帧率设置。`width`, `height`, `frame_rate`。 |
| `exposure` | object | 曝光设置。`value_us` 为曝光时间（微秒）。 |
| `gain` | object | 增益设置。`value_db` 为增益值 (dB)。 |
| `gamma` | object | Gamma校正值。`value` 为Gamma系数。 |
| `network` | object | GEV网络传输参数。`packet_size_bytes` (包大小) 和 `packet_delay_us` (包延迟)。 |

---

### 4. `camera_rois` (相机ROI配置)

此部分将每个相机（通过其逻辑索引）映射到包含其ROI坐标的特定文件。

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `"0"`, `"1"`, `"2"`... | string | 键是相机的**逻辑索引**（字符串形式）。值是包含该相机所有ROI坐标的 `.json` 文件路径。 |

---

### 5. `server_config` (服务器通信配置)

定义了与后端管理服务器通信的所有端点和网络参数。

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `server` | string | 后端服务器的IP地址或域名。 |
| `listen_host` | string | 本地API服务器监听的IP地址。`0.0.0.0` 表示监听所有网络接口。 |
| `listen_port` | integer | 本地API服务器监听的端口号。 |
| `cors_origins` | array | 跨域资源共享 (CORS) 设置，允许来自指定源的Web前端访问API。`["*"]` 表示允许所有源。 |
| `upload_url` | string | 上传缺陷报告（图片和JSON）的完整URL地址。 |
| `stats_push_url` | string | 定时推送产量和剔废统计数据的完整URL地址。 |
| `heartbeat_url` | string | 发送心跳状态（运行/停止等）的完整URL地址。 |

---

### 6. `rejection_params` (剔废控制参数)

控制物理剔废装置的行为。

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `max_defect_size_mm`| float | **自动剔废阈值**：当检测到的缺陷尺寸（长度或宽度）大于等于此值（毫米）时，触发自动剔废。 |
| `REJECTION_PULSE_MS`| integer | 剔废执行机构（如电磁阀）的脉冲信号持续时间（毫秒）。 |
| `REJECTION_DELAY_S` | float | 从检测到需要剔废的信号到实际发送剔废脉冲之间的延迟时间（秒）。用于匹配物品从相机位置移动到剔废位置所需的时间。 |

---

### 7. `defect_detection_params` (缺陷检测算法参数)

这是算法的核心，控制着从图像预处理到缺陷分类的全过程。

| 模块 | 参数 | 说明 |
| :--- | :--- | :--- |
| **`preprocess_params`** | | **图像预处理** |
| `GAMMA_VALUE` | float | Gamma校正值，用于调整图像亮暗部对比度。 |
| `CANNY_SIGMA` | float | Canny边缘检测算法的sigma值，影响其自适应阈值范围，从而控制边缘检测的灵敏度。 |
| `CLAHE_CLIP_LIMIT` | float | 限制对比度自适应直方图均衡化(CLAHE)的对比度增强幅度。 |
| `CLAHE_TILE_TARGET_PX`| integer | CLAHE算法的网格目标尺寸，影响局部对比度增强的区域大小。 |
| **`dbscan_params`** | | **DBSCAN聚类** (用于边缘点降噪) |
| `eps` | float | DBSCAN算法中，一个点被视为邻域点的最大距离。 |
| `min_samples` | integer | 一个点要成为核心点所需的邻域内最小点数。 |
| `min_points_per_cluster`| integer | 一个有效的点簇（被认为是真实边缘）所需的最小点数。 |
| **`contour_params`** | | **轮廓筛选** |
| `min_area` | integer | 有效玻璃轮廓的最小面积（像素）。小于此值的轮廓将被忽略。 |
| **`defect_params`** | | **缺陷定义与分类** |
| `ANGLE_*` | float | 定义了**角部缺陷**的各种角度阈值、容差和修正参数。 |
| `DEFECT_STD_DEV_THRESHOLD`| integer | 定义**亮度缺陷**的阈值。当一个像素的亮度低于区域平均值减去 `k` 倍标准差时，被视为异常。 |
| `MIN_DEFECT_AREA_MM2`| float | 亮度缺陷被识别所需的最小面积（平方毫米）。 |
| `MIN_DEFECT_DIMENSION_MM`| float | 亮度缺陷被识别所需的最小长度（毫米）。 |
| `ORIENTATION_PARALLEL_THRESHOLD` | float | 用于区分'B'类和'L'类缺陷的角度阈值。 |
| `BRIGHTNESS_*` | | 在检测亮度缺陷之前，对ROI进行的独立图像增强参数。 |
| **`drawing_params`** | | **结果可视化** |
| `DRAW_*` | boolean | 控制是否在结果图上绘制原始轮廓、简化轮廓等。 |
| `HIGHLIGHT_DEFECT_PIXELS`| boolean | 是否用半透明颜色叠加高亮显示检测到的亮度缺陷区域。 |
| `*_COLOR`, `*_THICKNESS`| | 控制各种绘制元素的颜色和线条粗细。 |

***

## 系统运行原理简介

该系统是一个基于多进程和多线程架构的高性能视觉检测系统，其工作流程可以概括为以下几个核心阶段：

1.  **启动与初始化 (`main.py`)**
    *   程序启动时，主进程首先读取 `config.json` 文件。
    *   它会创建用于进程间通信的共享内存对象，如数据队列 (`task_queue`, `results_queue`)、事件 (`stop_event`, `run_event`) 和状态变量。
    *   主进程会启动一系列独立的子进程来分担不同的任务，实现并行处理。

2.  **相机采集 (`camera_process.py`)**
    *   一个独立的**相机池进程**负责管理所有物理相机。
    *   启动时，它会使用 `MultiCameraSetup` 工具类扫描网络，根据 `camera_bindings` 中的MAC地址识别并自动配置每台相机的IP地址。
    *   随后，它为每一台相机再创建一个专属的**采集子进程**。
    *   每个采集子进程独立地打开相机、设置 `unified_params` 参数，并进入一个循环，不断地从相机获取原始图像帧，然后将这些帧放入共享的 `task_queue` 中。

3.  **并行计算与缺陷检测 (`processing_worker.py` & `image_processor_optimized.py`)**
    *   多个**计算工作进程**（数量由 `process_workers` 定义）同时运行，它们是系统的计算核心。
    *   这些进程从 `task_queue` 中竞争获取原始图像帧。为了保证实时性，它们会采用“帧合并”策略，丢弃队列中旧的帧，只处理每个相机的最新一帧。
    *   获取到图像后，工作进程会根据相机索引加载对应的ROI模板。
    *   它使用一个内部的**线程池**（线程数由 `roi_threads` 定义），对一张图中的所有ROI进行并行处理。
    *   每个ROI都经过复杂的图像处理流水线：**预处理**（增强对比度）、**边缘检测**、**轮廓提取与拟合**，最后进行**缺陷分类**（包括基于几何角度的角部缺陷和基于统计亮度的表面缺陷）。
    *   处理完成后，生成一份包含缺陷信息的JSON报告和一张标注了缺陷位置的可视化图像。这些结果被放入共享的 `results_queue`。

4.  **状态管理、通信与控制 (`state_machine.py` & `api_server.py`)**
    *   主进程中运行着一个**API服务器**，它负责接收外部命令（如启停、模式切换）并提供系统状态查询。
    *   服务器内部的一个关键后台线程是**状态机线程**。它持续从 `results_queue` 中获取已处理完毕的结果。
    *   该线程通过分析结果中的轮廓信息，维护一个简单的状态机（`WAITING_FOR_PANE` -> `PANE_DETECTED`），以判断玻璃的进入和离开。
    *   它负责更新总产量和剔废数量，并将统计数据持久化保存。
    *   同时，它将带有标注的预览图通过WebSocket实时推送到Web前端，供操作员监视。
    *   当检测到需要剔废的缺陷时，它会将一个剔废指令放入 `rejection_queue`。

5.  **剔废执行 (`rejection_controller.py`)**
    *   一个轻量级的**剔废处理器线程**专门监听 `rejection_queue`。
    *   一旦收到剔废指令，它会等待 `REJECTION_DELAY_S` 所配置的延迟时间，然后通过 `RejectionController` 模块与物理硬件交互，发送一个持续 `REJECTION_PULSE_MS` 毫秒的脉冲信号，完成最终的剔废动作。

通过这种高度并行化的设计，系统能够最大限度地利用多核CPU资源，实现从图像采集到缺陷报告输出的低延迟处理。