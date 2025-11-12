# 多主体（多片玻璃）分离与处理设计

本文档描述在同一相机视野/同一帧中出现多片玻璃时的算法与系统改造方案，目标是：
- 能在一帧内识别出多个独立“主体”（pane）
- 对每个主体独立计算边框、角点、缺陷与状态（OK/NG）
- 报告结构支持多主体；与现有上游/下游保持兼容
- 逐步改造状态机与剔废控制逻辑，最终实现精准剔废与准确产量统计

---

## 0. 术语
- 主体（Pane）：一片玻璃的几何与缺陷集合。
- 轮廓（Contour）：在单帧图像上通过阈值/边缘/连通性得到的封闭区域边界。

---

## 1. 分阶段里程碑

1) M1（检测侧最小可用）
- 在 `image_processor_hough.py` 内部，保留多个外部轮廓（`cv2.RETR_EXTERNAL`），对每个轮廓做独立的主边/角点/缺陷评估。
- 返回 `report.panes[]`：每个 pane 含 bbox、四角点、缺陷数组、state_code/image_status。
- 顶层 `report.image_status` 仍保留（例如取 panes 中最"严重"者，NG 优先）。
- 增加配置开关：`use_multi_subject: true`；关闭时保持旧行为（仅选择最大轮廓）。

2) M2（绘制与持久化）
- 可视化图层为每个主体使用不同颜色叠加；文件命名支持 `_p{piece_id}` 后缀（如 `cam0_ts123_p1.jpg`）。
- 本地保存/上传仍和旧路径一致，但当存在多个 NG 时，按主体分别输出一套图片+JSON（即多个样本）。

3) M3（状态机与上传）
- 状态机从“单 pane”扩展为“多 pane 活动列表”：
  - 帧间通过 IoU/质心距离做简单数据关联，给每个 pane 分配短期追踪 id（`track_id`）。
  - 每个 track 独立维护生命周期（enter/exit）与 NG 缓存；上传/统计按 track 归档。

4) M4（剔废控制）
- 如果硬件支持多通道或可在同一时间窗精准定位多片：
  - 按每个 track 的 NG 事件时间独立计算剔废延迟并触发。
- 如果硬件仅单脉冲：
  - 同帧多片 NG 时，采用最早到达剔废位的一片触发，或配置“全部剔废”策略。

---

## 2. 算法改造要点（M1）

### 2.1 多轮廓保留
- 预处理 + 边缘（Canny/聚类清理）后，使用 `cv2.findContours(..., RETR_EXTERNAL, ...)`。
- 过滤规则：`area >= min_area`；可根据 ROI 与画面尺寸做自适应。
- 若两片距离很近，可在边缘图上做一次轻微 `morphology_open`（核尺寸根据 `pixels_per_mm` 缩放）以断开细窄桥接。
- 如仍粘连，可选 `distance transform + watershed` 做分割（作为降级方案开关）。

### 2.2 线段与角点在主体域内求解
- 对每个轮廓生成 `mask` / `bbox`，在该局部范围内运行原有的 Hough/主边合并与角点选择逻辑：
  - 所有线段候选限制在该 `mask` 内；避免把另一片的边纳入同一合并簇。
  - 角点交叉验证时，仅在当前 `bbox` 内评分。

### 2.3 亮度缺陷在主体域内统计
- 计算均值/标准差与连通域时只在 pane `mask` 内进行，避免背景或其他 pane 拉偏统计。
- 最小面积/最小尺寸阈值继续使用毫米换算，保持与单主体时一致的物理意义。

### 2.4 报告结构（向后兼容）
```jsonc
{
  "image_status": "NG",       // 兼容旧逻辑：从 panes[].image_status 汇总（例如 NG 优先）
  "state_code": 1,             // 可取 panes 的合并（如 max）或仅保留供旧路径判断
  "panes": [
    {
      "pane_id": 1,           // 帧内连续编号；进入状态机后可映射到 track_id
      "bbox": [x, y, w, h],
      "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "defects": [ ... ],
      "image_status": "NG",
      "state_code": 1,
      "rois": { /* 可选：当前 pane 内部的 ROI 统计 */ }
    },
    { /* pane 2 ... */ }
  ],
  // 其他旧字段：保持或标记为 deprecated
}
```

---

## 3. 处理链路与系统改造（M2-M3）

### 3.1 处理结果传播（processing_worker.py）
- 将 `panes[]` 透传；预览图 `annotated_image_buffer` 渲染所有 pane（不同颜色）。
- 若存在 NG 的多个 pane：
  - 在结果队列中保留单条记录（包含所有 panes），由状态机拆分；或者
  - 直接拆分为多条记录（每个 pane 一条），二者择一，推荐第一种便于 UI。

### 3.2 状态机最小改造（M3）
- 活动 pane 列表：`active_tracks: Dict[int, Track]`，每帧匹配 `panes[].bbox` 与已存在 tracks：
  - IoU > 阈值 或 质心距离 < 阈值 → 关联更新；否则新建 track。
  - 失配若干帧则结束 track，触发上传/统计。
- NG 累积：对每个 track，缓冲对应 pane 的 NG 帧（图片/JSON）。
- 上传：
  - 结束时对每个 track 上传一批；文件命名 `cam{idx}_ts{ms}_p{trackId}.jpg`。

### 3.3 剔废（M4）
- 单通道设备：同一时间窗只允许一次脉冲；可配置：
  - 选择“优先到达剔废位的 NG track”。
  - 或“同窗全部剔废”。
- 多通道设备：每个 track 计算独立延迟，分别触发。

---

## 4. 配置项
```yaml
system_params:
  use_multi_subject: true          # 打开多主体分离
  split_strategy: "report-only"    # 或 "split-to-records"（把每个 pane 拆成独立记录）
  iou_match_threshold: 0.3
  centroid_match_px: 40
  min_pane_area_px: 5000
  morphology_open_kernel_px: 3     # = 0 则关闭
  use_watershed_on_merge: false
```

---

## 5. 质量保障
- 单主体回归：确保在 `use_multi_subject=false` 下输出与现有版本完全一致。
- 双主体样例：至少两类样本（并排/堆叠），验证 `panes.length == 2` 且缺陷/角点合理。
- 性能：测评多主体开启后单帧耗时；如需，限制最多 N 个 pane 进入后续精细流程。

---

## 6. 风险与回退
- 若分割仍粘连（极近距离）：启用 watershed 或阈值自适应增加分离。
- 若上游/服务器暂不支持多主体：保留旧顶层字段并仅上传最严重 pane；同时本地完整保存 `_p{}` 样本用于追溯。

---

## 7. 建议的实现顺序（开发清单）
1. `image_processor_hough.py`：实现 `panes[]`（受 `use_multi_subject` 控制），绘制多颜色叠加。
2. `processing_worker.py`：透传 `panes[]`，保持 WebSocket 轻量。
3. `state_machine.py`：新增简易 IoU 关联的 `active_tracks`，分 pane 上传与命名。
4. 文档与配置：README + 本文档；示例配置开启 `use_multi_subject`。
5. 回归与样例：新增 2-3 张双主体图片，输出校验。

---

以上为总体设计。若需，我可以按此拆分逐步提交变更，并先从 M1 开始（纯检测与报告扩展，最小侵入，支持回滚）。
