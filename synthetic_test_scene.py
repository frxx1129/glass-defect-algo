import numpy as np
import cv2
import random


class SyntheticGlassScene:
    """全景多相机模拟 (参考原 generate_test_image)：
    特性:
        * 黑底
        * 半透明玻璃：内部值 120，alpha≈0.6，边框值 20（更暗）
        * 玻璃高度 = cam_h * pane_height_mul (默认 2x) 自上向下滚动再循环
        * 缺陷：概率 0.7 生成 1~4 个；偏向四边分布 (top/bottom 0.3, left/right 0.2)
                  形状分布: 矩形0.2 / 椭圆0.3 / 不规则多边形0.5；灰度 5~15（暗色块）
    * ROI 先画，玻璃与缺陷覆盖其上（ROI 位于下层）
        * 支持 roi_only 跳过玻璃/缺陷，仅输出黑底+ROI
        * 提供 speed_multiplier 放大纵向速度（不改 config）
    """

    def __init__(self, num_cams: int, cam_w: int, cam_h: int, step_px: int = 16,
                 pane_height_mul: float = 2.0, defect_count: int = 4,
                 pane_width_factor: float = 0.90, pane_margin_px: int | None = None,
                 rois_per_cam: list | None = None,
                 roi_only: bool = False,
                 glass_alpha: float = 0.60,
                 glass_inner_val: int = 120,
                 glass_border_val: int = 8,  # 边框更深
                 enable_defects: bool = True,
                 speed_multiplier: float = 2.0,
                 defect_presence_prob: float = 0.7):
        self.num_cams = max(1, num_cams)
        self.cam_w = int(cam_w)
        self.cam_h = int(cam_h)
        self.total_w = self.cam_w * self.num_cams
        self.base_step_px = max(1, int(step_px))
        self.speed_multiplier = max(0.1, float(speed_multiplier))
        self.rois_per_cam = rois_per_cam or []
        self.roi_only = roi_only
        # 玻璃尺寸/位置
        self.pane_h = int(self.cam_h * pane_height_mul)
        self.pane_w = int(self.total_w * pane_width_factor)
        if pane_margin_px is not None:
            self.pane_x = pane_margin_px
            self.pane_w = min(self.total_w - 2 * pane_margin_px, self.pane_w)
        else:
            self.pane_x = (self.total_w - self.pane_w) // 2
        self.defect_count = int(defect_count)
        self.enable_defects = enable_defects and (self.defect_count > 0)
        self.glass_alpha = float(max(0.05, min(glass_alpha, 0.95)))
        self.glass_inner_val = int(max(0, min(glass_inner_val, 255)))
        self.glass_border_val = int(max(0, min(glass_border_val, 255)))
        self.defect_presence_prob = float(max(0.0, min(defect_presence_prob, 1.0)))
        self.y_offset = 0
        self._cycle_span = self.cam_h + self.pane_h  # 一个完整循环跨度（用于 wrap）
        self.defects = []
        if self.enable_defects:
            self._build_defects()

    # ---------------- internal helpers -----------------
    def _build_defects(self):
        self.defects.clear()
        if random.random() > self.defect_presence_prob:
            return
        n = random.randint(1, self.defect_count)
        for _ in range(n):
            edge_type = random.choices(['top', 'bottom', 'left', 'right'], weights=[0.3,0.3,0.2,0.2])[0]
            if edge_type in ('top','bottom'):
                rel_x = random.randint(100, max(100, self.pane_w - 200))
                if edge_type == 'top':
                    rel_y = random.randint(50, max(50, self.pane_h//8))
                else:
                    rel_y = random.randint(max(self.pane_h*7//8, 0), max(self.pane_h - 100, 0))
            else:  # left/right
                rel_y = random.randint(100, max(100, self.pane_h - 200))
                if edge_type == 'left':
                    rel_x = random.randint(50, max(50, self.pane_w//8))
                else:
                    rel_x = random.randint(max(self.pane_w*7//8, 0), max(self.pane_w - 100, 0))
            shape_type = random.choices([0,1,2], weights=[0.2,0.3,0.5])[0]
            block_w = random.randint(20, 60)
            block_h = random.randint(15, 30)
            color = random.randint(5, 15)  # 暗色
            self.defects.append({
                'shape': shape_type,
                'rect': (rel_x, rel_y, block_w, block_h),
                'color': color
            })

    def reset(self):
        self.y_offset = 0
        if self.enable_defects:
            self._build_defects()

    def _draw_rois_on_canvas(self, canvas: np.ndarray):
        if not self.rois_per_cam:
            return
        for cam_i, rois in enumerate(self.rois_per_cam):
            if not rois:
                continue
            base_x = cam_i * self.cam_w
            for r in rois:
                try:
                    x = int(r.get('x', r.get('X', 0)))
                    y = int(r.get('y', r.get('Y', 0)))
                    w = int(r.get('width', r.get('w', 0)))
                    h = int(r.get('height', r.get('h', 0)))
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                gx1 = max(0, base_x + x)
                gy1 = max(0, y)
                gx2 = min(self.total_w, base_x + x + w)
                gy2 = min(self.cam_h, y + h)
                if gx2 <= gx1 or gy2 <= gy1:
                    continue
                canvas[gy1:gy2, gx1:gx2] = 255  # 纯白覆写

    def _overlay_glass(self, canvas: np.ndarray, pane_y: int):
        # pane_y 为玻璃顶端在全局坐标（可为负）
        gy1 = pane_y
        gy2 = pane_y + self.pane_h
        if gy2 <= 0 or gy1 >= self.cam_h:
            return  # 完全不在视野
        clip_y1 = max(0, gy1)
        clip_y2 = min(self.cam_h, gy2)
        if clip_y2 <= clip_y1:
            return
        px1 = self.pane_x
        px2 = self.pane_x + self.pane_w
        region = canvas[clip_y1:clip_y2, px1:px2]
        # glass 内部半透明叠加
        overlay = region.copy()
        overlay[:, :] = self.glass_inner_val
        cv2.addWeighted(overlay, self.glass_alpha, region, 1.0 - self.glass_alpha, 0, region)
        # 边框 (2px, 深色)
        region[0:2, :] = self.glass_border_val
        region[-2:, :] = self.glass_border_val
        region[:, 0:2] = self.glass_border_val
        region[:, -2:] = self.glass_border_val
        # 缺陷绘制（暗色块，不规则形状）
        if self.enable_defects and self.defects:
            for d in self.defects:
                rel_x, rel_y, bw, bh = d['rect']
                shape = d['shape']
                color = d['color']
                by1 = pane_y + rel_y
                by2 = by1 + bh
                if by2 <= 0 or by1 >= self.cam_h:
                    continue
                cy1 = max(0, by1)
                cy2 = min(self.cam_h, by2)
                if cy2 <= cy1:
                    continue
                # 水平坐标
                cx1 = px1 + rel_x
                cx2 = cx1 + bw
                dx1 = cx1
                dx2 = cx2
                # 裁剪
                if dx2 <= 0 or dx1 >= self.total_w:
                    continue
                dx1_clip = max(0, dx1)
                dx2_clip = min(self.total_w, dx2)
                # 在 big canvas 上绘制，这里 region 对应子区域，直接写 canvas
                if shape == 0:  # 矩形
                    canvas[cy1:cy2, dx1_clip:dx2_clip] = color
                elif shape == 1:  # 椭圆
                    cx_center = (dx1 + dx2) // 2
                    cy_center = (by1 + by2) // 2
                    axes = (max(1, bw//2), max(1, bh//2))
                    cv2.ellipse(canvas, (cx_center, cy_center), axes, 0, 0, 360, color, -1)
                else:  # 多边形
                    pts = []
                    num_pts = random.randint(5,8)
                    cx_center = (dx1 + dx2) / 2.0
                    cy_center = (by1 + by2) / 2.0
                    radius = min(bw, bh) / 2.0
                    for i in range(num_pts):
                        ang = 2*np.pi*i/num_pts
                        rad = radius*(0.7 + 0.3*random.random())
                        px = int(cx_center + rad*np.cos(ang))
                        py = int(cy_center + rad*np.sin(ang))
                        pts.append([px, py])
                    pts = np.array(pts, dtype=np.int32).reshape(-1,1,2)
                    cv2.fillPoly(canvas, [pts], color)

    # --------------- public ---------------
    def next_row_frames(self):
        """生成一帧：黑底→ROI→(可选)玻璃(含缺陷)→拆分相机帧"""
        big = np.zeros((self.cam_h, self.total_w), dtype=np.uint8)
        # 需求：ROI 在玻璃下面 -> 先画 ROI，再叠加玻璃与缺陷
        self._draw_rois_on_canvas(big)
        if not self.roi_only:
            pane_y = (self.y_offset % self._cycle_span) - self.pane_h
            self._overlay_glass(big, pane_y)
            step_effective = int(self.base_step_px * self.speed_multiplier)
            step_effective = max(1, step_effective)
            self.y_offset += step_effective
            if (self.y_offset % self._cycle_span) < step_effective and self.enable_defects:
                self._build_defects()
        frames = []
        for i in range(self.num_cams):
            sx = i * self.cam_w
            frames.append(big[:, sx:sx + self.cam_w].copy())
        return frames
