# --- START OF FILE image_processor_hough.py ---
import cv2
import numpy as np
import json
import os
from itertools import combinations
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ====================================================================================
# --- 全局字体加载 ---
# ====================================================================================

def _get_font(font_size=36):
    if not PIL_AVAILABLE:
        return None
    
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except IOError:
            continue
    
    print("警告: 未找到中文字体, 标注可能无法正确显示中文。")
    try:
        return ImageFont.load_default()
    except Exception:
        return None

ANNOTATION_FONT = _get_font(font_size=32)
import math

# ====================================================================================
# --- 水平延长“栅栏”历史缓存 (基于前若干帧竖直边位置) ---
# ====================================================================================
# 目标: 当当前帧缺失某块玻璃的竖直主边时, 避免水平主边过度延长并跨到另一块玻璃的竖直边形成伪角。
# 策略:
#   1. 前 N(默认10) 帧内, 记录每个 ROI 内所有近竖直主边的 x 中点位置。
#   2. 在达到 N 帧时, 对累计的 x 位置进行简单聚类(阈值像素内归并)形成稳定“栅栏”集合, 每个栅栏代表一条可能的竖直分界线。
#   3. 后续水平延长时, 若尝试延长方向存在栅栏, 则限制延长不跨越该栅栏(保留少量 margin)。
# 变量:
#   _roi_vertical_history[(roi_w, roi_h)] -> list[list[x_positions]]
#   _roi_frame_count[(roi_w, roi_h)] -> int 已记录帧数
#   _roi_fences[(roi_w, roi_h)] -> sorted list[float] 稳定栅栏 x 坐标
# 注意: 以 (roi_w, roi_h) 作为简化的 ROI key; 若存在同尺寸多 ROI 则会共享栅栏, 如需更细粒度可以在调用层传入 ROI ID 并改为 (roi_id, roi_w, roi_h).

_roi_vertical_history = {}
_roi_frame_count = {}
_roi_fences = {}
_roi_glass_boundary_history = {}
_roi_glass_boundaries = {}

def _update_glass_boundaries(roi_key, vertical_x_list, params):
    """记录多玻璃之间的候选分隔线(边界), 与栅栏不同: 边界是两块玻璃之间的中线。
    当帧内存在至少两条近竖直边且最大间隙>=配置阈值时, 取该最大间隙的中点作为候选边界。累积若干帧后取中位数稳定化。
    """
    # 若已生成边界，不再累积历史，避免内存泄漏
    if roi_key in _roi_glass_boundaries:
        return

    try:
        need_frames = int(params.get('DEFECT_DETECTION', {}).get('GLASS_BOUNDARY_STABLE_FRAMES', 5))
    except Exception:
        need_frames = 5
    try:
        gap_thr = float(params.get('DEFECT_DETECTION', {}).get('GLASS_BOUNDARY_MIN_GAP_PX', 60.0))
    except Exception:
        gap_thr = 60.0
    xs = sorted(float(x) for x in vertical_x_list if np.isfinite(x))
    if len(xs) < 2:
        return
    gaps = []
    for i in range(len(xs)-1):
        g = xs[i+1] - xs[i]
        gaps.append((g, 0.5*(xs[i+1]+xs[i])))
    if not gaps:
        return
    max_gap, mid_pt = max(gaps, key=lambda t: t[0])
    if max_gap < gap_thr:
        return
    hist = _roi_glass_boundary_history.setdefault(roi_key, [])
    hist.append(mid_pt)
    if len(hist) >= need_frames:
        try:
            _roi_glass_boundaries[roi_key] = float(np.median(np.array(hist, dtype=float)))
        except Exception:
            _roi_glass_boundaries[roi_key] = mid_pt
        # 生成后清空历史以释放内存
        _roi_glass_boundary_history[roi_key] = []

def _get_glass_boundary(roi_key):
    return _roi_glass_boundaries.get(roi_key, None)

def _update_vertical_fences(roi_key, vertical_x_list, params):
    """更新竖直边历史并在达到设定帧数后生成栅栏。"""
    # 若已生成栅栏，不再累积历史，避免内存泄漏
    if roi_key in _roi_fences:
        return

    try:
        fence_frames = int(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_FENCE_INIT_FRAMES', 10))
    except Exception:
        fence_frames = 10
    try:
        cluster_tol = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_FENCE_CLUSTER_TOL_PX', 18.0))
    except Exception:
        cluster_tol = 18.0
    xs = [float(x) for x in vertical_x_list if np.isfinite(x)]
    if not xs:
        # 仍递增帧计数, 但没有新数据
        _roi_frame_count[roi_key] = _roi_frame_count.get(roi_key, 0) + 1
        return
    hist = _roi_vertical_history.setdefault(roi_key, [])
    hist.append(xs)
    fc = _roi_frame_count.get(roi_key, 0) + 1
    _roi_frame_count[roi_key] = fc
    # 仅在首次达到 fence_frames 时生成栅栏 (保持稳定, 不滚动窗口)
    if fc == fence_frames:
        try:
            all_x = np.concatenate([np.array(h, dtype=float) for h in hist])
            all_x.sort()
            clusters = []
            for x in all_x:
                placed = False
                for cl in clusters:
                    # 用聚类中心的当前均值做距离阈值判定
                    center = float(np.mean(cl))
                    if abs(x - center) <= cluster_tol:
                        cl.append(x); placed = True; break
                if not placed:
                    clusters.append([x])
            fences = [float(np.median(cl)) for cl in clusters]
            fences.sort()
            _roi_fences[roi_key] = fences
        except Exception:
            _roi_fences[roi_key] = []
        # 生成后清空历史以释放内存
        _roi_vertical_history[roi_key] = []

def _get_fences_for_roi(roi_key):
    return _roi_fences.get(roi_key, [])



# ====================================================================================
# --- 几何学与分析辅助函数 ---
# ====================================================================================

# --- 单位换算与配置读取辅助 ---
def _mm_to_px(val_mm: float, pixels_per_mm: float) -> float:
    return float(val_mm) * float(pixels_per_mm)

def _mm2_to_px2(val_mm2: float, pixels_per_mm: float) -> float:
    ppm = float(pixels_per_mm)
    return float(val_mm2) * (ppm * ppm)

def _get_dist_px(cfg: dict, key_mm: str, key_px: str | None, default_mm: float | None, pixels_per_mm: float, default_px: float | None = None) -> float:
    """优先读毫米键, 否则回退像素键并原样使用；若都无则用默认并换算。"""
    if key_mm in cfg:
        return _mm_to_px(cfg[key_mm], pixels_per_mm)
    if key_px and key_px in cfg:
        return float(cfg[key_px])
    if default_mm is not None:
        return _mm_to_px(default_mm, pixels_per_mm)
    if default_px is not None:
        return float(default_px)
    return 0.0

def _get_area_px2(cfg: dict, key_mm2: str, key_px2: str | None, default_mm2: float | None, pixels_per_mm: float, default_px2: float | None = None) -> float:
    if key_mm2 in cfg:
        return _mm2_to_px2(cfg[key_mm2], pixels_per_mm)
    if key_px2 and key_px2 in cfg:
        return float(cfg[key_px2])
    if default_mm2 is not None:
        return _mm2_to_px2(default_mm2, pixels_per_mm)
    if default_px2 is not None:
        return float(default_px2)
    return 0.0

def _kernel_mm_to_px_odd(kernel_mm: list[float] | tuple[float, float], pixels_per_mm: float, fallback_px: list[int] | None = None) -> tuple[int, int]:
    """将以毫米配置的核尺寸转换为奇数像素尺寸；若无毫米键则回退像素尺寸。"""
    if kernel_mm is not None:
        try:
            kx_px = int(round(float(kernel_mm[0]) * float(pixels_per_mm)))
            ky_px = int(round(float(kernel_mm[1]) * float(pixels_per_mm)))
            if kx_px < 1: kx_px = 1
            if ky_px < 1: ky_px = 1
            if kx_px % 2 == 0: kx_px += 1
            if ky_px % 2 == 0: ky_px += 1
            return (kx_px, ky_px)
        except Exception:
            pass
    if fallback_px is not None:
        try:
            kx_px = int(fallback_px[0]); ky_px = int(fallback_px[1])
            return (kx_px, ky_px)
        except Exception:
            return (5, 5)
    return (5, 5)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    if dist == 0: return
    
    delta = (np.array(pt2) - np.array(pt1)) / dist
    
    current_pos = np.array(pt1)
    segment_length = 0
    while segment_length < dist:
        start_point = current_pos
        end_point = current_pos + delta * dash_length
        if segment_length + dash_length > dist:
            end_point = pt2
        
        cv2.line(img, tuple(map(int, start_point)), tuple(map(int, end_point)), color, thickness)
        
        current_pos = current_pos + delta * (2 * dash_length)
        segment_length += 2 * dash_length

def find_line_intersection(line1, line2):
    x1, y1, x2, y2 = line1; x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

def calculate_vertex_angle(p_prev, p_curr, p_next):
    v1 = np.array(p_prev) - np.array(p_curr); v2 = np.array(p_next) - np.array(p_curr)
    v1_mag, v2_mag = np.linalg.norm(v1), np.linalg.norm(v2)
    if v1_mag < 1e-9 or v2_mag < 1e-9: return 180.0
    cosine_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def _adjust_angle_near_90(angle_deg: float) -> float:
    """角度修约规则：
    - 若 86°..94°，直接视为 90°；
    - 若 80°..100°，将与 90° 的差值缩小一半（向 90° 拉近）；
    - 其他范围保持不变。
    """
    try:
        a = float(angle_deg)
    except Exception:
        return angle_deg
    if 86.0 <= a <= 94.0:
        return 90.0
    if 80.0 <= a <= 100.0:
        dev = a - 90.0
        return 90.0 + 0.5 * dev
    return a

def calculate_angle_between_lines(line1, line2):
    v1 = np.array(line1[2:]) - np.array(line1[:2])
    v2 = np.array(line2[2:]) - np.array(line2[:2])
    v1_mag = np.linalg.norm(v1); v2_mag = np.linalg.norm(v2)
    if v1_mag < 1e-9 or v2_mag < 1e-9: return 0.0
    cosine_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle)
    if angle_deg > 90.0: angle_deg = 180.0 - angle_deg
    # 应用 90° 邻域修约
    return _adjust_angle_near_90(angle_deg)

def get_point_line_segment_projection(point, line_segment):
    p = np.array(point); p1 = np.array(line_segment[:2]); p2 = np.array(line_segment[2:])
    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-8: return p1, np.linalg.norm(p - p1)
    t = np.dot(p - p1, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_point = p1 + t * line_vec
    return proj_point, np.linalg.norm(p - proj_point)

def get_point_line_perpendicular_distance(point, line_segment):
    """计算点到由线段两端点确定的直线的垂直距离（无限延长线）。
    当线段退化（端点重合）时，退化为点到该点的距离。
    参数:
        point: (x, y)
        line_segment: [x1, y1, x2, y2]
    返回:
        距离（像素，float）
    """
    p = np.array(point, dtype=float)
    a = np.array(line_segment[:2], dtype=float)
    b = np.array(line_segment[2:], dtype=float)
    v = b - a
    denom = float(np.linalg.norm(v))
    if denom < 1e-8:
        return float(np.linalg.norm(p - a))
    # 2D 叉积的模 |v x (p-a)| / |v|
    cross = float(v[0] * (p[1] - a[1]) - v[1] * (p[0] - a[0]))
    return abs(cross) / denom

def scan_edge_for_luminosity_defects(roi_gray, edge, params, pixels_per_mm: float):
    p = params["DEFECT_DETECTION"]
    
    p1 = np.array(edge[:2]); p2 = np.array(edge[2:])
    # 宽度以毫米配置, 转像素
    scan_width = _get_dist_px(p, "LUMINOSITY_SCAN_WIDTH_MM", "LUMINOSITY_SCAN_WIDTH", None, pixels_per_mm)
    scan_width = max(1, scan_width)  # Ensure scan width is at least 1 pixel
    line_vec = p2 - p1; line_length = np.linalg.norm(line_vec)
    if line_length < 1e-6:
        return []

    unit_vec = line_vec / line_length
    normal_vec = np.array([-unit_vec[1], unit_vec[0]])
    half_width_vec = (scan_width / 2.0) * normal_vec

    # 四个顶点（按法线方向正侧为 c1-c2，与边线 p1-p2 组成正侧半带；另一侧为 p1-p2-c3-c4）
    c1 = p1 + half_width_vec; c2 = p2 + half_width_vec
    c3 = p2 - half_width_vec; c4 = p1 - half_width_vec

    # 分别构建两侧的扫描掩膜，取平均亮度更低的一侧
    mask_plus = np.zeros_like(roi_gray)
    poly_plus = np.array([c1, c2, p2, p1], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_plus, [poly_plus], 255)

    mask_minus = np.zeros_like(roi_gray)
    poly_minus = np.array([p1, p2, c3, c4], dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_minus, [poly_minus], 255)

    mean_plus = cv2.mean(roi_gray, mask=mask_plus)[0]
    mean_minus = cv2.mean(roi_gray, mask=mask_minus)[0]
    scan_mask = mask_plus if mean_plus < mean_minus else mask_minus

    # 从扫描掩膜中剔除边缘线本体及两端点的圆形区域
    # 动态忽略宽度（仅用于 B 类型亮度缺陷的边线屏蔽）：
    # 以边线“中点”到整图中心（或提供的全局中心）直线距离线性映射：
    #   距离 0px -> 忽略 0mm
    #   距离 2000px -> 忽略 4mm
    # 即忽略宽度(mm) = clamp(dist_px * 0.002, 0, 4)
    # 支持参数传入全局图像尺寸 FRAME_WIDTH, FRAME_HEIGHT 与 ROI 偏移 ROI_OFFSET_X/Y；
    # 若未提供则退回使用当前 roi 的中心作为“整图中心”近似。
    edge_mid = (p1 + p2) / 2.0
    frame_w = float(params.get('FRAME_WIDTH', roi_gray.shape[1]))
    frame_h = float(params.get('FRAME_HEIGHT', roi_gray.shape[0]))
    off_x = float(params.get('ROI_OFFSET_X', 0.0))
    off_y = float(params.get('ROI_OFFSET_Y', 0.0))
    # 计算全局中点与全局中心
    edge_mid_global = edge_mid + np.array([off_x, off_y], dtype=float)
    global_center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=float)
    dist_px_dynamic = float(np.linalg.norm(edge_mid_global - global_center))
    ignore_width_mm = min(4.0, max(0.0, dist_px_dynamic * 0.002))  # 0.002 mm/px
    if pixels_per_mm and pixels_per_mm > 0:
        edge_ignore_px = ignore_width_mm * pixels_per_mm
    else:
        # 若无标定，直接用像素距离的一个保守比例（等效 1px = 0.002mm，假设 1mm≈1px -> 宽度≈dist_px*0.002）
        edge_ignore_px = ignore_width_mm  # 作为像素近似
    # 保留向后兼容：若配置显式要求固定值，可通过设置 OVERRIDE_LUMINOSITY_EDGE_IGNORE_MM 忽略动态计算
    try:
        override_mm = params.get('DEFECT_DETECTION', {}).get('OVERRIDE_LUMINOSITY_EDGE_IGNORE_MM', None)
        if override_mm is not None:
            ov_mm = float(override_mm)
            if ov_mm >= 0:
                edge_ignore_px = ov_mm * (pixels_per_mm if pixels_per_mm else 1.0)
    except Exception:
        pass
    endpoint_exclude_r = _get_dist_px(p, "LUMINOSITY_ENDPOINT_EXCLUDE_RADIUS_MM", None, None, pixels_per_mm, default_px=0.0)
    if endpoint_exclude_r is None or endpoint_exclude_r <= 0:
        # 缺省：按扫描带宽度的 0.3 比例，限制上限 15px，下限 3px
        endpoint_exclude_r = max(3, int(min(0.3 * scan_width, 15)))

    ignore_mask = np.zeros_like(roi_gray)
    if edge_ignore_px > 0.5:
        thickness_px = int(math.ceil(edge_ignore_px))
        # 安全约束：OpenCV 需要 thickness>=1；再限制一个上限防止异常配置
        if thickness_px < 1:
            thickness_px = 1
        if thickness_px > 256:
            thickness_px = 256
        cv2.line(ignore_mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=thickness_px)
    # 两端点圆形区域直接去除
    cv2.circle(ignore_mask, tuple(map(int, p1)), int(round(endpoint_exclude_r)), 255, thickness=-1)
    cv2.circle(ignore_mask, tuple(map(int, p2)), int(round(endpoint_exclude_r)), 255, thickness=-1)

    mean, std_dev = cv2.meanStdDev(roi_gray, mask=scan_mask)

    initial_contours = []
    if std_dev[0][0] > 3:
        threshold_low = mean[0][0] - p["LUMINOSITY_STD_DEV_MULTIPLIER"] * std_dev[0][0]
        potential_defects = (roi_gray < threshold_low).astype(np.uint8) * 255
        defect_mask = cv2.bitwise_and(potential_defects, scan_mask)
        # 去除边缘线和端点区域
        defect_mask = cv2.subtract(defect_mask, ignore_mask)

        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            min_gradient_threshold = p.get("LUMINOSITY_MIN_GRADIENT", 15.0)

            blurred = cv2.medianBlur(roi_gray, 3)
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            min_area_px2 = _get_area_px2(p, "LUMINOSITY_MIN_AREA_MM2", "LUMINOSITY_MIN_AREA", None, pixels_per_mm)
            for cnt in contours:
                if cv2.contourArea(cnt) > min_area_px2:
                    contour_mask = np.zeros_like(roi_gray)
                    cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                    mean_grad_val = cv2.mean(grad_mag, mask=contour_mask)[0]
                    if mean_grad_val > min_gradient_threshold:
                        initial_contours.append(cnt)
    
    return initial_contours


def scan_edge_for_chipping_blocks(roi_gray, edge, params, pixels_per_mm: float):
    """块状(按长度5mm或配置)统计对比的崩边检测。
    思路:
      1. 沿主边线构建一条贴边扫描带(宽度 BLOCK_BAND_WIDTH_MM)。
      2. 按 BLOCK_LENGTH_MM(默认5mm)切分为若干长条块(矩形)。
      3. 对每块计算特征: 灰度均值、灰度标准差、平均梯度幅值、边缘密度(Canny)。
      4. 以每种特征的中位数+MAD(中位绝对偏差)做鲁棒离群检测；任一特征偏离超过对应倍数则判定为异常块。
      5. 将连续异常块合并为更长矩形区域，输出其多边形轮廓。
    参数(DEFECT_DETECTION下可配置):
      BLOCK_BASED_CHIPPING_ENABLED: 开关 (bool)
      BLOCK_LENGTH_MM: 块长度(默认5.0mm)
      BLOCK_BAND_WIDTH_MM: 扫描带宽度(默认复用 LUMINOSITY_SCAN_WIDTH_MM)
      BLOCK_MEAN_MAD_K: 灰度均值离群倍数(默认3.0)
      BLOCK_STD_MAD_K: 灰度标准差离群倍数(默认3.0)
      BLOCK_GRAD_MAD_K: 梯度幅值离群倍数(默认2.5)
      BLOCK_EDGE_DENSITY_MAD_K: 边缘密度离群倍数(默认2.5)
      BLOCK_MIN_CONSECUTIVE: 合并时最少连续块数(默认1)
    返回: list[np.ndarray] 轮廓列表
    """
    p_def = params.get("DEFECT_DETECTION", {})
    p1 = np.array(edge[:2], dtype=float); p2 = np.array(edge[2:], dtype=float)
    line_vec = p2 - p1; line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-6:
        return []
    unit_vec = line_vec / line_len
    normal_vec = np.array([-unit_vec[1], unit_vec[0]], dtype=float)
    normal_angle_deg = float((np.degrees(np.arctan2(normal_vec[1], normal_vec[0])) + 180.0) % 180.0)
    # 宽度: 若无单独配置则复用亮度法的扫描带宽度
    scan_width_px = _get_dist_px(p_def, "BLOCK_BAND_WIDTH_MM", "BLOCK_BAND_WIDTH", None, pixels_per_mm)
    if scan_width_px is None or scan_width_px <= 0:
        scan_width_px = _get_dist_px(p_def, "LUMINOSITY_SCAN_WIDTH_MM", "LUMINOSITY_SCAN_WIDTH", None, pixels_per_mm)
    scan_width_px = max(2.0, float(scan_width_px))
    half_width_vec = (scan_width_px / 2.0) * np.array([-unit_vec[1], unit_vec[0]])
    # 块长度
    block_len_px = _get_dist_px(p_def, "BLOCK_LENGTH_MM", "BLOCK_LENGTH", None, pixels_per_mm)
    if block_len_px is None or block_len_px <= 0:
        block_len_px = 5.0 * float(pixels_per_mm if pixels_per_mm else 1.0)
    block_len_px = max(4.0, float(block_len_px))
    # 至少需要两个块才有意义(否则无法形成“异常”)；不足则返回空
    n_blocks = int(math.floor(line_len / block_len_px))
    if n_blocks < 2:
        return []
    # 预计算梯度与 Canny
    try:
        blurred = cv2.medianBlur(roi_gray, 3)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        edge_img = cv2.Canny(blurred, 50, 150)
    except Exception:
        grad_x = np.zeros_like(roi_gray, dtype=np.float32)
        grad_y = np.zeros_like(roi_gray, dtype=np.float32)
        grad_mag = np.zeros_like(roi_gray, dtype=np.float32)
        edge_img = np.zeros_like(roi_gray, dtype=np.uint8)
    # 可选：自适应边缘屏蔽（复用亮度法的动态忽略宽度+端点剔除）
    use_edge_ignore = bool(p_def.get('BLOCK_USE_EDGE_IGNORE', True))
    ignore_mask = np.zeros_like(roi_gray, dtype=np.uint8)
    if use_edge_ignore:
        # 动态忽略宽度：基于边中点到全图中心距离（px -> mm，0px->0mm, 2000px->4mm）
        edge_mid = (p1 + p2) / 2.0
        frame_w = float(params.get('FRAME_WIDTH', roi_gray.shape[1]))
        frame_h = float(params.get('FRAME_HEIGHT', roi_gray.shape[0]))
        off_x = float(params.get('ROI_OFFSET_X', 0.0))
        off_y = float(params.get('ROI_OFFSET_Y', 0.0))
        edge_mid_global = edge_mid + np.array([off_x, off_y], dtype=float)
        global_center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=float)
        dist_px_dynamic = float(np.linalg.norm(edge_mid_global - global_center))
        ignore_width_mm = min(4.0, max(0.0, dist_px_dynamic * 0.002))
        if pixels_per_mm and pixels_per_mm > 0:
            edge_ignore_px = ignore_width_mm * pixels_per_mm
        else:
            edge_ignore_px = ignore_width_mm
        # 允许通过 OVERRIDE_LUMINOSITY_EDGE_IGNORE_MM 覆盖
        try:
            override_mm = params.get('DEFECT_DETECTION', {}).get('OVERRIDE_LUMINOSITY_EDGE_IGNORE_MM', None)
            if override_mm is not None:
                ov_mm = float(override_mm)
                if ov_mm >= 0:
                    edge_ignore_px = ov_mm * (pixels_per_mm if pixels_per_mm else 1.0)
        except Exception:
            pass
        endpoint_exclude_r = _get_dist_px(p_def, "LUMINOSITY_ENDPOINT_EXCLUDE_RADIUS_MM", None, None, pixels_per_mm, default_px=0.0)
        if endpoint_exclude_r is None or endpoint_exclude_r <= 0:
            # 缺省：按扫描带宽度的0.3比例，限制上限15px，下限3px
            endpoint_exclude_r = max(3, int(min(0.3 * scan_width_px, 15)))
        # 画忽略线与端点圆
        if edge_ignore_px > 0.5:
            thickness_px = int(math.ceil(edge_ignore_px))
            if thickness_px < 1: thickness_px = 1
            if thickness_px > 256: thickness_px = 256
            cv2.line(ignore_mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=thickness_px)
        cv2.circle(ignore_mask, tuple(map(int, p1)), int(round(endpoint_exclude_r)), 255, thickness=-1)
        cv2.circle(ignore_mask, tuple(map(int, p2)), int(round(endpoint_exclude_r)), 255, thickness=-1)

    features = []  # 每块: {idx, mean, std, grad, edge_density, poly, dir_ratio}
    # 构建块并提取特征
    for bi in range(n_blocks):
        s = bi * block_len_px
        e = s + block_len_px
        if e > line_len:
            e = line_len
        seg_p1 = p1 + unit_vec * s
        seg_p2 = p1 + unit_vec * e
        # 矩形四点: seg_p1±half_width_vec, seg_p2±half_width_vec
        q1 = seg_p1 + half_width_vec
        q2 = seg_p2 + half_width_vec
        q3 = seg_p2 - half_width_vec
        q4 = seg_p1 - half_width_vec
        poly = np.array([q1, q2, q3, q4], dtype=np.int32).reshape((-1, 1, 2))
        # ROI 裁剪掩膜
        mask = np.zeros_like(roi_gray, dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        if use_edge_ignore:
            # 在块统计前剔除贴边忽略区和端点区域
            mask = cv2.subtract(mask, ignore_mask)
        # 特征计算
        mean_val = cv2.mean(roi_gray, mask=mask)[0]
        # 标准差: 使用 masked 像素
        pixels = roi_gray[mask == 255]
        if pixels.size < 4:
            continue
        std_val = float(np.std(pixels))
        grad_mean = cv2.mean(grad_mag, mask=mask)[0]
        edge_roi = edge_img[mask == 255]
        edge_density = float(np.count_nonzero(edge_roi)) / float(edge_roi.size if edge_roi.size else 1.0)
        # 结构一致性：梯度方向是否与边法线一致
        dir_ratio = 0.0
        try:
            # 仅统计 mask 内且为边缘像素的位置
            h_, w_ = roi_gray.shape
            edge_positions = np.where((mask == 255) & (edge_img > 0))
            cnt = int(edge_positions[0].size)
            if cnt >= 10:
                gy = grad_y[edge_positions].astype(np.float32)
                gx = grad_x[edge_positions].astype(np.float32)
                ang = np.degrees(np.arctan2(gy, gx))
                ang = np.mod(ang + 180.0, 180.0)  # 0..180
                diff = np.abs(ang - normal_angle_deg)
                diff = np.minimum(diff, 180.0 - diff)
                tol = float(p_def.get('BLOCK_GRAD_DIR_TOL_DEG', 35.0))
                dir_ratio = float(np.mean(diff <= tol))
        except Exception:
            dir_ratio = 0.0

        features.append({"idx": bi, "mean": mean_val, "std": std_val, "grad": grad_mean, "edge_density": edge_density, "poly": poly, "dir_ratio": dir_ratio})
    if len(features) < 2:
        return []
    def _median(values):
        return float(np.median(np.array(values, dtype=float)))
    def _mad(values, med):
        v = np.abs(np.array(values, dtype=float) - med)
        m = np.median(v)
        return float(m if m > 1e-6 else 1e-6)
    means = [f["mean"] for f in features]; m_mean = _median(means); mad_mean = _mad(means, m_mean)
    stds = [f["std"] for f in features]; m_std = _median(stds); mad_std = _mad(stds, m_std)
    grads = [f["grad"] for f in features]; m_grad = _median(grads); mad_grad = _mad(grads, m_grad)
    eds = [f["edge_density"] for f in features]; m_ed = _median(eds); mad_ed = _mad(eds, m_ed)
    k_mean = float(p_def.get("BLOCK_MEAN_MAD_K", 3.0))
    k_std = float(p_def.get("BLOCK_STD_MAD_K", 3.0))
    k_grad = float(p_def.get("BLOCK_GRAD_MAD_K", 2.5))
    k_ed = float(p_def.get("BLOCK_EDGE_DENSITY_MAD_K", 2.5))
    structural_mode = bool(p_def.get("BLOCK_STRUCTURAL_MODE", False))
    # 结构优先模式：忽略亮度均值/标准差离群，只看梯度与边缘密度（或再加最少有效像素比例）
    min_effective_ratio = float(p_def.get("BLOCK_MIN_EFFECTIVE_PIXEL_RATIO", 0.3)) if structural_mode else 0.0
    grad_dir_min_ratio = float(p_def.get("BLOCK_GRAD_DIR_MIN_RATIO", 0.5)) if structural_mode else 0.0
    # 至少需要同时满足一定数量特征的离群判定，才能认为该块异常
    min_feat_outliers = int(p_def.get("BLOCK_MIN_FEATURE_OUTLIERS", 2))
    abnormal_indices = []
    for f in features:
        dev_mean = abs(f["mean"] - m_mean) / mad_mean
        dev_std = abs(f["std"] - m_std) / mad_std
        dev_grad = abs(f["grad"] - m_grad) / mad_grad
        dev_ed = abs(f["edge_density"] - m_ed) / mad_ed
        outlier_count = 0
        if structural_mode:
            # 计算该块有效像素比例(用于剔除被屏蔽后过窄的块)
            # poly 掩膜已经减去 ignore_mask 后统计的 pixels.size
            effective_count = int(np.count_nonzero(roi_gray[f["poly"].reshape(-1,2)[:,1].clip(0,roi_gray.shape[0]-1), f["poly"].reshape(-1,2)[:,0].clip(0,roi_gray.shape[1]-1)])) if False else None
            # 简化：直接用特征计算阶段的像素数与理论块面积估计比例
            # 理论面积 ~ block_len_px * scan_width_px；像素数可用 std 计算时的 pixels.size
            # (此处不重新计算，保守跳过最少比例判断，后续可改进为传入 pixels.size 和面积)
            if dev_grad > k_grad: outlier_count += 1
            if dev_ed > k_ed: outlier_count += 1
            # 方向一致性约束：需要一定比例的边缘梯度方向与边法线一致
            if f.get('dir_ratio', 0.0) < grad_dir_min_ratio:
                outlier_count = -9999  # 强制不达标
        else:
            if dev_mean > k_mean: outlier_count += 1
            if dev_std > k_std: outlier_count += 1
            if dev_grad > k_grad: outlier_count += 1
            if dev_ed > k_ed: outlier_count += 1
        if outlier_count >= max(1, min_feat_outliers):
            abnormal_indices.append(f["idx"])
    if not abnormal_indices:
        return []
    # 合并连续块
    abnormal_indices.sort()
    min_consecutive = int(p_def.get("BLOCK_MIN_CONSECUTIVE", 1))
    merged_polys = []
    group_start = None; prev = None
    for idx in abnormal_indices:
        if group_start is None:
            group_start = idx; prev = idx; continue
        if idx == prev + 1:
            prev = idx; continue
        # 结束前一组
        if prev - group_start + 1 >= min_consecutive:
            merged_polys.append((group_start, prev))
        group_start = idx; prev = idx
    if group_start is not None and prev is not None and prev - group_start + 1 >= min_consecutive:
        merged_polys.append((group_start, prev))
    contours_out = []
    for (gs, ge) in merged_polys:
        s = gs * block_len_px
        e = (ge + 1) * block_len_px
        if e > line_len:
            e = line_len
        seg_p1 = p1 + unit_vec * s
        seg_p2 = p1 + unit_vec * e
        q1 = seg_p1 + half_width_vec
        q2 = seg_p2 + half_width_vec
        q3 = seg_p2 - half_width_vec
        q4 = seg_p1 - half_width_vec
        poly = np.array([q1, q2, q3, q4], dtype=np.int32)
        contours_out.append(poly.reshape((-1, 1, 2)))
    return contours_out


def find_gradient_endpoint(start_point, line_vec_normalized, roi_gray_blurred, max_search_dist, search_width=2, gradient_stop_threshold=20.0):
    h, w = roi_gray_blurred.shape
    perp_vec = np.array([-line_vec_normalized[1], line_vec_normalized[0]])
    half_width = (search_width - 1) // 2

    last_avg_intensity = -1.0
    for i in range(3, int(max_search_dist)):
        current_center = start_point + i * line_vec_normalized
        
        if not (0 <= current_center[0] < w and 0 <= current_center[1] < h):
            break

        sample_points_coords = [current_center + j * perp_vec for j in range(-half_width, half_width + 1)]
        
        intensities = []
        valid_coords = []
        for p in sample_points_coords:
            px, py = int(p[0]), int(p[1])
            if 0 <= px < w and 0 <= py < h:
                intensities.append(roi_gray_blurred[py, px])
        
        if not intensities:
            continue

        current_avg_intensity = np.mean(intensities)

        if last_avg_intensity >= 0:
            grad = abs(current_avg_intensity - last_avg_intensity)
            
            if grad > gradient_stop_threshold:
                return np.mean(valid_coords, axis=0)
        
        last_avg_intensity = current_avg_intensity

    return None

# ====================================================================================
# --- 核心处理流程 ---
# ====================================================================================
def preprocess_for_hough_enhanced(roi_gray, params):
    p = params["PREPROCESSING"]
    grid_size = tuple(p.get("CLAHE_GRID_SIZE", [8, 8]))
    # 取消 Otsu：直接对原始灰度进行中值滤波 + CLAHE，再做 Canny
    blurred = cv2.medianBlur(roi_gray, p["MEDIAN_BLUR_KSIZE"])
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP_LIMIT"], tileGridSize=grid_size)
    enhanced_contrast = clahe.apply(blurred)
    edges = cv2.Canny(enhanced_contrast, p["CANNY_THRESHOLD_LOW"], p["CANNY_THRESHOLD_HIGH"])
    return denoise_edge_map(edges, params)

def denoise_edge_map(edge_img: np.ndarray, params) -> np.ndarray:
    """快速对 Canny 边缘图进行降噪：移除孤立像素与极小簇，减少离群点对后续检测干扰。
    可配置 DEFECT_DETECTION 下参数：
      CANNY_DENOISE_ENABLE (bool, 默认 True)
      CANNY_DENOISE_MIN_NEIGHBORS (含自身的3x3邻域最少白点数, 默认 2)
      CANNY_DENOISE_MIN_CLUSTER_PIXELS (连通域像素下限, 默认 5)
      CANNY_DENOISE_MAX_ITER (最大迭代次数, 默认 1)
    过程：
      1) 3x3 邻域计数过滤孤立或过稀疏的点。
      2) 连通域分析剔除过小簇。
      3) 可迭代一次（避免频繁重计算）。
    保持速度：全部操作在二值图上，卷积与 connectedComponentsO(像素数)。"""
    try:
        pdef = params.get('DEFECT_DETECTION', {})
        if not bool(pdef.get('CANNY_DENOISE_ENABLE', True)):
            return edge_img
        min_neighbors = int(pdef.get('CANNY_DENOISE_MIN_NEIGHBORS', 2))  # 3x3包含自身的白点数阈值
        min_cluster = int(pdef.get('CANNY_DENOISE_MIN_CLUSTER_PIXELS', 15))
        max_iter = int(pdef.get('CANNY_DENOISE_MAX_ITER', 1))
    except Exception:
        return edge_img

    if edge_img is None or edge_img.size == 0:
        return edge_img
    # 转为二值（0/1）
    bin_img = (edge_img > 0).astype(np.uint8)
    H, W = bin_img.shape[:2]
    if H*W <= 0:
        return edge_img
    kernel3 = np.ones((3,3), dtype=np.uint8)
    work = bin_img.copy()
    for _ in range(max(1, max_iter)):
        # 邻域计数：保留计数>min_neighbors的点
        neigh_counts = cv2.filter2D(work, -1, kernel3, borderType=cv2.BORDER_CONSTANT)
        work = ((neigh_counts > min_neighbors) & (work > 0)).astype(np.uint8)
        # 连通域剔除小簇
        try:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
            if num_labels > 1:
                mask_keep = np.zeros_like(work)
                for lbl in range(1, num_labels):
                    area = int(stats[lbl, cv2.CC_STAT_AREA])
                    if area >= min_cluster:
                        mask_keep[labels == lbl] = 1
                work = mask_keep
        except Exception:
            pass
    # 恢复到 0/255 形式
    out = (work * 255).astype(np.uint8)
    return out


def _snap_line_to_canny(seg: np.ndarray, edge_img: np.ndarray, stripe_half_px: int,
                        min_points: int, max_angle_deg: float) -> np.ndarray:
    """将合并后的直线在 Canny 边缘图上进行细调，减少角度与位置偏移。"""
    if edge_img is None or edge_img.size == 0:
        return seg
    h_img, w_img = edge_img.shape[:2]
    if h_img <= 0 or w_img <= 0:
        return seg
    if edge_img.dtype != np.uint8:
        edge_bin = (edge_img > 0).astype(np.uint8)
    else:
        edge_bin = edge_img
    thickness = max(1, int(stripe_half_px) * 2 + 1)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    p1 = (int(round(float(seg[0]))), int(round(float(seg[1]))))
    p2 = (int(round(float(seg[2]))), int(round(float(seg[3]))))
    cv2.line(mask, p1, p2, 255, thickness=thickness)
    overlap = cv2.bitwise_and(edge_bin, edge_bin, mask=mask)
    ys, xs = np.where(overlap > 0)
    if xs.size < max(2, int(min_points)):
        return seg
    pts = np.column_stack((xs, ys)).astype(np.float32)
    try:
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    except Exception:
        return seg
    norm = math.hypot(float(vx), float(vy))
    if norm < 1e-6:
        return seg
    vx = float(vx) / norm
    vy = float(vy) / norm
    angle_old = math.degrees(math.atan2(float(seg[3]) - float(seg[1]), float(seg[2]) - float(seg[0])))
    if angle_old < 0:
        angle_old += 180.0
    angle_new = math.degrees(math.atan2(vy, vx))
    if angle_new < 0:
        angle_new += 180.0
    angle_diff = abs(angle_new - angle_old)
    if angle_diff > 90.0:
        angle_diff = 180.0 - angle_diff
    if angle_diff > float(max_angle_deg):
        return seg
    base = np.array([float(x0), float(y0)], dtype=float)
    direction = np.array([vx, vy], dtype=float)

    def _project(pt: np.ndarray) -> float:
        return float(np.dot(pt - base, direction))

    p1_arr = np.array(seg[:2], dtype=float)
    p2_arr = np.array(seg[2:], dtype=float)
    t1 = _project(p1_arr)
    t2 = _project(p2_arr)
    t_min = min(t1, t2)
    t_max = max(t1, t2)
    new_p1 = base + t_min * direction
    new_p2 = base + t_max * direction
    new_seg = np.array([new_p1[0], new_p1[1], new_p2[0], new_p2[1]], dtype=float)
    new_seg[0] = float(np.clip(new_seg[0], 0.0, w_img - 1.0))
    new_seg[1] = float(np.clip(new_seg[1], 0.0, h_img - 1.0))
    new_seg[2] = float(np.clip(new_seg[2], 0.0, w_img - 1.0))
    new_seg[3] = float(np.clip(new_seg[3], 0.0, h_img - 1.0))
    return new_seg


def _fit_line_from_edges(edge_img: np.ndarray, seg: np.ndarray, band_half: int,
                         min_points: int, max_angle_dev_deg: float) -> np.ndarray|None:
    """在水平段附近用 Canny 点拟合直线，允许轻微斜率以贴合真实边缘。"""
    if edge_img is None or edge_img.size == 0:
        return None
    H, W = edge_img.shape[:2]
    band_half = max(1, int(band_half))
    x1, y1, x2, y2 = map(float, seg)
    x_min = int(max(0, math.floor(min(x1, x2))))
    x_max = int(min(W - 1, math.ceil(max(x1, x2))))
    y_med = int(round(0.5 * (y1 + y2)))
    y0 = max(0, y_med - band_half)
    y1b = min(H - 1, y_med + band_half)
    if x_max - x_min < 1 or y1b - y0 < 1:
        return None
    roi = edge_img[y0:y1b+1, x_min:x_max+1]
    ys, xs = np.where(roi > 0)
    if xs.size < max(2, int(min_points)):
        return None
    xs = xs.astype(np.float32) + float(x_min)
    ys = ys.astype(np.float32) + float(y0)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    try:
        vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    except Exception:
        return None
    norm = math.hypot(float(vx), float(vy))
    if norm < 1e-6:
        return None
    vx = float(vx) / norm; vy = float(vy) / norm
    ang = abs(math.degrees(math.atan2(vy, vx)))
    if ang > 90.0:
        ang = 180.0 - ang
    if ang > float(max_angle_dev_deg):
        return None
    base = np.array([float(cx), float(cy)], dtype=float)
    direction = np.array([vx, vy], dtype=float)
    t_vals = np.dot(pts - base, direction)
    t_min = float(np.min(t_vals)); t_max = float(np.max(t_vals))
    p1n = base + t_min * direction
    p2n = base + t_max * direction
    seg_new = np.array([p1n[0], p1n[1], p2n[0], p2n[1]], dtype=float)
    seg_new[0] = float(np.clip(seg_new[0], 0.0, W - 1.0))
    seg_new[1] = float(np.clip(seg_new[1], 0.0, H - 1.0))
    seg_new[2] = float(np.clip(seg_new[2], 0.0, W - 1.0))
    seg_new[3] = float(np.clip(seg_new[3], 0.0, H - 1.0))
    return seg_new


def _detect_edge_notches(edge_img: np.ndarray, segments: list, pixels_per_mm: float, params: dict) -> list:
    """在主边上查找较长的“凹进/缺段”并标记为缺角(Q)。"""
    if edge_img is None or edge_img.size == 0 or not segments:
        return []
    try:
        stripe_half = int(params.get('DEFECT_DETECTION', {}).get('NOTCH_STRIPE_HALF_PX', 3))
    except Exception:
        stripe_half = 3
    try:
        min_gap = int(params.get('DEFECT_DETECTION', {}).get('NOTCH_MIN_GAP_PX', 12))
    except Exception:
        min_gap = 12
    stripe_half = max(1, int(stripe_half))
    min_gap = max(4, int(min_gap))
    H, W = edge_img.shape[:2]
    defects = []

    def _runs_of_false(mask: np.ndarray):
        # mask: 1d bool
        if mask.size == 0:
            return []
        diff = np.diff(mask.astype(np.int8))
        run_starts = list(np.where(diff == -1)[0] + 1) if mask[0] else [0] + list(np.where(diff == -1)[0] + 1)
        run_ends = list(np.where(diff == 1)[0] + 1) if not mask[-1] else list(np.where(diff == 1)[0] + 1) + [mask.size]
        if len(run_starts) != len(run_ends):
            m = min(len(run_starts), len(run_ends))
            run_starts = run_starts[:m]; run_ends = run_ends[:m]
        return [(s, e) for s, e in zip(run_starts, run_ends) if e > s]

    for seg in segments:
        try:
            x1,y1,x2,y2 = map(float, seg)
            dx, dy = x2 - x1, y2 - y1
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            if ang > 90.0:
                ang = 180.0 - ang
            near_vert = ang >= 45.0
            if near_vert:
                y0 = int(max(0, math.floor(min(y1, y2))))
                y1i = int(min(H - 1, math.ceil(max(y1, y2))))
                if y1i - y0 + 1 < min_gap:
                    continue
                x_med = int(round((x1 + x2) * 0.5))
                x0 = max(0, x_med - stripe_half); x1c = min(W - 1, x_med + stripe_half)
                stripe = edge_img[y0:y1i+1, x0:x1c+1]
                if stripe.size == 0:
                    continue
                presence = np.any(stripe > 0, axis=1)
                runs = _runs_of_false(presence)
                for s, e in runs:
                    gap_len = e - s
                    if gap_len < min_gap:
                        continue
                    if presence[:s].any() and presence[e:].any():
                        ys0 = y0 + s; ys1 = y0 + e
                        cx = 0.5 * (x0 + x1c)
                        cy = 0.5 * (ys0 + ys1)
                        w_px = float(x1c - x0 + 1)
                        h_px = float(gap_len)
                        rect = ((cx, cy), (w_px, h_px), 0.0)
                        box = cv2.boxPoints(rect)
                        defects.append({
                            'type': 'Q',
                            'origin': 'notch_vertical_gap',
                            'min_area_rect': rect,
                            'box_points': np.int32(box),
                            'center': (int(round(cx)), int(round(cy))),
                            'length_mm': h_px / float(pixels_per_mm) if pixels_per_mm else 0.0,
                            'width_mm': w_px / float(pixels_per_mm) if pixels_per_mm else 0.0,
                            'notch_run_px': int(gap_len)
                        })
            else:
                x0i = int(max(0, math.floor(min(x1, x2))))
                x1i = int(min(W - 1, math.ceil(max(x1, x2))))
                if x1i - x0i + 1 < min_gap:
                    continue
                y_med = int(round((y1 + y2) * 0.5))
                y0c = max(0, y_med - stripe_half); y1c = min(H - 1, y_med + stripe_half)
                stripe = edge_img[y0c:y1c+1, x0i:x1i+1]
                if stripe.size == 0:
                    continue
                presence = np.any(stripe > 0, axis=0)
                runs = _runs_of_false(presence)
                for s, e in runs:
                    gap_len = e - s
                    if gap_len < min_gap:
                        continue
                    if presence[:s].any() and presence[e:].any():
                        xs0 = x0i + s; xs1 = x0i + e
                        cx = 0.5 * (xs0 + xs1)
                        cy = 0.5 * (y0c + y1c)
                        w_px = float(gap_len)
                        h_px = float(y1c - y0c + 1)
                        rect = ((cx, cy), (w_px, h_px), 0.0)
                        box = cv2.boxPoints(rect)
                        defects.append({
                            'type': 'Q',
                            'origin': 'notch_horizontal_gap',
                            'min_area_rect': rect,
                            'box_points': np.int32(box),
                            'center': (int(round(cx)), int(round(cy))),
                            'length_mm': w_px / float(pixels_per_mm) if pixels_per_mm else 0.0,
                            'width_mm': h_px / float(pixels_per_mm) if pixels_per_mm else 0.0,
                            'notch_run_px': int(gap_len)
                        })
        except Exception:
            continue
    return defects

def merge_lines_and_get_main_edges(lines, params, pixels_per_mm: float, edge_img=None):
    if lines is None or len(lines) < 1: return []
    p = params["LINE_MERGING"]
    # 方向容忍（用于判断近竖直/近水平）
    try:
        v_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
    except Exception:
        v_tol_deg = 10.0
    try:
        h_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', v_tol_deg))
    except Exception:
        h_tol_deg = v_tol_deg

    # mm/px 合并与分离阈值（可按方向区分）
    def _px_dist_mm(cfg_key_mm: str, cfg_key_px: str, default_mm: float|None, default_px: float|None=None):
        return _get_dist_px(p, cfg_key_mm, cfg_key_px, default_mm, pixels_per_mm, default_px=default_px)

    # 竖直粗边“横向”合并范围（默认 10mm）
    vertical_thick_merge_px = _px_dist_mm('VERTICAL_THICK_MERGE_MM', 'VERTICAL_THICK_MERGE_PX', 10.0, default_px=0.0)
    # 自适应基于“横向厚度”的扩展系数（用于近竖直双线的合并），以及封顶
    try:
        vertical_thick_merge_scale = float(p.get('VERTICAL_THICK_MERGE_SCALE', 1.5))
    except Exception:
        vertical_thick_merge_scale = 1.5
    vertical_thick_merge_cap_px = _px_dist_mm('VERTICAL_THICK_MERGE_MAX_MM', 'VERTICAL_THICK_MERGE_MAX_PX', 30.0, default_px=0.0)
    # 近水平线：不对“轴向”距离做硬切（避免拆散同一条长水平边）；仅依赖横向距离与 Canny 缝隙检查
    # 近竖直线合并时允许的“轴向”最大间隙（默认 20mm）；便于将同一粗竖边上下段合并
    vertical_max_ax_gap_px = _px_dist_mm('VERTICAL_MAX_AXIAL_GAP_MM', 'VERTICAL_MAX_AXIAL_GAP', 20.0, default_px=40.0)
    # 轴向锁定/精修配置
    vertical_lock_axis = bool(p.get('VERTICAL_LOCK_AXIS', True))
    horizontal_lock_axis = bool(p.get('HORIZONTAL_LOCK_AXIS', True))
    try:
        canny_refine_half_px = int(params.get('LINE_MERGING', {}).get('CANNY_AXIS_REFINE_HALF_PX', 3))
    except Exception:
        canny_refine_half_px = 3
    try:
        snap_enable = bool(params.get('LINE_MERGING', {}).get('CANNY_SNAP_ENABLE', True))
    except Exception:
        snap_enable = True
    try:
        snap_half_px = int(params.get('LINE_MERGING', {}).get('CANNY_SNAP_HALF_STRIPE_PX', 4))
    except Exception:
        snap_half_px = 4
    try:
        snap_min_points = int(params.get('LINE_MERGING', {}).get('CANNY_SNAP_MIN_POINTS', 12))
    except Exception:
        snap_min_points = 12
    try:
        snap_max_angle = float(params.get('LINE_MERGING', {}).get('CANNY_SNAP_MAX_ANGLE_DIFF_DEG', 8.0))
    except Exception:
        snap_max_angle = 8.0
    try:
        horiz_fit_enable = bool(params.get('LINE_MERGING', {}).get('HORIZONTAL_LOCK_FIT_ENABLE', True))
    except Exception:
        horiz_fit_enable = True
    try:
        horiz_fit_half_px = int(params.get('LINE_MERGING', {}).get('HORIZONTAL_LOCK_FIT_HALF_PX', 4))
    except Exception:
        horiz_fit_half_px = 4
    try:
        horiz_fit_min_pts = int(params.get('LINE_MERGING', {}).get('HORIZONTAL_LOCK_FIT_MIN_POINTS', 18))
    except Exception:
        horiz_fit_min_pts = 18
    try:
        horiz_fit_max_angle = float(params.get('LINE_MERGING', {}).get('HORIZONTAL_LOCK_MAX_ANGLE_DEV_DEG', h_tol_deg))
    except Exception:
        horiz_fit_max_angle = h_tol_deg
    try:
        min_vertical_gap_px = float(params.get('LINE_MERGING', {}).get('MIN_VERTICAL_EDGE_GAP_PX', 10.0))
    except Exception:
        min_vertical_gap_px = 10.0
    merged_min_support_px = _px_dist_mm('MERGED_MIN_SUPPORT_MM', 'MERGED_MIN_SUPPORT_PX', 5.0, default_px=0.0)

    def _is_near_vertical(angle_deg: float) -> bool:
        a = angle_deg % 180.0
        return abs(90.0 - a) <= v_tol_deg

    def _is_near_horizontal(angle_deg: float) -> bool:
        a = angle_deg % 180.0
        return min(a, 180.0 - a) <= h_tol_deg
    lines_np = np.array(lines).reshape(-int(len(lines)), 4)
    angles = np.rad2deg(np.arctan2(lines_np[:, 3] - lines_np[:, 1], lines_np[:, 2] - lines_np[:, 0]))
    angles[angles < 0] += 180
    angle_clusters = {}
    for i, angle in enumerate(angles):
        placed = False
        for cluster_angle in angle_clusters:
            if min(abs(angle - cluster_angle), 180 - abs(angle - cluster_angle)) < p["ANGLE_TOLERANCE"]:
                angle_clusters[cluster_angle].append(lines_np[i]); placed = True; break
        if not placed: angle_clusters[angle] = [lines_np[i]]
    final_line_groups = []
    for angle, segments in angle_clusters.items():
        if not segments: continue
        segments.sort(key=lambda s: np.linalg.norm(s[2:4] - s[0:2]), reverse=True)
        proximity_groups = [ [segments.pop(0)] ] if segments else []
        for segment in segments:
            mid_point = np.array([(segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2])
            placed = False
            for group in proximity_groups:
                ref_line = group[0]; p1, p2 = ref_line[0:2], ref_line[2:4]
                vec_line = p2 - p1; line_length = np.linalg.norm(vec_line)
                if line_length > 1e-6:
                    # 使用统一的“点到直线的垂直距离”作为聚类准则，阈值仅依赖 MAX_LATERAL_DISTANCE[_MM]
                    max_lat_dist_px = _get_dist_px(p, "MAX_LATERAL_DISTANCE_MM", "MAX_LATERAL_DISTANCE", None, pixels_per_mm)
                    vec2 = p1 - mid_point
                    cross_product_2d = vec_line[0] * vec2[1] - vec_line[1] * vec2[0]
                    dist_val = abs(cross_product_2d) / line_length
                    if dist_val < max_lat_dist_px:
                        # 基于 Canny 的“缝隙”检查：若沿法线方向存在足够长的无边缘像素区间，则不合并
                        allowed_merge = True
                        # 新增：按方向增加“轴向间隙”约束，避免远距离共线被合并（特别是水平边）
                        try:
                            u = (vec_line / line_length).astype(float)
                            # 计算该 group 当前所有成员在参考线方向上的投影范围
                            g_points = np.array([pt for ln in group for pt in (ln[0:2], ln[2:4])], dtype=float)
                            g_proj = (g_points[:, 0] - p1[0]) * u[0] + (g_points[:, 1] - p1[1]) * u[1]
                            g_min = float(np.min(g_proj)); g_max = float(np.max(g_proj))
                            t_proj = float(np.dot((mid_point - p1), u))
                            if _is_near_vertical(angle):
                                # 竖直边：允许更大的轴向连接（将同一粗竖边上下段拼接）
                                if (t_proj < g_min - vertical_max_ax_gap_px) or (t_proj > g_max + vertical_max_ax_gap_px):
                                    allowed_merge = False
                        except Exception:
                            pass
                        if edge_img is not None:
                                # 参考线单位方向 u 及其法线方向 w（从垂足到候选中点）
                                u = (vec_line / line_length).astype(float)
                                # 候选中点在参考线上的垂足 p_perp
                                t_proj = float(np.dot(mid_point - p1, u))
                                p_perp = p1 + t_proj * u
                                d_vec = mid_point - p_perp
                                gap_len = float(np.linalg.norm(d_vec))
                                if gap_len > 1.0:
                                    steps = int(np.ceil(gap_len))
                                    step_vec = d_vec / steps
                                    # 通道半宽复用现有参数（像素）
                                    stripe_half = int(max(1, int(params.get('DEFECT_DETECTION', {}).get('Q_CANNY_STRIPE_HALF_WIDTH_PX', 2))))
                                    GAP_NO_EDGE_PX = 30  # 硬编码：最大连续无边缘长度阈值
                                    max_run = 0
                                    run = 0
                                    h, w_img = edge_img.shape[:2]
                                    for i in range(steps + 1):
                                        c = p_perp + step_vec * i
                                        cx = int(round(float(c[0])))
                                        cy = int(round(float(c[1])))
                                        hit = False
                                        x0 = max(0, cx - stripe_half); x1 = min(w_img - 1, cx + stripe_half)
                                        y0 = max(0, cy - stripe_half); y1 = min(h - 1, cy + stripe_half)
                                        if x0 <= x1 and y0 <= y1:
                                            roi = edge_img[y0:y1+1, x0:x1+1]
                                            # 任一像素有边缘即视为命中
                                            if np.any(roi > 0):
                                                hit = True
                                        if hit:
                                            run = 0
                                        else:
                                            run += 1
                                            if run > max_run:
                                                max_run = run
                                                if max_run >= GAP_NO_EDGE_PX:
                                                    allowed_merge = False
                                                    break
                        # 若过程中任何异常，保持 allowed_merge 默认值（True）
                        if allowed_merge:
                            group.append(segment); placed = True; break
            if not placed: proximity_groups.append([segment])
        # 针对“竖直粗边”在小横向距离内做二次合并（把双线/多线合成单线）
        if _is_near_vertical(angle) and len(proximity_groups) > 1 and vertical_thick_merge_px and vertical_thick_merge_px > 0:
            def _group_stats(g):
                pts = np.array([pt for ln in g for pt in (ln[0:2], ln[2:4])], dtype=float)
                xs = (pts[:,0])
                ys = (pts[:,1])
                x_rep = float(np.median(xs))
                # 以 MAD 近似厚度（横向散布），稳健性较强
                x_mad = float(np.median(np.abs(xs - x_rep))) * 2.0
                y_min = float(np.min(ys)); y_max = float(np.max(ys))
                return x_rep, x_mad, y_min, y_max
            used = [False]*len(proximity_groups)
            merged_groups = []
            for i_gp in range(len(proximity_groups)):
                if used[i_gp]:
                    continue
                xi, xi_mad, yi0, yi1 = _group_stats(proximity_groups[i_gp])
                cur = list(proximity_groups[i_gp])
                used[i_gp] = True
                for j_gp in range(i_gp+1, len(proximity_groups)):
                    if used[j_gp]:
                        continue
                    xj, xj_mad, yj0, yj1 = _group_stats(proximity_groups[j_gp])
                    # 动态允许的横向阈值：max(静态阈值, scale*(xi_mad+xj_mad))，并受上限cap约束
                    dyn_lat_allow = max(float(vertical_thick_merge_px), vertical_thick_merge_scale * (xi_mad + xj_mad))
                    if vertical_thick_merge_cap_px and vertical_thick_merge_cap_px > 0:
                        dyn_lat_allow = min(dyn_lat_allow, float(vertical_thick_merge_cap_px))
                    # 横向距离小于合并阈值，且在竖直方向上有足够重叠
                    if abs(xi - xj) <= dyn_lat_allow:
                        overlap = max(0.0, min(yi1, yj1) - max(yi0, yj0))
                        span = max(1.0, max(yi1, yj1) - min(yi0, yj0))
                        if (overlap / span) >= float(p.get('VERTICAL_THICK_MIN_OVERLAP_RATIO', 0.2)):
                            cur.extend(proximity_groups[j_gp])
                            used[j_gp] = True
                            # 合并后更新代表统计，便于继续吞并更多相邻组（提高合并完整度）
                            xi, xi_mad, yi0, yi1 = _group_stats(cur)
                merged_groups.append(cur)
            final_line_groups.extend(merged_groups)
        else:
            final_line_groups.extend(proximity_groups)
    merged_lines_with_scores = []
    for group in final_line_groups:
        points = np.array([pt for line in group for pt in (line[0:2], line[2:4])], dtype=float)
        if points.shape[0] < 2:
            continue
        # 先用 fitLine 粗估方向
        try:
            line_params = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = map(float, line_params.flatten())
        except Exception:
            # 回退：用端点 PCA
            cov = np.cov(points.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            idx = int(np.argmax(eigvals))
            vx, vy = map(float, eigvecs[:, idx])
            x0, y0 = map(float, np.mean(points, axis=0))

        ang = abs(np.degrees(np.arctan2(vy, vx)))
        if ang > 90.0:
            ang = 180.0 - ang

        # 默认用投影端点
        projected = (points[:, 0] - x0) * vx + (points[:, 1] - y0) * vy
        p_min = points[int(np.argmin(projected))]; p_max = points[int(np.argmax(projected))]
        final_merged_line = np.array([p_min[0], p_min[1], p_max[0], p_max[1]], dtype=float)

        # 轴向锁定 + Canny 细化：防止角度漂移、利用边缘图稳定位置
        if edge_img is not None:
            h_img, w_img = edge_img.shape[:2]
            if vertical_lock_axis and _is_near_vertical(ang):
                y0_seg = max(0, int(np.floor(points[:,1].min())))
                y1_seg = min(h_img-1, int(np.ceil(points[:,1].max())))
                # 以中位 x 为中心，在 ±canny_refine_half_px 内寻找边缘计数最大列
                x_med = int(round(float(np.median(points[:,0]))))
                cx_best = x_med; best_sum = -1
                for cx in range(max(0, x_med - canny_refine_half_px), min(w_img-1, x_med + canny_refine_half_px) + 1):
                    col_sum = int(np.count_nonzero(edge_img[y0_seg:y1_seg+1, cx]))
                    if col_sum > best_sum:
                        best_sum = col_sum; cx_best = cx
                final_merged_line = np.array([cx_best, y0_seg, cx_best, y1_seg], dtype=float)
            elif horizontal_lock_axis and _is_near_horizontal(ang):
                x0_seg = max(0, int(np.floor(points[:,0].min())))
                x1_seg = min(w_img-1, int(np.ceil(points[:,0].max())))
                y_med = int(round(float(np.median(points[:,1]))))
                # 先尝试基于 Canny 的线拟合，允许轻微斜率以紧贴真实边缘
                fitted = None
                if horiz_fit_enable:
                    fitted = _fit_line_from_edges(edge_img, np.array([x0_seg, y_med, x1_seg, y_med], dtype=float),
                                                  horiz_fit_half_px, horiz_fit_min_pts, horiz_fit_max_angle)
                if fitted is not None:
                    final_merged_line = fitted
                    cy_best = int(round(float((fitted[1] + fitted[3]) * 0.5)))
                else:
                    cy_best = y_med; best_sum = -1
                    for cy in range(max(0, y_med - canny_refine_half_px), min(h_img-1, y_med + canny_refine_half_px) + 1):
                        row_sum = int(np.count_nonzero(edge_img[cy, x0_seg:x1_seg+1]))
                        if row_sum > best_sum:
                            best_sum = row_sum; cy_best = cy
                    # 防止 cy_best 未命中任何行的极端情况
                    if cy_best is None:
                        cy_best = y_med
                    final_merged_line = np.array([x0_seg, cy_best, x1_seg, cy_best], dtype=float)
                cy_best = int(round(float(cy_best)))

                # ================= 新增：水平线段合并连接性校验 =================
                # 若本合并将多个原子线段跨越较大 gap，而 gap 区域缺乏 Canny 白点，则拆分回原 group（保留支持度最高的代表）
                try:
                    connectivity_enable = bool(params.get('LINE_MERGING', {}).get('HORIZONTAL_CONNECTIVITY_ENABLE', True))
                except Exception:
                    connectivity_enable = True
                if connectivity_enable and edge_img is not None and len(group) >= 2:
                    # 计算按投影排序后的段列表与相邻 gap
                    segs_proj = []
                    for ln in group:
                        x1l,y1l,x2l,y2l = map(float, ln)
                        xp_min = min(x1l,x2l); xp_max = max(x1l,x2l)
                        segs_proj.append((xp_min,xp_max, ln))
                    segs_proj.sort(key=lambda t: t[0])
                    # 参数：最小检查 gap 像素 & 允许的无边缘最大连续像素
                    try:
                        gap_min_px = float(params.get('LINE_MERGING', {}).get('HORIZ_CONNECT_MIN_GAP_PX', 8.0))
                    except Exception:
                        gap_min_px = 8.0
                    try:
                        gap_no_edge_allow = float(params.get('LINE_MERGING', {}).get('HORIZ_CONNECT_MAX_NO_EDGE_RUN_PX', 25.0))
                    except Exception:
                        gap_no_edge_allow = 25.0
                    try:
                        stripe_half = int(params.get('LINE_MERGING', {}).get('HORIZ_CONNECT_STRIPE_HALF_PX', 2))
                    except Exception:
                        stripe_half = 2
                    connectivity_ok = True
                    for k in range(len(segs_proj)-1):
                        a0,a1,_ = segs_proj[k]
                        b0,b1,_ = segs_proj[k+1]
                        gap_len = b0 - a1
                        if gap_len < gap_min_px:
                            continue
                        # 在 [a1, b0] 区间采样，统计是否存在足够的 Canny 边缘
                        steps = int(max(1, gap_len))
                        no_edge_run = 0
                        for s in range(steps+1):
                            x = a1 + (gap_len * s / max(1, steps))
                            cx = int(round(x))
                            cy = int(round(cy_best))
                            hE, wE = edge_img.shape[:2]
                            x0 = max(0, cx - stripe_half); x1l2 = min(wE-1, cx + stripe_half)
                            y0 = max(0, cy - stripe_half); y1l3 = min(hE-1, cy + stripe_half)
                            hit = False
                            if x0 <= x1l2 and y0 <= y1l3:
                                roiE = edge_img[y0:y1l3+1, x0:x1l2+1]
                                if np.any(roiE > 0):
                                    hit = True
                            if hit:
                                no_edge_run = 0
                            else:
                                no_edge_run += 1
                                if no_edge_run >= gap_no_edge_allow:
                                    connectivity_ok = False
                                    break
                        if not connectivity_ok:
                            break
                    if not connectivity_ok:
                        # 回退：仅保留原 group 中支持度最高的水平线段（按长度挑选）
                        try:
                            lengths = [float(np.hypot(ln[2]-ln[0], ln[3]-ln[1])) for ln in group]
                            idx_best = int(np.argmax(lengths))
                            final_merged_line = group[idx_best].astype(float)
                        except Exception:
                            pass

        support_score = float(sum(np.linalg.norm(l[2:4] - l[0:2]) for l in group))
        # 过滤极短支持（减少零散直线）
        if merged_min_support_px and merged_min_support_px > 0.0:
            if support_score < float(merged_min_support_px):
                continue
        merged_lines_with_scores.append({'line': final_merged_line, 'score': support_score})
    # 先按支持度排序（强边在前）
    merged_lines_with_scores.sort(key=lambda item: item['score'], reverse=True)

    # 改进：取消原有“竖直与水平线段相交距离过滤”逻辑，仅对完全重合/重复的线段进行去重，保留几何直觉交叉
    # 新增参数：LINE_MERGING.DUPLICATE_MERGE_MAX_OFFSET_PX（默认 2.0）控制重合判定的平移容差
    try:
        dup_offset_px = float(params.get('LINE_MERGING', {}).get('DUPLICATE_MERGE_MAX_OFFSET_PX', 2.0))
    except Exception:
        dup_offset_px = 2.0
    try:
        dup_angle_tol_deg = float(params.get('LINE_MERGING', {}).get('DUPLICATE_MERGE_ANGLE_TOL_DEG', 3.0))
    except Exception:
        dup_angle_tol_deg = 3.0

    def _angle_deg(seg):
        x1,y1,x2,y2 = map(float, seg)
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if ang < 0: ang += 180.0
        return ang

    def _proj_length(seg):
        return float(np.hypot(float(seg[2]) - float(seg[0]), float(seg[3]) - float(seg[1])))

    def _merge_vertical_lines(seg_a, score_a, seg_b, score_b):
        """将过近的竖线合并为单条；x 取加权平均，y 取全范围。"""
        w = max(1e-6, float(score_a) + float(score_b))
        x_a = float(seg_a[0] + seg_a[2]) / 2.0
        x_b = float(seg_b[0] + seg_b[2]) / 2.0
        x_new = (x_a * float(score_a) + x_b * float(score_b)) / w
        y_vals = [float(seg_a[1]), float(seg_a[3]), float(seg_b[1]), float(seg_b[3])]
        y_min = min(y_vals)
        y_max = max(y_vals)
        return np.array([x_new, y_min, x_new, y_max], dtype=float)

    # 配置：水平重复合并的最大垂距与最小轴向重叠比
    horiz_dup_max_off_px = _px_dist_mm('HORIZONTAL_DUP_MERGE_MAX_OFFSET_MM', 'HORIZONTAL_DUP_MERGE_MAX_OFFSET_PX', 2.0, default_px=4.0)
    try:
        horiz_dup_min_overlap_ratio = float(p.get('HORIZONTAL_DUP_MIN_OVERLAP_RATIO', 0.2))
    except Exception:
        horiz_dup_min_overlap_ratio = 0.2

    # 将线段按支持度排序后做重复检测：角度接近且距离差小，判为重复，保留强的一条
    filtered = []
    for item in merged_lines_with_scores:
        seg = item['line']
        ang = _angle_deg(seg)
        keep = True
        sx1, sy1, sx2, sy2 = map(float, seg)
        for kept in filtered:
            kseg = kept['line']
            kang = _angle_deg(kseg)
            ang_diff = min(abs(ang - kang), 180.0 - abs(ang - kang))
            if ang_diff > dup_angle_tol_deg:
                continue
            # 近水平的重复判定：垂直距离小且沿轴投影重叠足够
            if _is_near_horizontal(ang) and _is_near_horizontal(kang):
                # 计算两线的最小垂距（取一个端点到另一线的垂距的最小值近似）
                def _perp_dist_to_line(pt, line):
                    ax, ay, bx, by = map(float, line)
                    a = np.array([ax, ay]); b = np.array([bx, by]); p = np.array([pt[0], pt[1]])
                    v = b - a; lv = float(np.dot(v, v))
                    if lv < 1e-6:
                        return float(np.linalg.norm(p - a))
                    t = float(np.dot(p - a, v) / lv)
                    proj = a + t * v
                    # 使用点到“无限延长线”的垂距
                    return float(abs(v[0]*(a[1]-p[1]) - v[1]*(a[0]-p[0])) / (math.sqrt(lv)))
                d_perp = min(
                    _perp_dist_to_line((sx1, sy1), kseg),
                    _perp_dist_to_line((sx2, sy2), kseg)
                )
                if d_perp <= float(horiz_dup_max_off_px):
                    # 轴向重叠：比较x投影（水平线）
                    sx_min = min(sx1, sx2); sx_max = max(sx1, sx2)
                    kx_min = min(float(kseg[0]), float(kseg[2])); kx_max = max(float(kseg[0]), float(kseg[2]))
                    overlap = max(0.0, min(sx_max, kx_max) - max(sx_min, kx_min))
                    span = max(1.0, max(sx_max, kx_max) - min(sx_min, kx_min))
                    if (overlap / span) >= horiz_dup_min_overlap_ratio:
                        keep = False
                        break
            # 计算本线段四个端点到已保留线段的最小距离，用于判断是否近乎重合
            ax1, ay1, ax2, ay2 = map(float, kseg)
            a = np.array([ax1, ay1]); b = np.array([ax2, ay2])
            v = b - a
            lv = float(np.dot(v, v))
            if lv < 1e-6:
                continue
            def _pt_dist(pt):
                t = float(np.dot(pt - a, v) / lv)
                t_clamped = max(0.0, min(1.0, t))
                proj = a + t_clamped * v
                return float(np.linalg.norm(pt - proj))
            d1 = _pt_dist(np.array([sx1, sy1]))
            d2 = _pt_dist(np.array([sx2, sy2]))
            if d1 <= dup_offset_px and d2 <= dup_offset_px:
                # 判为重复：仅保留已存在的（更强的）
                keep = False
                break
        if keep:
            filtered.append(item)
    # 保持支持度排序
    filtered.sort(key=lambda it: it['score'], reverse=True)

    # 基于 Canny 对合并结果进行二次微调，保证线段紧贴真实边缘
    if edge_img is not None and snap_enable and snap_half_px > 0 and filtered:
        snapped = []
        for item in filtered:
            seg = item['line']
            ang = _angle_deg(seg)
            if _is_near_vertical(ang) or _is_near_horizontal(ang):
                seg = _snap_line_to_canny(seg, edge_img, snap_half_px, snap_min_points, snap_max_angle)
            snapped.append({'line': seg, 'score': item['score']})
        filtered = snapped

    if min_vertical_gap_px and min_vertical_gap_px > 0.0 and filtered:
        filtered_with_gap = []
        for item in filtered:
            seg = item['line']
            score = item['score']
            if not _is_near_vertical(_angle_deg(seg)):
                filtered_with_gap.append({'line': seg, 'score': score})
                continue
            x_center = float(seg[0] + seg[2]) / 2.0
            merged = False
            for existing in filtered_with_gap:
                ex_seg = existing['line']
                if not _is_near_vertical(_angle_deg(ex_seg)):
                    continue
                ex_center = float(ex_seg[0] + ex_seg[2]) / 2.0
                if abs(x_center - ex_center) < float(min_vertical_gap_px):
                    existing['line'] = _merge_vertical_lines(ex_seg, existing['score'], seg, score)
                    existing['score'] += score
                    merged = True
                    break
            if not merged:
                filtered_with_gap.append({'line': seg, 'score': score})
        filtered = filtered_with_gap

    # 根据是否与其它直线（实际或小范围延长后）相交，决定用于角度/扫描/过滤的有效线段：
    # - 不相交：保持原始长度
    # - 相交（含延长容差内相交）：使用“交点 → 原始线段最远端点”的新线段
    edges = [item['line'] for item in filtered[:p["TOP_N_EDGES"]]]
    # 多样性保护：若 Top-N 中没有近竖直主边，而候选集中存在强近竖直主边，则用其替换最弱一条
    try:
        ensure_vertical = bool(params.get('LINE_MERGING', {}).get('ENSURE_VERTICAL_PRESENCE', False))
    except Exception:
        ensure_vertical = False
    if ensure_vertical and len(filtered) > 0 and len(edges) > 0:
        try:
            v_tol = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
        except Exception:
            v_tol = 10.0
        def _a_deg(seg):
            x1,y1,x2,y2 = map(float, seg)
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang > 90.0:
                ang = 180.0 - ang
            return ang
        def _is_vertical(seg):
            return _a_deg(seg) >= (90.0 - v_tol)
        if not any(_is_vertical(e) for e in edges):
            # 找到候选集中（按得分排序后的 filtered）第一个近竖直，但不在 edges 中的线
            cur_set = {tuple(map(float, e)) for e in edges}
            candidate = None; candidate_idx = None
            for idx_f, item in enumerate(filtered):
                seg = item['line']
                if _is_vertical(seg) and tuple(map(float, seg)) not in cur_set:
                    candidate = seg; candidate_idx = idx_f; break
            if candidate is not None:
                # 替换 edges 中末尾（最弱）一条
                edges[-1] = candidate
    m = len(edges)
    if m <= 1:
        return edges

    # 用于后续“交点裁剪”步骤的辅助：短延长容差与垂距/参数计算
    extend_margin_px_default = _mm_to_px(5.0, pixels_per_mm)
    def _t_param_and_perp_dist(pt, a, b):
        v = b - a
        denom = float(np.dot(v, v))
        if denom < 1e-8:
            return 0.0, float(np.linalg.norm(pt - a)), 0.0
        t = float(np.dot(pt - a, v) / denom)
        proj = a + t * v
        return t, float(np.linalg.norm(pt - proj)), float(np.linalg.norm(v))

    best_intersection = [None] * m  # (inter_pt, t_i, d_perp_i, len_i)
    # 复用上面定义的 _t_param_and_perp_dist 与 extend 容差
    def _inter_score(t: float):
        # 优先选择落在段内的交点；若不在段内，选择距离[0,1]最近者
        if 0.0 <= t <= 1.0:
            return (0, 0.0)
        # 距离[0,1]的外侧距离
        return (1, min(abs(t - 0.0), abs(t - 1.0)))

    # 简易两主体聚类：以边中点为依据，在 x 或 y 维度上寻找最大间隙；若大于阈值则分两组
    cluster_gap_px = _px_dist_mm('GLASS_CLUSTER_GAP_MM', 'GLASS_CLUSTER_GAP_PX', 40.0, default_px=0.0)
    mids = np.array([[(e[0]+e[2])/2.0, (e[1]+e[3])/2.0] for e in edges], dtype=float)
    labels = np.zeros((m,), dtype=int)
    if m >= 4 and cluster_gap_px and cluster_gap_px > 0:
        for dim in [0,1]:
            order = np.argsort(mids[:,dim])
            vals = mids[order, dim]
            gaps = np.diff(vals)
            if gaps.size > 0:
                k = int(np.argmax(gaps))
                if gaps[k] >= float(cluster_gap_px):
                    # 分割点在 k 与 k+1 之间
                    labels[order[:k+1]] = 0
                    labels[order[k+1:]] = 1
                    break

    for i in range(m):
        li = edges[i]
        ai = np.array(li[:2], dtype=float); bi = np.array(li[2:], dtype=float)
        for j in range(i + 1, m):
            lj = edges[j]
            # 禁止跨主体（两组）求交，避免第一片水平与第二片竖直相交
            if labels[i] != labels[j]:
                continue
            aj = np.array(lj[:2], dtype=float); bj = np.array(lj[2:], dtype=float)

            inter = find_line_intersection(li, lj)
            if inter is None:
                continue

            t_i, d_perp_i, len_i = _t_param_and_perp_dist(inter, ai, bi)
            t_j, d_perp_j, len_j = _t_param_and_perp_dist(inter, aj, bj)
            if len_i < 1e-6 or len_j < 1e-6:
                continue

            ext_tol_i = extend_margin_px_default / len_i
            ext_tol_j = extend_margin_px_default / len_j
            # 放宽：对“近正交”线对（近似竖直×水平），取消相交的距离限制（d_perp）
            try:
                def _ang_deg(seg):
                    x1,y1,x2,y2 = map(float, seg)
                    a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if a < 0: a += 180.0
                    return a
                diff_deg = min(abs(_ang_deg(li) - _ang_deg(lj)), 180.0 - abs(_ang_deg(li) - _ang_deg(lj)))
                ortho_accept_deg = float(params.get('LINE_MERGING', {}).get('INTERSECTION_ORTHO_ACCEPT_MIN_DIFF_DEG', 60.0))
            except Exception:
                diff_deg = 0.0; ortho_accept_deg = 60.0
            near_line = (d_perp_i < 1.5) and (d_perp_j < 1.5)
            if diff_deg >= ortho_accept_deg:
                # 近似竖直×水平：视为有交，不要求近线距离
                near_line = True
            within_i = (-ext_tol_i <= t_i <= 1.0 + ext_tol_i)
            within_j = (-ext_tol_j <= t_j <= 1.0 + ext_tol_j)
            if near_line and within_i and within_j:
                score_i = _inter_score(t_i)
                prev_i = best_intersection[i]
                if (prev_i is None) or (score_i < _inter_score(prev_i[1])):
                    best_intersection[i] = (inter.astype(float), t_i, d_perp_i, len_i)
                score_j = _inter_score(t_j)
                prev_j = best_intersection[j]
                if (prev_j is None) or (score_j < _inter_score(prev_j[1])):
                    best_intersection[j] = (inter.astype(float), t_j, d_perp_j, len_j)

    adjusted_edges = []
    for idx, seg in enumerate(edges):
        bi = best_intersection[idx]
        if bi is None:
            adjusted_edges.append(seg)
            continue
        inter_pt = bi[0]
        p1 = np.array(seg[:2], dtype=float); p2 = np.array(seg[2:], dtype=float)
        # 选择“交点到原始段最远端点”的新线段
        d1 = float(np.linalg.norm(inter_pt - p1))
        d2 = float(np.linalg.norm(inter_pt - p2))
        far = p1 if d1 >= d2 else p2
        adjusted_edges.append(np.array([inter_pt[0], inter_pt[1], far[0], far[1]], dtype=float))

    return adjusted_edges

def find_and_analyze_defects(edges, roi_gray, roi_dims, params, pixels_per_mm: float, binary_edges=None):
    p_defect = params["DEFECT_DETECTION"]; p_crack = params["CRACK_CLASSIFICATION"]
    num_edges = len(edges); roi_h, roi_w = roi_dims
    # 调试开关（控制台打印）：默认关闭，可通过 params.DEBUG.PRINT_CORNERS 开启
    try:
        _DBG_PRINT = bool(params.get('DEBUG', {}).get('PRINT_CORNERS', params.get('DEFECT_DETECTION', {}).get('DEBUG_PRINT', False)))
    except Exception:
        _DBG_PRINT = False
    try:
        _DBG_LEVEL = int(params.get('DEBUG', {}).get('PRINT_LEVEL', 1))
    except Exception:
        _DBG_LEVEL = 1
    def _dprint(*a, **k):
        if _DBG_PRINT:
            try:
                print(*a, **k)
            except Exception:
                pass
    
    # 去除通过几何（直线）判断裂纹（L）的功能
    crack_defects = []
    crack_indices = set()

    true_edges = [edge for i, edge in enumerate(edges) if i not in crack_indices]
    # 记录修改前的主边拷贝，用于后续判断“水平边的延长部分”（相对原坐标）
    true_edges_before_ext = [edge.copy() for edge in true_edges]

    # ================= 恢复：基于主边中点的主体(cluster)划分（中点间隙法） =================
    # 仍沿用 merge 阶段的 GLASS_CLUSTER_GAP_MM 阈值；若 gap 不足则视为单主体
    cluster_gap_px_cfg = 0.0
    try:
        cluster_gap_px_cfg = float(_get_dist_px(params.get('LINE_MERGING', {}), 'GLASS_CLUSTER_GAP_MM', 'GLASS_CLUSTER_GAP_PX', None, pixels_per_mm))
    except Exception:
        cluster_gap_px_cfg = 0.0
    mids_all = np.array([[(e[0]+e[2])/2.0, (e[1]+e[3])/2.0] for e in true_edges], dtype=float) if true_edges else np.zeros((0,2), dtype=float)
    cluster_labels = np.zeros((len(true_edges),), dtype=int)

    # 动态阈值：若未配置，则用 8% 的 ROI 宽度；优先用近竖直边的 x 方向间隙进行二分
    try:
        Hc_dyn, Wc_dyn = roi_h, roi_w
    except Exception:
        Hc_dyn, Wc_dyn = (0, 0)
    eff_gap_px = float(cluster_gap_px_cfg) if cluster_gap_px_cfg and cluster_gap_px_cfg > 0 else (0.08 * float(Wc_dyn if Wc_dyn else 1000.0))

    def _is_vert_for_cluster(seg):
        try:
            tol = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
        except Exception:
            tol = 10.0
        x1,y1,x2,y2 = map(float, seg)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang > 90.0:
            ang = 180.0 - ang
        return ang >= (90.0 - tol)

    if len(true_edges) >= 2 and mids_all.shape[0] > 0:
        try:
            vert_idx = [i for i,e in enumerate(true_edges) if _is_vert_for_cluster(e)]
            if len(vert_idx) >= 2:
                xs = np.array([mids_all[i,0] for i in vert_idx], dtype=float)
                order = np.argsort(xs)
                xs_sorted = xs[order]
                gaps = np.diff(xs_sorted)
                if gaps.size > 0 and float(np.max(gaps)) >= eff_gap_px:
                    k = int(np.argmax(gaps))
                    # 依据 x 中位数划分所有边
                    x_thresh = 0.5 * (xs_sorted[k] + xs_sorted[k+1])
                    for i in range(len(true_edges)):
                        mx = mids_all[i,0]
                        cluster_labels[i] = 0 if mx <= x_thresh else 1
            # 若未能按竖直边分，则回退到“X/Y 最大间隙”策略
            if np.max(cluster_labels) == 0 and len(true_edges) >= 4 and eff_gap_px > 0:
                for dim in [0,1]:
                    order = np.argsort(mids_all[:,dim])
                    vals = mids_all[order, dim]
                    gaps = np.diff(vals)
                    if gaps.size > 0:
                        k = int(np.argmax(gaps))
                        if gaps[k] >= eff_gap_px:
                            cluster_labels[order[:k+1]] = 0
                            cluster_labels[order[k+1:]] = 1
                            break
        except Exception:
            pass
    # 计算每组的中心（中点质心），便于后续射线朝向选择
    cluster_centers = {}
    if mids_all.shape[0] > 0:
        for c in np.unique(cluster_labels):
            pts_c = mids_all[cluster_labels == c]
            if pts_c.size > 0:
                cluster_centers[int(c)] = np.mean(pts_c, axis=0)
    # 若只有单主体则 cluster_centers 只含 0；后续逻辑自动退回原轮廓中心判定
    if _DBG_PRINT:
        try:
            uniq = list(map(int, np.unique(cluster_labels))) if len(cluster_labels) > 0 else []
            counts = [(int(c), int((cluster_labels==c).sum())) for c in uniq]
            _dprint(f"[DBG] clusters(gap)={counts} edges={len(true_edges)}")
        except Exception:
            pass

    # ========== 条件性凸包验证：仅在同时存在多个竖直和多个水平主边时启用 ==========
    qx_blocked = False
    try:
        # 统计近竖直与近水平条数
        try:
            v_tol_deg_chk = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
        except Exception:
            v_tol_deg_chk = 10.0
        try:
            h_tol_deg_chk = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', v_tol_deg_chk))
        except Exception:
            h_tol_deg_chk = v_tol_deg_chk
        def _ang_x(seg):
            x1,y1,x2,y2 = map(float, seg)
            a = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            return a if a <= 90.0 else 180.0 - a
        cnt_v = sum(1 for e in true_edges if _ang_x(e) >= (90.0 - v_tol_deg_chk))
        cnt_h = sum(1 for e in true_edges if _ang_x(e) <= h_tol_deg_chk)
        enable_hull_check = (cnt_v >= 2 and cnt_h >= 2)
        if enable_hull_check and (len(np.unique(cluster_labels)) >= 2):
            clusters_points = {}
            for idx, seg in enumerate(true_edges):
                cid = int(cluster_labels[idx]) if len(cluster_labels) > idx else 0
                a = np.array(seg[:2], dtype=float); b = np.array(seg[2:], dtype=float)
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    continue
                clusters_points.setdefault(cid, []).append(a)
                clusters_points[cid].append(b)
            hulls = {}
            for cid, pts in clusters_points.items():
                if len(pts) < 3:
                    continue
                P = np.vstack(pts).astype(np.float32)
                try:
                    hull = cv2.convexHull(P)
                except Exception:
                    hull = None
                if hull is not None and isinstance(hull, np.ndarray) and hull.shape[0] >= 3:
                    hulls[cid] = hull.astype(np.float32)
            keys = sorted(hulls.keys())
            for i_k in range(len(keys)):
                for j_k in range(i_k+1, len(keys)):
                    h1 = hulls[keys[i_k]]; h2 = hulls[keys[j_k]]
                    try:
                        area, _ = cv2.intersectConvexConvex(h1, h2)
                        if area is not None and float(area) > 0.0:
                            qx_blocked = True
                            raise StopIteration
                    except StopIteration:
                        raise
                    except Exception:
                        continue
        if _DBG_PRINT:
            _dprint(f"[DBG] hull-check enable={enable_hull_check} cntV={cnt_v} cntH={cnt_h} qx_blocked={bool(qx_blocked)}")
    except StopIteration:
        pass
    except Exception:
        qx_blocked = False
    
    corner_defects = []; num_true_edges = len(true_edges)
    edges_for_drawing = [edge.copy() for edge in true_edges]
    endpoint_paired_status = {i: [False, False] for i in range(num_true_edges)}
    # 收集未产生 Q 的直线交点，用于后续过滤其附近的 B 误检
    non_q_intersections = []
    # 记录已确认的角点，供 Q 检测阶段复用，避免重复计算交点
    paired_corners = []  # 列表元素: (i_idx, j_idx, np.array([x,y]))
    
    def get_line_quadrant(line, w, h):
        center_x, center_y = w / 2, h / 2
        mid_x, mid_y = (line[0] + line[2]) / 2, (line[1] + line[3]) / 2
        if mid_y < center_y:
            return 'TL' if mid_x < center_x else 'TR'
        else:
            return 'BL' if mid_x < center_x else 'BR'

    def get_quadrant_compatibility(q1, q2):
        if q1 == q2: return 0
        pair = frozenset([q1, q2])
        if pair in [frozenset(['TL', 'TR']), frozenset(['BL', 'BR']), 
                    frozenset(['TL', 'BL']), frozenset(['TR', 'BR'])]:
            return 1
        return 2

    edge_quadrants = [get_line_quadrant(edge, roi_w, roi_h) for edge in true_edges]

    # 新增：为缺角(Q)检测稳定角点，将可靠的“近竖直直线”延长到当前 ROI 边界内
    # 开关：DEFECT_DETECTION.ENABLE_VERTICAL_EXTENSION_FOR_Q (默认 True)
    # 最短长度门槛：DEFECT_DETECTION.VERTICAL_EXTEND_MIN_LEN_MM (默认 5mm)
    try:
        enable_v_ext = bool(params.get('DEFECT_DETECTION', {}).get('ENABLE_VERTICAL_EXTENSION_FOR_Q', True))
    except Exception:
        enable_v_ext = True
    try:
        v_ext_min_len_px = _get_dist_px(params.get('DEFECT_DETECTION', {}), 'VERTICAL_EXTEND_MIN_LEN_MM', None, 5.0, pixels_per_mm)
    except Exception:
        v_ext_min_len_px = 5.0 * float(pixels_per_mm if pixels_per_mm else 1.0)

    def _clip_infinite_line_to_roi(seg, w, h):
        """将由 seg 两点确定的无限直线裁剪到 ROI 边界内，返回裁剪后的线段(两端在边界上)。
        若直线与 ROI 无交则返回原始 seg。"""
        try:
            x1, y1, x2, y2 = map(float, seg)
            dx, dy = (x2 - x1), (y2 - y1)
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return seg
            candidates = []
            # 与 x=0, x=w-1 相交
            if abs(dx) >= 1e-9:
                for xk in (0.0, float(w - 1)):
                    t = (xk - x1) / dx
                    yk = y1 + t * dy
                    if 0.0 <= yk <= float(h - 1):
                        candidates.append((xk, yk))
            # 与 y=0, y=h-1 相交
            if abs(dy) >= 1e-9:
                for yk in (0.0, float(h - 1)):
                    t = (yk - y1) / dy
                    xk = x1 + t * dx
                    if 0.0 <= xk <= float(w - 1):
                        candidates.append((xk, yk))
            # 去重并选择两个最远点
            if len(candidates) < 2:
                return seg
            # 唯一化
            uniq = []
            for pt in candidates:
                if not any(abs(pt[0]-q[0]) < 0.5 and abs(pt[1]-q[1]) < 0.5 for q in uniq):
                    uniq.append(pt)
            if len(uniq) < 2:
                return seg
            # 选相互最远的两点
            pts = np.array(uniq, dtype=float)
            idx0, idx1 = 0, 1
            maxd = -1.0
            for i0 in range(len(pts)):
                for i1 in range(i0+1, len(pts)):
                    d = float(np.hypot(pts[i1,0]-pts[i0,0], pts[i1,1]-pts[i0,1]))
                    if d > maxd:
                        maxd = d; idx0, idx1 = i0, i1
            a, b = pts[idx0], pts[idx1]
            return np.array([a[0], a[1], b[0], b[1]], dtype=float)
        except Exception:
            return seg

    if enable_v_ext and true_edges:
        try:
            # 与“斜边分类”一致使用的竖直容忍角度
            try:
                vertical_tol_deg_local = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
            except Exception:
                vertical_tol_deg_local = 10.0
            def _angle_to_x_axis_deg_local(line):
                x1, y1, x2, y2 = map(float, line)
                dx, dy = (x2 - x1), (y2 - y1)
                ang = abs(np.degrees(np.arctan2(dy, dx)))
                if ang > 90.0:
                    ang = 180.0 - ang
                return ang
            H_roi, W_roi = roi_h, roi_w
            for idx in range(len(true_edges)):
                seg = true_edges[idx]
                # 长度门槛
                try:
                    if v_ext_min_len_px and v_ext_min_len_px > 0:
                        if float(np.hypot(seg[2]-seg[0], seg[3]-seg[1])) < float(v_ext_min_len_px):
                            continue
                except Exception:
                    pass
                ang = _angle_to_x_axis_deg_local(seg)
                # 近竖直
                if ang >= (90.0 - vertical_tol_deg_local):
                    ext = _clip_infinite_line_to_roi(seg, W_roi, H_roi)
                    true_edges[idx] = np.array(ext, dtype=float)
        except Exception:
            pass

    # ========= 收集竖直边 x 中点, 用于水平延长的“栅栏”推断 =========
    try:
        fence_enable = bool(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_FENCE_ENABLED', True))
    except Exception:
        fence_enable = True
    vertical_x_for_history = []
    if fence_enable:
        try:
            v_tol_for_hist = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
        except Exception:
            v_tol_for_hist = 10.0
        for seg in true_edges:
            x1,y1,x2,y2 = map(float, seg)
            ang_hist = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if ang_hist > 90.0: ang_hist = 180.0 - ang_hist
            if ang_hist >= (90.0 - v_tol_for_hist):
                vertical_x_for_history.append(0.5 * (x1 + x2))
        roi_id = params.get('ROI_ID', None)
        roi_key = (roi_id, roi_w, roi_h) if roi_id is not None else (roi_w, roi_h)
        _update_vertical_fences(roi_key, vertical_x_for_history, params)
        # 记录多玻璃分隔线候选
        _update_glass_boundaries(roi_key, vertical_x_for_history, params)
        fences = _get_fences_for_roi(roi_key)
        glass_boundary = _get_glass_boundary(roi_key)
    else:
        fences = []
        glass_boundary = None

    # 新增：当存在“理想竖直边”（近竖直主边）时，将近水平主边延长到与这些竖直边相交（仅限同主体 cluster 且采用最近邻策略）
    # - ENABLE_HORIZONTAL_EXTENSION_TO_VERTICAL (默认 True)
    # - HORIZONTAL_EXTEND_MAX_GAP_MM (默认 40mm)
    # - HORIZONTAL_EXTEND_STRIPE_HALF_PX / HORIZONTAL_EXTEND_MAX_NO_EDGE_RUN_PX 用于 Canny 连续性校验
    if true_edges:
        try:
            # 判定近竖直：沿用 vertical_tol_deg_local
            def _is_vertical(seg):
                ang = _angle_to_x_axis_deg_local(seg)
                return ang >= (90.0 - vertical_tol_deg_local)
            # 缺角交点对齐容差（px）用于处理 1px 边稍错位导致的未相交问题
            try:
                q_align_tol_px = int(params.get('DEFECT_DETECTION', {}).get('Q_CORNER_ALIGNMENT_TOL_PX', 3))
            except Exception:
                q_align_tol_px = 3

            def _fuzzy_hv_intersection(hseg, vseg, tol_px: int):
                """在严格直线求交失败时，为水平与竖直近似线段提供基于像素容差的交点补偿。
                仅在 hseg 近水平且 vseg 近竖直时尝试，tol_px 默认 3。
                返回 np.array([x,y]) 或 None。"""
                try:
                    hx1, hy1, hx2, hy2 = map(float, hseg)
                    vx1, vy1, vx2, vy2 = map(float, vseg)
                    # 快速角度判定（使用本地函数）
                    if _angle_to_x_axis_deg_local(hseg) > 15.0:
                        return None
                    ang_v = _angle_to_x_axis_deg_local(vseg)
                    if not (ang_v >= 75.0):  # 近竖直
                        return None
                    # 水平段 y 范围与竖直段 x 范围
                    hy = (hy1 + hy2) / 2.0
                    vx = (vx1 + vx2) / 2.0
                    # 判定是否有“近似交点”：
                    # 1) vx 位于水平段投影范围 x[min,max] 扩展±tol
                    h_xmin = min(hx1, hx2) - tol_px
                    h_xmax = max(hx1, hx2) + tol_px
                    if not (h_xmin <= vx <= h_xmax):
                        return None
                    # 2) hy 位于竖直段投影范围 y[min,max] 扩展±tol
                    v_ymin = min(vy1, vy2) - tol_px
                    v_ymax = max(vy1, vy2) + tol_px
                    if not (v_ymin <= hy <= v_ymax):
                        return None
                    # 生成“修正交点”并限制在 ROI 有效区域（调用处外层保证）
                    return np.array([vx, hy], dtype=float)
                except Exception:
                    return None
            # 携带索引，便于按 cluster 过滤
            vertical_ideals = [(vi, seg.copy()) for vi, seg in enumerate(true_edges) if _is_vertical(seg)]
            if vertical_ideals:
                # 水平判定：角度接近 0°（不再使用可配置容忍）
                def _is_horizontal(seg):
                    ang = _angle_to_x_axis_deg_local(seg)
                    return ang <= 10.0  # 固定阈值 10° 内视为水平
                # 距离与连接性阈值
                try:
                    max_gap_px = float(_get_dist_px(params.get('DEFECT_DETECTION', {}), 'HORIZONTAL_EXTEND_MAX_GAP_MM', None, 40.0, pixels_per_mm))
                except Exception:
                    max_gap_px = 40.0 * float(pixels_per_mm if pixels_per_mm else 1.0)
                try:
                    stripe_half_ext = int(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_STRIPE_HALF_PX', 2))
                except Exception:
                    stripe_half_ext = 2
                try:
                    max_no_edge_run_ext = int(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_MAX_NO_EDGE_RUN_PX', 25))
                except Exception:
                    max_no_edge_run_ext = 25
                for idx, seg in enumerate(true_edges):
                    if not _is_horizontal(seg):
                        continue
                    if _DBG_PRINT and _DBG_LEVEL >= 2:
                        try:
                            _dprint(f"[DBG] H-extend on idx={idx} c={int(cluster_labels[idx]) if len(cluster_labels)>idx else -1}")
                        except Exception:
                            pass
                    # 仅使用与该水平线同主体(cluster)的竖直边
                    cand_verticals = vertical_ideals
                    try:
                        if cluster_labels.size > idx:
                            cid = int(cluster_labels[idx])
                            cand_verticals = [(vi, vseg) for (vi, vseg) in vertical_ideals if (cluster_labels.size > vi and int(cluster_labels[vi]) == cid)]
                    except Exception:
                        cand_verticals = vertical_ideals
                    if not cand_verticals:
                        continue
                    x1, y1, x2, y2 = map(float, seg)
                    dx = x2 - x1; dy = y2 - y1
                    if abs(dx) < 1e-6:
                        continue
                    # 读取栅栏, 计算当前水平线可延长的左右最大边界
                    try:
                        fence_margin = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_FENCE_MARGIN_PX', 4.0))
                    except Exception:
                        fence_margin = 4.0
                    fence_left_bound = None
                    fence_right_bound = None
                    if fences:
                        seg_xmin = min(x1, x2); seg_xmax = max(x1, x2)
                        # 找最近左侧和右侧栅栏
                        left_candidates = [fx for fx in fences if fx < seg_xmin]
                        right_candidates = [fx for fx in fences if fx > seg_xmax]
                        if left_candidates:
                            fence_left_bound = max(left_candidates) - fence_margin
                        if right_candidates:
                            fence_right_bound = min(right_candidates) + fence_margin
                    # 多玻璃边界: 若存在稳定分隔线, 根据水平线中点判断所属玻璃, 设置对侧禁止跨越的边界
                    if glass_boundary is not None:
                        mid_x = 0.5 * (x1 + x2)
                        if mid_x <= glass_boundary:  # 属于左玻璃, 不跨右侧边界
                            if fence_right_bound is None or fence_right_bound > glass_boundary:
                                fence_right_bound = glass_boundary + fence_margin
                        else:  # 属于右玻璃
                            if fence_left_bound is None or fence_left_bound < glass_boundary:
                                fence_left_bound = glass_boundary - fence_margin
                    if _DBG_PRINT and _DBG_LEVEL >= 2 and (glass_boundary is not None):
                        _dprint(f"  - glass_boundary={glass_boundary:.1f} left_bound={fence_left_bound} right_bound={fence_right_bound}")
                    # 构造无限延长后的水平直线，与候选竖直边求交点（使用竖直边的延长线）
                    intersections = []  # (pt, vi, vseg)
                    for (vi, vseg) in cand_verticals:
                        vx1, vy1, vx2, vy2 = map(float, vseg)
                        inter = find_line_intersection(seg, vseg)
                        if inter is None:
                            # 回退：尝试模糊交点（处理 1px 边轻微错位）
                            inter = _fuzzy_hv_intersection(seg, vseg, q_align_tol_px)
                            if inter is None:
                                continue
                        # 要求交点 y 在竖直边段范围内（±1px 缓冲）
                        vy_min = min(vy1, vy2) - 1.0; vy_max = max(vy1, vy2) + 1.0
                        if vy_min <= inter[1] <= vy_max:
                            # 栅栏过滤: 若有边界限制则不得跨越
                            if fence_left_bound is not None and inter[0] < fence_left_bound:
                                continue
                            if fence_right_bound is not None and inter[0] > fence_right_bound:
                                continue
                            intersections.append((inter.astype(float), vi, vseg))
                    if _DBG_PRINT and _DBG_LEVEL >= 2:
                        _dprint(f"  - candV={len(cand_verticals)} inter_inseg={len(intersections)}")
                    if not intersections:
                        # 无真实段内交点：保守不延长
                        if _DBG_PRINT and _DBG_LEVEL >= 2:
                            _dprint("  - no in-segment intersections; skip extend")
                        continue
                    seg_xmin = min(x1, x2); seg_xmax = max(x1, x2)
                    # 最近邻延长：左侧取最接近 seg_xmin 的交点(最大的小于 seg_xmin)；右侧取最接近 seg_xmax 的交点(最小的大于 seg_xmax)
                    left_candidates  = [(pt,vi,vseg) for (pt,vi,vseg) in intersections if pt[0] < seg_xmin - 0.5]
                    right_candidates = [(pt,vi,vseg) for (pt,vi,vseg) in intersections if pt[0] > seg_xmax + 0.5]
                    left_ext  = max(left_candidates,  key=lambda t: t[0][0]) if left_candidates  else None
                    right_ext = min(right_candidates, key=lambda t: t[0][0]) if right_candidates else None
                    if left_ext is None and right_ext is None:
                        continue
                    # 原线斜率（用于 y 外推）
                    slope = dy / dx if abs(dx) > 1e-6 else 0.0
                    def _y_at(new_x):
                        return y1 + slope * (new_x - x1)
                    new_x1 = x1; new_y1 = y1; new_x2 = x2; new_y2 = y2
                    if left_ext is not None:
                        pt_left = left_ext[0]
                        gap_len_px = float(seg_xmin - pt_left[0])
                        if fence_left_bound is not None and pt_left[0] < fence_left_bound:
                            # 超越栅栏, 忽略此延长
                            left_ext = None
                        if left_ext is None:
                            pass
                        if gap_len_px <= max_gap_px:
                            ok_conn = True
                            if isinstance(binary_edges, np.ndarray) and binary_edges.size > 0:
                                cy = int(round(_y_at(seg_xmin)))
                                x_a = int(round(pt_left[0])); x_b = int(round(seg_xmin))
                                x0 = max(0, min(x_a, x_b)); x1c = min(binary_edges.shape[1]-1, max(x_a, x_b))
                                no_edge_run = 0
                                for cx in range(x0, x1c+1):
                                    y0 = max(0, cy - stripe_half_ext); y1e = min(binary_edges.shape[0]-1, cy + stripe_half_ext)
                                    roi_be = binary_edges[y0:y1e+1, max(0,cx-stripe_half_ext):min(binary_edges.shape[1]-1,cx+stripe_half_ext)+1]
                                    if roi_be.size > 0 and np.any(roi_be > 0):
                                        no_edge_run = 0
                                    else:
                                        no_edge_run += 1
                                        if no_edge_run >= max_no_edge_run_ext:
                                            ok_conn = False; break
                            if ok_conn:
                                target_x = max(0.0, min(float(roi_w - 1), pt_left[0]))
                                if x1 < x2:
                                    new_x1 = target_x
                                    new_y1 = _y_at(new_x1)
                                else:
                                    new_x2 = target_x
                                    new_y2 = _y_at(new_x2)
                            if _DBG_PRINT and _DBG_LEVEL >= 2:
                                _dprint(f"  - choose LEFT x={pt_left[0]:.1f} gap={gap_len_px:.1f}px ok={ok_conn}")
                    if right_ext is not None:
                        pt_right = right_ext[0]
                        gap_len_px = float(pt_right[0] - seg_xmax)
                        if fence_right_bound is not None and pt_right[0] > fence_right_bound:
                            right_ext = None
                        if right_ext is None:
                            pass
                        if gap_len_px <= max_gap_px:
                            ok_conn = True
                            if isinstance(binary_edges, np.ndarray) and binary_edges.size > 0:
                                cy = int(round(_y_at(seg_xmax)))
                                x_a = int(round(seg_xmax)); x_b = int(round(pt_right[0]))
                                x0 = max(0, min(x_a, x_b)); x1c = min(binary_edges.shape[1]-1, max(x_a, x_b))
                                no_edge_run = 0
                                for cx in range(x0, x1c+1):
                                    y0 = max(0, cy - stripe_half_ext); y1e = min(binary_edges.shape[0]-1, cy + stripe_half_ext)
                                    roi_be = binary_edges[y0:y1e+1, max(0,cx-stripe_half_ext):min(binary_edges.shape[1]-1,cx+stripe_half_ext)+1]
                                    if roi_be.size > 0 and np.any(roi_be > 0):
                                        no_edge_run = 0
                                    else:
                                        no_edge_run += 1
                                        if no_edge_run >= max_no_edge_run_ext:
                                            ok_conn = False; break
                            if ok_conn:
                                target_x = max(0.0, min(float(roi_w - 1), pt_right[0]))
                                if x1 < x2:
                                    new_x2 = target_x
                                    new_y2 = _y_at(new_x2)
                                else:
                                    new_x1 = target_x
                                    new_y1 = _y_at(new_x1)
                            if _DBG_PRINT and _DBG_LEVEL >= 2:
                                _dprint(f"  - choose RIGHT x={pt_right[0]:.1f} gap={gap_len_px:.1f}px ok={ok_conn}")
                    true_edges[idx] = np.array([new_x1, new_y1, new_x2, new_y2], dtype=float)
        except Exception as _e_horiz_ext:
            # 失败安全：不中断后续流程
            pass

    # 注意：我们在上面可能修改了 true_edges（竖直/水平的延长）。
    # 因此需要在此处刷新用于绘制与配对的边集合与象限归属，确保后续角点/缺角逻辑使用更新后的线段。
    try:
        edges_for_drawing = [edge.copy() for edge in true_edges]
        edge_quadrants = [get_line_quadrant(edge, roi_w, roi_h) for edge in true_edges]
    except Exception:
        pass
    
    # 限制“用于角点/配对”的主边：每个主体(cluster)最多保留 1 条近竖直 + 1 条近水平（均取长度最大者）。
    # 说明：仅影响角点/缺角配对与补充交点；不影响后续 E/B 扫描（仍使用 true_edges 全集）。
    allowed_pair_lines = set()
    try:
        # 角度容忍
        v_tol_deg_lm = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
        h_tol_deg_lm = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', v_tol_deg_lm))
    except Exception:
        v_tol_deg_lm = 10.0; h_tol_deg_lm = 10.0
    def _angle_to_x_axis_deg_lm(seg):
        x1,y1,x2,y2 = map(float, seg)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        return ang if ang <= 90.0 else 180.0 - ang
    # 为每个 cluster 选出最长的垂直与水平
    per_cluster_best = {}
    for i, seg in enumerate(true_edges):
        try:
            cid = int(cluster_labels[i]) if len(cluster_labels) > i else 0
        except Exception:
            cid = 0
        ang = _angle_to_x_axis_deg_lm(seg)
        L = float(np.hypot(seg[2]-seg[0], seg[3]-seg[1]))
        entry = per_cluster_best.setdefault(cid, {'v':(-1, -1.0), 'h':(-1, -1.0)})
        if ang >= (90.0 - v_tol_deg_lm):
            if L > entry['v'][1]:
                entry['v'] = (i, L)
        elif ang <= h_tol_deg_lm:
            if L > entry['h'][1]:
                entry['h'] = (i, L)
    for cid, pick in per_cluster_best.items():
        if pick['v'][0] >= 0:
            allowed_pair_lines.add(int(pick['v'][0]))
        if pick['h'][0] >= 0:
            allowed_pair_lines.add(int(pick['h'][0]))
    if _DBG_PRINT and _DBG_LEVEL >= 1:
        try:
            _dprint("[DBG] allowed pair lines per cluster:")
            for cid, pick in per_cluster_best.items():
                _dprint(f"  - cluster {cid}: V={pick['v'][0]}(L={pick['v'][1]:.1f}) H={pick['h'][0]}(L={pick['h'][1]:.1f})")
        except Exception:
            pass
    if _DBG_PRINT and true_edges:
        def _ang_deg_for(seg):
            x1,y1,x2,y2 = map(float, seg)
            a = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            return a if a <= 90.0 else 180.0 - a
        for i, e in enumerate(true_edges):
            try:
                L = float(np.hypot(e[2]-e[0], e[3]-e[1]))
                cid = int(cluster_labels[i]) if len(cluster_labels) > i else -1
                _dprint(f"  [DBG] edge#{i:02d} c={cid} ang={_ang_deg_for(e):.2f} len={L:.1f} seg=({e[0]:.1f},{e[1]:.1f})-({e[2]:.1f},{e[3]:.1f})")
            except Exception:
                pass

    # 新增：按角度将主边分类为 平行 / 垂直 / 斜边，并为斜边生成覆盖条带+Canny白点范围的缺陷
    skew_line_defects = []
    try:
        # 默认容忍角度改为 10°（可通过 DEFECT_DETECTION.VERTICAL_ANGLE_TOL_DEG 覆盖）
        vertical_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
    except Exception:
        vertical_tol_deg = 10.0

    def _angle_to_x_axis_deg(line):
        x1, y1, x2, y2 = map(float, line)
        dx, dy = (x2 - x1), (y2 - y1)
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang > 90.0:
            ang = 180.0 - ang
        return ang  # [0,90]

    def _axis_aligned_box_points_for_line(seg, pad: int = 2):
        # 用线段端点生成窄矩形，再取其 axis-aligned 外接框
        p1 = np.array(seg[:2], dtype=float); p2 = np.array(seg[2:], dtype=float)
        v = p2 - p1
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            x = int(round(min(p1[0], p2[0]))); y = int(round(min(p1[1], p2[1])))
            w = h = max(1, int(pad*2))
        else:
            u = v / L
            n = np.array([-u[1], u[0]])
            half_w = float(max(1, pad))
            quad = np.array([
                p1 + n * half_w,
                p2 + n * half_w,
                p2 - n * half_w,
                p1 - n * half_w
            ], dtype=np.float32)
            x, y, w, h = cv2.boundingRect(np.int32(quad))
        box = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
        return box

    # 计算“参考垂直角”：用当前 ROI 中被判为垂直的边的角度均值；若无，则用 90°
    try:
        all_angles_deg = []
        for e in true_edges:
            all_angles_deg.append(_angle_to_x_axis_deg(e))
        vertical_angles = [a for a in all_angles_deg if a >= (90.0 - vertical_tol_deg)]
        vertical_ref_deg = (float(np.mean(vertical_angles)) if vertical_angles else 90.0)
    except Exception:
        vertical_ref_deg = 90.0

    # 分类并收集边缘异常（非平行且非垂直）
    for e in true_edges:
        try:
            a_deg = _angle_to_x_axis_deg(e)
            is_parallel = (a_deg <= vertical_tol_deg)
            is_vertical = (abs(a_deg - vertical_ref_deg) <= vertical_tol_deg)
            if not is_parallel and not is_vertical:
                angle_to_vertical = float(abs(a_deg - vertical_ref_deg))
                # 构建沿斜边的“拓宽+延长”条带，并用该条带内的 Canny 白点作为完整覆盖范围
                p1 = np.array(e[:2], dtype=float); p2 = np.array(e[2:], dtype=float)
                v = p2 - p1; L = float(np.linalg.norm(v))
                if L < 1e-6:
                    continue
                u = v / L
                n = np.array([-u[1], u[0]], dtype=float)
                # 宽度与延长长度（毫米配置转像素）
                try:
                    half_w_px = 0.5 * _get_dist_px(params.get('DEFECT_DETECTION', {}), 'E_STRIPE_WIDTH_MM', None, 6.0, pixels_per_mm)
                except Exception:
                    half_w_px = max(3.0, 3.0 * float(pixels_per_mm if pixels_per_mm else 1.0))
                try:
                    extend_px = _get_dist_px(params.get('DEFECT_DETECTION', {}), 'E_EXTEND_MARGIN_MM', None, 10.0, pixels_per_mm)
                except Exception:
                    extend_px = 10.0 * float(pixels_per_mm if pixels_per_mm else 1.0)
                # 生成扩展后的四点条带
                s = p1 - u * float(extend_px)
                t = p2 + u * float(extend_px)
                q1 = s + n * float(half_w_px)
                q2 = t + n * float(half_w_px)
                q3 = t - n * float(half_w_px)
                q4 = s - n * float(half_w_px)
                band_poly = np.array([q1, q2, q3, q4], dtype=np.int32).reshape((-1,1,2))

                # 在条带内取 Canny 白点；如无 Canny，回退到细长框
                min_rect = None; box_pts = None
                if binary_edges is not None and isinstance(binary_edges, np.ndarray) and binary_edges.size > 0:
                    try:
                        mask = np.zeros(roi_gray.shape, dtype=np.uint8)
                        cv2.fillPoly(mask, [band_poly], 255)
                        # 条带内边缘
                        band_edges = cv2.bitwise_and(binary_edges, mask)
                        # 1) 条带内膨胀，尽量闭合
                        try:
                            k_band = int(params.get('DEFECT_DETECTION', {}).get('E_BAND_DILATE_KSIZE', 3))
                        except Exception:
                            k_band = 3
                        k_band = max(1, k_band | 1)  # 确保奇数
                        try:
                            it_band = int(params.get('DEFECT_DETECTION', {}).get('E_BAND_DILATE_ITERS', 1))
                        except Exception:
                            it_band = 1
                        kernel_band = np.ones((k_band, k_band), dtype=np.uint8)
                        band_dil = cv2.dilate(band_edges, kernel_band, iterations=max(1, it_band))
                        # 2) 连接条带外部：对全局边缘轻度膨胀，找与 band_dil 连通的整块
                        try:
                            k_conn = int(params.get('DEFECT_DETECTION', {}).get('E_CONNECT_DILATE_KSIZE', 3))
                        except Exception:
                            k_conn = 3
                        k_conn = max(1, k_conn | 1)
                        try:
                            it_conn = int(params.get('DEFECT_DETECTION', {}).get('E_CONNECT_DILATE_ITERS', 1))
                        except Exception:
                            it_conn = 1
                        kernel_conn = np.ones((k_conn, k_conn), dtype=np.uint8)
                        global_conn = cv2.dilate(binary_edges, kernel_conn, iterations=max(1, it_conn))
                        # 连通域标签
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((global_conn > 0).astype(np.uint8), connectivity=8)
                        if num_labels > 1:
                            # 找与 band_dil 有重叠的标签集合（去除背景0）
                            overlap_labels = np.unique(labels[(band_dil > 0)])
                            overlap_labels = overlap_labels[overlap_labels != 0]
                            union_mask = (band_dil > 0).astype(np.uint8)
                            for lbl in overlap_labels:
                                union_mask |= (labels == int(lbl)).astype(np.uint8)
                        else:
                            union_mask = (band_dil > 0).astype(np.uint8)
                        # 若 ROI 存在水平边，则先构造水平带状屏蔽掩码，并从联合掩码中剔除
                        try:
                            try:
                                h_tol_deg_local = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', 10.0))
                            except Exception:
                                h_tol_deg_local = 10.0
                            try:
                                stripe_w_px = int(params.get('DEFECT_DETECTION', {}).get('E_HORIZONTAL_MASK_STRIPE_PX', 5))
                            except Exception:
                                stripe_w_px = 5
                            stripe_half = max(1.0, float(stripe_w_px) / 2.0)
                            hmask = np.zeros_like(union_mask, dtype=np.uint8)
                            for he in (true_edges or []):
                                ang_h = _angle_to_x_axis_deg(he)
                                if ang_h <= h_tol_deg_local:
                                    hp1 = np.array(he[:2], dtype=float); hp2 = np.array(he[2:], dtype=float)
                                    hv = hp2 - hp1
                                    hL = float(np.linalg.norm(hv))
                                    if hL <= 1e-6:
                                        continue
                                    hu = hv / hL
                                    hn = np.array([-hu[1], hu[0]], dtype=float)
                                    hq1 = hp1 + hn * stripe_half
                                    hq2 = hp2 + hn * stripe_half
                                    hq3 = hp2 - hn * stripe_half
                                    hq4 = hp1 - hn * stripe_half
                                    hpoly = np.array([hq1, hq2, hq3, hq4], dtype=np.int32).reshape((-1,1,2))
                                    cv2.fillPoly(hmask, [hpoly], 1)
                            if int(np.sum(hmask)) > 0:
                                union_mask = (union_mask & (1 - hmask)).astype(np.uint8)
                        except Exception:
                            pass

                        # 生成区域轮廓并取凸包→最小外接矩形
                        if int(cv2.countNonZero(union_mask)) >= 10:
                            contours, _ = cv2.findContours((union_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                biggest = max(contours, key=cv2.contourArea)
                                hull = cv2.convexHull(biggest)
                                if isinstance(hull, np.ndarray) and hull.shape[0] >= 3:
                                    min_rect = cv2.minAreaRect(hull)
                                    box_pts = cv2.boxPoints(min_rect).astype(np.int32)
                                else:
                                    min_rect = None; box_pts = None
                        # 无法构建则回退
                    except Exception:
                        min_rect = None; box_pts = None
                # 回退：若 Canny 未覆盖，使用原有细窄外接框
                if min_rect is None or box_pts is None:
                    # 直接使用线段的轴对齐外接框
                    box_pts = _axis_aligned_box_points_for_line(e, pad=3)
                    skew_line_defects.append({
                        'type': 'E',
                        'box_points': box_pts,
                        'skew_angle_deg': angle_to_vertical
                    })
                else:
                    # 改为输出轴对齐 bounding box：由 minAreaRect 的四点再取 boundingRect
                    try:
                        rect = cv2.boundingRect(box_pts.astype(np.int32))
                        xbb, ybb, wbb, hbb = rect
                        aabb = np.array([[xbb, ybb], [xbb + wbb, ybb], [xbb + wbb, ybb + hbb], [xbb, ybb + hbb]], dtype=np.int32)
                    except Exception:
                        # 回退：若异常则直接用原 box_pts（通常已接近旋转矩形），仍作为多边形输出
                        aabb = box_pts.astype(np.int32)
                    skew_line_defects.append({
                        'type': 'E',
                        'box_points': aabb,
                        'skew_angle_deg': angle_to_vertical
                    })
        except Exception:
            continue

    # 合并重叠的 E 类型 bounding box，并生成 size_label
    if skew_line_defects:
        try:
            e_items = [d for d in skew_line_defects if d.get('type') == 'E' and d.get('box_points') is not None]
            others_e = [d for d in skew_line_defects if not (d.get('type') == 'E' and d.get('box_points') is not None)]
            rects = []  # (x,y,w,h,angle,area)
            for ed in e_items:
                try:
                    box = np.array(ed.get('box_points'), dtype=np.int32)
                    x,y,w,h = cv2.boundingRect(box)
                    ang = float(ed.get('skew_angle_deg', 0.0))
                    area = float(max(1,w)*max(1,h))
                    rects.append([x,y,w,h,ang,area])
                except Exception:
                    continue
            n_e = len(rects)
            if n_e > 0:
                parent = list(range(n_e))
                def find(a):
                    while parent[a] != a:
                        parent[a] = parent[parent[a]]
                        a = parent[a]
                    return a
                def union(a,b):
                    ra,rb = find(a),find(b)
                    if ra!=rb: parent[rb]=ra
                def overlap(r1,r2):
                    x1,y1,w1,h1 = r1[0],r1[1],r1[2],r1[3]
                    x2,y2,w2,h2 = r2[0],r2[1],r2[2],r2[3]
                    ax1,ay1,ax2,ay2 = x1,y1,x1+w1,y1+h1
                    bx1,by1,bx2,by2 = x2,y2,x2+w2,y2+h2
                    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
                    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
                    return (ix2-ix1) > 0 and (iy2-iy1) > 0
                for i in range(n_e):
                    for j in range(i+1,n_e):
                        try:
                            if overlap(rects[i], rects[j]):
                                union(i,j)
                        except Exception:
                            continue
                groups = {}
                for i in range(n_e):
                    r = find(i)
                    groups.setdefault(r, []).append(i)
                merged = []
                for _, idxs in groups.items():
                    xs=[]; ys=[]; x2s=[]; y2s=[]; angle_sum=0.0; area_sum=0.0
                    for k in idxs:
                        x,y,w,h,ang,area = rects[k]
                        xs.append(x); ys.append(y); x2s.append(x+w); y2s.append(y+h)
                        angle_sum += ang * area; area_sum += area
                    X=min(xs); Y=min(ys); X2=max(x2s); Y2=max(y2s)
                    W=max(1, X2-X); H=max(1, Y2-Y)
                    aabb = np.array([[X,Y],[X+W,Y],[X+W,Y+H],[X,Y+H]], dtype=np.int32)
                    avg_ang = float(angle_sum/area_sum) if area_sum>1e-6 else 0.0
                    merged.append({'type':'E','box_points':aabb,'skew_angle_deg':avg_ang})
                skew_line_defects = others_e + merged
        except Exception:
            pass
    # 为 E 类型添加尺寸，供后续输出（size_label 不在算法侧生成）
    try:
        for ed in (skew_line_defects or []):
            if ed.get('type')!='E' or ed.get('box_points') is None:
                continue
            box = np.array(ed.get('box_points'), dtype=np.int32)
            x,y,w,h = cv2.boundingRect(box)
            ed['length_px'] = float(max(w,h)); ed['width_px'] = float(min(w,h))
    except Exception:
        pass

    rect_q_defects = []

    # 新逻辑A（保留）：利用已获得角部交点与两条主边，与玻璃轮廓求最近交点生成三角形缺角区域
    # 收集玻璃主体轮廓（与矩形差法重复一次，后续可优化成复用）
    corner_contour_q_defects = []
    try:
        kernel_qc = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges_qc = preprocess_for_hough_enhanced(roi_gray, params)
        edges_qc_dil = cv2.dilate(edges_qc, kernel_qc, iterations=1)
        cnts_qc, _ = cv2.findContours(edges_qc_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_qc:
            main_cnt_qc = max(cnts_qc, key=cv2.contourArea)
            if cv2.contourArea(main_cnt_qc) > 10.0 and cv2.arcLength(main_cnt_qc, True) >= 1200.0:
                mask_cnt_qc = np.zeros(roi_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask_cnt_qc, [main_cnt_qc], -1, 255, thickness=-1)

                # 轮廓点数组（便于几何计算）
                cnt_pts = main_cnt_qc.reshape(-1,2).astype(float)

                def _ray_intersect_contour(origin: np.ndarray, dir_vec: np.ndarray, contour_points: np.ndarray, t_min: float = 2.0):
                    # 与多边形每条边求交，返回 t 最小的交点与所在边索引
                    d = dir_vec.astype(float)
                    n = len(contour_points)
                    if n < 2:
                        return None
                    d_norm = np.linalg.norm(d)
                    if d_norm < 1e-9:
                        return None
                    d = d / d_norm
                    best = None
                    for k in range(n):
                        a = contour_points[k]
                        b = contour_points[(k+1) % n]
                        e = b - a
                        M = np.array([[d[0], -e[0]],[d[1], -e[1]]], dtype=float)
                        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
                        if abs(det) < 1e-9:
                            continue
                        rhs = (a - origin).astype(float)
                        t = ( rhs[0]*M[1,1] - rhs[1]*M[0,1]) / det
                        u = ( M[0,0]*rhs[1] - M[1,0]*rhs[0]) / det
                        if t >= t_min and 0.0 <= u <= 1.0:
                            P = origin + d * t
                            if (best is None) or (t < best[0]):
                                best = (t, P, k)
                    return best  # (t_min, point, edge_index)

                def _nearest_contour_edge_index(P: np.ndarray, contour_points: np.ndarray) -> int:
                    # 估计 P 所在的最近多边形边索引
                    n = len(contour_points)
                    best_idx = 0; best_d = 1e18
                    for k in range(n):
                        a = contour_points[k]
                        b = contour_points[(k+1) % n]
                        ab = b - a
                        L2 = float(np.dot(ab, ab))
                        if L2 < 1e-12:
                            d = float(np.linalg.norm(P - a))
                        else:
                            t = np.clip(float(np.dot(P - a, ab) / L2), 0.0, 1.0)
                            proj = a + ab * t
                            d = float(np.linalg.norm(P - proj))
                        if d < best_d:
                            best_d = d; best_idx = k
                    return best_idx

                def _ray_intersect_contour_thick(origin: np.ndarray, dir_vec: np.ndarray, contour_points: np.ndarray, t_min: float = 2.0, stripe_half_px: int = 2):
                    # 使用加粗的射线（条带）与轮廓边界相交检测，避免共线退化；返回首个命中点
                    d = dir_vec.astype(float)
                    if np.linalg.norm(d) < 1e-9:
                        return None
                    d = d / np.linalg.norm(d)
                    nvec = np.array([-d[1], d[0]], dtype=float)
                    H, W = roi_gray.shape
                    # 绘制轮廓边界到掩码（细线），仅用于几何检测，不参与最终可视化
                    edge_mask = np.zeros((H, W), dtype=np.uint8)
                    try:
                        cv2.polylines(edge_mask, [main_cnt_qc], True, 255, 1)
                    except Exception:
                        pass
                    T_max = float(max(H, W) * 2.0)
                    step = 1.0
                    t = float(max(t_min, 0.0))
                    while t <= T_max:
                        p = origin + d * t
                        for off in range(-int(stripe_half_px), int(stripe_half_px) + 1):
                            q = p + nvec * float(off)
                            qx, qy = int(round(q[0])), int(round(q[1]))
                            if 0 <= qx < W and 0 <= qy < H and edge_mask[qy, qx] != 0:
                                hit_pt = np.array([float(qx), float(qy)], dtype=float)
                                idx = _nearest_contour_edge_index(hit_pt, contour_points)
                                return (t, hit_pt, idx)
                        t += step
                    return None

                def _build_contour_path(i1_idx: int, i1_pt: np.ndarray, i2_idx: int, i2_pt: np.ndarray, contour_points: np.ndarray):
                    n = len(contour_points)
                    # forward: from i1_idx+1 ... to i2_idx (inclusive)
                    path_f = [i1_pt]
                    k = (i1_idx + 1) % n
                    while k != ((i2_idx + 1) % n):
                        path_f.append(contour_points[k])
                        k = (k + 1) % n
                    path_f.append(i2_pt)
                    # backward: from i1_idx down to i2_idx+1
                    path_b = [i1_pt]
                    k = i1_idx
                    while k != i2_idx:
                        path_b.append(contour_points[k])
                        k = (k - 1 + n) % n
                    path_b.append(i2_pt)
                    def _path_len(path):
                        return float(sum(np.linalg.norm(np.array(path[m+1],dtype=float) - np.array(path[m],dtype=float)) for m in range(len(path)-1)))
                    return (path_f, _path_len(path_f)), (path_b, _path_len(path_b))

                # 基于之前确认的角点集合逐一处理缺角：避免在此阶段再次枚举所有主边组合
                corner_inters = []  # 列表元素: (i, j, cp)
                H_roi2, W_roi2 = roi_gray.shape[:2]
                for (ii, jj, cp_arr) in paired_corners:
                    try:
                        xi, yi = float(cp_arr[0]), float(cp_arr[1])
                        if 0 <= xi < W_roi2 and 0 <= yi < H_roi2:
                            # 角点跨主体（cluster）则丢弃，避免玻璃1水平与玻璃2竖直的交点
                            if cluster_labels.size > max(ii, jj) and cluster_labels[ii] == cluster_labels[jj]:
                                if _DBG_PRINT and _DBG_LEVEL >= 1:
                                    try:
                                        _dprint(f"[DBG] reuse paired corner ({xi:.1f},{yi:.1f}) from ({ii},{jj}) in same cluster {int(cluster_labels[ii])}")
                                    except Exception:
                                        pass
                                corner_inters.append((ii, jj, np.array([xi, yi], dtype=float)))
                    except Exception:
                        continue

                # 若上游配对阶段因端点门槛未覆盖到由“理想竖直边 + 水平边”形成的交点，
                # 这里补充一轮基于角度类别的交点收集，确保理想竖直边参与 Q 检测。
                try:
                    vertical_tol_deg_aug = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
                except Exception:
                    vertical_tol_deg_aug = 10.0
                try:
                    horizontal_tol_deg_aug = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', vertical_tol_deg_aug))
                except Exception:
                    horizontal_tol_deg_aug = vertical_tol_deg_aug

                def _is_vertical_aug(seg):
                    ang = _angle_to_x_axis_deg(seg)
                    return ang >= (90.0 - vertical_tol_deg_aug)

                def _is_horizontal_aug(seg):
                    ang = _angle_to_x_axis_deg(seg)
                    return ang <= float(horizontal_tol_deg_aug)

                idx_vertical = [(iv, e) for iv, e in enumerate(edges_for_drawing) if _is_vertical_aug(e) and (iv in allowed_pair_lines)]
                idx_horizontal = [(ih, e) for ih, e in enumerate(edges_for_drawing) if _is_horizontal_aug(e) and (ih in allowed_pair_lines)]

                # 已有角点坐标集合用于去重
                existing_corners = [np.array(cp_arr, dtype=float) for (_, _, cp_arr) in corner_inters]

                Hc, Wc = roi_gray.shape[:2]
                for iv, vseg in idx_vertical:
                    for ih, hseg in idx_horizontal:
                        if iv == ih:
                            continue
                        # 主体过滤：不同 cluster 不再补充交点
                        if cluster_labels.size > max(iv, ih) and cluster_labels[iv] != cluster_labels[ih]:
                            continue
                        inter_pt = find_line_intersection(vseg, hseg)
                        if inter_pt is None:
                            # 容差补偿：以水平线段在前调用模糊交点（内部进行角度判定）
                            inter_pt = _fuzzy_hv_intersection(hseg, vseg, q_align_tol_px)
                            if inter_pt is None:
                                continue
                        xi, yi = float(inter_pt[0]), float(inter_pt[1])
                        if not (0.0 <= xi < float(Wc) and 0.0 <= yi < float(Hc)):
                            continue
                        # 与已收集角点去重
                        is_dup = False
                        for ec in existing_corners:
                            if float(np.linalg.norm(ec - np.array([xi, yi], dtype=float))) <= 3.0:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        if _DBG_PRINT and _DBG_LEVEL >= 1:
                            try:
                                cv = int(cluster_labels[iv]) if len(cluster_labels)>iv else -1
                                ch = int(cluster_labels[ih]) if len(cluster_labels)>ih else -1
                                _dprint(f"[DBG] add corner (supplement) V{iv}(c{cv})-H{ih}(c{ch}) -> ({xi:.1f},{yi:.1f})")
                            except Exception:
                                pass
                        corner_inters.append((iv, ih, np.array([xi, yi], dtype=float)))
                        existing_corners.append(np.array([xi, yi], dtype=float))

                # 计算玻璃主体轮廓质心，作为“内侧”参考方向
                try:
                    m_main = cv2.moments(main_cnt_qc)
                    if float(m_main.get('m00', 0.0)) != 0.0:
                        cnt_center = np.array([
                            float(m_main.get('m10', 0.0)) / float(m_main.get('m00', 1.0)),
                            float(m_main.get('m01', 0.0)) / float(m_main.get('m00', 1.0))
                        ], dtype=float)
                    else:
                        cnt_center = (np.mean(cnt_pts, axis=0).astype(float) if cnt_pts.size > 0 else np.array([W_roi2/2.0, H_roi2/2.0], dtype=float))
                except Exception:
                    cnt_center = np.array([W_roi2/2.0, H_roi2/2.0], dtype=float)

                for (idx_i, idx_j, cp) in corner_inters:
                    # 直接使用该交点对应的两条主边
                    chosen = [(idx_i, edges_for_drawing[idx_i]), (idx_j, edges_for_drawing[idx_j])]
                    # 新增规则：若角点在玻璃主体轮廓上或距离轮廓<=6px，则跳过该角的Q检测，避免边缘轻微毛刺被误判为缺角
                    try:
                        dist_min = float(np.min(np.linalg.norm(cnt_pts - cp, axis=1))) if cnt_pts.size > 0 else 9999.0
                    except Exception:
                        dist_min = 9999.0
                    # 从配置读取角点到主体轮廓的最小距离阈值（像素），默认 16.0
                    try:
                        min_corner_dist_px = float(params.get('DEFECT_DETECTION', {}).get('Q_CORNER_CONTOUR_MIN_DIST_PX', 16.0))
                    except Exception:
                        min_corner_dist_px = 16.0
                    if dist_min <= float(min_corner_dist_px):
                        continue
                    # 方向判定改为“基于轮廓的双向试探”：对每条主边，分别沿端点方向发射射线，选择命中距离更近的一侧
                    inter_hits = []  # (point, edge_index)
                    ray_hits_dbg = []
                    chosen_dirs = []
                    for _, seg in chosen:
                        p1 = np.array(seg[:2], dtype=float); p2 = np.array(seg[2:], dtype=float)
                        # 候选方向：指向两个端点
                        cands = []
                        for tgt in (p1, p2):
                            v = tgt - cp
                            n = float(np.linalg.norm(v))
                            if n <= 1e-6:
                                continue
                            u0 = v / n
                            hit = _ray_intersect_contour(cp, u0, cnt_pts, t_min=1.0)
                            if hit is not None:
                                cands.append((hit, u0))  # ((t, P, edge_idx), dir)
                        # 若未命中，使用“加粗条带”相交作为退路
                        if not cands:
                            tmp_cands = []
                            for tgt in (p1, p2):
                                v = tgt - cp
                                n = float(np.linalg.norm(v))
                                if n <= 1e-6:
                                    continue
                                u0 = v / n
                                # 加宽射线条带到 11px
                                hit2 = _ray_intersect_contour_thick(cp, u0, cnt_pts, t_min=1.0, stripe_half_px=7)
                                if hit2 is not None:
                                    tmp_cands.append((hit2, u0))
                            cands = tmp_cands

                        if cands:
                            # 优先选择“指向本主体(cluster)中心”的候选；否则退回最短 t；主体中心缺失时退回轮廓中心
                            cluster_center_ref = None
                            try:
                                # 取该线段的 cluster id（选第一条线即可，因为 corner 两条线同主体）
                                c_id = int(cluster_labels[idx_i]) if cluster_labels.size > idx_i else None
                                if c_id is not None and c_id in cluster_centers:
                                    cluster_center_ref = cluster_centers[c_id]
                            except Exception:
                                cluster_center_ref = None
                            inward_ref = (cluster_center_ref - cp) if cluster_center_ref is not None else (cnt_center - cp)
                            inward_norm = float(np.linalg.norm(inward_ref))
                            if inward_norm > 1e-6:
                                inward_dir = inward_ref / inward_norm
                                inward_cands = [it for it in cands if float(np.dot(it[1], inward_dir)) > 0.0]
                                selected = inward_cands if inward_cands else cands
                            else:
                                selected = cands
                            selected.sort(key=lambda it: it[0][0])
                            (t_sel, pt_sel, ei_sel), u_sel = selected[0]
                            inter_hits.append((pt_sel, ei_sel))
                            chosen_dirs.append(u_sel)
                            ray_hits_dbg.append({'seg': (tuple(map(int, cp)), (int(round(pt_sel[0])), int(round(pt_sel[1]))))})
                        else:
                            chosen_dirs.append(None)
                    # 若双边均获得命中则走三角形法；否则尝试“单射线平移平行四边形”法
                    if len(inter_hits) != 2 or any(d is None for d in chosen_dirs):
                        # 新方法：当竖直边方向的射线未命中而水平边命中时，
                        # 使用理想竖直边从角点出发裁剪一段固定长度，结合水平命中点与角点构成三角形作为缺角候选。
                        try:
                            if len(inter_hits) == 1 and any(d is not None for d in chosen_dirs):
                                hit_idx_local = 0 if (chosen_dirs[0] is not None) else 1
                                miss_idx_local = 1 - hit_idx_local
                                seg_hit = chosen[hit_idx_local][1]
                                seg_miss = chosen[miss_idx_local][1]

                                def _is_vertical_local(seg):
                                    ang = _angle_to_x_axis_deg(seg)
                                    return ang >= (90.0 - vertical_tol_deg)
                                def _is_horizontal_local(seg):
                                    ang = _angle_to_x_axis_deg(seg)
                                    return ang <= float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', vertical_tol_deg))

                                # 仅处理“未命中为竖直，命中为水平”的情况
                                if not (_is_vertical_local(seg_miss) and _is_horizontal_local(seg_hit)):
                                    raise RuntimeError('new-tri: pattern not matched (need vertical-miss & horizontal-hit)')

                                # 命中点（水平射线）
                                pt_hit, _ = inter_hits[0]
                                pt_hit = np.array(pt_hit, dtype=float)

                                # 侧向选择策略更新：按 ROI 上/下两半的平均亮度选择较暗侧对应端点（默认）。
                                # 可选保留：'horizontal' 使用“水平边法线”两侧的 Canny 白点多数侧；否则回退到旧的 Canny 上/下多数侧。
                                vm_p1 = np.array(seg_miss[:2], dtype=float)
                                vm_p2 = np.array(seg_miss[2:], dtype=float)
                                v_vec = vm_p2 - vm_p1
                                v_norm = float(np.linalg.norm(v_vec))
                                v_dir = (v_vec / v_norm) if v_norm > 1e-6 else np.array([0.0, 1.0], dtype=float)
                                edge_map_loc = edges_qc_dil if 'edges_qc_dil' in locals() else edges_qc
                                v_clip = None
                                # 首选：基于亮度的上下半区选择
                                try:
                                    H_g, W_g = roi_gray.shape[:2]
                                    xs = np.arange(W_g, dtype=np.float32)
                                    ys = np.arange(H_g, dtype=np.float32)
                                    Xg, Yg = np.meshgrid(xs, ys)
                                    dots_v = (Xg - float(cp[0])) * float(v_dir[0]) + (Yg - float(cp[1])) * float(v_dir[1])
                                    mask_up = dots_v > 0.0
                                    mask_dn = dots_v < 0.0
                                    # 端点投影判定“上/下”端
                                    t1 = float(np.dot(vm_p1 - cp, v_dir))
                                    t2 = float(np.dot(vm_p2 - cp, v_dir))
                                    v_up = vm_p1 if t1 >= t2 else vm_p2
                                    v_dn = vm_p2 if v_up is vm_p1 else vm_p1
                                    # 计算两半区的平均亮度
                                    if np.any(mask_up) and np.any(mask_dn):
                                        mean_up = float(np.mean(roi_gray[mask_up]))
                                        mean_dn = float(np.mean(roi_gray[mask_dn]))
                                        # 选择较暗侧对应的端点
                                        v_clip = v_up if mean_up < mean_dn else v_dn
                                        if _DBG_PRINT and _DBG_LEVEL >= 1:
                                            _dprint(f"[DBG] vertical-clip by brightness: mean_up={mean_up:.1f} mean_dn={mean_dn:.1f} -> pick={'UP' if mean_up < mean_dn else 'DOWN'}")
                                except Exception:
                                    v_clip = None
                                # 若亮度路径未能判定，则尝试保留旧的 Canny 统计方式
                                if v_clip is None and edge_map_loc is not None and edge_map_loc.size > 0:
                                    ys, xs = np.nonzero(edge_map_loc)
                                    if ys.size > 0:
                                        sel_mode = str(params.get('DEFECT_DETECTION', {}).get('Q_TRI_VERTICAL_CLIP_SIDE_MODE', 'vertical')).lower()
                                        dx_all = xs.astype(np.float32) - float(cp[0])
                                        dy_all = ys.astype(np.float32) - float(cp[1])
                                        if sel_mode == 'horizontal':
                                            # 以“水平边法线”划分两侧
                                            sh_p1 = np.array(seg_hit[:2], dtype=float)
                                            sh_p2 = np.array(seg_hit[2:], dtype=float)
                                            sh_vec = sh_p2 - sh_p1
                                            sh_nrm = float(np.linalg.norm(sh_vec))
                                            if sh_nrm > 1e-6:
                                                u_h = sh_vec / sh_nrm
                                            else:
                                                u_h = np.array([1.0, 0.0], dtype=float)
                                            n_h = np.array([-u_h[1], u_h[0]], dtype=float)  # 水平边法线
                                            dots_h = dx_all * float(n_h[0]) + dy_all * float(n_h[1])
                                            c_pos = int((dots_h > 0.0).sum())
                                            c_neg = int((dots_h < 0.0).sum())
                                            s1 = float(np.dot(vm_p1 - cp, n_h))
                                            s2 = float(np.dot(vm_p2 - cp, n_h))
                                            if c_pos != c_neg:
                                                # 多数侧对应的端点（端点在 n_h 正侧则匹配 c_pos，多数为负侧则匹配 c_neg）
                                                v_clip = vm_p1 if (s1 >= 0 and c_pos > c_neg) or (s1 < 0 and c_neg > c_pos) else vm_p2
                                            # 若持平则后续进入 cluster/面积回退
                                        else:
                                            # 默认 vertical：以上下两侧统计
                                            dots_v = dx_all * float(v_dir[0]) + dy_all * float(v_dir[1])
                                            c_up = int((dots_v > 0.0).sum())
                                            c_dn = int((dots_v < 0.0).sum())
                                            # 端点投影，用于确定哪端在"上"或"下"
                                            t1 = float(np.dot(vm_p1 - cp, v_dir))
                                            t2 = float(np.dot(vm_p2 - cp, v_dir))
                                            v_up = vm_p1 if t1 >= t2 else vm_p2
                                            v_dn = vm_p2 if v_up is vm_p1 else vm_p1
                                            if c_up != c_dn:
                                                v_clip = v_up if c_up > c_dn else v_dn
                                        # 若统计不分胜负或前述未定，使用 cluster 中心方向偏好
                                        if v_clip is None:
                                            try:
                                                c_id_local = int(cluster_labels[idx_i]) if cluster_labels.size > idx_i else None
                                                cc_ref = cluster_centers.get(c_id_local, None)
                                                if cc_ref is not None:
                                                    cand_scores = {}
                                                    for cand in [vm_p1, vm_p2]:
                                                        vec_c = (cc_ref - cp); nc = float(np.linalg.norm(vec_c))
                                                        if nc > 1e-6:
                                                            cand_scores[tuple(cand.tolist())] = float(np.dot((cand - cp)/float(np.linalg.norm(cand - cp)+1e-9), vec_c / nc))
                                                    if cand_scores:
                                                        v_clip = np.array(max(cand_scores.items(), key=lambda kv: kv[1])[0], dtype=float)
                                            except Exception:
                                                pass
                                if v_clip is None:
                                    # 退回面积更大端点
                                    def _tri_area(cp_pt, a, b):
                                        tri = np.vstack([cp_pt, a, b]).astype(np.float32)
                                        tri_cnt = tri.reshape((-1,1,2)).astype(np.int32)
                                        return float(cv2.contourArea(tri_cnt))
                                    area1 = _tri_area(cp, pt_hit, vm_p1)
                                    area2 = _tri_area(cp, pt_hit, vm_p2)
                                    v_clip = vm_p1 if area1 >= area2 else vm_p2

                                # 构造三角形：cp, pt_hit(水平命中), v_clip(竖直裁剪)
                                tri = np.vstack([cp, pt_hit, v_clip]).astype(np.float32)
                                tri_cnt = tri.reshape((-1,1,2)).astype(np.int32)
                                area_px = float(cv2.contourArea(tri_cnt))
                                if area_px <= 1.0:
                                    raise RuntimeError('new-tri: area too small')
                                if _DBG_PRINT and _DBG_LEVEL >= 1:
                                    _dprint(f"[DBG] Q-tri vertical-clip cp=({cp[0]:.1f},{cp[1]:.1f}) hit=({pt_hit[0]:.1f},{pt_hit[1]:.1f}) vclip=({v_clip[0]:.1f},{v_clip[1]:.1f}) area_px={area_px:.1f}")

                                r3 = cv2.minAreaRect(tri_cnt)
                                (cx3, cy3), (rw3, rh3), ang3 = r3
                                width_mm3 = (min(rw3, rh3) / float(pixels_per_mm)) if pixels_per_mm else 0.0
                                length_mm3 = (max(rw3, rh3) / float(pixels_per_mm)) if pixels_per_mm else 0.0
                                area_mm2_3 = (rw3 * rh3) / float(pixels_per_mm * pixels_per_mm) if pixels_per_mm else 0.0
                                try:
                                    q_cfg3 = float(params.get('DEFECT_DETECTION', {}).get('Q_MAX_SIDE_MM', 0.0))
                                except Exception:
                                    q_cfg3 = 0.0
                                if q_cfg3 is not None and q_cfg3 > 0.0:
                                    max_side_ok3 = (length_mm3 <= q_cfg3 and width_mm3 <= q_cfg3)
                                else:
                                    max_side_ok3 = True
                                if width_mm3 >= 5.0 and area_mm2_3 >= 25.0 and max_side_ok3:
                                    box3 = cv2.boxPoints(r3).astype(np.int32)
                                    corner_contour_q_defects.append({
                                        'type': 'Q',
                                        'origin': 'tri_vertical_clip',
                                        'min_area_rect': r3,
                                        'box_points': box3,
                                        'region_contour': tri_cnt,
                                        'center': (int(round(cx3)), int(round(cy3))),
                                        'corner_point': tuple(map(int, cp)),
                                        'intersections': [tuple(map(int, pt_hit))],
                                        'ray_segments': [ {'seg': (tuple(map(int, cp)), (int(round(pt_hit[0])), int(round(pt_hit[1]))))} ]
                                    })
                        except Exception:
                            pass
                        continue
                    # 可选内收微调：将方向向双角平分方向内收 Q_RAY_INWARD_DEG（默认1.8°），仅用于可视化射线，命中点沿原命中点保持
                    try:
                        inward_deg = float(params.get('DEFECT_DETECTION', {}).get('Q_RAY_INWARD_DEG', 1.8))
                    except Exception:
                        inward_deg = 1.8
                    def _rotate(vec: np.ndarray, deg: float) -> np.ndarray:
                        th = np.deg2rad(deg)
                        c, s = float(np.cos(th)), float(np.sin(th))
                        R = np.array([[c, -s], [s, c]], dtype=float)
                        return (R @ vec.reshape(2,)).reshape(2,)
                    if all(d is not None for d in chosen_dirs):
                        m = chosen_dirs[0] + chosen_dirs[1]
                        mn = float(np.linalg.norm(m))
                        if mn > 1e-6:
                            m_dir = m / mn
                            ray_hits_dbg = []
                            # 重建仅用于可视化的“内收”射线段
                            for k, (pt_hit, _) in enumerate(inter_hits):
                                u = chosen_dirs[k]
                                cand1 = _rotate(u, inward_deg)
                                cand2 = _rotate(u, -inward_deg)
                                u_vis = cand1 if float(np.dot(cand1, m_dir)) >= float(np.dot(cand2, m_dir)) else cand2
                                ray_hits_dbg.append({'seg': (tuple(map(int, cp)), (int(round(pt_hit[0])), int(round(pt_hit[1]))))})
                    (i1_pt, i1_idx), (i2_pt, i2_idx) = inter_hits[0], inter_hits[1]

                    # 任一交点与角点距离 < 5mm 则跳过此角的Q检测（避免微小切角被误判）
                    try:
                        th_px = float(5.0 * pixels_per_mm) if pixels_per_mm else None
                    except Exception:
                        th_px = None
                    if th_px is not None and th_px > 0:
                        d1 = float(np.linalg.norm(np.array(i1_pt, dtype=float) - cp))
                        d2 = float(np.linalg.norm(np.array(i2_pt, dtype=float) - cp))
                        if d1 < th_px or d2 < th_px:
                            continue

                    # 改为三角形缺角区域：角点 + 两射线命中点
                    tri = np.array([
                        [cp[0], cp[1]],
                        [i1_pt[0], i1_pt[1]],
                        [i2_pt[0], i2_pt[1]]
                    ], dtype=np.float32)
                    tri_cnt = tri.reshape((-1,1,2)).astype(np.int32)
                    tri_area_px = float(cv2.contourArea(tri_cnt))
                    if tri_area_px <= 1.0:
                        continue
                    if _DBG_PRINT and _DBG_LEVEL >= 1:
                        _dprint(f"[DBG] Q-tri cp=({cp[0]:.1f},{cp[1]:.1f}) i1=({i1_pt[0]:.1f},{i1_pt[1]:.1f}) i2=({i2_pt[0]:.1f},{i2_pt[1]:.1f}) area={tri_area_px:.1f}")
                    # 计算最小外接矩形及尺寸
                    r2 = cv2.minAreaRect(tri_cnt)
                    (cx, cy), (rw, rh), ang = r2
                    width_mm = (min(rw, rh) / float(pixels_per_mm)) if pixels_per_mm else 0.0
                    length_mm = (max(rw, rh) / float(pixels_per_mm)) if pixels_per_mm else 0.0
                    area_mm2 = (rw * rh) / float(pixels_per_mm * pixels_per_mm) if pixels_per_mm else 0.0
                    try:
                        p_def_local2 = params.get('DEFECT_DETECTION', {})
                        q_cfg2 = float(p_def_local2.get('Q_MAX_SIDE_MM', 0.0))
                    except Exception:
                        q_cfg2 = 0.0
                    # 同样取消硬性上限：按配置生效，否则不限制
                    if q_cfg2 is not None and q_cfg2 > 0.0:
                        max_side_ok2 = (length_mm <= q_cfg2 and width_mm <= q_cfg2)
                    else:
                        max_side_ok2 = True
                    # 基本阈值：短边>=5mm，面积>=25mm^2，长短边均不超过上限
                    if width_mm >= 5.0 and area_mm2 >= 25.0 and max_side_ok2:
                        box2 = cv2.boxPoints(r2).astype(np.int32)
                        corner_contour_q_defects.append({
                            'type': 'Q',
                            'min_area_rect': r2,
                            'box_points': box2,
                            'region_contour': tri_cnt,  # 三角形轮廓
                            'center': (int(round(cx)), int(round(cy))),
                            'origin': 'corner_contour',
                            'corner_point': tuple(map(int, cp)),
                            'triangle_area_px': int(round(tri_area_px)),
                            'intersections': [tuple(map(int, i1_pt)), tuple(map(int, i2_pt))],
                            'ray_segments': [rh['seg'] for rh in ray_hits_dbg]
                        })
                # 去重：同一角点附近的 Q 仅保留一个（按中心距离阈值选三角像素面积较大者）
                try:
                    if corner_contour_q_defects:
                        dist_thr = float(params.get('DEFECT_DETECTION', {}).get('Q_DEDUP_CENTER_DIST_PX', 12.0))
                        dedup = []
                        for nd in corner_contour_q_defects:
                            c_new = np.array(nd.get('center', (0,0)), dtype=float)
                            picked = False
                            for idx_old, od in enumerate(dedup):
                                c_old = np.array(od.get('center', (0,0)), dtype=float)
                                if float(np.linalg.norm(c_new - c_old)) <= dist_thr:
                                    # 替换为面积更大者
                                    a_new = float(nd.get('triangle_area_px', 0.0) or 0.0)
                                    a_old = float(od.get('triangle_area_px', 0.0) or 0.0)
                                    if a_new > a_old:
                                        dedup[idx_old] = nd
                                    picked = True
                                    break
                            if not picked:
                                dedup.append(nd)
                        corner_contour_q_defects = dedup
                except Exception:
                    pass
    except Exception:
        pass

    def _dedup_q(existing, new_list, dist_px: float = 20.0):
        kept = existing[:]
        for nd in new_list:
            c_new = np.array(nd.get('center', (0,0)), dtype=float)
            duplicated = False
            for od in kept:
                if od.get('type') != 'Q':
                    continue
                c_old = np.array(od.get('center', (0,0)), dtype=float)
                if np.linalg.norm(c_new - c_old) < dist_px:
                    duplicated = True
                    break
            if not duplicated:
                kept.append(nd)
        return kept

    # 方法B（HV 闭合区域）已禁用：为提升速度，仅保留 corner_contour 方法
    hv_corner_q_defects = []

    # 仅使用角点 + 轮廓三角法作为缺角(Q)检测结果
    rect_q_defects = corner_contour_q_defects

    # 若聚类凸包相交，则阻断 Q 生成（相机级别过滤在上层进行）
    if 'qx_blocked' in locals() and bool(qx_blocked):
        rect_q_defects = []

    # 新增：对 Q 缺陷进行“平行四边形 + 边缘点集群”验证过滤
    try:
        def _parallelogram_from_triangle(tri_pts: np.ndarray):
            # tri_pts: (3,2) float32 in ROI coords
            A, B, C = tri_pts[0].astype(float), tri_pts[1].astype(float), tri_pts[2].astype(float)
            # 找到最长边与第三点
            dAB = float(np.linalg.norm(A - B))
            dBC = float(np.linalg.norm(B - C))
            dCA = float(np.linalg.norm(C - A))
            if dAB >= dBC and dAB >= dCA:
                P, Q, R = A, B, C
            elif dBC >= dAB and dBC >= dCA:
                P, Q, R = B, C, A
            else:
                P, Q, R = C, A, B
            M = (P + Q) / 2.0
            D = (2.0 * M) - R  # 关于对角线中点的对称点
            pts = np.vstack([P, Q, R, D]).astype(np.float32)
            # 以质心排序，确保顶点顺序形成简单多边形
            cen = np.mean(pts, axis=0)
            ang = np.arctan2(pts[:,1] - cen[1], pts[:,0] - cen[0])
            order = np.argsort(ang)
            return pts[order]

        def _q_parallelogram_cluster_ok(q_def: dict, edges_img: np.ndarray) -> bool:
            # 从 q_def['region_contour'] 还原三角形顶点
            tri_cnt = q_def.get('region_contour', None)
            if tri_cnt is None or not isinstance(tri_cnt, np.ndarray) or tri_cnt.size < 6:
                return True  # 无法验证则放行
            tri_pts = tri_cnt.reshape(-1,2).astype(np.float32)
            if tri_pts.shape[0] != 3:
                return True
            H, W = edges_img.shape[:2]
            pg = _parallelogram_from_triangle(tri_pts)
            # 构造内区掩码（排除边界，使用腐蚀1px）
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [pg.astype(np.int32)], 255)
            kernel = np.ones((3,3), dtype=np.uint8)
            inner = cv2.erode(mask, kernel, iterations=1)
            # 在验证内区连通域前，剔除“靠近水平/竖直边（延长后）的条带区域”以避免边界噪声干扰
            try:
                stripe_half = int(params.get('DEFECT_DETECTION', {}).get('Q_PARALLELOGRAM_EXCLUDE_STRIPE_HALF_PX', 5))
            except Exception:
                stripe_half = 5
            stripe_half = max(0, int(stripe_half))
            if stripe_half > 0:
                exclude = np.zeros((H, W), dtype=np.uint8)
                # 优先使用 q_def['ray_segments']（两条射线，方向分别近水平/近竖直）
                rays = q_def.get('ray_segments', None)
                norm_rays = []
                if isinstance(rays, (list, tuple)) and len(rays) > 0:
                    for r in rays:
                        try:
                            if isinstance(r, dict) and 'seg' in r:
                                p1, p2 = r['seg']
                            else:
                                p1, p2 = r  # 形如((x1,y1),(x2,y2))
                            x1, y1 = float(p1[0]), float(p1[1])
                            x2, y2 = float(p2[0]), float(p2[1])
                            norm_rays.append(((x1, y1), (x2, y2)))
                        except Exception:
                            continue
                # 若没有射线，则用 corner_point → intersections 构造两条近似射线
                if not norm_rays:
                    try:
                        cp = q_def.get('corner_point', None)
                        inters = q_def.get('intersections', []) or []
                        if cp is not None and len(inters) >= 1:
                            for ip in inters[:2]:
                                norm_rays.append(((float(cp[0]), float(cp[1])), (float(ip[0]), float(ip[1]))))
                    except Exception:
                        pass
                # 画“延长后”的加粗条带
                if norm_rays:
                    L = float(max(H, W) * 2.0)
                    thick = int(2 * stripe_half + 1)
                    for (x1, y1), (x2, y2) in norm_rays[:2]:  # 仅取两条
                        dx, dy = float(x2 - x1), float(y2 - y1)
                        nrm = float(np.hypot(dx, dy))
                        if nrm < 1e-3:
                            continue
                        ux, uy = dx / nrm, dy / nrm
                        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                        sx, sy = cx - ux * L, cy - uy * L
                        ex, ey = cx + ux * L, cy + uy * L
                        p1e = (int(round(sx)), int(round(sy)))
                        p2e = (int(round(ex)), int(round(ey)))
                        cv2.line(exclude, p1e, p2e, 255, thickness=thick, lineType=cv2.LINE_AA)
                # 将条带从内区中剔除
                inner = cv2.bitwise_and(inner, cv2.bitwise_not(exclude))
            # 取内区的边缘点
            cand = cv2.bitwise_and(edges_img, edges_img, mask=inner)
            # 可选形态学连接（轻度）
            try:
                use_dilate = bool(params.get('DEFECT_DETECTION', {}).get('Q_PARALLELOGRAM_USE_DILATE', True))
            except Exception:
                use_dilate = True
            if use_dilate:
                cand = cv2.dilate(cand, kernel, iterations=1)
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((cand>0).astype(np.uint8), connectivity=8)
            if num_labels <= 1:
                return False
            # 阈值
            try:
                min_pixels = int(params.get('DEFECT_DETECTION', {}).get('Q_PARALLELOGRAM_MIN_EDGE_PIXELS', 50))
            except Exception:
                min_pixels = 50
            # 沿对角线的投影跨度要求
            P, Q = None, None
            # 取与三角形最长边一致的对角线方向
            A, B, C = tri_pts[0], tri_pts[1], tri_pts[2]
            dAB = float(np.linalg.norm(A - B))
            dBC = float(np.linalg.norm(B - C))
            dCA = float(np.linalg.norm(C - A))
            if dAB >= dBC and dAB >= dCA:
                P, Q = A, B
            elif dBC >= dAB and dBC >= dCA:
                P, Q = B, C
            else:
                P, Q = C, A
            diag_vec = (Q - P).astype(float)
            diag_len = float(np.linalg.norm(diag_vec))
            if diag_len <= 1.0:
                return False
            u = diag_vec / diag_len
            try:
                span_frac = float(params.get('DEFECT_DETECTION', {}).get('Q_PARALLELOGRAM_MIN_SPAN_FRAC', 0.25))
            except Exception:
                span_frac = 0.25
            need_span = float(span_frac * diag_len)
            # 遍历每个连通域，检查像素数与跨度
            for lbl in range(1, num_labels):
                cnt = int(stats[lbl, cv2.CC_STAT_AREA])
                if cnt < min_pixels:
                    continue
                ys, xs = np.where(labels == lbl)
                if xs.size == 0:
                    continue
                pts = np.vstack([xs.astype(float), ys.astype(float)]).T
                proj = np.dot(pts - P.reshape(1,2), u.reshape(2,))
                span = float(np.max(proj) - np.min(proj))
                if span >= need_span:
                    return True
            return False

        # 准备边缘图
        edges_pf = preprocess_for_hough_enhanced(roi_gray, params)
        filtered = []
        for nd in (rect_q_defects or []):
            try:
                if nd.get('type') != 'Q':
                    filtered.append(nd)
                    continue
                if _q_parallelogram_cluster_ok(nd, edges_pf):
                    filtered.append(nd)
                else:
                    # 过滤该 Q
                    pass
            except Exception:
                filtered.append(nd)
        rect_q_defects = filtered
    except Exception:
        # 任意错误不影响主流程，保守放行
        pass

    def sort_key_func(pair_indices):
        i, j = pair_indices
        line1, line2 = true_edges[i], true_edges[j]
        compatibility = get_quadrant_compatibility(edge_quadrants[i], edge_quadrants[j])
        min_dist = min(np.linalg.norm(line1[:2] - line2[:2]), np.linalg.norm(line1[:2] - line2[2:]),
                       np.linalg.norm(line1[2:] - line2[:2]), np.linalg.norm(line1[2:] - line2[2:]))
        return (compatibility, min_dist)
    
    if num_true_edges >= 2:
        # 仅在“允许配对”的主边集合中做组合（每 cluster 仅 1V+1H）。
        pairable_idx = sorted([idx for idx in range(num_true_edges) if idx in allowed_pair_lines])
        potential_pairs = sorted([(i, j) for i, j in combinations(pairable_idx, 2)], key=sort_key_func)
    else:
        potential_pairs = []

    for i, j in potential_pairs:
        # 新增：禁止跨主体的线对参与角点配对，避免玻璃1水平与玻璃2竖直配对
        try:
            if cluster_labels.size > max(i, j) and cluster_labels[i] != cluster_labels[j]:
                if _DBG_PRINT and _DBG_LEVEL >= 2:
                    try:
                        _dprint(f"[DBG] skip pair i={i} j={j} due to cluster mismatch ({int(cluster_labels[i])}-{int(cluster_labels[j])})")
                    except Exception:
                        pass
                continue
        except Exception:
            pass
        # 栅栏/分隔线逻辑适配：若存在稳定 glass_boundary 或 fences，防止跨玻璃或越界配对形成伪角
        def _is_vertical_for_pair(seg):
            try:
                ang = abs(np.degrees(np.arctan2(float(seg[3]-seg[1]), float(seg[2]-seg[0]))))
                if ang > 90.0: ang = 180.0 - ang
                v_tol = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
                return ang >= (90.0 - v_tol)
            except Exception:
                return False
        def _is_horizontal_for_pair(seg):
            try:
                ang = abs(np.degrees(np.arctan2(float(seg[3]-seg[1]), float(seg[2]-seg[0]))))
                if ang > 90.0: ang = 180.0 - ang
                h_tol = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_ANGLE_TOL_DEG', params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0)))
                return ang <= h_tol
            except Exception:
                return False
        # 仅在一条水平+一条竖直组合时应用边界约束
        seg_i = true_edges[i]; seg_j = true_edges[j]
        hv_combo = (_is_horizontal_for_pair(seg_i) and _is_vertical_for_pair(seg_j)) or (_is_horizontal_for_pair(seg_j) and _is_vertical_for_pair(seg_i))
        if hv_combo:
            # 获取边界参考
            fence_margin_local = float(params.get('DEFECT_DETECTION', {}).get('HORIZONTAL_EXTEND_FENCE_MARGIN_PX', 4.0)) if params.get('DEFECT_DETECTION', {}) else 4.0
            # 针对水平线所在玻璃, 计算禁止跨越的边界区间
            # 取水平线
            if _is_horizontal_for_pair(seg_i):
                h_seg = seg_i; v_seg_other = seg_j
            else:
                h_seg = seg_j; v_seg_other = seg_i
            xh1, yh1, xh2, yh2 = map(float, h_seg)
            h_mid_x = 0.5 * (xh1 + xh2)
            # fences 和 glass_boundary 若存在
            # 变量可能在上游定义: fences, glass_boundary；若不存在使用空或 None
            try:
                current_fences = fences if 'fences' in locals() else []
            except Exception:
                current_fences = []
            try:
                gb = glass_boundary if 'glass_boundary' in locals() else None
            except Exception:
                gb = None
            # 计算水平线所属侧并设置允许的 x 范围
            allow_min = -1e9; allow_max = 1e9
            if gb is not None:
                if h_mid_x <= gb:  # 左玻璃
                    allow_max = gb + fence_margin_local
                else:              # 右玻璃
                    allow_min = gb - fence_margin_local
            # 使用 fences 进一步收紧(只取距离水平线最近的内侧栅栏)
            if current_fences:
                seg_xmin = min(xh1, xh2); seg_xmax = max(xh1, xh2)
                left_cand = [fx for fx in current_fences if fx < seg_xmin]
                right_cand = [fx for fx in current_fences if fx > seg_xmax]
                if left_cand:
                    allow_min = max(allow_min, max(left_cand) - fence_margin_local)
                if right_cand:
                    allow_max = min(allow_max, min(right_cand) + fence_margin_local)
        else:
            allow_min = -1e9; allow_max = 1e9
        if all(endpoint_paired_status.get(i, [True,True])) or all(endpoint_paired_status.get(j, [True,True])):
            continue
        # 使用最新坐标的线段参与角点计算：优先采用 edges_for_drawing
        line1 = edges_for_drawing[i] if i < len(edges_for_drawing) else true_edges[i]
        line2 = edges_for_drawing[j] if j < len(edges_for_drawing) else true_edges[j]

        angle_between = calculate_angle_between_lines(line1, line2)
        max_extension_dist = _get_dist_px(
            p_defect,
            "CORNER_MAX_EXTENSION_DIST_PERPENDICULAR_MM" if abs(angle_between - 90.0) < p_defect["PERPENDICULAR_ANGLE_TOLERANCE"] else "CORNER_MAX_EXTENSION_DIST_NORMAL_MM",
            "CORNER_MAX_EXTENSION_DIST_PERPENDICULAR" if abs(angle_between - 90.0) < p_defect["PERPENDICULAR_ANGLE_TOLERANCE"] else "CORNER_MAX_EXTENSION_DIST_NORMAL",
            None,
            pixels_per_mm
        )

        intersection = find_line_intersection(line1, line2)
        if intersection is None:
            continue
        # 栅栏/边界过滤：若交点超出允许范围直接丢弃
        if not (allow_min - 1e-6 <= intersection[0] <= allow_max + 1e-6):
            if _DBG_PRINT and _DBG_LEVEL >= 2:
                _dprint(f"[DBG] skip pair i={i} j={j} intersection x={intersection[0]:.1f} outside [{allow_min:.1f},{allow_max:.1f}]")
            continue

        try:
            p1a = np.array(line1[:2], dtype=float); p1b = np.array(line1[2:], dtype=float)
            v1 = p1b - p1a; L1 = float(np.dot(v1, v1))
            t1 = float(np.dot(intersection - p1a, v1) / L1) if L1 > 1e-9 else -999.0
            p2a = np.array(line2[:2], dtype=float); p2b = np.array(line2[2:], dtype=float)
            v2 = p2b - p2a; L2 = float(np.dot(v2, v2))
            t2 = float(np.dot(intersection - p2a, v2) / L2) if L2 > 1e-9 else -999.0
        except Exception:
            t1 = -999.0; t2 = -999.0

        dists_i = [np.linalg.norm(intersection - line1[:2]), np.linalg.norm(intersection - line1[2:])]; endpoint_idx_i = np.argmin(dists_i)
        dists_j = [np.linalg.norm(intersection - line2[:2]), np.linalg.norm(intersection - line2[2:])]; endpoint_idx_j = np.argmin(dists_j)

        if endpoint_paired_status[i][endpoint_idx_i] or endpoint_paired_status[j][endpoint_idx_j]:
            continue

        corner_gap_px = _get_dist_px(p_defect, "CORNER_MAX_PHYSICAL_GAP_MM", "CORNER_MAX_PHYSICAL_GAP", None, pixels_per_mm)
        roi_ext_allow = 0.2 * float(roi_w)
        max_extend_eff = max(float(max_extension_dist), float(roi_ext_allow))
        is_physical = dists_i[endpoint_idx_i] < corner_gap_px and dists_j[endpoint_idx_j] < corner_gap_px
        is_valid_virtual = dists_i[endpoint_idx_i] < max_extend_eff and dists_j[endpoint_idx_j] < max_extend_eff

        inside_i = (0.0 <= t1 <= 1.0)
        inside_j = (0.0 <= t2 <= 1.0)
        accept_inside = inside_i and inside_j

        if (accept_inside or is_physical or is_valid_virtual) and (0 <= intersection[0] < roi_w and 0 <= intersection[1] < roi_h):
            endpoint_paired_status[i][endpoint_idx_i] = True; endpoint_paired_status[j][endpoint_idx_j] = True
            if not accept_inside:
                edges_for_drawing[i][endpoint_idx_i*2:(endpoint_idx_i*2)+2] = intersection
                edges_for_drawing[j][endpoint_idx_j*2:(endpoint_idx_j*2)+2] = intersection
            p1_near = line1[:2] if endpoint_idx_i == 0 else line1[2:]; p2_near = line2[:2] if endpoint_idx_j == 0 else line2[2:]
            p1_far = line1[2:] if endpoint_idx_i == 0 else line1[:2]
            p2_far = line2[2:] if endpoint_idx_j == 0 else line2[:2]
            if _DBG_PRINT and _DBG_LEVEL >= 1:
                try:
                    ci = int(cluster_labels[i]) if len(cluster_labels)>i else -1
                    cj = int(cluster_labels[j]) if len(cluster_labels)>j else -1
                    _dprint(f"[DBG] paired i={i} j={j} at ({intersection[0]:.1f},{intersection[1]:.1f}) inside={accept_inside} phys={is_physical} virt={is_valid_virtual} c=({ci},{cj})")
                except Exception:
                    pass
            try:
                # 记录角点（仅一次）供后续 Q 检测使用；索引基于 edges_for_drawing（已刷新过）
                paired_corners.append((int(i), int(j), np.array([float(intersection[0]), float(intersection[1])], dtype=float)))
            except Exception:
                pass

            def _handle_as_x_defect():
                angle = calculate_vertex_angle(p1_far, intersection, p2_far)
                # 仅在合理范围考虑 X（避免尖角/钝角极端值）
                if angle < 5.0 or angle > 175.0:
                    return
                # 按规则对 90° 邻域进行修约
                corrected_angle = _adjust_angle_near_90(angle)
                corrected_deviation_final = abs(corrected_angle - 90.0)
                angle_tolerance = p_defect.get("ANGLE_DEVIATION_TOLERANCE", 4.0)
                if corrected_deviation_final > angle_tolerance:
                    # X 型缺陷仅保留角度信息，不输出尺寸/size_label
                    corner_defects.append({"type": "X", "center": tuple(map(int, intersection)), "angle": corrected_angle})

            # 删除 Harris 缺角检测：统一仅按角度偏差尝试判定 X，不再生成 Q
            if 'qx_blocked' not in locals() or not bool(qx_blocked):
                _handle_as_x_defect()
            try:
                # 记录该交点用于后续过滤其附近的 B 误检
                non_q_intersections.append((float(intersection[0]), float(intersection[1])))
            except Exception:
                pass
        

    # 将斜边缺陷并入后续缺陷列表
    all_chipping_contours = []; chipping_defects = []
    use_block_based = bool(params.get("DEFECT_DETECTION", {}).get("BLOCK_BASED_CHIPPING_ENABLED", False))
    for edge in true_edges:
        if use_block_based:
            all_chipping_contours.extend(scan_edge_for_chipping_blocks(roi_gray, edge, params, pixels_per_mm))
        else:
            all_chipping_contours.extend(scan_edge_for_luminosity_defects(roi_gray, edge, params, pixels_per_mm))
    if all_chipping_contours:
        defect_canvas = np.zeros(roi_dims, dtype=np.uint8)
        cv2.drawContours(defect_canvas, all_chipping_contours, -1, 255, -1)
        kernel_mm = p_defect.get("MERGE_DEFECTS_KERNEL_MM")
        kernel = np.ones(_kernel_mm_to_px_odd(kernel_mm, pixels_per_mm, p_defect.get("MERGE_DEFECTS_KERNEL_SIZE", [5, 5])), np.uint8)
        merged_mask = cv2.morphologyEx(defect_canvas, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in final_contours:
            min_area_px2 = _get_area_px2(p_defect, "LUMINOSITY_MIN_AREA_MM2", "LUMINOSITY_MIN_AREA", None, pixels_per_mm)
            if cv2.contourArea(cnt) < min_area_px2:
                continue

            min_area_rect = cv2.minAreaRect(cnt)
            (w, h) = min_area_rect[1]
            width = min(w, h)
            length = max(w, h)
            aspect_ratio = length / width if width > 1e-6 else 0.0
            
            
            box_points = cv2.boxPoints(min_area_rect)
            box_points = np.intp(box_points)
            
            chipping_defects.append({
                "type": "B", 
                "box_points": box_points,
                "min_area_rect": min_area_rect,
                "contour": cnt 
            })
    
    # 取消崩边(B)的边缘屏蔽：不再按距离端点/边缘的阈值剔除
    # 并新增：剔除与主边缘过远(>5mm，或由配置 DEFECT_DETECTION.B_MAX_DISTANCE_TO_EDGE_MM 指定)的崩边误检
    # 计算每个 B 缺陷的最小“矩形点(四角+中心)”到任一主边缘线段的距离，超过阈值则丢弃
    b_max_edge_dist_px = _get_dist_px(p_defect, "B_MAX_DISTANCE_TO_EDGE_MM", None, 5.0, pixels_per_mm)

    def _min_dist_rect_to_edges(b_defect_obj, edge_list):
        try:
            # 取 minAreaRect 的四角点与中心点
            box = b_defect_obj.get('box_points')
            rect = b_defect_obj.get('min_area_rect')
            pts = []
            if box is not None and len(box) >= 4:
                pts.extend([np.array(p, dtype=float) for p in box])
            # 中心点：优先从 minAreaRect 读取，否则用四点均值
            if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                cx, cy = rect[0]
                pts.append(np.array([float(cx), float(cy)], dtype=float))
            elif box is not None and len(box) >= 4:
                center = np.mean(np.array(box, dtype=float), axis=0)
                pts.append(center)
            if not pts:
                return float('inf')

            min_dist = float('inf')
            for e in edge_list:
                a = np.array(e[:2], dtype=float); b = np.array(e[2:], dtype=float)
                for pt in pts:
                    # 使用点到直线（无限延长）的垂直距离
                    d = get_point_line_perpendicular_distance(pt, [a[0], a[1], b[0], b[1]])
                    if d < min_dist:
                        min_dist = d
                        if min_dist <= b_max_edge_dist_px:
                            return min_dist
            return min_dist
        except Exception:
            return float('inf')

    final_chipping_defects = []
    if true_edges:
        # 仅当 B 细长且窄时，才启用“距主边距离”门控；距离按中心点到直线（无限延长）计算
        try:
            ar_min_for_gate = float(p_defect.get('B_FILTER_PARALLEL_AR_MIN', 10.0))
        except Exception:
            ar_min_for_gate = 10.0
        try:
            min_side_mm_for_gate = float(p_defect.get('B_FILTER_PARALLEL_MIN_SIDE_MM', 2.0))
        except Exception:
            min_side_mm_for_gate = 2.0

        for bd in chipping_defects:
            rect = bd.get('min_area_rect')
            box = bd.get('box_points')
            # 计算中心点（优先用 minAreaRect 中心）
            if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                cx, cy = rect[0]
                center_pt = np.array([float(cx), float(cy)], dtype=float)
                w_px = float(rect[1][0] or 0.0)
                h_px = float(rect[1][1] or 0.0)
            else:
                if box is not None and len(box) >= 4:
                    box_np = np.array(box, dtype=float).reshape(-1, 2)
                    center_pt = np.mean(box_np, axis=0)
                    # 回推 w,h
                    try:
                        tmp_rect = cv2.minAreaRect(box_np.astype(np.float32))
                        w_px = float(tmp_rect[1][0] or 0.0)
                        h_px = float(tmp_rect[1][1] or 0.0)
                    except Exception:
                        w_px = 0.0; h_px = 0.0
                else:
                    # 缺乏几何信息，保守保留
                    final_chipping_defects.append(bd)
                    continue

            width_px = float(min(w_px, h_px))
            length_px = float(max(w_px, h_px))
            width_mm = width_px / float(pixels_per_mm if pixels_per_mm else 1.0)
            ar = (length_px / width_px) if width_px > 1e-6 else float('inf')

            need_distance_gate = (ar > ar_min_for_gate) and (width_mm < min_side_mm_for_gate)

            if need_distance_gate:
                # 计算中心点到所有主边直线的最小垂直距离（像素）
                min_d_px = float('inf')
                for e in true_edges:
                    a = np.array(e[:2], dtype=float); b = np.array(e[2:], dtype=float)
                    dpx = get_point_line_perpendicular_distance(center_pt, [a[0], a[1], b[0], b[1]])
                    if dpx < min_d_px:
                        min_d_px = dpx
                        if min_d_px <= b_max_edge_dist_px:
                            break
                if min_d_px <= b_max_edge_dist_px:
                    # 距主边足够近，保留该 B
                    final_chipping_defects.append(bd)
                else:
                    # 距主边过远，过滤该 B
                    pass
            else:
                # 不细长或不够窄：不启用距离门控，直接保留
                final_chipping_defects.append(bd)
    else:
        # 没有主边缘时，保守处理：保留原 B 列表（通常此场景不会出现，因为扫描依赖主边）
        final_chipping_defects = chipping_defects

    surviving_chipping_defects = []
    if corner_defects and final_chipping_defects:
        q_mask = np.zeros(roi_dims, dtype=np.uint8)
        for q_defect in corner_defects:
            if q_defect.get('type') == 'Q' and 'endpoints' in q_defect:
                q_contour = np.array([
                    q_defect['center'], 
                    q_defect['endpoints'][0], 
                    q_defect['endpoints'][1]
                ], dtype=np.int32)
                cv2.fillPoly(q_mask, [q_contour], 255)

        for b_defect in final_chipping_defects:
            b_mask = np.zeros(roi_dims, dtype=np.uint8)
            b_contour = b_defect.get('box_points')
            if b_contour is None:
                surviving_chipping_defects.append(b_defect)
                continue
            
            cv2.fillPoly(b_mask, [np.array(b_contour)], 255)
            
            intersection = cv2.bitwise_and(q_mask, b_mask)
            
            if cv2.countNonZero(intersection) == 0:
                surviving_chipping_defects.append(b_defect)
    else:
        surviving_chipping_defects = final_chipping_defects

    # 非Q交点附近(默认20mm)的 B 被视为误检并过滤
    if non_q_intersections and surviving_chipping_defects:
        b_near_nonq_radius_px = _get_dist_px(p_defect, "B_FILTER_NEAR_NONQ_INTERSECTION_RADIUS_MM", None, 20.0, pixels_per_mm)
        filtered_b = []
        for b_defect in surviving_chipping_defects:
            if b_defect.get('type') != 'B':
                filtered_b.append(b_defect)
                continue
            rect = b_defect.get('min_area_rect')
            center = None
            if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                cx, cy = rect[0]
                center = np.array([float(cx), float(cy)], dtype=float)
            else:
                box = b_defect.get('box_points')
                if box is not None and len(box) >= 4:
                    center = np.mean(np.array(box, dtype=float), axis=0)
            if center is None:
                filtered_b.append(b_defect)
                continue
            min_d = float('inf')
            for (ix, iy) in non_q_intersections:
                d = float(np.hypot(center[0] - ix, center[1] - iy))
                if d < min_d:
                    min_d = d
                    if min_d <= b_near_nonq_radius_px:
                        break
            if min_d <= b_near_nonq_radius_px:
                # 过滤掉该 B
                continue
            filtered_b.append(b_defect)
        surviving_chipping_defects = filtered_b

    # 将重叠/相交的 B 缺陷在数据层面进行合并，输出为单一缺陷（凸包 + minAreaRect 融合）
    if surviving_chipping_defects:
        # 仅对 B 进行合并；其他类型保持不变
        b_list = [bd for bd in surviving_chipping_defects if bd.get('type') == 'B']
        others = [bd for bd in surviving_chipping_defects if bd.get('type') != 'B']

        # 新增：只有尺寸至少 2mm x 2mm 的 B 才参与“邻域/重叠合并”（可配 DEFECT_DETECTION.B_MERGE_MIN_SIDE_MM）
        try:
            b_merge_min_side_mm = float(params.get('DEFECT_DETECTION', {}).get('B_MERGE_MIN_SIDE_MM', 2.0))
        except Exception:
            b_merge_min_side_mm = 2.0

        b_merge_list = []
        b_small_list = []  # 不满足 2mm x 2mm 的 B，不参与合并但保留原状
        for bd in b_list:
            rect = bd.get('min_area_rect')
            if rect is None or not isinstance(rect, tuple) or len(rect) < 2:
                b_small_list.append(bd)
                continue
            try:
                w_px, h_px = float(rect[1][0] or 0.0), float(rect[1][1] or 0.0)
                w_mm = w_px / float(pixels_per_mm if pixels_per_mm else 1.0)
                h_mm = h_px / float(pixels_per_mm if pixels_per_mm else 1.0)
                if min(w_mm, h_mm) >= b_merge_min_side_mm:
                    b_merge_list.append(bd)
                else:
                    b_small_list.append(bd)
            except Exception:
                b_small_list.append(bd)

        def _rotated_rect_intersect(rect1, rect2, box1, box2):
            """判断两个旋转矩形是否相交/包含。
            优先使用 cv2.rotatedRectangleIntersection；若返回不稳定，再使用凸多边形相交测试(cv2.intersectConvexConvex)。
            注意：彻底弃用基于 AABB 的 cv2.boundingRect 回退，避免与最短边过滤规则不一致。
            """
            try:
                ret, pts = cv2.rotatedRectangleIntersection(rect1, rect2)
                # ret: 0-不相交，1-相交，2-包含
                if ret in (1, 2):
                    if pts is not None and len(pts) >= 3:
                        # 面积很小的交集也算合并，避免碎片
                        area = cv2.contourArea(pts)
                        if area >= 1.0:
                            return True
                    else:
                        # 包含情形有时不会返回 pts，多数情况下可视为相交
                        return True

                # 回退：使用凸多边形相交（以 minAreaRect 的 box_points 为准）
                box1f = np.array(box1, dtype=np.float32).reshape(-1, 2)
                box2f = np.array(box2, dtype=np.float32).reshape(-1, 2)
                try:
                    inter_area, _ = cv2.intersectConvexConvex(box1f, box2f)
                    return float(inter_area) >= 1.0
                except Exception:
                    return False
            except Exception:
                return False

        # 并查集分组
        n = len(b_merge_list)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            rect_i = b_merge_list[i].get('min_area_rect')
            box_i = b_merge_list[i].get('box_points')
            if rect_i is None or box_i is None: 
                continue
            for j in range(i + 1, n):
                rect_j = b_merge_list[j].get('min_area_rect')
                box_j = b_merge_list[j].get('box_points')
                if rect_j is None or box_j is None:
                    continue
                if _rotated_rect_intersect(rect_i, rect_j, box_i, box_j):
                    union(i, j)

        groups = {}
        for idx in range(n):
            r = find(idx)
            groups.setdefault(r, []).append(idx)

        fused_b_list = []
        for _, indices in groups.items():
            if len(indices) == 1:
                fused_b_list.append(b_merge_list[indices[0]])
                continue
            # 融合：将所有 box_points 聚合做凸包，然后用 minAreaRect 得到新的旋转矩形
            pts = []
            for k in indices:
                box = b_merge_list[k].get('box_points')
                if box is not None and len(box) >= 4:
                    pts.extend([p for p in box])
            if not pts:
                # 回退：直接合并第一个
                fused_b_list.append(b_merge_list[indices[0]])
                continue
            pts_np = np.array(pts, dtype=np.float32)
            hull = cv2.convexHull(pts_np)
            fused_rect = cv2.minAreaRect(hull)
            fused_box = cv2.boxPoints(fused_rect)
            fused_box = np.intp(fused_box)

            fused_b_list.append({
                "type": "B",
                "box_points": fused_box,
                "min_area_rect": fused_rect,
                "contour": hull.astype(np.int32)
            })

        # 合并结果 = 其他类型 + 不参与合并的小 B + 合并后的 B
        surviving_chipping_defects = others + b_small_list + fused_b_list
            
    # 新增：检测主边上的凹进/缺段，标记为缺角
    try:
        edges_notch_img = preprocess_for_hough_enhanced(roi_gray, params)
        notch_defects = _detect_edge_notches(edges_notch_img, edges_for_drawing, pixels_per_mm, params)
        if notch_defects:
            corner_defects.extend(notch_defects)
    except Exception:
        pass

    # 合并输出缺陷：角点/崩边/斜边（直线）
    combined_defects = corner_defects + surviving_chipping_defects + skew_line_defects + rect_q_defects

    # Q/B/L 最短边（宽度）过滤：过滤掉宽度 < 5mm 的缺陷
    def _measure_width_mm_for_defect(d: dict, ppm: float) -> float:
        try:
            if d is None or ppm is None or ppm <= 0:
                return 0.0
            t = d.get('type')
            # Q：由交点 + 两端点组成三角形，取其 minAreaRect 的短边
            if t == 'Q':
                try:
                    ctr = d.get('center')
                    eps = d.get('endpoints')
                    if ctr is not None and isinstance(eps, tuple) and len(eps) == 2 and eps[0] is not None and eps[1] is not None:
                        tri = np.array([ctr, eps[0], eps[1]], dtype=np.float32).reshape(-1, 2)
                        rectq = cv2.minAreaRect(tri)
                        w_px = float(rectq[1][0] or 0.0)
                        h_px = float(rectq[1][1] or 0.0)
                        width_px = float(min(w_px, h_px))
                        return width_px / float(ppm)
                except Exception:
                    pass
            # 优先使用 minAreaRect
            rect = d.get('min_area_rect')
            if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                w_px = float(rect[1][0] or 0.0)
                h_px = float(rect[1][1] or 0.0)
                width_px = float(min(w_px, h_px))
                return width_px / float(ppm)
            # 其次使用 box_points
            box = d.get('box_points')
            if box is not None and len(box) >= 4:
                box_np = np.array(box, dtype=np.float32).reshape(-1, 2)
                rect2 = cv2.minAreaRect(box_np)
                w_px = float(rect2[1][0] or 0.0)
                h_px = float(rect2[1][1] or 0.0)
                width_px = float(min(w_px, h_px))
                return width_px / float(ppm)
            # 再次使用 contour
            cnt = d.get('contour')
            if cnt is not None and len(cnt) >= 3:
                rect3 = cv2.minAreaRect(np.array(cnt, dtype=np.float32))
                w_px = float(rect3[1][0] or 0.0)
                h_px = float(rect3[1][1] or 0.0)
                width_px = float(min(w_px, h_px))
                return width_px / float(ppm)
        except Exception:
            return 0.0
        return 0.0

    filtered_defects = []
    for d in combined_defects:
        if d is None:
            continue
        t = d.get('type')
        if t in ('Q', 'B', 'L'):
            width_mm = _measure_width_mm_for_defect(d, pixels_per_mm)
            if width_mm >= 5.0:
                filtered_defects.append(d)
            else:
                # 过滤宽度小于 5mm 的 Q/B/L
                pass
        else:
            filtered_defects.append(d)

    return edges_for_drawing, filtered_defects, paired_corners


def process_roi_hough_based(roi_idx, roi_template, image_gray, params, pixels_per_mm):
    # 兼容多种 ROI 表达：dict/list/tuple
    def _parse_roi(rt):
        if isinstance(rt, (list, tuple)) and len(rt) >= 4:
            return int(rt[0]), int(rt[1]), int(rt[2]), int(rt[3])
        if isinstance(rt, dict):
            if 'x' in rt or 'y' in rt or 'width' in rt or 'height' in rt:
                x0 = int(rt.get('x', 0)); y0 = int(rt.get('y', 0))
                w0 = int(rt.get('width', rt.get('w', 0)) or 0)
                h0 = int(rt.get('height', rt.get('h', 0)) or 0)
                return x0, y0, w0, h0
            if all(k in rt for k in ('left','top','right','bottom')):
                left = int(rt.get('left', 0)); top = int(rt.get('top', 0))
                right = int(rt.get('right', left)); bottom = int(rt.get('bottom', top))
                return left, top, max(0, right - left), max(0, bottom - top)
        return 0, 0, 0, 0
    x, y, w, h = _parse_roi(roi_template)
    # 基本越界裁剪
    H, W = image_gray.shape[:2]
    if w < 0: w = 0
    if h < 0: h = 0
    if x < 0:
        w = max(0, w + x)
        x = 0
    if y < 0:
        h = max(0, h + y)
        y = 0
    if x + w > W:
        w = max(0, W - x)
    if y + h > H:
        h = max(0, H - y)
    roi_gray = image_gray[y:y+h, x:x+w]
    
    p_hough = params["HOUGH_TRANSFORM"]
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    # 保证传入 HoughLinesP 的参数为整数类型（OpenCV 要求 threshold 为 int，其他也用 int 更稳妥）
    min_len_pixels = roi_gray.shape[1] * p_hough.get("MIN_LINE_LENGTH_RATIO", 0.05)
    try:
        min_len_pixels_i = int(max(1, round(float(min_len_pixels))))
    except Exception:
        min_len_pixels_i = max(1, int(roi_gray.shape[1] * 0.05))

    max_line_gap_px = _get_dist_px(p_hough, "MAX_LINE_GAP_MM", "MAX_LINE_GAP", None, pixels_per_mm)
    try:
        max_line_gap_px_i = int(max(0, round(float(max_line_gap_px)))) if max_line_gap_px is not None else 0
    except Exception:
        max_line_gap_px_i = 0

    try:
        hough_threshold_i = int(round(float(p_hough.get("THRESHOLD", 50))))
    except Exception:
        hough_threshold_i = 50

    raw_lines = cv2.HoughLinesP(
        binary_edges,
        1,
        np.pi / 180,
        hough_threshold_i,
        minLineLength=min_len_pixels_i,
        maxLineGap=max_line_gap_px_i,
    )
    
    main_edges = merge_lines_and_get_main_edges(raw_lines, params, pixels_per_mm, edge_img=binary_edges)

    # 运行时：根据上层注入的 DEFECT_DETECTION.Q_ENABLED 控制是否生成/绘制 Q
    try:
        _q_enabled_runtime = bool(params.get('DEFECT_DETECTION', {}).get('Q_ENABLED', True))
    except Exception:
        _q_enabled_runtime = True

    # 接入“跨 ROI 统一竖直虚拟边”：将全局共享竖直边裁剪到本 ROI 并并入主边
    try:
        p_def_sh = params.get('DEFECT_DETECTION', {})
        cross_enabled = bool(p_def_sh.get('CROSS_ROI_VERTICAL_ENABLED', True))
    except Exception:
        cross_enabled = True

    def _clip_infinite_line_to_roi_local(seg, w_loc, h_loc):
        try:
            x1, y1, x2, y2 = map(float, seg)
            dx, dy = (x2 - x1), (y2 - y1)
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return None
            candidates = []
            if abs(dx) >= 1e-9:
                for xk in (0.0, float(w_loc - 1)):
                    t = (xk - x1) / dx
                    yk = y1 + t * dy
                    if 0.0 <= yk <= float(h_loc - 1):
                        candidates.append((xk, yk))
            if abs(dy) >= 1e-9:
                for yk in (0.0, float(h_loc - 1)):
                    t = (yk - y1) / dy
                    xk = x1 + t * dx
                    if 0.0 <= xk <= float(w_loc - 1):
                        candidates.append((xk, yk))
            uniq = []
            for pt in candidates:
                if not any(abs(pt[0]-q[0]) < 0.5 and abs(pt[1]-q[1]) < 0.5 for q in uniq):
                    uniq.append(pt)
            if len(uniq) < 2:
                return None
            pts = np.array(uniq, dtype=float)
            idx0, idx1 = 0, 1
            maxd = -1.0
            for i0 in range(len(pts)):
                for i1 in range(i0+1, len(pts)):
                    d = float(np.hypot(pts[i1,0]-pts[i0,0], pts[i1,1]-pts[i0,1]))
                    if d > maxd:
                        maxd = d; idx0, idx1 = i0, i1
            a, b = pts[idx0], pts[idx1]
            return np.array([a[0], a[1], b[0], b[1]], dtype=float)
        except Exception:
            return None

    if cross_enabled:
        try:
            shared_global = params.get('CROSS_ROI_SHARED_VERTICAL_GLOBAL_EDGES', []) or []
            if shared_global:
                # 竖直判定容忍角（与其他流程一致）
                try:
                    vertical_tol_deg_local2 = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
                except Exception:
                    vertical_tol_deg_local2 = 10.0
                # 将全局共享直线变换到本 ROI 坐标系并裁剪
                H_roi_loc, W_roi_loc = roi_gray.shape[:2]
                appended = []
                for gseg in shared_global:
                    try:
                        gx1, gy1, gx2, gy2 = map(float, gseg)
                        lseg = [gx1 - x, gy1 - y, gx2 - x, gy2 - y]
                        clipped = _clip_infinite_line_to_roi_local(lseg, W_roi_loc, H_roi_loc)
                        if clipped is None:
                            continue
                        # 仅保留近竖直
                        dx = float(clipped[2] - clipped[0])
                        dy = float(clipped[3] - clipped[1])
                        ang = abs(np.degrees(np.arctan2(dy, dx)))
                        if ang > 90.0:
                            ang = 180.0 - ang
                        if ang < (90.0 - vertical_tol_deg_local2):
                            continue
                        appended.append(np.array(clipped, dtype=float))
                    except Exception:
                        continue
                if appended:
                    # 简单去重：若与现有主边距离很近则跳过
                    def _seg_perp_dist(a, b, c, d):
                        A = np.array([a, b], dtype=float); B = np.array([c, d], dtype=float)
                        v = B - A
                        L = float(np.hypot(v[0], v[1]))
                        if L < 1e-6:
                            return 1e9
                        mid = (A + B) * 0.5
                        best = 1e9
                        for e in main_edges:
                            E1 = np.array(e[:2], dtype=float); E2 = np.array(e[2:], dtype=float)
                            ev = E2 - E1
                            el = float(np.hypot(ev[0], ev[1]))
                            if el < 1e-6:
                                continue
                            # 点到直线距离
                            cross = float(ev[0]*(mid[1]-E1[1]) - ev[1]*(mid[0]-E1[0]))
                            dperp = abs(cross)/el
                            if dperp < best:
                                best = dperp
                        return best
                    for cseg in appended:
                        try:
                            d0 = _seg_perp_dist(cseg[0], cseg[1], cseg[2], cseg[3])
                        except Exception:
                            d0 = 1e9
                        if d0 > 2.0:
                            main_edges.append(cseg)
        except Exception:
            pass

    # 额外增强：为崩边检测补充“理想竖直边”（基于本 ROI 内的 Canny 白点列峰），
    # 仅当当前主边中“近竖直”数量不足 2 条时才尝试生成；避免重复
    try:
        p_def_evb = params.get('DEFECT_DETECTION', {})
        enable_ev = bool(p_def_evb.get('B_USE_IDEAL_VERTICAL_FROM_CANNY', True))
    except Exception:
        enable_ev = True
    if enable_ev and binary_edges is not None and binary_edges.size > 0:
        try:
            # 统计当前近竖直主边数量
            def _is_near_vertical(seg, tol_deg=10.0):
                x1,y1,x2,y2 = map(float, seg)
                ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if ang > 90.0:
                    ang = 180.0 - ang
                return ang >= (90.0 - tol_deg)
            tol_v = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
            current_v = sum(1 for e in (main_edges or []) if _is_near_vertical(e, tol_v))
            if current_v < 2:
                H_be, W_be = binary_edges.shape[:2]
                col_sum = np.sum((binary_edges > 0).astype(np.uint16), axis=0).astype(np.float32)
                # 简单平滑
                k = int(max(3, round(0.03 * W_be)))
                if k % 2 == 0:
                    k += 1
                if k > 1:
                    pad = k // 2
                    col_pad = np.pad(col_sum, (pad, pad), mode='edge')
                    ker = np.ones(k, dtype=np.float32) / float(k)
                    col_smooth = np.convolve(col_pad, ker, mode='valid')
                else:
                    col_smooth = col_sum
                # 阈值：列白点数量需达到高度的一定比例
                min_ratio = float(params.get('DEFECT_DETECTION', {}).get('B_IDEAL_VERTICAL_MIN_COL_SUM_RATIO', 0.15))
                thr = max(1.0, min_ratio * float(H_be))
                # 选择两个相距足够远的峰值
                min_sep = float(params.get('DEFECT_DETECTION', {}).get('B_IDEAL_VERTICAL_MIN_SEPARATION_PX', max(8.0, 0.2 * W_be)))
                idx_sorted = list(np.argsort(-col_smooth))  # 降序
                picks = []
                for idx in idx_sorted:
                    if col_smooth[int(idx)] < thr:
                        break
                    if not picks:
                        picks.append(int(idx))
                    else:
                        far = True
                        for p in picks:
                            if abs(int(idx) - int(p)) < min_sep:
                                far = False
                                break
                        if far:
                            picks.append(int(idx))
                    if len(picks) >= 2:
                        break
                # 将挑选到的列转为本 ROI 内的竖直段，并去重加入 main_edges
                if picks:
                    # 去重：若与已有主边近似重合（按 x 距离 < 2px）则跳过
                    def _x_mid(seg):
                        return 0.5 * (float(seg[0]) + float(seg[2]))
                    existing_xs = []
                    for e in (main_edges or []):
                        if _is_near_vertical(e, tol_v):
                            existing_xs.append(_x_mid(e))
                    for px_i in picks:
                        x_new = float(px_i)
                        if any(abs(float(ex) - x_new) <= 2.0 for ex in existing_xs):
                            continue
                        seg_new = np.array([x_new, 0.0, x_new, float(H_be - 1)], dtype=float)
                        main_edges.append(seg_new)
        except Exception:
            pass
    # 计算本帧（该 ROI）主边之间的有效相交点（仅在本 ROI 范围内）
    intersections_frame = []
    if main_edges and len(main_edges) >= 2:
        H_roi, W_roi = roi_gray.shape[:2]
        for i in range(len(main_edges)):
            for j in range(i+1, len(main_edges)):
                inter = find_line_intersection(main_edges[i], main_edges[j])
                if inter is None:
                    continue
                x_int, y_int = float(inter[0]), float(inter[1])
                if 0 <= x_int < W_roi and 0 <= y_int < H_roi:
                    intersections_frame.append((int(round(x_int)), int(round(y_int))))
    # 打印每帧交点信息（ROI级别）
    # 不再打印 intersections_frame 调试信息
    edges_for_drawing, all_defects, paired_corners = find_and_analyze_defects(main_edges, roi_gray, roi_gray.shape, params, pixels_per_mm, binary_edges)

    # 构建“主边端点扫描带”掩膜：长度默认20mm（可通过 DEFECT_DETECTION.L_ENDPOINT_BELT_LENGTH_MM 配置），
    # 宽度等于亮度扫描带宽 LUMINOSITY_SCAN_WIDTH_MM；用于过滤位于边端扫描带内的 L 型缺陷（视为误检）。
    p_def_for_belt = params.get("DEFECT_DETECTION", {})
    try:
        scan_width_px_for_belt = _get_dist_px(p_def_for_belt, "LUMINOSITY_SCAN_WIDTH_MM", "LUMINOSITY_SCAN_WIDTH", None, pixels_per_mm)
    except Exception:
        scan_width_px_for_belt = 0.0
    try:
        belt_len_px = _get_dist_px(p_def_for_belt, "L_ENDPOINT_BELT_LENGTH_MM", None, 20.0, pixels_per_mm)
    except Exception:
        belt_len_px = 0.0

    l_endpoint_belt_mask = None
    if scan_width_px_for_belt and belt_len_px and scan_width_px_for_belt > 0 and belt_len_px > 0 and len(main_edges) > 0:
        l_endpoint_belt_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        for e in main_edges:
            try:
                p1 = np.array(e[:2], dtype=float); p2 = np.array(e[2:], dtype=float)
                v = p2 - p1; L_line = float(np.linalg.norm(v))
                if L_line < 1e-6:
                    continue
                u = v / L_line
                n = np.array([-u[1], u[0]])
                half_w_vec = (scan_width_px_for_belt / 2.0) * n
                d = float(min(belt_len_px, L_line))
                # 两端的短矩形：p1->p1+u*d 与 p2->p2-u*d
                segs = [(p1, p1 + u * d), (p2, p2 - u * d)]
                for s_pt, e_pt in segs:
                    poly = np.array([
                        s_pt + half_w_vec,
                        e_pt + half_w_vec,
                        e_pt - half_w_vec,
                        s_pt - half_w_vec
                    ], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(l_endpoint_belt_mask, [poly], 255)
            except Exception:
                # 忽略个别主边异常，继续其它主边
                pass
    
    final_defects_for_report = []
    for defect in all_defects:
        # 若当前相机禁用 Q，则直接跳过所有 Q 缺陷（不进入后续绘制与上报）
        try:
            if (not _q_enabled_runtime) and str(defect.get('type','')).upper() == 'Q':
                continue
        except Exception:
            pass
        new_defect = {'type': defect['type']}
        location = {}
        
        # --- MODIFICATION START: Introduce a flag to mark defects for filtering ---
        should_be_filtered = False
        # --- MODIFICATION END ---
        if defect['type'] in ('X', 'E'):
            # 支持两种 X：
            # 1) 交点型（有 center + angle）
            # 2) 斜边型（有 box_points + skew_angle_deg）
            if 'box_points' in defect:
                box = defect.get('box_points')
                if isinstance(box, (list, np.ndarray)) and len(box) >= 4:
                    box_np = np.array(box, dtype=float).reshape(-1, 2)
                    center = np.mean(box_np, axis=0)
                    location['x'] = int(center[0] + x)
                    location['y'] = int(center[1] + y)
                    # 记录角度：曲边则为“曲率角”，其余为与垂直参考的夹角
                    if str(defect.get('skew_subtype', '')) == 'curved':
                        location['subtype'] = 'curved'
                        try:
                            location['angle'] = float(round(defect.get('skew_angle_deg', 0.0), 2))
                        except Exception:
                            location['angle'] = 0.0
                    else:
                        try:
                            location['angle'] = float(round(defect.get('skew_angle_deg', 0.0), 2))
                        except Exception:
                            location['angle'] = float(round(defect.get('angle', 0.0), 2))
                    # 仅对 E 类型计算并填充尺寸与 size_label
                    if defect.get('type') == 'E':
                        try:
                            length_px, width_px = 0.0, 0.0
                            # 优先用已有 box_points 的 AABB
                            if isinstance(box_np, np.ndarray) and box_np.size >= 8:
                                xbb, ybb, wbb, hbb = cv2.boundingRect(box_np.astype(np.int32))
                                length_px, width_px = float(max(wbb, hbb)), float(min(wbb, hbb))
                            else:
                                # 回退：若仅有 min_area_rect，则先还原四点再取 AABB
                                r = defect.get('min_area_rect')
                                if r is not None and isinstance(r, tuple) and len(r) >= 2:
                                    pts = cv2.boxPoints(r).astype(np.int32)
                                    xbb, ybb, wbb, hbb = cv2.boundingRect(pts)
                                    length_px, width_px = float(max(wbb, hbb)), float(min(wbb, hbb))
                            if pixels_per_mm and pixels_per_mm > 0 and (length_px > 0 or width_px > 0):
                                location['length_mm'] = float(round(length_px / pixels_per_mm, 2))
                                location['width_mm']  = float(round(width_px  / pixels_per_mm, 2))
                        except Exception:
                            pass
                else:
                    location['x'] = x
                    location['y'] = y
                    if str(defect.get('skew_subtype', '')) == 'curved':
                        location['subtype'] = 'curved'
                        location['angle'] = float(round(defect.get('skew_angle_deg', 0.0), 2))
                    else:
                        location['angle'] = float(round(defect.get('skew_angle_deg', 0.0), 2))
            else:
                center = defect.get('center', (0, 0))
                location['x'] = int(center[0] + x)
                location['y'] = int(center[1] + y)
                location['angle'] = float(round(defect.get('angle', 0.0), 2))
                # X 型不输出尺寸/size_label（仅角度）

            # 按需求：取消对 E 类型的 ROI 级过滤（不在此处基于主边/距离/尺寸过滤）
        else:
            length_px, width_px = 0.0, 0.0
            if defect['type'] == 'Q':
                center = defect.get('center', (0, 0))
                location['x'] = int(center[0] + x)
                location['y'] = int(center[1] + y)

                # 若有区域轮廓（如三角形），计算像素面积与面积(mm^2)
                try:
                    if 'region_contour' in defect and pixels_per_mm:
                        rc = defect['region_contour']
                        area_px_poly = float(cv2.contourArea(np.array(rc, dtype=np.int32)))
                        if area_px_poly > 0:
                            location['pixel_area'] = round(area_px_poly, 2)
                            location['area_mm2'] = round(area_px_poly / float(pixels_per_mm * pixels_per_mm), 2)
                except Exception:
                    pass

                if 'endpoints' in defect:
                    center_np = np.array(center)
                    p1_orig, p2_orig = np.array(defect["endpoints"][0]), np.array(defect["endpoints"][1])
                    v1_final, v2_final = p1_orig, p2_orig
                    
                    pixel_area = 0.5 * abs(center_np[0]*(v1_final[1]-v2_final[1]) + v1_final[0]*(v2_final[1]-center_np[1]) + v2_final[0]*(center_np[1]-v1_final[1]))
                    pixel_area_mm2 = pixel_area / (pixels_per_mm * pixels_per_mm)
                    location['pixel_area'] = round(pixel_area, 2)
                    location['area_mm2'] = round(pixel_area_mm2, 2)

                    q_min_area_mm2 = params["DEFECT_DETECTION"].get("Q_TRIANGLE_MIN_AREA_MM2", 2.0)
                    if pixel_area_mm2 < q_min_area_mm2: continue
                    
                    dist_leg1_px = np.linalg.norm(center_np - v1_final)
                    dist_leg2_px = np.linalg.norm(center_np - v2_final)
                    length_px = max(dist_leg1_px, dist_leg2_px)
                    width_px  = min(dist_leg1_px, dist_leg2_px)
                else:
                    # 新增：支持基于矩形差法产生的 Q（携带 min_area_rect 或 box_points）
                    r = defect.get('min_area_rect')
                    if r is not None and isinstance(r, tuple) and len(r) >= 2:
                        rw, rh = float(r[1][0] or 0.0), float(r[1][1] or 0.0)
                        length_px, width_px = max(rw, rh), min(rw, rh)
                    else:
                        box = defect.get('box_points')
                        if box is not None and len(box) >= 4:
                            box_np = np.array(box, dtype=np.float32).reshape(-1, 2)
                            rect2 = cv2.minAreaRect(box_np)
                            w_px2 = float(rect2[1][0] or 0.0); h_px2 = float(rect2[1][1] or 0.0)
                            length_px, width_px = max(w_px2, h_px2), min(w_px2, h_px2)
                        else:
                            length_px, width_px = 0.0, 0.0
            
            elif defect['type'] in ['B']:
                box = defect.get('box_points', [])
                if len(box) >= 4:
                    center = np.mean(box, axis=0)
                    location['x'] = int(center[0] + x)
                    location['y'] = int(center[1] + y)
                    d1 = np.linalg.norm(np.array(box[0]) - np.array(box[1]))
                    d2 = np.linalg.norm(np.array(box[1]) - np.array(box[2]))
                    length_px, width_px = max(d1, d2), min(d1, d2)

            location['length_mm'] = float(round(length_px / pixels_per_mm, 2))
            location['width_mm'] = float(round(width_px / pixels_per_mm, 2))

        new_defect['location'] = location
        
        # 新增：按最短边（width_mm）过滤缺陷，默认阈值 5mm，可通过 DEFECT_DETECTION.MIN_WIDTH_MM 配置
        try:
            min_width_mm_rule = float(params.get("DEFECT_DETECTION", {}).get("MIN_WIDTH_MM", 5.0))
        except Exception:
            min_width_mm_rule = 5.0
        # 规则仅对 Q 以及“未被重分类为 L 的 B”生效：
        # 这里先对 Q 进行早期过滤；B 的过滤放到后续 B 分支且“未转为 L”时执行。
        if new_defect['type'] == 'Q':
            if float(location.get('width_mm', 0.0) or 0.0) < min_width_mm_rule:
                continue

        min_size_mm = params["DEFECT_DETECTION"].get("MIN_DEFECT_SIZE_MM", 3.0)
        
        if new_defect['type'] == 'B':
            # 注：已恢复再分类策略——若后续计算得出 B 的长宽比 > 7.5，将其改判为 L（裂纹）
            pass
        
        new_defect['raw_defect'] = defect 
        
        if new_defect.get('raw_defect', {}).get('type') == 'B':
            min_extent_ratio = params["DEFECT_DETECTION"].get("SHADOW_FILTER_MIN_EXTENT_RATIO", 0.25)
            contour = defect.get('contour')
            min_area_rect_for_extent = defect.get('min_area_rect')
            if contour is not None and min_area_rect_for_extent is not None:
                contour_area = cv2.contourArea(contour)
                rect_w, rect_h = min_area_rect_for_extent[1]
                rect_area = rect_w * rect_h
                if rect_area > 1e-6 and (contour_area / rect_area) < min_extent_ratio:
                    continue

            # 平行过滤：若 B 的长边与最近主边近似平行（<=容差），且(长宽比>10 或 窄边<2mm) 则直接丢弃
            try:
                # 使用 main_edges（本函数作用域的主边列表）
                box = defect.get('box_points')
                if box is not None and len(box) >= 4 and len(main_edges) > 0:
                    box_np = np.array(box, dtype=float).reshape(-1, 2)
                    adj_pairs = [
                        (box_np[0], box_np[1]),
                        (box_np[1], box_np[2]),
                        (box_np[2], box_np[3]),
                        (box_np[3], box_np[0])
                    ]
                    lengths = [np.linalg.norm(b - a) for a, b in adj_pairs]
                    idx_long = int(np.argmax(lengths))
                    long_a, long_b = adj_pairs[idx_long]
                    rect_long_seg = [float(long_a[0]), float(long_a[1]), float(long_b[0]), float(long_b[1])]

                    # 选用矩形中心作为参考点寻找最近主边
                    rect = defect.get('min_area_rect')
                    if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                        cx, cy = rect[0]
                        center_pt = np.array([float(cx), float(cy)], dtype=float)
                    else:
                        center_pt = np.mean(box_np, axis=0)

                    min_d = float('inf'); nearest_edge = None
                    for e in main_edges:
                        a = np.array(e[:2], dtype=float); b = np.array(e[2:], dtype=float)
                        _, d = get_point_line_segment_projection(center_pt, [a[0], a[1], b[0], b[1]])
                        if d < min_d:
                            min_d = d; nearest_edge = [float(a[0]), float(a[1]), float(b[0]), float(b[1])]

                    if nearest_edge is not None:
                        angle_deg = calculate_angle_between_lines(rect_long_seg, nearest_edge)  # [0,90]
                        p_def = params.get('DEFECT_DETECTION', {})
                        parallel_tol = float(p_def.get('B_FILTER_PARALLEL_TOLERANCE_DEG', 10.0))
                        is_parallel = angle_deg <= parallel_tol

                        # 取当前 B 的长短边（mm）
                        length_mm = location.get('length_mm', 0.0)
                        width_mm = location.get('width_mm', 0.0)
                        ar = (length_mm / width_mm) if width_mm > 1e-6 else float('inf')
                        ar_min = float(p_def.get('B_FILTER_PARALLEL_AR_MIN', 10.0))
                        min_side_mm = float(p_def.get('B_FILTER_PARALLEL_MIN_SIDE_MM', 2.0))
                        if is_parallel and (ar > ar_min or width_mm < min_side_mm):
                            continue
            except Exception:
                pass

        if new_defect['type'] == 'Q':
            # 双重保险：若禁用 Q，提前跳过
            if not _q_enabled_runtime:
                continue
            length_mm = location.get('length_mm', 0); width_mm = location.get('width_mm', 0)
            area_mm2 = length_mm * width_mm; aspect_ratio = length_mm / width_mm if width_mm > 1e-6 else float('inf')
            if area_mm2 < 2.25: continue
            if aspect_ratio > 20.0: continue
            if min(length_mm, width_mm) < 2.0: continue
            if length_mm < min_size_mm: continue
            
        elif new_defect['type'] in ['B']:
            if location.get('length_mm', 0) < min_size_mm:
                continue

        if new_defect['type'] == 'B':
            length_mm = location.get('length_mm', 0); width_mm = location.get('width_mm', 0)
            area_mm2 = length_mm * width_mm; aspect_ratio = length_mm / width_mm if width_mm > 1e-6 else float('inf')

            # 条件2：长边需与最近主边近似垂直（90°±容差）
            is_perp_to_nearest_edge = False
            try:
                raw_b = new_defect.get('raw_defect', {})
                box = raw_b.get('box_points')
                rect = raw_b.get('min_area_rect')
                if box is not None and len(box) >= 4 and len(main_edges) > 0:
                    box_np = np.array(box, dtype=float).reshape(-1, 2)
                    # 选取长边段（在相邻点之间取最大长度的边）
                    adj_pairs = [
                        (box_np[0], box_np[1]),
                        (box_np[1], box_np[2]),
                        (box_np[2], box_np[3]),
                        (box_np[3], box_np[0])
                    ]
                    lengths = [np.linalg.norm(b - a) for a, b in adj_pairs]
                    idx_long = int(np.argmax(lengths))
                    long_a, long_b = adj_pairs[idx_long]
                    rect_long_seg = [float(long_a[0]), float(long_a[1]), float(long_b[0]), float(long_b[1])]

                    # 取矩形中心
                    if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                        cx, cy = rect[0]
                        center_pt = np.array([float(cx), float(cy)], dtype=float)
                    else:
                        center_pt = np.mean(box_np, axis=0)

                    # 找最近主边（按中心点到线段的最短距离）
                    min_d = float('inf'); nearest_edge = None
                    for e in main_edges:
                        a = np.array(e[:2], dtype=float); b = np.array(e[2:], dtype=float)
                        _, d = get_point_line_segment_projection(center_pt, [a[0], a[1], b[0], b[1]])
                        if d < min_d:
                            min_d = d; nearest_edge = [float(a[0]), float(a[1]), float(b[0]), float(b[1])]

                    if nearest_edge is not None:
                        angle_deg = calculate_angle_between_lines(rect_long_seg, nearest_edge)  # 返回[0,90]
                        tol_deg = float(params.get('DEFECT_DETECTION', {}).get('B_TO_L_PERP_TOLERANCE_DEG', 10.0))
                        # 接近90°：angle >= 90 - tol
                        if angle_deg >= (90.0 - tol_deg):
                            is_perp_to_nearest_edge = True
            except Exception:
                is_perp_to_nearest_edge = False

            # 恢复再分类：细长且长边近似垂直主边的 B 视为 L（裂纹）
            if aspect_ratio > 7.5 and is_perp_to_nearest_edge:
                new_defect['type'] = 'L'
                # L 型缺陷若位于主边端点附近的扫描带内（默认20mm），视为误检，进行过滤
                try:
                    if l_endpoint_belt_mask is not None:
                        box = defect.get('box_points')
                        if box is not None and len(box) >= 4:
                            tmp_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
                            cv2.fillPoly(tmp_mask, [np.int32(box)], 255)
                            inter = cv2.bitwise_and(tmp_mask, l_endpoint_belt_mask)
                            if cv2.countNonZero(inter) > 0:
                                should_be_filtered = True
                except Exception:
                    pass
            else:
                # 仅对仍为 B 的缺陷应用 B 专属筛选
                # 宽度过滤（仅对未被转为 L 的 B 生效）
                if width_mm < min_width_mm_rule:
                    continue
                if area_mm2 < 25 and width_mm < min_size_mm: continue
        
        # 新增：L 型缺陷过滤——在完成 B→L 重分类之后，屏蔽主边缘附近(≤阈值，默认5mm)的所有 L 型缺陷
        if new_defect.get('type') == 'L':
            try:
                # 若该 L 由 B 重分类而来，则复用 B 的后置规则（最短边过滤除外）
                try:
                    if new_defect.get('raw_defect', {}).get('type') == 'B':
                        length_mm = float(location.get('length_mm', 0.0) or 0.0)
                        width_mm = float(location.get('width_mm', 0.0) or 0.0)
                        area_mm2 = length_mm * width_mm
                        aspect_ratio = (length_mm / width_mm) if width_mm > 1e-6 else float('inf')
                        # 对应 B 分支中的面积/比例启发式（不包含 MIN_WIDTH_MM 最短边过滤）
                        if (area_mm2 < 25 and width_mm < 2.0):
                            should_be_filtered = True
                except Exception:
                    pass

                raw_l = new_defect.get('raw_defect', {})
                box = raw_l.get('box_points')
                rect = raw_l.get('min_area_rect')
                if box is not None and len(box) >= 4 and len(main_edges) > 0:
                    box_np = np.array(box, dtype=float).reshape(-1, 2)
                    # 距离采样点：仅中心
                    if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                        cx, cy = rect[0]
                        center_pt = np.array([float(cx), float(cy)], dtype=float)
                    else:
                        center_pt = np.mean(box_np, axis=0)

                    max_dist_mm = float(params.get('DEFECT_DETECTION', {}).get('L_FILTER_MAX_DISTANCE_TO_EDGE_MM', 5.0))
                    # 遍历所有主边，使用“中心点到直线（无限延长）”垂直距离
                    for e in main_edges:
                        a = np.array(e[:2], dtype=float); b = np.array(e[2:], dtype=float)
                        d_px = get_point_line_perpendicular_distance(center_pt, [a[0], a[1], b[0], b[1]])
                        d_mm = d_px / float(pixels_per_mm if pixels_per_mm else 1.0)
                        if d_mm <= max_dist_mm:
                            should_be_filtered = True
                            break
            except Exception:
                pass

        # --- MODIFICATION START: Final check of the filter flag ---
        if not should_be_filtered:
            final_defects_for_report.append(new_defect)
        # --- MODIFICATION END ---

    # 统计“近竖直”的主边数量（0~2 常见）：基于主边段方向角(相对x轴 0~90°)，角度>=90°-tol 视为近竖直
    try:
        # 默认容忍角度与主流程保持一致：10°（可通过 DEFECT_DETECTION.VERTICAL_ANGLE_TOL_DEG 覆盖）
        vertical_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
    except Exception:
        vertical_tol_deg = 10.0
    def _line_angle_deg(line):
        x1, y1, x2, y2 = map(float, line)
        dx, dy = (x2 - x1), (y2 - y1)
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang > 90.0:
            ang = 180.0 - ang
        return ang  # [0,90]
    near_vertical_count = 0
    try:
        for e in main_edges:
            ang = _line_angle_deg(e)
            if ang >= (90.0 - vertical_tol_deg):
                near_vertical_count += 1
    except Exception:
        near_vertical_count = 0

    roi_report = {
        "roi_idx": roi_idx, "x": x, "y": y, "w": w, "h": h,
        "defects": [d.copy() for d in final_defects_for_report],
        "edges_found": len(main_edges),
        "near_vertical_line_count": int(near_vertical_count)
    }
    for d in roi_report['defects']:
        d.pop('raw_defect', None)

    roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # 可视化：用亮黄色标出 Canny 边缘（用于缺角二次验证）
    # if binary_edges is not None:
    #     roi_color[binary_edges > 0] = [0, 255, 255]


    # 可视化颜色：'E'（边缘异常）使用红色；注意为 BGR 通道顺序
    DEFECT_COLORS_BGR = {'Q': (0, 0, 255), 'E': (0, 0, 255), 'X': (255, 0, 0), 'L': (255, 0, 255), 'B': (0, 165, 255)}
    p_vis = params["VISUALIZATION"]
    THICKNESS = 1
    
    alpha = p_vis["DEFECT_OVERLAY_ALPHA"]; beta = 1 - alpha
    
    # 绘制主边直线、角点（移除调试打印）
    annotations_to_draw = []
    # try:
    #    for i, seg in enumerate(edges_for_drawing or []):
    #        x1,y1,x2,y2 = map(float, seg)
    #        dx, dy = (x2-x1), (y2-y1)
    #        ang = abs(np.degrees(np.arctan2(dy, dx)))
    #        if ang > 90.0: ang = 180.0 - ang
    #        length_px = float(np.hypot(dx, dy))
    #        length_mm = (length_px / float(pixels_per_mm)) if pixels_per_mm else 0.0

    #        color = (200,200,200)
    #        if ang >= 80.0:
    #            color = (0,255,0)
    #        elif ang <= 10.0:
    #            color = (255,0,0)
    #        cv2.line(roi_color, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 1)

    #        mx, my = int(round((x1+x2)/2.0)), int(round((y1+y2)/2.0))
    #        try:
    #            cv2.putText(roi_color, f"L{i}", (mx+3, my-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    #        except Exception:
    #            pass

    #    for (ii, jj, cp_arr) in (paired_corners or []):
    #       try:
    #           cx, cy = float(cp_arr[0]), float(cp_arr[1])
    #           cv2.circle(roi_color, (int(round(cx)), int(round(cy))), 5, (255,0,255), -1)
    #           cv2.putText(roi_color, f"C({ii},{jj})", (int(round(cx))+4, int(round(cy))-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1, cv2.LINE_AA)
    #       except Exception:
    #           continue

    # except Exception:
    #    pass
    
    for defect_report in final_defects_for_report:
        defect = defect_report['raw_defect']
        color_bgr = DEFECT_COLORS_BGR.get(defect_report["type"], (255, 255, 255))
        
        loc = defect_report['location']
        defect_type_map = {'Q': '缺角', 'B': '崩边', 'E': '边缘异常', 'X': '斜边', 'L': '裂纹'}
        type_str = defect_type_map.get(defect_report['type'], '未知')

        try:
            if defect_report['type'] == 'Q' and isinstance(defect.get('ray_segments'), (list, tuple)):
                for seg_entry in defect.get('ray_segments'):
                    try:
                        if isinstance(seg_entry, dict) and 'seg' in seg_entry:
                            p0, p1 = seg_entry['seg']
                        else:
                            p0, p1 = seg_entry
                        x0,y0 = int(p0[0]), int(p0[1])
                        x1,y1 = int(p1[0]), int(p1[1])
                        cv2.arrowedLine(roi_color, (x0,y0), (x1,y1), (0,255,255), 1, tipLength=0.25)
                    except Exception:
                        continue
        except Exception:
            pass
#
        if defect_report['type'] in ('E', 'X'):
            # 区分 E 与 X 的标注：均显示角度；E 还需显示长宽；X 为“混合型”也显示长宽
            if loc.get('subtype') == 'curved' or (defect.get('skew_subtype', '') == 'curved'):
                angle_part = f"曲度: {loc.get('angle', 0.0):.1f}°"
            else:
                angle_part = f"角度: {loc.get('angle', 0.0):.1f}°"
            if 'length_mm' in loc and 'width_mm' in loc and (loc.get('length_mm') or loc.get('width_mm')):
                text = f"{type_str}: ({loc['x']}, {loc['y']}), {angle_part}, 尺寸: {loc.get('length_mm',0):.1f}x{loc.get('width_mm',0):.1f}mm"
            else:
                text = f"{type_str}: ({loc['x']}, {loc['y']}), {angle_part}"
        elif defect_report['type'] == 'Q' and 'pixel_area' in loc:
            text = f"{type_str}: ({loc['x']}, {loc['y']}), 尺寸: {loc['length_mm']:.1f}x{loc['width_mm']:.1f}mm"
        else:
            text = f"{type_str}: ({loc['x']}, {loc['y']}), 尺寸: {loc['length_mm']:.1f}x{loc['width_mm']:.1f}mm"
        
        annotations_to_draw.append({'text': text, 'color': color_bgr})

        # Q 不再强制绘制三角形，统一用 region_contour；若有 barrier 信息，附加 barrier 显示
        if defect_report["type"] == "Q" and "region_contour" in defect:
            # 使用像素块的真实轮廓高亮（不使用 minAreaRect 来圈出）
            region = defect["region_contour"]
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [region], color_bgr)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            # 可选：画细边框帮助观察（维持同色细线）——如完全不需要可去掉下一行
            cv2.drawContours(roi_color, [region], 0, color_bgr, THICKNESS)
            # 绘制 Q 检测条带（thick ray）与玻璃轮廓，便于调试观察
            try:
                raw_q = defect
                # 新：绘制玻璃轮廓（来自 barrier_contour）
                bc = raw_q.get('barrier_contour')
                if bc:
                    bc_np = np.array(bc, dtype=np.int32)
                    cv2.polylines(roi_color, [bc_np], True, (0, 255, 255), 1)
            except Exception:
                pass
        elif defect["type"] == "X" and "center" in defect:
            cv2.circle(roi_color, defect["center"], 15, color_bgr, THICKNESS)
        elif defect_report["type"] in ["L", "B", "X", "E"] and "box_points" in defect:
            # 其它缺陷继续使用自身 box_points 可视化
            box_points = defect["box_points"]
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [box_points], color_bgr)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            cv2.drawContours(roi_color, [box_points], 0, color_bgr, THICKNESS)

    # 绘制识别到的主直线和“理想直线边”（可配置开关）
    try:
        draw_main = False  
    except Exception:
        draw_main = False
    try:
        draw_ideal = False
    except Exception:
        draw_ideal = False
    try:
        draw_shared = False
    except Exception:
        draw_shared = False

    # 仍保留：若某些缺陷对象自身提供 center/region_contour，则额外用绿色标点以示区分
    try:
        for defect_report in final_defects_for_report:
            if defect_report.get('type') not in ('X','Q'):
                continue
            raw = defect_report.get('raw_defect', {})
            center_pt = None
            if 'center' in raw and raw['center'] is not None:
                try:
                    cx, cy = int(raw['center'][0]), int(raw['center'][1])
                    center_pt = (cx, cy)
                except Exception:
                    center_pt = None
            if center_pt is None and 'region_contour' in raw and raw['region_contour'] is not None:
                try:
                    cnt = np.array(raw['region_contour'], dtype=np.int32)
                    m = cv2.moments(cnt)
                    if m['m00'] != 0:
                        cx = int(m['m10']/m['m00']); cy = int(m['m01']/m['m00'])
                        center_pt = (cx, cy)
                except Exception:
                    center_pt = None
            if center_pt is not None:
            #    cv2.circle(roi_color, center_pt, 6, (0,255,0), -1)
                pass
    except Exception:
        pass

    # 共享竖直虚拟边（跨 ROI）以青色虚线绘制
    if draw_shared:
        try:
            shared_global = params.get('CROSS_ROI_SHARED_VERTICAL_GLOBAL_EDGES', []) or []
            if shared_global:
                H_loc, W_loc = roi_gray.shape[:2]
                for g in shared_global:
                    try:
                        gx1,gy1,gx2,gy2 = map(float, g)
                        loc_seg = [gx1 - x, gy1 - y, gx2 - x, gy2 - y]
                        clipped = None
                        try:
                            clipped = _clip_infinite_line_to_roi_local(loc_seg, W_loc, H_loc)
                        except Exception:
                            # 若局部裁剪函数不可用，则退回为直接裁剪到 ROI 矩形边界
                            clipped = None
                        if clipped is None:
                            # 粗略范围检查
                            continue
                        x1,y1,x2,y2 = map(int, map(round, clipped))
                        draw_dashed_line(roi_color, (x1,y1), (x2,y2), (255,255,0), thickness=1, dash_length=10)
                    except Exception:
                        continue
        except Exception:
            pass

    if annotations_to_draw and PIL_AVAILABLE and ANNOTATION_FONT:
        pil_img = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        y_text = 10; padding = 10
        for ann in annotations_to_draw:
            text = ann['text']
            color_rgb = tuple(reversed(ann['color']))
            
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0,0), text, font=ANNOTATION_FONT)
                text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]
            else:
                 text_width, text_height = draw.textsize(text, font=ANNOTATION_FONT)

            x_text = w - text_width - 10
            draw.text((x_text, y_text), text, font=ANNOTATION_FONT, fill=color_rgb)
            y_text += text_height + padding
            
        roi_color = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return roi_report, roi_color

def process_image_from_memory_parallel(image_gray, template_rois, config):
    final_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    report = {"image_status": "OK", "defects": [] , "state_code": 0, "rois": []}

    hough_params = config.get('hough_inspector_params')
    if not hough_params:
        raise ValueError("Configuration error: 'hough_inspector_params' section not found in the config file.")

    sys_params = config.get('system_params', {})
    pixels_per_mm = float(sys_params.get('pixels_per_mm', 1.0))

    try:
        roi_threads_cfg = int(sys_params.get('roi_threads', 0) or 0)
    except Exception:
        roi_threads_cfg = 0
    
    cpu_workers = multiprocessing.cpu_count()
    auto_workers = min(cpu_workers, len(template_rois))
    num_workers = auto_workers if roi_threads_cfg <= 0 else max(1, min(roi_threads_cfg, len(template_rois)))

    def _safe_roi_hough(i, r):
        try:
            return process_roi_hough_based(i, r, image_gray, hough_params, pixels_per_mm)
        except Exception as e:
            print(f"Error processing ROI {i}: {e}")
            # 兼容多种 ROI 表达，尽可能返回一个安全的占位 ROI
            try:
                if isinstance(r, (list, tuple)) and len(r) >= 4:
                    x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
                elif isinstance(r, dict):
                    if 'x' in r or 'y' in r or 'width' in r or 'height' in r:
                        x = int(r.get('x', 0)); y = int(r.get('y', 0))
                        w = int(r.get('width', r.get('w', 0)) or 0)
                        h = int(r.get('height', r.get('h', 0)) or 0)
                    elif all(k in r for k in ('left','top','right','bottom')):
                        left = int(r.get('left', 0)); top = int(r.get('top', 0))
                        right = int(r.get('right', left)); bottom = int(r.get('bottom', top))
                        x, y, w, h = left, top, max(0, right-left), max(0, bottom-top)
                    else:
                        x, y, w, h = 0, 0, 0, 0
                else:
                    x, y, w, h = 0, 0, 0, 0
            except Exception:
                x, y, w, h = 0, 0, 0, 0
            roi_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                roi_gray_crop = image_gray[y:y+h, x:x+w]
                roi_bgr = cv2.cvtColor(roi_gray_crop, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass
            return ({"roi_idx": i, "x": x, "y": y, "w": w, "h": h, "defects": [], "edges_found": 0, "near_vertical_line_count": 0}, roi_bgr)

    # 预收集跨 ROI 的共享竖直直线（全局坐标）
    try:
        p_def_cross = hough_params.get('DEFECT_DETECTION', {})
        cross_enabled = bool(p_def_cross.get('CROSS_ROI_VERTICAL_ENABLED', True))
    except Exception:
        cross_enabled = True

    shared_vertical_global = []
    # 移除跨帧预注入的共享竖直边（不再支持跨帧基线复用）
    if cross_enabled and template_rois:
        # 角度容忍与最短长度
        try:
            vertical_tol_deg_cross = float(hough_params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 10.0))
        except Exception:
            vertical_tol_deg_cross = 10.0
        try:
            min_len_px_cross = _get_dist_px(hough_params.get('DEFECT_DETECTION', {}), 'CROSS_ROI_VERTICAL_MIN_LEN_MM', None, 5.0, pixels_per_mm)
        except Exception:
            min_len_px_cross = 5.0 * float(pixels_per_mm if pixels_per_mm else 1.0)
        try:
            cluster_x_px = float(hough_params.get('DEFECT_DETECTION', {}).get('CROSS_ROI_VERTICAL_CLUSTER_XPX', 12.0))
        except Exception:
            cluster_x_px = 12.0

        def _parse_roi(rt):
            if isinstance(rt, (list, tuple)) and len(rt) >= 4:
                return int(rt[0]), int(rt[1]), int(rt[2]), int(rt[3])
            if isinstance(rt, dict):
                if 'x' in rt or 'y' in rt or 'width' in rt or 'height' in rt:
                    x0 = int(rt.get('x', 0)); y0 = int(rt.get('y', 0))
                    w0 = int(rt.get('width', rt.get('w', 0)) or 0)
                    h0 = int(rt.get('height', rt.get('h', 0)) or 0)
                    return x0, y0, w0, h0
                if all(k in rt for k in ('left','top','right','bottom')):
                    left = int(rt.get('left', 0)); top = int(rt.get('top', 0))
                    right = int(rt.get('right', left)); bottom = int(rt.get('bottom', top))
                    return left, top, max(0, right - left), max(0, bottom - top)
            return 0, 0, 0, 0

        cand_global = []  # 每条为 [x1,y1,x2,y2] 全局
        for r in template_rois:
            try:
                rx, ry, rw, rh = _parse_roi(r)
                if rw <= 0 or rh <= 0:
                    continue
                roi_gray = image_gray[ry:ry+rh, rx:rx+rw]
                edge_img = preprocess_for_hough_enhanced(roi_gray, hough_params)
                p_h = hough_params.get('HOUGH_TRANSFORM', {})
                min_len_pixels = roi_gray.shape[1] * p_h.get('MIN_LINE_LENGTH_RATIO', 0.05)
                try:
                    min_len_pixels_i = int(max(1, round(float(min_len_pixels))))
                except Exception:
                    min_len_pixels_i = max(1, int(roi_gray.shape[1] * 0.05))
                max_line_gap_px = _get_dist_px(p_h, 'MAX_LINE_GAP_MM', 'MAX_LINE_GAP', None, pixels_per_mm)
                try:
                    max_line_gap_px_i = int(max(0, round(float(max_line_gap_px)))) if max_line_gap_px is not None else 0
                except Exception:
                    max_line_gap_px_i = 0
                try:
                    hough_threshold_i = int(round(float(p_h.get('THRESHOLD', 50))))
                except Exception:
                    hough_threshold_i = 50
                raw = cv2.HoughLinesP(
                    edge_img,
                    1,
                    np.pi/180,
                    hough_threshold_i,
                    minLineLength=min_len_pixels_i,
                    maxLineGap=max_line_gap_px_i,
                )
                merged = merge_lines_and_get_main_edges(raw, hough_params, pixels_per_mm, edge_img=edge_img)
                for seg in merged:
                    try:
                        x1,y1,x2,y2 = map(float, seg)
                        dx = x2 - x1; dy = y2 - y1
                        ang = abs(np.degrees(np.arctan2(dy, dx)))
                        if ang > 90.0:
                            ang = 180.0 - ang
                        if ang < (90.0 - vertical_tol_deg_cross):
                            continue
                        L = float(np.hypot(dx, dy))
                        if L < float(min_len_px_cross):
                            continue
                        # 转全局
                        cand_global.append([x1 + rx, y1 + ry, x2 + rx, y2 + ry])
                    except Exception:
                        continue
            except Exception:
                continue

        if cand_global:
            try:
                slant_bias = float(hough_params.get('DEFECT_DETECTION', {}).get('CROSS_ROI_VERTICAL_SLANT_BIAS', 0.5))
            except Exception:
                slant_bias = 0.5
            xs = [0.5 * (float(s[0]) + float(s[2])) for s in cand_global]
            angles = []
            for s in cand_global:
                dx = float(s[2]) - float(s[0]); dy = float(s[3]) - float(s[1])
                ang = abs(np.degrees(np.arctan2(dy, dx)))
                if ang > 90.0:
                    ang = 180.0 - ang
                angles.append(ang)
            order = np.argsort(np.array(xs))
            groups = []
            for idx in order:
                xm = xs[int(idx)]; ang = angles[int(idx)]; seg = cand_global[int(idx)]
                placed = False
                for g in groups:
                    if abs(xm - float(np.mean(g['xs']))) <= cluster_x_px:
                        g['xs'].append(xm); g['segs'].append(seg); g['angs'].append(ang)
                        placed = True; break
                if not placed:
                    groups.append({'xs':[xm],'segs':[seg],'angs':[ang]})
            scored_lines = []
            for g in groups:
                pts = []
                total_len = 0.0
                for s in g['segs']:
                    pts.append([float(s[0]), float(s[1])]); pts.append([float(s[2]), float(s[3])])
                    total_len += float(np.hypot(float(s[2]) - float(s[0]), float(s[3]) - float(s[1])))
                pts_np = np.array(pts, dtype=np.float32)
                try:
                    vx, vy, x0, y0 = cv2.fitLine(pts_np, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                except Exception:
                    vx, vy, x0, y0 = 0.0, 1.0, float(np.mean(g['xs'])), float(np.mean(pts_np[:,1]))
                norm = math.hypot(float(vx), float(vy))
                if norm < 1e-6:
                    vx, vy = 0.0, 1.0
                else:
                    vx, vy = float(vx) / norm, float(vy) / norm
                ang_fit = abs(np.degrees(math.atan2(vy, vx)))
                if ang_fit > 90.0:
                    ang_fit = 180.0 - ang_fit
                y_min = float(np.min(pts_np[:,1])); y_max = float(np.max(pts_np[:,1]))
                if y_max - y_min < 1.0:
                    continue
                if abs(vy) < 1e-3:
                    x_min = x_max = float(np.mean(g['xs']))
                else:
                    x_min = float(x0 + vx / vy * (y_min - y0))
                    x_max = float(x0 + vx / vy * (y_max - y0))
                line_fit = [x_min, y_min, x_max, y_max]
                ang_dev = max(0.0, float(abs(90.0 - ang_fit)))
                score = total_len * (1.0 + slant_bias * (ang_dev / max(1e-3, vertical_tol_deg_cross)))
                scored_lines.append({'line': line_fit, 'score': score, 'ang_dev': ang_dev})
            # 按倾斜度优先的得分排序，保留全部，但确保不重复过近的 x
            scored_lines.sort(key=lambda d: d['score'], reverse=True)
            kept = []
            for item in scored_lines:
                lx = 0.5 * (float(item['line'][0]) + float(item['line'][2]))
                if any(abs(lx - 0.5 * (float(k[0]) + float(k[2]))) < cluster_x_px * 0.5 for k in kept):
                    continue
                kept.append(item['line'])
            shared_vertical_global.extend(kept)

        # 不再与外部预注入的共享竖直边合并（删除跨帧逻辑）

    # 将共享竖直边放入参数供 ROI 线程读取
    try:
        # 无论是否为空，都同步当前帧的共享竖直边到参数，供 ROI 线程绘制使用
        if cross_enabled:
            hough_params['CROSS_ROI_SHARED_VERTICAL_GLOBAL_EDGES'] = shared_vertical_global
        else:
            hough_params.pop('CROSS_ROI_SHARED_VERTICAL_GLOBAL_EDGES', None)
    except Exception:
        pass

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_safe_roi_hough, i, r) for i, r in enumerate(template_rois)]
        results = [future.result() for future in futures]

    max_edges_found = 0
    for roi_report, roi_color in results:
        if "x" not in roi_report: continue
        x, y, w, h = roi_report["x"], roi_report["y"], roi_report["w"], roi_report["h"]
        if w > 0 and h > 0: final_image[y:y+h, x:x+w] = roi_color

        # 按新需求：不再绘制/使用 envelope 信息
        
        max_edges_found = max(max_edges_found, roi_report.get("edges_found", 0))
        
        if roi_report.get("defects"):
            report["image_status"] = "NG"
            report["defects"].extend(roi_report["defects"])
            
    # 将 near_vertical_line_count 也带入状态机可用的精简 ROI 报告
        slim_report = {k: roi_report.get(k) for k in ("roi_idx","x","y","w","h","edges_found","near_vertical_line_count")}
        report['rois'].append(slim_report)

    # 取消针对 E 类型的帧级竖直主边进入判定：恢复为仅依据是否有主边
    report["state_code"] = 1 if max_edges_found > 0 else 0
    # 移除跨帧输出：不再在报告中携带共享竖直边
    
    return report, final_image