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
    edge_ignore_px = _get_dist_px(p, "LUMINOSITY_EDGE_IGNORE_WIDTH_MM", "LUMINOSITY_EDGE_IGNORE_WIDTH", 0.0, pixels_per_mm)
    endpoint_exclude_r = _get_dist_px(p, "LUMINOSITY_ENDPOINT_EXCLUDE_RADIUS_MM", None, None, pixels_per_mm, default_px=0.0)
    if endpoint_exclude_r is None or endpoint_exclude_r <= 0:
        # 缺省：按扫描带宽度的 0.3 比例，限制上限 15px，下限 3px
        endpoint_exclude_r = max(3, int(min(0.3 * scan_width, 15)))

    ignore_mask = np.zeros_like(roi_gray)
    if edge_ignore_px > 0:
        cv2.line(ignore_mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=int(round(edge_ignore_px)))
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
    return cv2.Canny(enhanced_contrast, p["CANNY_THRESHOLD_LOW"], p["CANNY_THRESHOLD_HIGH"])

def merge_lines_and_get_main_edges(lines, params, pixels_per_mm: float, edge_img=None):
    if lines is None or len(lines) < 1: return []
    p = params["LINE_MERGING"]
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
        final_line_groups.extend(proximity_groups)
    merged_lines_with_scores = []
    for group in final_line_groups:
        points = np.array([pt for line in group for pt in (line[0:2], line[2:4])], dtype=np.float32)
        if len(points) < 2:
            continue
        # 简化：直接使用 fitLine 估计方向与基点
        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()
        projected = (points[:, 0] - x0) * vx + (points[:, 1] - y0) * vy
        pt1, pt2 = points[np.argmin(projected)], points[np.argmax(projected)]
        final_merged_line = np.array([pt1[0], pt1[1], pt2[0], pt2[1]])
        support_score = sum(np.linalg.norm(l[2:4] - l[0:2]) for l in group)
        merged_lines_with_scores.append({'line': final_merged_line, 'score': support_score})
    # 先按支持度排序（强边在前）
    merged_lines_with_scores.sort(key=lambda item: item['score'], reverse=True)

    # 过滤与其它直线中段相交（或延长后会相交）的干扰直线：保留更强的一条
    p_crack = params.get("CRACK_CLASSIFICATION", {})
    endpoint_margin_px = _get_dist_px(p_crack, "ENDPOINT_PROXIMITY_THRESHOLD_MM", None, 5.5, pixels_per_mm)
    # 允许的短延长容差（毫米）→ 像素，用于判定“延长后会相交”
    extend_margin_px_default = _mm_to_px(5.0, pixels_per_mm)

    n = len(merged_lines_with_scores)
    valid = [True] * n

    def _t_param_and_perp_dist(pt, a, b):
        v = b - a
        denom = float(np.dot(v, v))
        if denom < 1e-8:
            return 0.0, np.linalg.norm(pt - a), 0.0
        t = float(np.dot(pt - a, v) / denom)
        proj = a + t * v
        return t, float(np.linalg.norm(pt - proj)), float(np.linalg.norm(v))

    for i in range(n):
        if not valid[i]:
            continue
        li = merged_lines_with_scores[i]['line']
        ai = np.array(li[:2], dtype=float); bi = np.array(li[2:], dtype=float)
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            lj = merged_lines_with_scores[j]['line']
            aj = np.array(lj[:2], dtype=float); bj = np.array(lj[2:], dtype=float)

            inter = find_line_intersection(li, lj)
            if inter is None:
                continue

            t_i, d_perp_i, len_i = _t_param_and_perp_dist(inter, ai, bi)
            t_j, d_perp_j, len_j = _t_param_and_perp_dist(inter, aj, bj)
            if len_i < 1e-6 or len_j < 1e-6:
                continue

            # 要求“几乎相交”且参数落在段内或小范围延长内
            ext_tol_i = extend_margin_px_default / len_i
            ext_tol_j = extend_margin_px_default / len_j
            near_line = (d_perp_i < 1.5) and (d_perp_j < 1.5)
            within_i = (-ext_tol_i <= t_i <= 1.0 + ext_tol_i)
            within_j = (-ext_tol_j <= t_j <= 1.0 + ext_tol_j)
            if not (near_line and within_i and within_j):
                continue

            # 判断是否为“中间部分”相交（避免角点端点附近）
            inter = inter.astype(float)
            di_min = min(np.linalg.norm(inter - ai), np.linalg.norm(inter - bi))
            dj_min = min(np.linalg.norm(inter - aj), np.linalg.norm(inter - bj))

            # 若交点位于两条线段的内部（0..1）且都远离端点，则认为是干扰交叉；
            # 或者其中一条在内部且远离端点，另一条在小范围延长内，也认为是干扰。
            inside_i = (0.0 <= t_i <= 1.0)
            inside_j = (0.0 <= t_j <= 1.0)
            is_middle_cross = (
                (inside_i and inside_j and di_min > endpoint_margin_px and dj_min > endpoint_margin_px)
                or (inside_i and di_min > endpoint_margin_px and not inside_j)
                or (inside_j and dj_min > endpoint_margin_px and not inside_i)
            )

            if is_middle_cross:
                # 移除较弱的一条
                if merged_lines_with_scores[i]['score'] >= merged_lines_with_scores[j]['score']:
                    valid[j] = False
                else:
                    valid[i] = False
                    break

    filtered = [merged_lines_with_scores[k] for k in range(n) if valid[k]]
    filtered.sort(key=lambda item: item['score'], reverse=True)

    # 根据是否与其它直线（实际或小范围延长后）相交，决定用于角度/扫描/过滤的有效线段：
    # - 不相交：保持原始长度
    # - 相交（含延长容差内相交）：使用“交点 → 原始线段最远端点”的新线段
    edges = [item['line'] for item in filtered[:p["TOP_N_EDGES"]]]
    m = len(edges)
    if m <= 1:
        return edges

    best_intersection = [None] * m  # (inter_pt, t_i, d_perp_i, len_i)
    # 复用上面定义的 _t_param_and_perp_dist 与 extend 容差
    def _inter_score(t: float):
        # 优先选择落在段内的交点；若不在段内，选择距离[0,1]最近者
        if 0.0 <= t <= 1.0:
            return (0, 0.0)
        # 距离[0,1]的外侧距离
        return (1, min(abs(t - 0.0), abs(t - 1.0)))

    for i in range(m):
        li = edges[i]
        ai = np.array(li[:2], dtype=float); bi = np.array(li[2:], dtype=float)
        for j in range(i + 1, m):
            lj = edges[j]
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
            near_line = (d_perp_i < 1.5) and (d_perp_j < 1.5)
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
    
    # 去除通过几何（直线）判断裂纹（L）的功能
    crack_defects = []
    crack_indices = set()

    true_edges = [edge for i, edge in enumerate(edges) if i not in crack_indices]
    
    corner_defects = []; num_true_edges = len(true_edges)
    edges_for_drawing = [edge.copy() for edge in true_edges]
    endpoint_paired_status = {i: [False, False] for i in range(num_true_edges)}
    # 收集未产生 Q 的直线交点，用于后续过滤其附近的 B 误检
    non_q_intersections = []
    
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

    # 新增：按角度将主边分类为 平行 / 垂直 / 斜边，并为斜边生成基于 boundingRect 的缺陷
    skew_line_defects = []
    try:
        # 默认容忍角度由 15° 调整为 17°（可通过 DEFECT_DETECTION.VERTICAL_ANGLE_TOL_DEG 覆盖）
        vertical_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 17.0))
    except Exception:
        vertical_tol_deg = 17.0

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
                box_pts = _axis_aligned_box_points_for_line(e, pad=3)
                skew_line_defects.append({
                    'type': 'E',
                    'box_points': box_pts,
                    'skew_angle_deg': angle_to_vertical
                })
        except Exception:
            continue


    # 新增：基于“最小外接矩形与玻璃边缘轮廓差”的缺角(Q)检测
    def _detect_q_by_rect_diff(roi_gray_img, ppm, param_all):
        try:
            h, w = roi_gray_img.shape[:2]
            # 使用与主流程一致的预处理（Canny）获取边缘
            edges = preprocess_for_hough_enhanced(roi_gray_img, param_all)
            # 适度膨胀，便于形成封闭轮廓
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_dil = cv2.dilate(edges, kernel, iterations=1)
            # 取外轮廓
            contours, _ = cv2.findContours(edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
            # 选择面积最大的轮廓，视作玻璃边缘所在的主体轮廓
            main_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_cnt) < 1.0:
                return []
            # 面积阈值（mm^2）：对轮廓包围的最小面积做要求（默认 1600 mm^2）
            try:
                min_glass_area_mm2 = float(param_all.get('DEFECT_DETECTION', {}).get('GLASS_MIN_CONTOUR_AREA_MM2', 1600.0))
            except Exception:
                min_glass_area_mm2 = 1600.0
            if ppm and ppm > 0:
                main_area_mm2 = cv2.contourArea(main_cnt) / float(ppm * ppm)
                if main_area_mm2 < min_glass_area_mm2:
                    return []
            # 最小外接矩形（允许倾斜）
            rect = cv2.minAreaRect(main_cnt)
            rect_box = cv2.boxPoints(rect).astype(np.int32)
            # 生成掩膜：矩形掩膜与主体轮廓掩膜
            mask_rect = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_rect, [rect_box], 255)
            mask_cnt = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_cnt, [main_cnt], -1, 255, thickness=-1)
            # 差分：矩形中但不在主体轮廓内的区域，候选“缺角”区域
            diff = cv2.subtract(mask_rect, mask_cnt)
            # 形状修约：移除过细的“须状/细长”区域，得到更合理的块状形状
            try:
                p_def = param_all.get('DEFECT_DETECTION', {})
                min_thick_px = _get_dist_px(p_def, 'Q_REGION_MIN_THICKNESS_MM', 'Q_REGION_MIN_THICKNESS', 1.5, ppm)
                k = max(3, int(round(float(min_thick_px))))
                if k % 2 == 0:
                    k += 1  # 保证奇数核
                kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                # 开运算：先腐蚀后膨胀，去掉比核更细的突出物
                diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
                # 轻微闭运算：平滑边界并填补小凹陷
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            except Exception:
                # 回退最小方案：小核开运算
                diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)
            # 连通域/轮廓分析
            cand_cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            results = []
            for c in cand_cnts:
                area_px = cv2.contourArea(c)
                if area_px <= 1.0:
                    continue
                # 计算候选区域的最小外接矩形，得到长宽
                r2 = cv2.minAreaRect(c)
                (cx, cy), (rw, rh), ang = r2
                # 换算尺寸到 mm
                width_mm = (min(rw, rh) / float(ppm)) if ppm else 0.0
                length_mm = (max(rw, rh) / float(ppm)) if ppm else 0.0
                area_mm2 = (rw * rh) / float(ppm * ppm) if ppm else 0.0
                # 过滤规则：块状且尺寸通过门槛（最短边>=5mm，面积>=25mm^2；可选最大边上限 Q_MAX_SIDE_MM）
                try:
                    p_def_local = param_all.get('DEFECT_DETECTION', {})
                    q_cfg = float(p_def_local.get('Q_MAX_SIDE_MM', 0.0))
                except Exception:
                    q_cfg = 0.0
                # 硬性上限：>50mm 的 Q 一律过滤；若配置更严（<=50），按配置取更小者
                q_max_side_mm = 50.0 if (q_cfg is None or q_cfg <= 0.0 or q_cfg > 50.0) else q_cfg
                max_side_ok = (length_mm <= q_max_side_mm and width_mm <= q_max_side_mm)
                if width_mm >= 5.0 and area_mm2 >= 25.0 and max_side_ok:
                    box2 = cv2.boxPoints(r2).astype(np.int32)
                    # 生成用于高亮的像素块轮廓（而不是画 minAreaRect）：简化原始连通域轮廓
                    eps = max(2.0, 0.02 * float(cv2.arcLength(c, True)))
                    approx = cv2.approxPolyDP(c, eps, True)
                    if approx is None or len(approx) < 3:
                        approx = c  # 退回原始轮廓
                    results.append({
                        'type': 'Q',
                        'min_area_rect': r2,
                        'box_points': box2,
                        'region_contour': approx.astype(np.int32),
                        'center': (int(round(cx)), int(round(cy))),
                        'origin': 'rect_diff'
                    })
            return results
        except Exception:
            return []

    rect_q_defects = _detect_q_by_rect_diff(roi_gray, pixels_per_mm, params)

    # 新逻辑：利用已获得角部交点与两条主边，与玻璃轮廓求最近交点生成三角形缺角区域
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

                # 基于主边对的真实几何交点来逐一处理缺角：为每个 intersection 独立执行射线与轮廓求交与泛洪
                corner_inters = []  # 列表元素: (i, j, cp)
                H_roi2, W_roi2 = roi_gray.shape[:2]
                for i_e in range(len(edges_for_drawing)):
                    for j_e in range(i_e + 1, len(edges_for_drawing)):
                        inter = find_line_intersection(edges_for_drawing[i_e], edges_for_drawing[j_e])
                        if inter is None:
                            continue
                        xi, yi = float(inter[0]), float(inter[1])
                        if 0 <= xi < W_roi2 and 0 <= yi < H_roi2:
                            corner_inters.append((i_e, j_e, np.array([xi, yi], dtype=float)))

                for (idx_i, idx_j, cp) in corner_inters:
                    # 直接使用该交点对应的两条主边
                    chosen = [(idx_i, edges_for_drawing[idx_i]), (idx_j, edges_for_drawing[idx_j])]
                    # 两条线分别求与玻璃轮廓的射线-轮廓交点
                    inter_hits = []  # (point, edge_index)
                    ray_hits_dbg = []  # debug: store dir, t, stripe poly
                    # 计算两条主边从交点出发、远离交点的单位方向向量
                    dirs_local = []
                    for _, seg in chosen:
                        p1 = np.array(seg[:2], dtype=float); p2 = np.array(seg[2:], dtype=float)
                        d1 = np.linalg.norm(cp - p1); d2 = np.linalg.norm(cp - p2)
                        far_pt = p1 if d1 > d2 else p2
                        v = far_pt - cp
                        n = float(np.linalg.norm(v))
                        if n < 1e-6:
                            dirs_local.append(None)
                        else:
                            dirs_local.append(v / n)

                    # 计算“内侧”双角平分方向，用于微调射线方向
                    m_dir = None
                    if all(d is not None for d in dirs_local):
                        m = dirs_local[0] + dirs_local[1]
                        mn = float(np.linalg.norm(m))
                        if mn > 1e-6:
                            m_dir = m / mn

                    # 角度内收（默认 1.8°）
                    try:
                        inward_deg = float(params.get('DEFECT_DETECTION', {}).get('Q_RAY_INWARD_DEG', 1.8))
                    except Exception:
                        inward_deg = 1.8

                    def _rotate(vec: np.ndarray, deg: float) -> np.ndarray:
                        th = np.deg2rad(deg)
                        c, s = float(np.cos(th)), float(np.sin(th))
                        R = np.array([[c, -s], [s, c]], dtype=float)
                        return (R @ vec.reshape(2,)).reshape(2,)

                    # 发射两条经过“向内 1.5° 微调”的射线
                    for idx_ch, seg in enumerate(chosen):
                        u = dirs_local[idx_ch]
                        if u is None:
                            continue
                        u_cast = u.copy()
                        if m_dir is not None:
                            # 选择使 u 更贴近 m_dir 的旋转方向（±inward_deg 中选 dot 更大者）
                            cand1 = _rotate(u, inward_deg)
                            cand2 = _rotate(u, -inward_deg)
                            u_cast = cand1 if float(np.dot(cand1, m_dir)) >= float(np.dot(cand2, m_dir)) else cand2
                        # 使用细射线求交；适度降低起始门槛以捕捉近处边界
                        hit = _ray_intersect_contour(cp, u_cast, cnt_pts, t_min=1.0)
                        if hit is not None:
                            t_hit, pt_hit, edge_idx_hit = hit
                            inter_hits.append((pt_hit, edge_idx_hit))
                            # 记录用于可视化的射线段（cp -> 命中点）
                            ray_hits_dbg.append({'seg': (tuple(map(int, cp)), tuple(map(int, pt_hit)))})
                    if len(inter_hits) != 2:
                        continue
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
                    q_max_side_mm2 = 70.0 if (q_cfg2 is None or q_cfg2 <= 0.0 or q_cfg2 > 70.0) else q_cfg2
                    max_side_ok2 = (length_mm <= q_max_side_mm2 and width_mm <= q_max_side_mm2)
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
    except Exception:
        pass

    # 去重：基于质心距离（像素）避免重复 Q；若重叠，按需求以“第二种方法（corner_contour）”为准
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

    # 以 corner_contour 结果为基准，只有当 rect_diff 结果与其不重叠时才追加
    rect_q_defects = _dedup_q(corner_contour_q_defects, rect_q_defects)

    # 计算某条主边“平行四边形扫描带”的平均亮度，选择相对于缺角三角形质心的内侧（与三角形相反侧）半带，剔除边线与端点
    # 用作缺角(Q)亮度门控的对比基准
    def _edge_baseline_parallelogram_mean(edge_line, tri_centroid):
        try:
            p = params["DEFECT_DETECTION"]
            p1 = np.array(edge_line[:2], dtype=float)
            p2 = np.array(edge_line[2:], dtype=float)
            line_vec = p2 - p1
            line_len = float(np.linalg.norm(line_vec))
            if line_len <= 1e-6:
                return float(cv2.mean(roi_gray)[0])

            # 扫描带宽（mm 配置 → px）
            scan_width = _get_dist_px(p, "LUMINOSITY_SCAN_WIDTH_MM", "LUMINOSITY_SCAN_WIDTH", None, pixels_per_mm)
            scan_width = max(1, int(round(float(scan_width))))

            unit_vec = line_vec / line_len
            normal_vec = np.array([-unit_vec[1], unit_vec[0]], dtype=float)
            half_width_vec = (scan_width / 2.0) * normal_vec

            # 两侧半带四点（按 scan_edge_for_luminosity_defects 的构造方式）
            c1 = p1 + half_width_vec; c2 = p2 + half_width_vec
            c3 = p2 - half_width_vec; c4 = p1 - half_width_vec

            mask_plus = np.zeros(roi_gray.shape, dtype=np.uint8)
            poly_plus = np.array([c1, c2, p2, p1], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_plus, [poly_plus], 255)

            mask_minus = np.zeros(roi_gray.shape, dtype=np.uint8)
            poly_minus = np.array([p1, p2, c3, c4], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_minus, [poly_minus], 255)

            # 根据三角形质心在边线哪一侧，选择与其相反侧作为“玻璃内侧”基准
            # 侧性判定：法向量指向为 plus 侧，点到边线的有符号距离 s = dot((tri - p1), normal_vec)
            # 若 s > 0，质心在 plus 侧，则取 minus 侧；反之取 plus 侧
            s = float(np.dot((np.array(tri_centroid, dtype=float) - p1), normal_vec))
            scan_mask = mask_minus if s > 0 else mask_plus

            # 剔除边线本体与两端点圆形区域，以匹配崩边检测的有效区域
            edge_ignore_px = _get_dist_px(p, "LUMINOSITY_EDGE_IGNORE_WIDTH_MM", "LUMINOSITY_EDGE_IGNORE_WIDTH", 0.0, pixels_per_mm)
            endpoint_exclude_r = _get_dist_px(p, "LUMINOSITY_ENDPOINT_EXCLUDE_RADIUS_MM", None, None, pixels_per_mm, default_px=0.0)
            if endpoint_exclude_r is None or endpoint_exclude_r <= 0:
                endpoint_exclude_r = max(3, int(min(0.3 * scan_width, 15)))

            ignore_mask = np.zeros(roi_gray.shape, dtype=np.uint8)
            if edge_ignore_px and edge_ignore_px > 0:
                cv2.line(ignore_mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=int(round(edge_ignore_px)))
            cv2.circle(ignore_mask, tuple(map(int, p1)), int(round(endpoint_exclude_r)), 255, thickness=-1)
            cv2.circle(ignore_mask, tuple(map(int, p2)), int(round(endpoint_exclude_r)), 255, thickness=-1)

            eff_mask = cv2.subtract(scan_mask, ignore_mask)
            if cv2.countNonZero(eff_mask) <= 0:
                # 回退：用未剔除边线/端点的扫描带
                eff_mask = scan_mask

            return float(cv2.mean(roi_gray, mask=eff_mask)[0])
        except Exception:
            return float(cv2.mean(roi_gray)[0])

    def sort_key_func(pair_indices):
        i, j = pair_indices
        line1, line2 = true_edges[i], true_edges[j]
        compatibility = get_quadrant_compatibility(edge_quadrants[i], edge_quadrants[j])
        min_dist = min(np.linalg.norm(line1[:2] - line2[:2]), np.linalg.norm(line1[:2] - line2[2:]),
                       np.linalg.norm(line1[2:] - line2[:2]), np.linalg.norm(line1[2:] - line2[2:]))
        return (compatibility, min_dist)
    
    if num_true_edges >= 2:
        potential_pairs = sorted([(i, j) for i, j in combinations(range(num_true_edges), 2)], key=sort_key_func)
    else:
        potential_pairs = []

    # --- Canny 端点搜索辅助：沿主边方向的窄通道 run-length 检测 ---
    def _search_endpoint_canny(start_pt, dir_unit_vec, edge_img, max_search_dist_px: float,
                               stripe_half_w_px: int, min_edge_run_px: int, gap_min_px: int):
        if edge_img is None:
            return None
        h, w = edge_img.shape[:2]
        # 垂直方向单位向量（用于条带）
        perp = np.array([-dir_unit_vec[1], dir_unit_vec[0]], dtype=float)
        run_len = 0
        gap_len = 0
        last_edge_pos = None
        steps = int(max(1, int(round(float(max_search_dist_px)))))
        for i_step in range(1, steps + 1):
            p = start_pt + dir_unit_vec * i_step
            x, y = float(p[0]), float(p[1])
            if not (0 <= x < w and 0 <= y < h):
                break
            # 采样条带：[-half, +half]，取任何白点即认为该位置有边
            has_edge_here = False
            for off in range(-stripe_half_w_px, stripe_half_w_px + 1):
                q = p + perp * off
                qx, qy = int(round(q[0])), int(round(q[1]))
                if 0 <= qx < w and 0 <= qy < h:
                    if edge_img[qy, qx] != 0:
                        has_edge_here = True
                        break
            if has_edge_here:
                run_len += 1
                gap_len = 0
                last_edge_pos = p
            else:
                if run_len >= min_edge_run_px:
                    gap_len += 1
                    if gap_len >= gap_min_px:
                        # 在足够长的“完好边”之后出现连续缺失，即认为端点在缺失开始前的最后一处边
                        if last_edge_pos is not None:
                            return last_edge_pos
                        else:
                            return p
                # 边尚未形成稳定 run，继续前进
        return None

    for i, j in potential_pairs:
        if all(endpoint_paired_status.get(i, [True,True])) or all(endpoint_paired_status.get(j, [True,True])): continue
        line1, line2 = true_edges[i], true_edges[j]
        
        angle_between = calculate_angle_between_lines(line1, line2)
        # 角点延伸距离以毫米配置
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
        
        dists_i = [np.linalg.norm(intersection - line1[:2]), np.linalg.norm(intersection - line1[2:])]; endpoint_idx_i = np.argmin(dists_i)
        dists_j = [np.linalg.norm(intersection - line2[:2]), np.linalg.norm(intersection - line2[2:])]; endpoint_idx_j = np.argmin(dists_j)

        if endpoint_paired_status[i][endpoint_idx_i] or endpoint_paired_status[j][endpoint_idx_j]:
            continue

        corner_gap_px = _get_dist_px(p_defect, "CORNER_MAX_PHYSICAL_GAP_MM", "CORNER_MAX_PHYSICAL_GAP", None, pixels_per_mm)
        # 允许 ROI 外延：若线段被 ROI 截断，允许最多 20% ROI 宽度的“虚拟延长”参与交点
        roi_ext_allow = 0.2 * float(roi_w)
        max_extend_eff = max(float(max_extension_dist), float(roi_ext_allow))
        is_physical = dists_i[endpoint_idx_i] < corner_gap_px and dists_j[endpoint_idx_j] < corner_gap_px
        is_valid_virtual = dists_i[endpoint_idx_i] < max_extend_eff and dists_j[endpoint_idx_j] < max_extend_eff

        if (is_physical or is_valid_virtual) and (0 <= intersection[0] < roi_w and 0 <= intersection[1] < roi_h):
            endpoint_paired_status[i][endpoint_idx_i] = True; endpoint_paired_status[j][endpoint_idx_j] = True
            edges_for_drawing[i][endpoint_idx_i*2:(endpoint_idx_i*2)+2] = intersection
            edges_for_drawing[j][endpoint_idx_j*2:(endpoint_idx_j*2)+2] = intersection
            p1_near = line1[:2] if endpoint_idx_i == 0 else line1[2:]; p2_near = line2[:2] if endpoint_idx_j == 0 else line2[2:]
            # 预先计算反向端点（用于 Q 回溯及 X 角度判定）
            p1_far = line1[2:] if endpoint_idx_i == 0 else line1[:2]
            p2_far = line2[2:] if endpoint_idx_j == 0 else line2[:2]

            def _handle_as_x_defect():
                angle = calculate_vertex_angle(p1_far, intersection, p2_far)
                # 仅在合理范围考虑 X（避免尖角/钝角极端值）
                if angle < 20.0 or angle > 160.0:
                    return
                # 按规则对 90° 邻域进行修约
                corrected_angle = _adjust_angle_near_90(angle)
                corrected_deviation_final = abs(corrected_angle - 90.0)
                angle_tolerance = p_defect.get("ANGLE_DEVIATION_TOLERANCE", 4.0)
                if corrected_deviation_final > angle_tolerance:
                    corner_defects.append({"type": "X", "center": tuple(map(int, intersection)), "angle": corrected_angle})

            # 删除 Harris 缺角检测：统一仅按角度偏差尝试判定 X，不再生成 Q
            _handle_as_x_defect()
            try:
                # 记录该交点用于后续过滤其附近的 B 误检
                non_q_intersections.append((float(intersection[0]), float(intersection[1])))
            except Exception:
                pass

    # 将斜边缺陷并入后续缺陷列表
    all_chipping_contours = []; chipping_defects = []
    for edge in true_edges:
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

    return edges_for_drawing, filtered_defects


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
    min_len_pixels = roi_gray.shape[1] * p_hough.get("MIN_LINE_LENGTH_RATIO", 0.05)
    max_line_gap_px = _get_dist_px(p_hough, "MAX_LINE_GAP_MM", "MAX_LINE_GAP", None, pixels_per_mm)
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=max_line_gap_px)
    
    main_edges = merge_lines_and_get_main_edges(raw_lines, params, pixels_per_mm, edge_img=binary_edges)
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
    edges_for_drawing, all_defects = find_and_analyze_defects(main_edges, roi_gray, roi_gray.shape, params, pixels_per_mm, binary_edges)

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
            length_mm = location.get('length_mm', 0); width_mm = location.get('width_mm', 0)
            area_mm2 = length_mm * width_mm; aspect_ratio = length_mm / width_mm if width_mm > 1e-6 else float('inf')
            if area_mm2 < 2.25 or aspect_ratio > 3.6: continue
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
        # 默认容忍角度与主流程保持一致：17°（可通过 DEFECT_DETECTION.VERTICAL_ANGLE_TOL_DEG 覆盖）
        vertical_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('VERTICAL_ANGLE_TOL_DEG', 17.0))
    except Exception:
        vertical_tol_deg = 17.0
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
    # 不再绘制 intersections 与示例射线
    # 可视化颜色：'E'（边缘异常）使用红色；注意为 BGR 通道顺序
    DEFECT_COLORS_BGR = {'Q': (0, 0, 255), 'E': (0, 0, 255), 'X': (255, 0, 0), 'L': (255, 0, 255), 'B': (0, 165, 255)}
    p_vis = params["VISUALIZATION"]
    THICKNESS = 1
    
    alpha = p_vis["DEFECT_OVERLAY_ALPHA"]; beta = 1 - alpha
    
    # 按最新要求：不再绘制直线与打印端点
    annotations_to_draw = []
    
    for defect_report in final_defects_for_report:
        defect = defect_report['raw_defect']
        color_bgr = DEFECT_COLORS_BGR.get(defect_report["type"], (255, 255, 255))
        
        loc = defect_report['location']
        # 类型名称映射：新增 'E' -> 边缘异常；保留 'X' 兼容为“斜边（历史）”
        defect_type_map = {'Q': '缺角', 'B': '崩边', 'E': '边缘异常', 'X': '斜边', 'L': '裂纹'}
        type_str = defect_type_map.get(defect_report['type'], '未知')
        
        # 不再绘制射线段 ray_segments

        if defect_report['type'] in ('E', 'X'):
            # 若为曲边，展示曲度（曲率角）；否则展示与垂直参考的夹角
            if loc.get('subtype') == 'curved' or (defect.get('skew_subtype', '') == 'curved'):
                text = f"{type_str}：曲边: ({loc['x']}, {loc['y']}), 曲度: {loc.get('angle', 0.0):.1f}°"
            else:
                text = f"{type_str}: ({loc['x']}, {loc['y']}), 角度: {loc.get('angle', 0.0):.1f}°"
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

    # 在最终标注图上叠加“玻璃最大轮廓 + 最小外接矩形”，便于在结果视频中观察
    try:
        kernel_viz = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dil_viz = cv2.dilate(binary_edges, kernel_viz, iterations=1)
        cnts_viz, _ = cv2.findContours(edges_dil_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_viz:
            main_viz = max(cnts_viz, key=cv2.contourArea)
            area_px = cv2.contourArea(main_viz)
            if area_px > 10.0:
                ok_area = True  # 不再按轮廓面积门控，统一进入外接形状计算
                if ok_area:
                    # 按需求：不再绘制原始玻璃主体轮廓
                    pass

                    # 结合“主边缘直线 + 玻璃主体轮廓”计算一个“最小包裹轮廓且与主边平行”的平行四边形（最小外接平行包络）
                    try:
                        # 1) 计算主方向：优先使用 Hough 主边缘方向；若不可用，退回最小外接矩形方向
                        u_dir = None; v_dir = None; angle_deg = None
                        if main_edges and len(main_edges) > 0:
                            # 统计每条线的方向向量并做主方向聚合
                            dirs = []
                            for e in main_edges:
                                x1, y1, x2, y2 = map(float, e)
                                v = np.array([x2 - x1, y2 - y1], dtype=np.float32)
                                L = float(np.hypot(v[0], v[1]))
                                if L > 1e-6:
                                    v /= L
                                    # 方向无符号：把朝向统一到半圆
                                    if v[0] < 0:
                                        v = -v
                                    dirs.append(v)
                            if dirs:
                                mean_dir = np.mean(np.stack(dirs, axis=0), axis=0)
                                L = float(np.hypot(mean_dir[0], mean_dir[1]))
                                if L > 1e-6:
                                    u_dir = (mean_dir / L).astype(np.float32)
                                    v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
                                    angle_deg = float(np.degrees(np.arctan2(u_dir[1], u_dir[0])))
                        if u_dir is None:
                            rect_local = cv2.minAreaRect(main_viz)
                            box_local = cv2.boxPoints(rect_local).astype(np.float32)
                            e0 = box_local[1] - box_local[0]
                            L0 = float(np.hypot(e0[0], e0[1]))
                            if L0 < 1e-6:
                                raise RuntimeError('degenerate rect for envelope')
                            u_dir = (e0 / L0).astype(np.float32)
                            v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float32)
                            angle_deg = float(np.degrees(np.arctan2(u_dir[1], u_dir[0])))

                        # 2) 仅依靠主边缘直线生成平行四边形：按方向分组，取各自偏移极值构造两对平行线
                        pts = main_viz.reshape(-1, 2).astype(np.float32)
                        center = np.mean(pts, axis=0).astype(np.float32)

                        try:
                            angle_tol_deg = float(params.get('DEFECT_DETECTION', {}).get('ENVELOPE_SNAP_ANGLE_TOL_DEG', 8.0))
                        except Exception:
                            angle_tol_deg = 8.0
                        cos_tol = np.cos(np.deg2rad(angle_tol_deg))

                        group_u = []  # 与 u_dir 平行 → 用于 v=const 的上下边
                        group_v = []  # 与 v_dir 平行 → 用于 u=const 的左右边
                        for e in (main_edges or []):
                            x1, y1, x2, y2 = map(float, e)
                            p1 = np.array([x1, y1], dtype=np.float32)
                            p2 = np.array([x2, y2], dtype=np.float32)
                            vec = p2 - p1
                            L = float(np.hypot(vec[0], vec[1]))
                            if L <= 1e-6:
                                continue
                            dir_line = vec / L
                            if abs(float(np.dot(dir_line, u_dir))) >= cos_tol:
                                v1 = (p1 - center) @ v_dir; v2 = (p2 - center) @ v_dir
                                v_off = float(0.5 * (v1 + v2))
                                group_u.append({'line': e, 'v_off': v_off})
                            elif abs(float(np.dot(dir_line, v_dir))) >= cos_tol:
                                u1 = (p1 - center) @ u_dir; u2 = (p2 - center) @ u_dir
                                u_off = float(0.5 * (u1 + u2))
                                group_v.append({'line': e, 'u_off': u_off})

                        # 需要两组方向各至少两条直线，才能纯由主边生成一个闭合平行四边形
                        if len(group_u) >= 2 and len(group_v) >= 2:
                            group_u.sort(key=lambda it: it['v_off'])
                            group_v.sort(key=lambda it: it['u_off'])
                            v_top = float(group_u[0]['v_off']); v_bottom = float(group_u[-1]['v_off'])
                            u_left = float(group_v[0]['u_off']); u_right = float(group_v[-1]['u_off'])

                            # 生成无限长边线并求四角
                            T_local = float(max(roi_gray.shape) * 20)
                            def _line_from_v(v0):
                                c = center + v0 * v_dir
                                return np.array([c[0] - T_local*u_dir[0], c[1] - T_local*u_dir[1], c[0] + T_local*u_dir[0], c[1] + T_local*u_dir[1]])
                            def _line_from_u(u0):
                                c = center + u0 * u_dir
                                return np.array([c[0] - T_local*v_dir[0], c[1] - T_local*v_dir[1], c[0] + T_local*v_dir[0], c[1] + T_local*v_dir[1]])

                            lt = find_line_intersection(_line_from_u(u_left), _line_from_v(v_top))
                            rt = find_line_intersection(_line_from_u(u_right), _line_from_v(v_top))
                            rb = find_line_intersection(_line_from_u(u_right), _line_from_v(v_bottom))
                            lb = find_line_intersection(_line_from_u(u_left), _line_from_v(v_bottom))
                            if any(pt is None for pt in (lt, rt, rb, lb)):
                                raise RuntimeError('failed to intersect snapped lines')
                            quad_pts = np.array([
                                [int(round(lt[0])), int(round(lt[1]))],
                                [int(round(rt[0])), int(round(rt[1]))],
                                [int(round(rb[0])), int(round(rb[1]))],
                                [int(round(lb[0])), int(round(lb[1]))]
                            ], dtype=np.int32)

                            # 结果模式：仅由主边直线生成的平行四边形
                            mode = 'lines-parallelogram'

                            # 面积（px² 与 mm²）
                            width_px = float(u_right - u_left)
                            height_px = float(v_bottom - v_top)
                            area_px2 = max(0.0, width_px) * max(0.0, height_px)
                            area_mm2 = float(area_px2) / float(pixels_per_mm * pixels_per_mm) if pixels_per_mm else 0.0

                            # 按新需求：不再写入 envelope 到报告（屏蔽）
                            pass
                        else:
                            # 主边直线不足以闭合平行四边形：仅记录状态，不绘制、不使用 fallback
                            # 不写入 envelope 信息
                            pass
                    except Exception:
                        # 构建失败则静默跳过（不使用最小矩形回退，以满足“仅依赖主边直线”）
                        # 构建失败亦不写 envelope
                        pass
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
            
        slim_report = {k: roi_report.get(k) for k in ("roi_idx","x","y","w","h","edges_found")}
        report['rois'].append(slim_report)

    report["state_code"] = 1 if max_edges_found > 0 else 0
    
    return report, final_image