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
                valid_coords.append(p)

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
    blurred = cv2.medianBlur(roi_gray, p["MEDIAN_BLUR_KSIZE"])
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP_LIMIT"], tileGridSize=grid_size)
    enhanced_contrast = clahe.apply(blurred)
    return cv2.Canny(enhanced_contrast, p["CANNY_THRESHOLD_LOW"], p["CANNY_THRESHOLD_HIGH"])

def merge_lines_and_get_main_edges(lines, params, pixels_per_mm: float):
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
                    vec2 = p1 - mid_point
                    cross_product_2d = vec_line[0] * vec2[1] - vec_line[1] * vec2[0]
                    # 以毫米配置的横向距离容差
                    max_lat_dist_px = _get_dist_px(p, "MAX_LATERAL_DISTANCE_MM", "MAX_LATERAL_DISTANCE", None, pixels_per_mm)
                    if np.abs(cross_product_2d) / line_length < max_lat_dist_px:
                        group.append(segment); placed = True; break
            if not placed: proximity_groups.append([segment])
        final_line_groups.extend(proximity_groups)
    merged_lines_with_scores = []
    for group in final_line_groups:
        points = np.array([pt for line in group for pt in (line[0:2], line[2:4])], dtype=np.float32)
        if len(points) < 2: continue
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

def find_and_analyze_defects(edges, roi_gray, roi_dims, params, pixels_per_mm: float):
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

    # 计算某条主边缘“线段像素”的均值亮度（厚度=1px），用于缺角亮度门控
    def _edge_line_mean(edge_line):
        p1 = tuple(map(int, edge_line[:2]))
        p2 = tuple(map(int, edge_line[2:]))
        mask = np.zeros(roi_gray.shape, dtype=np.uint8)
        cv2.line(mask, p1, p2, 255, 1)
        cnt = cv2.countNonZero(mask)
        if cnt == 0:
            return float(cv2.mean(roi_gray)[0])
        return float(cv2.mean(roi_gray, mask=mask)[0])

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
        if intersection is None: continue
        
        dists_i = [np.linalg.norm(intersection - line1[:2]), np.linalg.norm(intersection - line1[2:])]; endpoint_idx_i = np.argmin(dists_i)
        dists_j = [np.linalg.norm(intersection - line2[:2]), np.linalg.norm(intersection - line2[2:])]; endpoint_idx_j = np.argmin(dists_j)

        if endpoint_paired_status[i][endpoint_idx_i] or endpoint_paired_status[j][endpoint_idx_j]:
            continue

        corner_gap_px = _get_dist_px(p_defect, "CORNER_MAX_PHYSICAL_GAP_MM", "CORNER_MAX_PHYSICAL_GAP", None, pixels_per_mm)
        is_physical = dists_i[endpoint_idx_i] < corner_gap_px and dists_j[endpoint_idx_j] < corner_gap_px
        is_valid_virtual = dists_i[endpoint_idx_i] < max_extension_dist and dists_j[endpoint_idx_j] < max_extension_dist

        if (is_physical or is_valid_virtual) and (0 <= intersection[0] < roi_w and 0 <= intersection[1] < roi_h):
            endpoint_paired_status[i][endpoint_idx_i] = True; endpoint_paired_status[j][endpoint_idx_j] = True
            edges_for_drawing[i][endpoint_idx_i*2:(endpoint_idx_i*2)+2] = intersection
            edges_for_drawing[j][endpoint_idx_j*2:(endpoint_idx_j*2)+2] = intersection
            p1_near = line1[:2] if endpoint_idx_i == 0 else line1[2:]; p2_near = line2[:2] if endpoint_idx_j == 0 else line2[2:]
            # 预先计算反向端点（用于 Q 回溯及 X 角度判定）
            p1_far = line1[2:] if endpoint_idx_i == 0 else line1[:2]
            p2_far = line2[2:] if endpoint_idx_j == 0 else line2[:2]
            q_created = False

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

            if is_valid_virtual and not is_physical:
                # 缺角(Q)亮度门控：取交点与上下左右4个点(共5点)的均值亮度
                ix, iy = int(round(float(intersection[0]))), int(round(float(intersection[1])))
                neighbor_coords = [
                    (ix, iy), (ix - 1, iy), (ix + 1, iy), (ix, iy - 1), (ix, iy + 1)
                ]
                samples = []
                for px, py in neighbor_coords:
                    if 0 <= px < roi_w and 0 <= py < roi_h:
                        samples.append(float(roi_gray[py, px]))
                inter_mean = float(np.mean(samples)) if samples else 0.0
                line_mean1 = _edge_line_mean(line1)
                line_mean2 = _edge_line_mean(line2)
                line_mean_ref = max(line_mean1, line_mean2)

                # 新规则：当找到 Q 的两个端点后，用“缺角三角形区域的平均亮度”对比“主边缘扫描带的平均亮度（背景）”做门控
                # 这里先进行预处理与向量准备，端点找到后再执行区域与扫描带的均值比较
                if inter_mean > line_mean_ref:
                    p_preproc = params["PREPROCESSING"]
                    roi_gray_blurred = cv2.medianBlur(roi_gray, p_preproc["MEDIAN_BLUR_KSIZE"])
                    vec1 = p1_far - intersection; norm1 = np.linalg.norm(vec1)
                    if norm1 > 1e-6: vec1 /= norm1
                    vec2 = p2_far - intersection; norm2 = np.linalg.norm(vec2)
                    if norm2 > 1e-6: vec2 /= norm2
                    grad_thresh = p_defect.get("Q_DEFECT_GRADIENT_THRESHOLD", 20.0)
                    search_w = int(max(1, int(p_defect.get("Q_DEFECT_SEARCH_WIDTH_PX", 2))))
                    new_p1 = find_gradient_endpoint(intersection, vec1, roi_gray_blurred, max_extension_dist, search_width=search_w, gradient_stop_threshold=grad_thresh)
                    new_p2 = find_gradient_endpoint(intersection, vec2, roi_gray_blurred, max_extension_dist, search_width=search_w, gradient_stop_threshold=grad_thresh)
                    # 取消回退：若任一端未找到有效梯度终止点，则认为是正常角点，不生成 Q
                    if (new_p1 is not None) and (new_p2 is not None):
                        # --- 计算缺角三角形区域与主边缘扫描带的平均亮度 ---
                        cx, cy = float(intersection[0]), float(intersection[1])
                        a = (int(round(cx)), int(round(cy)))
                        b = (int(round(new_p1[0])), int(round(new_p1[1])))
                        c = (int(round(new_p2[0])), int(round(new_p2[1])))

                        h_mask, w_mask = roi_h, roi_w
                        tri_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
                        cv2.fillPoly(tri_mask, [np.array([a, b, c], dtype=np.int32)], 255)
                        # 忽略三角形边缘：做一次轻微腐蚀
                        erode_px = max(1, int(search_w))
                        ksize = max(3, erode_px * 2 + 1)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                        tri_core = cv2.erode(tri_mask, kernel, iterations=1)

                        # 沿两条主边，使用“崩边扫描带”的定义构建背景带（与 scan_edge_for_luminosity_defects 一致）：
                        # 宽度来自 LUMINOSITY_SCAN_WIDTH(_MM)，选择亮度更低的一侧，并剔除边线本体与端点区域和三角核心。
                        scan_width = _get_dist_px(p_defect, "LUMINOSITY_SCAN_WIDTH_MM", "LUMINOSITY_SCAN_WIDTH", None, pixels_per_mm)
                        edge_ignore_px = _get_dist_px(p_defect, "LUMINOSITY_EDGE_IGNORE_WIDTH_MM", "LUMINOSITY_EDGE_IGNORE_WIDTH", 0.0, pixels_per_mm)
                        endpoint_exclude_r = _get_dist_px(p_defect, "LUMINOSITY_ENDPOINT_EXCLUDE_RADIUS_MM", None, None, pixels_per_mm, default_px=0.0)
                        if endpoint_exclude_r is None or endpoint_exclude_r <= 0:
                            endpoint_exclude_r = max(3, int(min(0.3 * scan_width, 15)))

                        def _make_chipping_belt_for_segment(p_start, p_end):
                            p1f = np.array([float(p_start[0]), float(p_start[1])])
                            p2f = np.array([float(p_end[0]), float(p_end[1])])
                            line_vec = p2f - p1f
                            line_length = float(np.linalg.norm(line_vec))
                            if line_length < 1e-6:
                                return np.zeros((h_mask, w_mask), dtype=np.uint8)
                            unit_vec = line_vec / line_length
                            normal_vec = np.array([-unit_vec[1], unit_vec[0]])
                            half_width_vec = (scan_width / 2.0) * normal_vec

                            c1 = p1f + half_width_vec; c2 = p2f + half_width_vec
                            c3 = p2f - half_width_vec; c4 = p1f - half_width_vec

                            mask_plus = np.zeros((h_mask, w_mask), dtype=np.uint8)
                            poly_plus = np.array([c1, c2, p2f, p1f], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask_plus, [poly_plus], 255)

                            mask_minus = np.zeros((h_mask, w_mask), dtype=np.uint8)
                            poly_minus = np.array([p1f, p2f, c3, c4], dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask_minus, [poly_minus], 255)

                            # 选择亮度更低的一侧作为背景（一致性：使用未模糊图计算侧均值）
                            mean_plus = cv2.mean(roi_gray, mask=mask_plus)[0]
                            mean_minus = cv2.mean(roi_gray, mask=mask_minus)[0]
                            scan_mask = mask_plus if mean_plus < mean_minus else mask_minus

                            # 剔除边线与端点邻域
                            ignore_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
                            if edge_ignore_px > 0:
                                cv2.line(ignore_mask, (int(round(p1f[0])), int(round(p1f[1]))), (int(round(p2f[0])), int(round(p2f[1]))), 255, thickness=int(round(edge_ignore_px)))
                            cv2.circle(ignore_mask, (int(round(p1f[0])), int(round(p1f[1]))), int(round(endpoint_exclude_r)), 255, thickness=-1)
                            cv2.circle(ignore_mask, (int(round(p2f[0])), int(round(p2f[1]))), int(round(endpoint_exclude_r)), 255, thickness=-1)

                            scan_mask = cv2.bitwise_and(scan_mask, cv2.bitwise_not(ignore_mask))
                            # 排除缺角三角形核心区域，避免把缺角本体计入背景
                            scan_mask = cv2.bitwise_and(scan_mask, cv2.bitwise_not(tri_core))
                            return scan_mask

                        belt1 = _make_chipping_belt_for_segment(a, b)
                        belt2 = _make_chipping_belt_for_segment(a, c)

                        # 计算均值（使用模糊后的灰度以降低噪声）
                        tri_vals = roi_gray_blurred[tri_core == 255]
                        belt1_vals = roi_gray_blurred[belt1 == 255]
                        belt2_vals = roi_gray_blurred[belt2 == 255]

                        valid_tri = tri_vals.size >= 10  # 至少若干像素才有统计意义
                        valid_b1 = belt1_vals.size >= 10
                        valid_b2 = belt2_vals.size >= 10

                        accept_q = False
                        if valid_tri and (valid_b1 or valid_b2):
                            tri_mean = float(np.mean(tri_vals))
                            # 背景取两条扫描带的均值（若两者都有），或仅用可用的一条
                            if valid_b1 and valid_b2:
                                bg_mean = float((np.mean(belt1_vals) + np.mean(belt2_vals)) / 2.0)
                            elif valid_b1:
                                bg_mean = float(np.mean(belt1_vals))
                            else:
                                bg_mean = float(np.mean(belt2_vals))
                            # 判定：缺角区域应当比背景更亮（与原先的“比主边线更亮”标准相近）
                            # 可按需加入余量，例如 p_defect.get('Q_TRIANGLE_BRIGHTNESS_MARGIN', 0.0)
                            margin = float(p_defect.get('Q_TRIANGLE_BRIGHTNESS_MARGIN', 0.0))
                            accept_q = (tri_mean > (bg_mean + margin))
                        else:
                            # 若统计不足，退化为旧规则（intersection 均值 vs 线均值）
                            accept_q = inter_mean > line_mean_ref

                        if accept_q:
                            corner_defects.append({
                                "type": "Q",
                                "center": tuple(map(int, intersection)),
                                "endpoints": (new_p1, new_p2),
                                "distances": (dists_i[endpoint_idx_i], dists_j[endpoint_idx_j])
                            })
                            q_created = True
                        else:
                            _handle_as_x_defect()
                    else:
                        # 可选：按原逻辑检测是否属于 X（若角度偏差较大）
                        _handle_as_x_defect()
                else:
                    _handle_as_x_defect()
            else:
                _handle_as_x_defect()

            # 若本交点未产生 Q，记录下来用于过滤其附近的 B 误检
            if not q_created:
                try:
                    non_q_intersections.append((float(intersection[0]), float(intersection[1])))
                except Exception:
                    pass

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
        for bd in chipping_defects:
            dmin = _min_dist_rect_to_edges(bd, true_edges)
            if dmin <= b_max_edge_dist_px:
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
            
    return edges_for_drawing, corner_defects + surviving_chipping_defects


def process_roi_hough_based(roi_idx, roi_template, image_gray, params, pixels_per_mm):
    x, y, w, h = int(roi_template['x']), int(roi_template['y']), int(roi_template['width']), int(roi_template['height'])
    roi_gray = image_gray[y:y+h, x:x+w]
    
    p_hough = params["HOUGH_TRANSFORM"]
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    min_len_pixels = roi_gray.shape[1] * p_hough.get("MIN_LINE_LENGTH_RATIO", 0.05)
    max_line_gap_px = _get_dist_px(p_hough, "MAX_LINE_GAP_MM", "MAX_LINE_GAP", None, pixels_per_mm)
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=max_line_gap_px)
    
    main_edges = merge_lines_and_get_main_edges(raw_lines, params, pixels_per_mm)
    edges_for_drawing, all_defects = find_and_analyze_defects(main_edges, roi_gray, roi_gray.shape, params, pixels_per_mm)

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

        if defect['type'] == 'X':
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
                if area_mm2 > 4000 and width_mm < 50: continue
                if area_mm2 > 1000 and aspect_ratio > 5.0: continue
        
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
                        if (area_mm2 < 25 and width_mm < min_size_mm) \
                           or (area_mm2 > 4000 and width_mm < 50) \
                           or (area_mm2 > 1000 and aspect_ratio > 5.0):
                            should_be_filtered = True
                except Exception:
                    pass

                raw_l = new_defect.get('raw_defect', {})
                box = raw_l.get('box_points')
                rect = raw_l.get('min_area_rect')
                if box is not None and len(box) >= 4 and len(main_edges) > 0:
                    box_np = np.array(box, dtype=float).reshape(-1, 2)
                    # 距离采样点：四角 + 中心
                    pts = [p for p in box_np]
                    if rect is not None and isinstance(rect, tuple) and len(rect) >= 2:
                        cx, cy = rect[0]
                        center_pt = np.array([float(cx), float(cy)], dtype=float)
                    else:
                        center_pt = np.mean(box_np, axis=0)
                    pts.append(center_pt)

                    max_dist_mm = float(params.get('DEFECT_DETECTION', {}).get('L_FILTER_MAX_DISTANCE_TO_EDGE_MM', 5.0))
                    # 与任一主边的最小“点到直线（无限延长）”垂直距离
                    for e in main_edges:
                        a = np.array(e[:2], dtype=float); b = np.array(e[2:], dtype=float)
                        for pt in pts:
                            d_px = get_point_line_perpendicular_distance(pt, [a[0], a[1], b[0], b[1]])
                            d_mm = d_px / float(pixels_per_mm if pixels_per_mm else 1.0)
                            if d_mm <= max_dist_mm:
                                should_be_filtered = True
                                break
                        if should_be_filtered:
                            break
            except Exception:
                pass

        # --- MODIFICATION START: Final check of the filter flag ---
        if not should_be_filtered:
            final_defects_for_report.append(new_defect)
        # --- MODIFICATION END ---
        
    # 统计“近竖直”的主边数量（0~2 常见）：基于主边段方向角(相对x轴 0~90°)，角度>=90°-tol 视为近竖直
    try:
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
    DEFECT_COLORS_BGR = {'Q': (0, 0, 255), 'X': (255, 0, 0), 'L': (255, 0, 255), 'B': (0, 165, 255)}
    p_vis = params["VISUALIZATION"]
    THICKNESS = 1
    
    alpha = p_vis["DEFECT_OVERLAY_ALPHA"]; beta = 1 - alpha
    
    #for edge in edges_for_drawing:
    #    pt1 = tuple(map(int, edge[:2]))
    #    pt2 = tuple(map(int, edge[2:]))
    #    cv2.line(roi_color, pt1, pt2, (0, 255, 0), 2)

    annotations_to_draw = []
    
    for defect_report in final_defects_for_report:
        defect = defect_report['raw_defect']
        color_bgr = DEFECT_COLORS_BGR.get(defect_report["type"], (255, 255, 255))
        
        loc = defect_report['location']
        defect_type_map = {'Q': '缺角', 'B': '崩边', 'X': '斜边', 'L': '裂纹'}
        type_str = defect_type_map.get(defect_report['type'], '未知')
        
        if defect_report['type'] == 'X':
            text = f"{type_str}: ({loc['x']}, {loc['y']}), 角度: {loc['angle']:.1f}°"
        elif defect_report['type'] == 'Q' and 'pixel_area' in loc:
            text = f"{type_str}: ({loc['x']}, {loc['y']}), 尺寸: {loc['length_mm']:.1f}x{loc['width_mm']:.1f}mm"
        else:
            text = f"{type_str}: ({loc['x']}, {loc['y']}), 尺寸: {loc['length_mm']:.1f}x{loc['width_mm']:.1f}mm"
        
        annotations_to_draw.append({'text': text, 'color': color_bgr})

        if defect["type"] == "Q" and "endpoints" in defect:
            center = np.array(defect["center"])
            p1_orig, p2_orig = np.array(defect["endpoints"][0]), np.array(defect["endpoints"][1])
            v1_final, v2_final = p1_orig, p2_orig 

            if "distances" in defect:
                retreat_threshold = p_vis.get("RETREAT_DISTANCE_THRESHOLD", 100.0)
                retreat_len = p_vis.get("EDGE_ENDPOINT_FIXED_LENGTH", 20)
                dist1, dist2 = defect["distances"]
                if dist1 > retreat_threshold:
                    vec1 = (p1_orig - center); norm_vec1 = np.linalg.norm(vec1)
                    if norm_vec1 > 1e-6: v1_final = center + (vec1 / norm_vec1) * retreat_len
                if dist2 > retreat_threshold:
                    vec2 = (p2_orig - center); norm_vec2 = np.linalg.norm(vec2)
                    if norm_vec2 > 1e-6: v2_final = center + (vec2 / norm_vec2) * retreat_len
                try:
                    defect_report.setdefault('_adjusted_q_lengths_px', [float(np.linalg.norm(v1_final - center)), float(np.linalg.norm(v2_final - center))])
                except Exception: pass

            triangle_vertices = np.array([tuple(map(int, center)), tuple(map(int, v1_final)), tuple(map(int, v2_final))], dtype=np.int32)
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [triangle_vertices], color_bgr)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            blue_color = (255, 0, 0)
            draw_dashed_line(roi_color, tuple(map(int, center)), tuple(map(int, v1_final)), blue_color, thickness=2, dash_length=8)
            draw_dashed_line(roi_color, tuple(map(int, center)), tuple(map(int, v2_final)), blue_color, thickness=2, dash_length=8)

        elif defect["type"] == "X" and "center" in defect:
            cv2.circle(roi_color, defect["center"], 15, color_bgr, THICKNESS)
            
        elif defect_report["type"] in ["L", "B"] and "box_points" in defect:
            box_points = defect["box_points"]
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [box_points], color_bgr)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            cv2.drawContours(roi_color, [box_points], 0, color_bgr, THICKNESS)

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
            x, y, w, h = int(r.get('x',0)), int(r.get('y',0)), int(r.get('width',0)), int(r.get('height',0))
            roi_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                roi_gray_crop = image_gray[y:y+h, x:x+w]
                roi_bgr = cv2.cvtColor(roi_gray_crop, cv2.COLOR_GRAY2BGR)
            except Exception:
                 pass
            return ({"roi_idx": i, "x": x, "y": y, "w": w, "h": h, "defects": [], "edges_found": 0}, roi_bgr)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_safe_roi_hough, i, r) for i, r in enumerate(template_rois)]
        results = [future.result() for future in futures]

    max_edges_found = 0
    for roi_report, roi_color in results:
        if "x" not in roi_report: continue
        x, y, w, h = roi_report["x"], roi_report["y"], roi_report["w"], roi_report["h"]
        if w > 0 and h > 0: final_image[y:y+h, x:x+w] = roi_color
        
        max_edges_found = max(max_edges_found, roi_report.get("edges_found", 0))
        
        if roi_report.get("defects"):
            report["image_status"] = "NG"
            report["defects"].extend(roi_report["defects"])
            
        slim_report = {k: roi_report.get(k) for k in ("roi_idx","x","y","w","h","edges_found")}
        report['rois'].append(slim_report)

    report["state_code"] = 1 if max_edges_found > 0 else 0
    
    return report, final_image