# --- START OF FILE image_processor_hough.py (Font Size and Spacing Corrected) ---
import cv2
import numpy as np
import json
import os
from itertools import combinations
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# FIX: Import Pillow for CJK character support
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ====================================================================================
# --- 全局字体加载 ---
# ====================================================================================

def _get_font(font_size=36):
    """
    Attempts to load a CJK-compatible font from common system paths.
    """
    if not PIL_AVAILABLE:
        return None
    
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',      # Microsoft YaHei on Windows
        'C:/Windows/Fonts/simsun.ttc',      # SimSun on Windows
        '/System/Library/Fonts/PingFang.ttc', # PingFang on macOS
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Noto Sans CJK on Linux
    ]
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except IOError:
            continue
    
    # If no specific font is found, use Pillow's default and print a warning.
    print("警告: 未找到中文字体, 标注可能无法正确显示中文。请安装或指定字体路径。")
    try:
        # For Pillow 10.0.0+, load_default() may require a size argument.
        # However, to maintain compatibility, we call it without arguments first.
        return ImageFont.load_default()
    except Exception:
        return None

# Load the font once when the script is imported
# --- FIX: Changed font size from 20 to 36 ---
ANNOTATION_FONT = _get_font(font_size=36)

# 全局缺陷日志（跨帧累积，不清除）
# 存储元素: (text, (R,G,B)) 兼容旧字符串形式
GLOBAL_DEFECT_LOG = []


# ====================================================================================
# --- 几何学与分析辅助函数 ---
# ====================================================================================

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """在图像上绘制虚线"""
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

def calculate_angle_between_lines(line1, line2):
    v1 = np.array(line1[2:]) - np.array(line1[:2])
    v2 = np.array(line2[2:]) - np.array(line2[:2])
    v1_mag = np.linalg.norm(v1); v2_mag = np.linalg.norm(v2)
    if v1_mag < 1e-9 or v2_mag < 1e-9: return 0.0
    cosine_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle)
    if angle_deg > 90.0: angle_deg = 180.0 - angle_deg
    return angle_deg

def get_point_line_segment_projection(point, line_segment):
    p = np.array(point); p1 = np.array(line_segment[:2]); p2 = np.array(line_segment[2:])
    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-8: return p1, np.linalg.norm(p - p1)
    t = np.dot(p - p1, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_point = p1 + t * line_vec
    return proj_point, np.linalg.norm(p - proj_point)

def scan_edge_for_luminosity_defects(roi_gray, edge, params):
    p = params["DEFECT_DETECTION"]
    
    p1 = np.array(edge[:2]); p2 = np.array(edge[2:])
    scan_width = p["LUMINOSITY_SCAN_WIDTH"]
    line_vec = p2 - p1; line_length = np.linalg.norm(line_vec)
    if line_length < 1e-6: return []

    unit_vec = line_vec / line_length; normal_vec = np.array([-unit_vec[1], unit_vec[0]])
    half_width_vec = (scan_width / 2.0) * normal_vec
    c1 = p1 + half_width_vec; c2 = p2 + half_width_vec
    c3 = p2 - half_width_vec; c4 = p1 - half_width_vec
    rect_points = np.array([c1, c2, c3, c4], dtype=np.int32).reshape((-1, 1, 2))
    
    scan_mask = np.zeros_like(roi_gray)
    cv2.fillPoly(scan_mask, [rect_points], 255)
    
    mean, std_dev = cv2.meanStdDev(roi_gray, mask=scan_mask)
    
    initial_contours = []
    if std_dev[0][0] > 3:
        threshold_low = mean[0][0] - p["LUMINOSITY_STD_DEV_MULTIPLIER"] * std_dev[0][0]
        potential_defects = (roi_gray < threshold_low).astype(np.uint8) * 255
        defect_mask = cv2.bitwise_and(potential_defects, scan_mask)

        if p.get("LUMINOSITY_EDGE_IGNORE_WIDTH", 0) > 0:
            ignore_mask = np.zeros_like(roi_gray)
            cv2.line(ignore_mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=p["LUMINOSITY_EDGE_IGNORE_WIDTH"])
            defect_mask = cv2.subtract(defect_mask, ignore_mask)

        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > p["LUMINOSITY_MIN_AREA"]:
                initial_contours.append(cnt)
    return initial_contours

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

def merge_lines_and_get_main_edges(lines, params):
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
                    if np.abs(cross_product_2d) / line_length < p["MAX_LATERAL_DISTANCE"]:
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
    merged_lines_with_scores.sort(key=lambda item: item['score'], reverse=True)
    return [item['line'] for item in merged_lines_with_scores[:p["TOP_N_EDGES"]]]

def find_and_analyze_defects(edges, roi_gray, roi_dims, params):
    p_defect = params["DEFECT_DETECTION"]; p_crack = params["CRACK_CLASSIFICATION"]
    num_edges = len(edges); roi_h, roi_w = roi_dims
    
    crack_defects = []; crack_indices = set(); endpoint_proximity_threshold = 15.0 
    if num_edges >= 2:
        for i, j in combinations(range(num_edges), 2):
            if i in crack_indices or j in crack_indices: continue
            line1, line2 = edges[i], edges[j]
            for point1 in [line1[:2], line1[2:]]:
                proj_point, dist = get_point_line_segment_projection(point1, line2)
                if dist < endpoint_proximity_threshold:
                    len_line2 = np.linalg.norm(line2[:2] - line2[2:])
                    if len_line2 > 1e-6:
                        shield2 = len_line2 * p_crack["ENDPOINT_SHIELD_RATIO"]
                        if np.linalg.norm(proj_point - line2[:2]) > shield2 and np.linalg.norm(proj_point - line2[2:]) > shield2:
                            x, y, w, h = cv2.boundingRect(line1.reshape(-1, 2).astype(np.int32));
                            crack_defects.append({"type": "L", "rect": (x,y,w,h)}); crack_indices.add(i); break
            if i in crack_indices: continue
            for point2 in [line2[:2], line2[2:]]:
                proj_point, dist = get_point_line_segment_projection(point2, line1)
                if dist < endpoint_proximity_threshold:
                    len_line1 = np.linalg.norm(line1[:2] - line1[2:])
                    if len_line1 > 1e-6:
                        shield1 = len_line1 * p_crack["ENDPOINT_SHIELD_RATIO"]
                        if np.linalg.norm(proj_point - line1[:2]) > shield1 and np.linalg.norm(proj_point - line1[2:]) > shield1:
                            x, y, w, h = cv2.boundingRect(line2.reshape(-1, 2).astype(np.int32))
                            crack_defects.append({"type": "L", "rect": (x,y,w,h)}); crack_indices.add(j); break

    true_edges = [edge for i, edge in enumerate(edges) if i not in crack_indices]
    
    corner_defects = []; num_true_edges = len(true_edges)
    edges_for_drawing = [edge.copy() for edge in true_edges]
    endpoint_paired_status = {i: [False, False] for i in range(num_true_edges)}
    potential_pairs = sorted([(i, j) for i, j in combinations(range(num_true_edges), 2)], key=lambda p: min(np.linalg.norm(true_edges[p[0]][:2] - true_edges[p[1]][:2]), np.linalg.norm(true_edges[p[0]][:2] - true_edges[p[1]][2:]), np.linalg.norm(true_edges[p[0]][2:] - true_edges[p[1]][:2]), np.linalg.norm(true_edges[p[0]][2:] - true_edges[p[1]][2:])))

    for i, j in potential_pairs:
        if all(endpoint_paired_status.get(i, [True,True])) or all(endpoint_paired_status.get(j, [True,True])): continue
        line1, line2 = true_edges[i], true_edges[j]
        
        angle_between = calculate_angle_between_lines(line1, line2)
        max_extension_dist = p_defect["CORNER_MAX_EXTENSION_DIST_PERPENDICULAR"] if abs(angle_between - 90.0) < p_defect["PERPENDICULAR_ANGLE_TOLERANCE"] else p_defect["CORNER_MAX_EXTENSION_DIST_NORMAL"]
        
        intersection = find_line_intersection(line1, line2)
        if intersection is None: continue
        
        dists_i = [np.linalg.norm(intersection - line1[:2]), np.linalg.norm(intersection - line1[2:])]; endpoint_idx_i = np.argmin(dists_i)
        dists_j = [np.linalg.norm(intersection - line2[:2]), np.linalg.norm(intersection - line2[2:])]; endpoint_idx_j = np.argmin(dists_j)

        if endpoint_paired_status[i][endpoint_idx_i] or endpoint_paired_status[j][endpoint_idx_j]: continue
        
        is_physical = dists_i[endpoint_idx_i] < p_defect["CORNER_MAX_PHYSICAL_GAP"] and dists_j[endpoint_idx_j] < p_defect["CORNER_MAX_PHYSICAL_GAP"]
        is_valid_virtual = dists_i[endpoint_idx_i] < max_extension_dist and dists_j[endpoint_idx_j] < max_extension_dist

        if (is_physical or is_valid_virtual) and (0 <= intersection[0] < roi_w and 0 <= intersection[1] < roi_h):
            endpoint_paired_status[i][endpoint_idx_i] = True; endpoint_paired_status[j][endpoint_idx_j] = True
            edges_for_drawing[i][endpoint_idx_i*2:(endpoint_idx_i*2)+2] = intersection
            edges_for_drawing[j][endpoint_idx_j*2:(endpoint_idx_j*2)+2] = intersection
            p1_near = line1[:2] if endpoint_idx_i == 0 else line1[2:]; p2_near = line2[:2] if endpoint_idx_j == 0 else line2[2:]

            if is_valid_virtual and not is_physical:
                corner_defects.append({"type": "Q", "center": tuple(map(int, intersection)), "endpoints": (p1_near, p2_near), "distances": (dists_i[endpoint_idx_i], dists_j[endpoint_idx_j])})
            else:
                p1_far = line1[2:] if endpoint_idx_i == 0 else line1[:2]; p2_far = line2[2:] if endpoint_idx_j == 0 else line2[:2]
                angle = calculate_vertex_angle(p1_far, intersection, p2_far)
                deviation = abs(angle - 90.0)
                
                angle_tolerance = p_defect.get("ANGLE_DEVIATION_TOLERANCE", 2.0)
                if deviation > angle_tolerance:
                    corner_defects.append({"type": "X", "center": tuple(map(int, intersection)), "angle": angle})

    all_chipping_contours = []; chipping_defects = []
    for edge in true_edges:
        all_chipping_contours.extend(scan_edge_for_luminosity_defects(roi_gray, edge, params))
    if all_chipping_contours:
        defect_canvas = np.zeros(roi_dims, dtype=np.uint8)
        cv2.drawContours(defect_canvas, all_chipping_contours, -1, 255, -1)
        kernel = np.ones(tuple(p_defect.get("MERGE_DEFECTS_KERNEL_SIZE", [5, 5])), np.uint8)
        merged_mask = cv2.morphologyEx(defect_canvas, cv2.MORPH_CLOSE, kernel, iterations=2)
        final_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_width = p_defect.get("FALSE_DEFECT_MAX_WIDTH", 8.0)
        min_aspect_ratio = p_defect.get("FALSE_DEFECT_MIN_ASPECT_RATIO", 5.0)

        for cnt in final_contours:
            if cv2.contourArea(cnt) < p_defect["LUMINOSITY_MIN_AREA"]: continue
            min_area_rect = cv2.minAreaRect(cnt)
            (w, h) = min_area_rect[1]
            width = min(w, h)
            length = max(w, h)
            if width < 1e-6: continue
            aspect_ratio = length / width
            if width < max_width and aspect_ratio > min_aspect_ratio: continue
            box_points = cv2.boxPoints(min_area_rect)
            box_points = np.intp(box_points)
            chipping_defects.append({"type": "B", "box_points": box_points})
    
    final_chipping_defects = []
    shield_radius = p_defect.get("CHIPPING_ENDPOINT_SHIELD_RADIUS", 30)
    all_endpoints = [np.array(edge[:2]) for edge in true_edges] + [np.array(edge[2:]) for edge in true_edges]

    if not all_endpoints:
        final_chipping_defects = chipping_defects
    else:
        for defect in chipping_defects:
            center = np.mean(defect["box_points"], axis=0)
            min_dist_to_endpoint = min(np.linalg.norm(center - ep) for ep in all_endpoints)
            if min_dist_to_endpoint > shield_radius:
                final_chipping_defects.append(defect)
            
    return edges_for_drawing, corner_defects + final_chipping_defects + crack_defects

def process_roi_hough_based(roi_idx, roi_template, image_gray, params, pixels_per_mm):
    x, y, w, h = int(roi_template['x']), int(roi_template['y']), int(roi_template['width']), int(roi_template['height'])
    roi_gray = image_gray[y:y+h, x:x+w]
    
    p_hough = params["HOUGH_TRANSFORM"]
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    min_len_pixels = roi_gray.shape[1] * p_hough["MIN_LINE_LENGTH_RATIO"]
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=p_hough["MAX_LINE_GAP"])
    
    main_edges = merge_lines_and_get_main_edges(raw_lines, params)
    edges_for_drawing, all_defects = find_and_analyze_defects(main_edges, roi_gray, roi_gray.shape, params)
    
    final_defects_for_report = []
    for defect in all_defects:
        new_defect = {'type': defect['type']}
        location = {}

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
                    p1, p2 = defect['endpoints']
                    dist_leg1_px = np.linalg.norm(np.array(center) - np.array(p1))
                    dist_leg2_px = np.linalg.norm(np.array(center) - np.array(p2))
                    # 修改：若绘制阶段会因为“过长”回退到固定长度，这里尺寸同样应用回退规则，确保报告尺寸与最终三角形一致。
                    p_vis_local = params.get("VISUALIZATION", {})
                    retreat_threshold = p_vis_local.get("RETREAT_DISTANCE_THRESHOLD", 100.0)
                    retreat_len_px = p_vis_local.get("EDGE_ENDPOINT_FIXED_LENGTH", 20)
                    adj_leg1 = retreat_len_px if dist_leg1_px > retreat_threshold else dist_leg1_px
                    adj_leg2 = retreat_len_px if dist_leg2_px > retreat_threshold else dist_leg2_px
                    length_px = max(adj_leg1, adj_leg2)
                    width_px  = min(adj_leg1, adj_leg2)
                else:
                    length_px, width_px = 0.0, 0.0
            elif defect['type'] == 'L':
                lx, ly, lw, lh = defect.get('rect', (0,0,0,0))
                location['x'] = int(lx + lw//2 + x)
                location['y'] = int(ly + lh//2 + y)
                length_px, width_px = max(lw, lh), min(lw, lh)
            elif defect['type'] == 'B':
                box = defect.get('box_points', [])
                if len(box) >= 4:
                    center = np.mean(box, axis=0)
                    location['x'] = int(center[0]) + x
                    location['y'] = int(center[1]) + y
                    d1 = np.linalg.norm(np.array(box[0]) - np.array(box[1]))
                    d2 = np.linalg.norm(np.array(box[1]) - np.array(box[2]))
                    length_px, width_px = max(d1, d2), min(d1, d2)

            location['length_mm'] = float(round(length_px / pixels_per_mm, 2))
            location['width_mm'] = float(round(width_px / pixels_per_mm, 2))

        new_defect['location'] = location
        
        if new_defect['type'] in ['Q', 'L', 'B']:
            min_size_mm = params["DEFECT_DETECTION"].get("MIN_DEFECT_SIZE_MM", 3.0)
            defect_size_mm = location.get('length_mm', 0)
            if defect_size_mm < min_size_mm:
                continue 

        new_defect['raw_defect'] = defect 
        final_defects_for_report.append(new_defect)
        
    roi_report = {
        "roi_idx": roi_idx, "x": x, "y": y, "w": w, "h": h,
        "defects": [d.copy() for d in final_defects_for_report],
        "edges_found": len(main_edges)
    }
    for d in roi_report['defects']:
        d.pop('raw_defect', None)

    roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    DEFECT_COLORS_BGR = {'Q': (0, 0, 255), 'X': (0, 255, 255), 'L': (255, 0, 255), 'B': (0, 165, 255)}
    p_vis = params["VISUALIZATION"]
    THICKNESS = 3
    
    alpha = p_vis["DEFECT_OVERLAY_ALPHA"]; beta = 1 - alpha
    
    # 不再在 ROI 局部绘制文字，只保留图形高亮
    annotations_to_draw = []
    
    for defect_report in final_defects_for_report:
        defect = defect_report['raw_defect']
        color_bgr = DEFECT_COLORS_BGR.get(defect["type"], (255, 255, 255))
        
        loc = defect_report['location']
        defect_type_map = {'Q': '缺角', 'B': '崩边', 'X': '斜边', 'L': '裂纹'}
        type_str = defect_type_map.get(defect_report['type'], '未知')
        
    # 这里不添加文字到 ROI 局部

        if defect["type"] == "Q" and "endpoints" in defect:
            center = np.array(defect["center"])
            p1_orig, p2_orig = np.array(defect["endpoints"][0]), np.array(defect["endpoints"][1])
            v1_final, v2_final = p1_orig, p2_orig 

            if "distances" in defect:
                retreat_threshold = p_vis.get("RETREAT_DISTANCE_THRESHOLD", 100.0)
                retreat_len = p_vis.get("EDGE_ENDPOINT_FIXED_LENGTH", 20)
                dist1, dist2 = defect["distances"]
                if dist1 > retreat_threshold:
                    vec1 = (p1_orig - center) / dist1
                    v1_final = center + vec1 * retreat_len
                if dist2 > retreat_threshold:
                    vec2 = (p2_orig - center) / dist2
                    v2_final = center + vec2 * retreat_len
                # 调试：写入调整后两条边的像素长度，供后续需要时参考（不进入最终上报）
                try:
                    defect_report.setdefault('_adjusted_q_lengths_px', [
                        float(np.linalg.norm(v1_final - center)),
                        float(np.linalg.norm(v2_final - center))
                    ])
                except Exception:
                    pass

            triangle_vertices = np.array([tuple(map(int, center)), tuple(map(int, v1_final)), tuple(map(int, v2_final))], dtype=np.int32)
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [triangle_vertices], color_bgr)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            blue_color = (255, 0, 0)
            draw_dashed_line(roi_color, tuple(map(int, center)), tuple(map(int, v1_final)), blue_color, thickness=2, dash_length=8)
            draw_dashed_line(roi_color, tuple(map(int, center)), tuple(map(int, v2_final)), blue_color, thickness=2, dash_length=8)

        elif defect["type"] == "X" and "center" in defect:
            cv2.circle(roi_color, defect["center"], 15, color_bgr, THICKNESS)
            
        elif defect["type"] == "L" and "rect" in defect:
            x_r, y_r, w_r, h_r = defect["rect"]
            overlay = roi_color.copy()
            cv2.rectangle(overlay, (x_r, y_r), (x_r + w_r, y_r + h_r), color_bgr, -1)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            cv2.rectangle(roi_color, (x_r, y_r), (x_r + w_r, y_r + h_r), color_bgr, THICKNESS)
            
        elif defect["type"] == "B" and "box_points" in defect:
            box_points = defect["box_points"]
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [box_points], color_bgr)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            cv2.drawContours(roi_color, [box_points], 0, color_bgr, THICKNESS)

    # 已取消 ROI 局部文字绘制

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

    # ===== 新增：将当前帧缺陷信息追加到全局日志，并统一在整幅图右上角绘制 =====
    if report.get("defects"):
        now_ts = datetime.now().strftime('%H:%M:%S')
        # 与 ROI 绘制相同的颜色表（BGR -> 转 RGB）
        defect_colors_bgr = {'Q': (0, 0, 255), 'X': (0, 255, 255), 'L': (255, 0, 255), 'B': (0, 165, 255)}
        for d in report["defects"]:
            t = d.get('type','?')
            #缺陷类型转化成中文
            t = {'Q': '缺角', 'X': '斜边', 'L': '裂纹', 'B': '崩边'}.get(t, t)
            loc = d.get('location', {})
            x_d = loc.get('x','?'); y_d = loc.get('y','?')
            if t == 'X':
                angle = loc.get('angle','?')
                text_line = f"{now_ts} X({x_d},{y_d}) angle={angle}°"
            else:
                l_mm = loc.get('length_mm','?'); w_mm = loc.get('width_mm','?')
                text_line = f"{now_ts} {t}({x_d},{y_d}) 尺寸={l_mm}x{w_mm}mm"
            # 将t转换为原英文标注
            t = {'缺角': 'Q', '斜边': 'X', '裂纹': 'L', '崩边': 'B'}.get(t, t)
            bgr = defect_colors_bgr.get(t, (255,255,0))
            rgb = (bgr[2], bgr[1], bgr[0])
            GLOBAL_DEFECT_LOG.append((text_line, rgb))

    if PIL_AVAILABLE and ANNOTATION_FONT and GLOBAL_DEFECT_LOG:
        pil_img_full = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        draw_full = ImageDraw.Draw(pil_img_full)
        padding = 6
        y_cursor = 10
        margin = 10
        # 逐行绘制（全部历史）
        for entry in GLOBAL_DEFECT_LOG:
            if isinstance(entry, tuple):
                log_line, color_rgb = entry
            else:  # 兼容旧简单字符串
                log_line, color_rgb = entry, (255,255,0)
            if hasattr(draw_full, 'textbbox'):
                bbox = draw_full.textbbox((0,0), log_line, font=ANNOTATION_FONT)
                text_w = bbox[2]-bbox[0]; text_h = bbox[3]-bbox[1]
            else:
                text_w, text_h = draw_full.textsize(log_line, font=ANNOTATION_FONT)
            x_pos = final_image.shape[1] - text_w - margin
            draw_full.text((x_pos, y_cursor), log_line, font=ANNOTATION_FONT, fill=color_rgb)
            y_cursor += text_h + padding
        final_image = cv2.cvtColor(np.array(pil_img_full), cv2.COLOR_RGB2BGR)

    return report, final_image
# --- END OF FILE image_processor_hough.py (Corrected for New Filtering Rules) ---