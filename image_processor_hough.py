# --- START OF FILE image_processor_hough.py (Corrected for KeyError: 'endpoints') ---
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

ANNOTATION_FONT = _get_font(font_size=36)


# ====================================================================================
# --- 几何学与分析辅助函数 ---
# ====================================================================================

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
        # 步骤1: 使用基于均值和标准差的动态阈值进行初步筛选
        threshold_low = mean[0][0] - p["LUMINOSITY_STD_DEV_MULTIPLIER"] * std_dev[0][0]
        potential_defects = (roi_gray < threshold_low).astype(np.uint8) * 255
        defect_mask = cv2.bitwise_and(potential_defects, scan_mask)

        if p.get("LUMINOSITY_EDGE_IGNORE_WIDTH", 0) > 0:
            ignore_mask = np.zeros_like(roi_gray)
            cv2.line(ignore_mask, tuple(map(int, p1)), tuple(map(int, p2)), 255, thickness=p["LUMINOSITY_EDGE_IGNORE_WIDTH"])
            defect_mask = cv2.subtract(defect_mask, ignore_mask)

        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            min_gradient_threshold = p.get("LUMINOSITY_MIN_GRADIENT", 15.0)
            # 从配置中获取严格的亮度阈值
            strict_brightness_threshold = p.get("LUMINOSITY_MAX_BRIGHTNESS_THRESHOLD", 75)

            blurred = cv2.medianBlur(roi_gray, 3)
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            for cnt in contours:
                if cv2.contourArea(cnt) > p["LUMINOSITY_MIN_AREA"]:
                    contour_mask = np.zeros_like(roi_gray)
                    cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
                    
                    # 步骤2: 对初步筛选出的缺陷，进行严格的亮度阈值二次筛选
                    mean_brightness_val = cv2.mean(roi_gray, mask=contour_mask)[0]
                    if mean_brightness_val < strict_brightness_threshold:
                        # 步骤3: 通过亮度筛选后，再进行梯度检查
                        mean_grad_val = cv2.mean(grad_mag, mask=contour_mask)[0]
                        if mean_grad_val > min_gradient_threshold:
                            initial_contours.append(cnt)
    
    return initial_contours


def find_gradient_endpoint(start_point, line_vec_normalized, roi_gray_blurred, max_search_dist, search_width=7, gradient_stop_threshold=20.0):
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
                proj_point, dist = (get_point_line_segment_projection)(point1, line2)
                if dist < endpoint_proximity_threshold:
                    len_line2 = np.linalg.norm(line2[:2] - line2[2:])
                    if len_line2 > 1e-6:
                        shield2 = len_line2 * p_crack["ENDPOINT_SHIELD_RATIO"]
                        if np.linalg.norm(proj_point - line2[:2]) > shield2 and np.linalg.norm(proj_point - line2[2:]) > shield2:
                            contour = line1.reshape(-1, 2).astype(np.int32)
                            min_area_rect = cv2.minAreaRect(contour)
                            box_points = np.intp(cv2.boxPoints(min_area_rect))
                            crack_defects.append({"type": "L", "box_points": box_points}); crack_indices.add(i); break

            if i in crack_indices: continue
            for point2 in [line2[:2], line2[2:]]:
                proj_point, dist = get_point_line_segment_projection(point2, line1)
                if dist < endpoint_proximity_threshold:
                    len_line1 = np.linalg.norm(line1[:2] - line1[2:])
                    if len_line1 > 1e-6:
                        shield1 = len_line1 * p_crack["ENDPOINT_SHIELD_RATIO"]
                        if np.linalg.norm(proj_point - line1[:2]) > shield1 and np.linalg.norm(proj_point - line1[2:]) > shield1:
                            contour = line2.reshape(-1, 2).astype(np.int32)
                            min_area_rect = cv2.minAreaRect(contour)
                            box_points = np.intp(cv2.boxPoints(min_area_rect))
                            crack_defects.append({"type": "L", "box_points": box_points}); crack_indices.add(j); break

    true_edges = [edge for i, edge in enumerate(edges) if i not in crack_indices]
    
    corner_defects = []; num_true_edges = len(true_edges)
    edges_for_drawing = [edge.copy() for edge in true_edges]
    endpoint_paired_status = {i: [False, False] for i in range(num_true_edges)}
    
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
                p1_far = line1[2:] if endpoint_idx_i == 0 else line1[:2]
                p2_far = line2[2:] if endpoint_idx_j == 0 else line2[:2]
                
                p_preproc = params["PREPROCESSING"]
                roi_gray_blurred = cv2.medianBlur(roi_gray, p_preproc["MEDIAN_BLUR_KSIZE"])
                
                vec1 = p1_far - intersection; norm1 = np.linalg.norm(vec1)
                if norm1 > 1e-6: vec1 /= norm1
                
                vec2 = p2_far - intersection; norm2 = np.linalg.norm(vec2)
                if norm2 > 1e-6: vec2 /= norm2

                grad_thresh = p_defect.get("Q_DEFECT_GRADIENT_THRESHOLD", 20.0)

                new_p1 = find_gradient_endpoint(intersection, vec1, roi_gray_blurred, max_extension_dist, gradient_stop_threshold=grad_thresh)
                new_p2 = find_gradient_endpoint(intersection, vec2, roi_gray_blurred, max_extension_dist, gradient_stop_threshold=grad_thresh)
                
                final_p1 = new_p1 if new_p1 is not None else p1_near
                final_p2 = new_p2 if new_p2 is not None else p2_near

                corner_defects.append({"type": "Q", "center": tuple(map(int, intersection)), "endpoints": (final_p1, final_p2), "distances": (dists_i[endpoint_idx_i], dists_j[endpoint_idx_j])})
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
            if cv2.contourArea(cnt) < p_defect["LUMINOSITY_MIN_AREA"]:
                continue

            min_area_rect = cv2.minAreaRect(cnt)
            (w, h) = min_area_rect[1]
            width = min(w, h)
            length = max(w, h)
            aspect_ratio = length / width if width > 1e-6 else 0.0
            
            if width < max_width and aspect_ratio > min_aspect_ratio:
                continue
            
            box_points = cv2.boxPoints(min_area_rect)
            box_points = np.intp(box_points)
            
            chipping_defects.append({
                "type": "B", 
                "box_points": box_points,
                "min_area_rect": min_area_rect 
            })
    
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

    surviving_chipping_defects = []
    if corner_defects and final_chipping_defects:
        q_mask = np.zeros(roi_dims, dtype=np.uint8)
        for q_defect in corner_defects:
            # --- FIX STARTS HERE ---
            # Check if the defect is a Q-type before accessing 'endpoints'
            if q_defect.get('type') == 'Q' and 'endpoints' in q_defect:
                q_contour = np.array([
                    q_defect['center'], 
                    q_defect['endpoints'][0], 
                    q_defect['endpoints'][1]
                ], dtype=np.int32)
                cv2.fillPoly(q_mask, [q_contour], 255)
            # --- FIX ENDS HERE ---

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
            
    return edges_for_drawing, corner_defects + surviving_chipping_defects + crack_defects

def process_roi_hough_based(roi_idx, roi_template, image_gray, params, pixels_per_mm):
    x, y, w, h = int(roi_template['x']), int(roi_template['y']), int(roi_template['width']), int(roi_template['height'])
    roi_gray = image_gray[y:y+h, x:x+w]
    
    p_hough = params["HOUGH_TRANSFORM"]
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    min_len_pixels = roi_gray.shape[1] * p_hough.get("MIN_LINE_LENGTH_RATIO", 0.05)
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=p_hough["MAX_LINE_GAP"])
    
    main_edges = merge_lines_and_get_main_edges(raw_lines, params)
    edges_for_drawing, all_defects = find_and_analyze_defects(main_edges, roi_gray, roi_gray.shape, params)
    
    final_defects_for_report = []
    for defect in all_defects:
        new_defect = {'type': defect['type']}
        location = {}

        if defect['type'] == 'L':
            box_pts = defect.get('box_points')
            if box_pts is not None and len(box_pts) > 0:
                x_r, y_r, w_r, h_r = cv2.boundingRect(np.array(box_pts))
                if w_r > 0 and h_r > 0:
                    defect_sub_roi = roi_gray[y_r:y_r+h_r, x_r:x_r+w_r]
                    mean_brightness = cv2.mean(defect_sub_roi)[0]
                    if mean_brightness > 50:
                        continue
                else: 
                    continue
            else:
                continue

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
                    location['pixel_area'] = round(pixel_area, 2)

                    if pixel_area < 15: continue
                    
                    dist_leg1_px = np.linalg.norm(center_np - v1_final)
                    dist_leg2_px = np.linalg.norm(center_np - v2_final)
                    length_px = max(dist_leg1_px, dist_leg2_px)
                    width_px  = min(dist_leg1_px, dist_leg2_px)
                else:
                    length_px, width_px = 0.0, 0.0
            
            elif defect['type'] in ['L', 'B']:
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
        
        min_size_mm = params["DEFECT_DETECTION"].get("MIN_DEFECT_SIZE_MM", 3.0)
        if new_defect['type'] == 'Q':
            length_mm = location.get('length_mm', 0)
            width_mm = location.get('width_mm', 0)
            area_mm2 = length_mm * width_mm
            aspect_ratio = length_mm / width_mm if width_mm > 1e-6 else float('inf')
            if area_mm2 < 10.0:
                continue
            if location.get('length_mm', 0) < min_size_mm:
                continue
            if location.get('width_mm', 0) < 0.1:
                continue
            if aspect_ratio > 3.0:
                continue
            
        elif new_defect['type'] in ['L', 'B']:
            if location.get('length_mm', 0) < min_size_mm:
                continue

        if new_defect['type'] == 'L':

            if location.get('width_mm', 0) < 0.1:
                continue
            if location.get('width_mm', 0) > 15.0:
                continue
        
        if new_defect['type'] == 'B':
            length_mm = location.get('length_mm', 0)
            width_mm = location.get('width_mm', 0)
            area_mm2 = length_mm * width_mm
            aspect_ratio = length_mm / width_mm if width_mm > 1e-6 else float('inf')
            if aspect_ratio > 10.0:
                continue
            if area_mm2 < 16.0:
                continue
            if width_mm > 4.0 and aspect_ratio > 6.0:
                continue
            
            
            p_reclass = params["DEFECT_DETECTION"].get("RECLASSIFY_B_AS_L_PARAMS", {})
            min_area_rect = defect.get("min_area_rect")

            if p_reclass.get("ENABLE", False) and min_area_rect and aspect_ratio > p_reclass.get("MIN_ASPECT_RATIO", 4.0):
                target_edge = None
                
                (w_rect, h_rect) = min_area_rect[1]
                angle_raw = min_area_rect[2]

                defect_angle = angle_raw
                if w_rect < h_rect: 
                    defect_angle += 90
                
                while defect_angle < 0: defect_angle += 180
                defect_angle %= 180

                for edge in main_edges:
                    defect_center = np.array(min_area_rect[0])
                    _, dist = get_point_line_segment_projection(defect_center, edge)
                    if dist < p_reclass.get("MAX_DISTANCE_PX", 40):
                        edge_vec = np.array(edge[2:]) - np.array(edge[:2])
                        edge_angle = np.degrees(np.arctan2(edge_vec[1], edge_vec[0]))
                        
                        while edge_angle < 0: edge_angle += 180
                        edge_angle %= 180
                        
                        angle_diff = abs(defect_angle - edge_angle)
                        angle_diff = min(angle_diff, 180 - angle_diff)
                        
                        if abs(angle_diff - 90.0) < p_reclass.get("ANGLE_TOLERANCE", 15.0):
                            new_defect['type'] = "L"
                            target_edge = edge
                            break
                        
                if new_defect['type'] == "L" and target_edge is not None:
                    center = np.array(min_area_rect[0])
                    defect_length = max(w_rect, h_rect)
                    defect_width = min(w_rect, h_rect)
                    
                    angle_rad = np.deg2rad(defect_angle)
                    vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                    ep1 = center + vec * defect_length / 2
                    ep2 = center - vec * defect_length / 2
                    
                    _, dist1 = get_point_line_segment_projection(ep1, target_edge)
                    _, dist2 = get_point_line_segment_projection(ep2, target_edge)
                    far_endpoint = ep1 if dist1 > dist2 else ep2

                    axis_line = np.hstack([ep1, ep2])
                    intersection_point = find_line_intersection(axis_line, target_edge)

                    if intersection_point is not None:
                        new_length = np.linalg.norm(far_endpoint - intersection_point)
                        new_center = (far_endpoint + intersection_point) / 2
                        
                        new_size = (new_length, defect_width) if w_rect > h_rect else (defect_width, new_length)
                        
                        new_min_area_rect = (tuple(new_center), new_size, angle_raw)
                        new_box_points = np.intp(cv2.boxPoints(new_min_area_rect))
                        
                        defect['box_points'] = new_box_points
                        length_px, width_px = new_length, defect_width
                        new_defect['location']['length_mm'] = float(round(length_px / pixels_per_mm, 2))
                        new_defect['location']['width_mm'] = float(round(width_px / pixels_per_mm, 2))


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
    DEFECT_COLORS_BGR = {'Q': (0, 0, 255), 'X': (255, 0, 0), 'L': (255, 0, 255), 'B': (0, 165, 255)}
    p_vis = params["VISUALIZATION"]
    THICKNESS = 1
    
    alpha = p_vis["DEFECT_OVERLAY_ALPHA"]; beta = 1 - alpha
    
    for edge in edges_for_drawing:
        pt1 = tuple(map(int, edge[:2]))
        pt2 = tuple(map(int, edge[2:]))
        #绘制直线
        cv2.line(roi_color, pt1, pt2, (0, 255, 0), 1)

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
            draw_dashed_line(roi_color, tuple(map(int, center)), tuple(map(int, v1_final)), blue_color, thickness=1, dash_length=8)
            draw_dashed_line(roi_color, tuple(map(int, center)), tuple(map(int, v2_final)), blue_color, thickness=1, dash_length=8)

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