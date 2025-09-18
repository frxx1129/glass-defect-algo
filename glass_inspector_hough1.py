# glass_inspector_hough.py (Final Intelligent Version with Advanced Rules)

import cv2
import numpy as np
import argparse
import json
import os
from tqdm import tqdm
from itertools import combinations
import math

# ====================================================================================
# --- 算法核心参数调节区 ---
# ====================================================================================
TUNABLE_PARAMETERS = {
    "PREPROCESSING": {
        "MEDIAN_BLUR_KSIZE": 3, "CLAHE_CLIP_LIMIT": 2.0, "CLAHE_GRID_SIZE": (8, 8),
        "CANNY_THRESHOLD_LOW": 30, "CANNY_THRESHOLD_HIGH": 90,
    },
    "HOUGH_TRANSFORM": {
        "THRESHOLD": 40, "MIN_LINE_LENGTH_RATIO": 0.05, "MAX_LINE_GAP": 25,
    },
    "LINE_MERGING": {
        "ANGLE_TOLERANCE": 5.0, "MAX_LATERAL_DISTANCE": 30, "TOP_N_EDGES": 8,
    },
    "CRACK_CLASSIFICATION": {
        "ENDPOINT_SHIELD_RATIO": 0.1,
    },
    "DEFECT_DETECTION": {
        "ANGLE_NORMAL_TOLERANCE": 2.0, "ANGLE_BEVEL_TOLERANCE": 2.5,
        "CORNER_MAX_PHYSICAL_GAP": 3,
        "CORNER_MAX_EXTENSION_DIST_NORMAL": 100,
        "CORNER_MAX_EXTENSION_DIST_PERPENDICULAR": 500,
        "PERPENDICULAR_ANGLE_TOLERANCE": 10.0,
        "CORNER_SECTOR_RADIUS": 40,
        
        "LUMINOSITY_SCAN_WIDTH": 30,
        "LUMINOSITY_STD_DEV_MULTIPLIER": 2.4,
        "LUMINOSITY_MIN_AREA": 50,
        "LUMINOSITY_EDGE_IGNORE_WIDTH": 2,
        "MERGE_DEFECTS_KERNEL_SIZE": (5, 5),
        
        "FALSE_DEFECT_MAX_WIDTH": 8.0,
        "FALSE_DEFECT_MIN_ASPECT_RATIO": 5.0,
        
        "CHIPPING_ENDPOINT_SHIELD_RADIUS": 30,
    },
    "VISUALIZATION": {
        "EDGE_ENDPOINT_FIXED_LENGTH": 20,
        "DEFECT_OVERLAY_ALPHA": 0.25,
        "RETREAT_DISTANCE_THRESHOLD": 100.0,
    }
}
# ====================================================================================
# --- 几何学与分析辅助函数 ---
# ====================================================================================

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
    if t < 0.0: proj_point = p1
    elif t > 1.0: proj_point = p2
    else: proj_point = p1 + t * line_vec
    return proj_point, np.linalg.norm(p - proj_point)

def get_line_angle(line):
    p1 = np.array(line[:2]); p2 = np.array(line[2:])
    angle = np.rad2deg(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    return angle % 180

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
def load_rois_from_file(roi_file_path):
    if not os.path.exists(roi_file_path): return None
    try:
        with open(roi_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        rois_list = []
        if isinstance(data, dict):
            for group_key in data:
                group_data = data[group_key]
                if isinstance(group_data, dict) and 'averaged_rois' in group_data:
                    rois_list.extend(group_data['averaged_rois'])
        if not rois_list: return None
        return rois_list
    except Exception: return None

def preprocess_for_hough_enhanced(roi_gray, params):
    p = params["PREPROCESSING"]
    blurred = cv2.medianBlur(roi_gray, p["MEDIAN_BLUR_KSIZE"])
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP_LIMIT"], tileGridSize=p["CLAHE_GRID_SIZE"])
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
    
    # --- 1. 裂纹检测 ---
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
    
    # --- 2. 角点检测 (Q, X) ---
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
                if deviation > p_defect["ANGLE_BEVEL_TOLERANCE"]:
                    corner_defects.append({"type": "Q", "center": tuple(map(int, intersection)), "endpoints": (p1_near, p2_near)})
                elif deviation > p_defect["ANGLE_NORMAL_TOLERANCE"]:
                    corner_defects.append({"type": "X", "center": tuple(map(int, intersection)), "angle": angle})

    # --- 3. 崩边检测 (B) ---
    all_chipping_contours = []; chipping_defects = []
    for edge in true_edges:
        all_chipping_contours.extend(scan_edge_for_luminosity_defects(roi_gray, edge, params))
    if all_chipping_contours:
        defect_canvas = np.zeros(roi_dims, dtype=np.uint8)
        cv2.drawContours(defect_canvas, all_chipping_contours, -1, 255, -1)
        kernel = np.ones(p_defect.get("MERGE_DEFECTS_KERNEL_SIZE", (5, 5)), np.uint8)
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
            
            if width < 1e-6: continue
            
            aspect_ratio = length / width
            
            if width < max_width and aspect_ratio > min_aspect_ratio:
                continue
            
            box_points = cv2.boxPoints(min_area_rect)
            box_points = np.intp(box_points)
            chipping_defects.append({"type": "B", "box_points": box_points})
    
    # --- 4. 过滤掉靠近端点的崩边缺陷 ---
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

def process_roi_hough_based(roi_gray, params):
    p_hough = params["HOUGH_TRANSFORM"]
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    min_len_pixels = roi_gray.shape[1] * p_hough["MIN_LINE_LENGTH_RATIO"]
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=p_hough["MAX_LINE_GAP"])
    main_edges = merge_lines_and_get_main_edges(raw_lines, params)
    edges_for_drawing, all_defects = find_and_analyze_defects(main_edges, roi_gray, roi_gray.shape, params)
    
    roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    DEFECT_COLORS = {'Q': (0, 0, 255), 'X': (0, 255, 255), 'L': (255, 0, 255), 'B': (0, 165, 255)}
    p_vis = params["VISUALIZATION"]
    
    fixed_endpoint_len = p_vis.get("EDGE_ENDPOINT_FIXED_LENGTH", 20)
    for edge in edges_for_drawing:
        p1 = np.array(edge[:2]); p2 = np.array(edge[2:])
        line_length = np.linalg.norm(p2 - p1)
        if line_length < 1e-6: continue
        
        draw_endpoint_len = min(fixed_endpoint_len, line_length / 2.0)
        
        vec = (p2 - p1) / line_length
        p1_inner = p1 + vec * draw_endpoint_len
        p2_inner = p2 - vec * draw_endpoint_len
        
        cv2.line(roi_color, tuple(map(int, p1)), tuple(map(int, p1_inner)), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(roi_color, tuple(map(int, p1_inner)), tuple(map(int, p2_inner)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(roi_color, tuple(map(int, p2_inner)), tuple(map(int, p2)), (255, 0, 0), 2, cv2.LINE_AA)

    alpha = p_vis["DEFECT_OVERLAY_ALPHA"]; beta = 1 - alpha
    for defect in all_defects:
        color = DEFECT_COLORS.get(defect["type"], (255, 255, 255))
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

            triangle_vertices = np.array([tuple(map(int, center)), tuple(map(int, v1_final)), tuple(map(int, v2_final))], dtype=np.int32)
            overlay = roi_color.copy(); cv2.fillPoly(overlay, [triangle_vertices], color)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)

        elif defect["type"] == "X" and "center" in defect:
            cv2.circle(roi_color, defect["center"], 15, color, 2)
            
        elif defect["type"] == "L" and "rect" in defect:
            x, y, w, h = defect["rect"]
            overlay = roi_color.copy(); cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            cv2.rectangle(roi_color, (x, y), (x + w, y + h), color, 2)
            
        elif defect["type"] == "B" and "box_points" in defect:
            box_points = defect["box_points"]
            overlay = roi_color.copy()
            cv2.fillPoly(overlay, [box_points], color)
            cv2.addWeighted(overlay, alpha, roi_color, beta, 0, roi_color)
            cv2.drawContours(roi_color, [box_points], 0, color, 2)
            
    return roi_color
    
def process_video_with_hough(video_path, roi_file_path, output_path, params):
    rois = load_rois_from_file(roi_file_path)
    if not rois: print(f"错误: 无法从 {roi_file_path} 加载 ROI。"); return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"错误: 无法打开视频文件 {video_path}。"); return
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    print(f"开始使用最终精 polished 算法处理视频: {video_path}")
    with tqdm(total=total_frames, desc="处理中") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for roi_def in rois:
                x, y, w, h = [int(roi_def.get(k, 0)) for k in ('x', 'y', 'width', 'height')]
                if w > 0 and h > 0:
                    x_c, y_c = max(0, x), max(0, y); w_c, h_c = min(w, fw - x), min(h, fh - y)
                    if w_c > 0 and h_c > 0:
                        roi_gray_c = gray_frame[y_c:y_c+h_c, x_c:x_c+w_c]
                        processed_roi = process_roi_hough_based(roi_gray_c, params)
                        frame[y_c:y_c+h_c, x_c:x_c+w_c] = processed_roi
            out.write(frame)
            pbar.update(1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成！视频已保存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用最终的、精细化规则的几何分析算法在视频ROI内检测所有类型的玻璃缺陷。")
    parser.add_argument('--video-in', type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument('--roi-file', type=str, default='cam5_roi_averaged_by_group.json', help="包含ROI定义的JSON文件路径。")
    parser.add_argument('--video-out', type=str, default='output_hough_polished.mp4', help="处理后输出的视频文件路径。")
    args = parser.parse_args()
    process_video_with_hough(args.video_in, args.roi_file, args.video_out, TUNABLE_PARAMETERS)