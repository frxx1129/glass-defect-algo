# glass_inspector_hough.py (Final Refined Version)

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
    # --- 裂纹分类参数已禁用 ---
    "DEFECT_DETECTION": {
        "ANGLE_NORMAL_TOLERANCE": 2.0, "ANGLE_BEVEL_TOLERANCE": 2.5,
        "CORNER_MAX_PHYSICAL_GAP": 20, "CORNER_MAX_EXTENSION_DIST": 100,
        
        # (回归) “窗口分析法”参数
        "CORNER_ANALYSIS_WINDOW_SIZE": 50, # 方形分析窗口大小
        "CORNER_MISSING_PIXEL_RATIO": 0.20, # 暗像素比例高于此值，则判为缺角(Q)
        "CORNER_SECTOR_RADIUS": 40, # (仅用于高亮) 扇形高亮区域的半径

        # --- (全新) 崩边缺陷扫描参数 (Edge Chipping) ---
        "CHIPPING_SCAN_WIDTH": 10,        # 内外扫描带的宽度（像素）
        "CHIPPING_MIN_LENGTH": 20,        # 崩边缺陷的最小长度（像素）
        "CHIPPING_GRADIENT_SENSITIVITY": 0.6, # 梯度灵敏度(0-1)。值越低越灵敏，更容易报缺陷。

        # --- 亮度缺陷 (L, B) ---
        "SCAN_WIDTH": 30, "IQR_MULTIPLIER": 2.5, "MIN_DEFECT_AREA": 40,
        "ORIENTATION_PARALLEL_THRESHOLD": 20.0,
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

def analyze_corner_for_chipping(roi_gray, corner_point, params):
    """(规则1: 回归) 窗口分析法，使用固定的暗像素比例判断缺角。"""
    p = params["DEFECT_DETECTION"]; w_size = p["CORNER_ANALYSIS_WINDOW_SIZE"]
    cx, cy = int(corner_point[0]), int(corner_point[1]); h, w = roi_gray.shape
    x_start, y_start = max(0, cx - w_size // 2), max(0, cy - w_size // 2)
    x_end, y_end = min(w, cx + w_size // 2), min(h, cy + w_size // 2)
    window = roi_gray[y_start:y_end, x_start:x_end]
    if window.size < (w_size * w_size) / 4: return False
    # 使用一个简单的阈值（例如128的一半）或Otsu来找到暗像素
    _, thresh = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_pixel_count = np.sum(thresh == 0)
    dark_ratio = dark_pixel_count / window.size
    return dark_ratio > p["CORNER_MISSING_PIXEL_RATIO"]

def scan_edge_for_chipping_defects(roi_gray, edge, params):
    """
    (全新逻辑) 使用内外双扫描带和梯度分析，检测边缘的崩边缺陷。
    """
    p = params["DEFECT_DETECTION"]
    defects_found = []
    
    # 1. 定义边缘的法线向量，用于确定内外侧
    p1 = edge[:2]; p2 = edge[2:]
    vec = p2 - p1
    if np.linalg.norm(vec) < 1e-6: return []
    normal = np.array([-vec[1], vec[0]]) / np.linalg.norm(vec)

    # 2. 生成内外侧扫描带的掩码
    scan_width = p["CHIPPING_SCAN_WIDTH"]
    inner_mask = np.zeros_like(roi_gray)
    outer_mask = np.zeros_like(roi_gray)
    
    # 将线段平移到内外两侧
    p1_inner, p2_inner = p1 - normal * scan_width, p2 - normal * scan_width
    p1_outer, p2_outer = p1 + normal * scan_width, p2 + normal * scan_width
    
    cv2.line(inner_mask, tuple(map(int, p1_inner)), tuple(map(int, p2_inner)), 255, scan_width)
    cv2.line(outer_mask, tuple(map(int, p1_outer)), tuple(map(int, p2_outer)), 255, scan_width)

    # 3. 确定哪个是玻璃侧（更亮），哪个是背景侧（更暗）
    mean_inner = cv2.mean(roi_gray, mask=inner_mask)[0]
    mean_outer = cv2.mean(roi_gray, mask=outer_mask)[0]
    
    if mean_inner > mean_outer:
        glass_mask, bg_mask = inner_mask, outer_mask
    else:
        glass_mask, bg_mask = outer_mask, inner_mask

    # 4. 沿边缘分段计算局部梯度
    num_steps = int(np.linalg.norm(vec) / 5) # 每5个像素检查一次
    if num_steps < 4: return []
    
    gradients = []
    points_along_edge = np.linspace(p1, p2, num_steps)
    
    for pt in points_along_edge:
        # 在每个点周围创建一个小的圆形分析区域
        local_mask = np.zeros_like(roi_gray)
        cv2.circle(local_mask, tuple(map(int, pt)), 10, 255, -1)
        
        # 计算该小区域内，玻璃侧和背景侧的平均亮度
        local_glass_pixels = roi_gray[np.logical_and(glass_mask, local_mask) > 0]
        local_bg_pixels = roi_gray[np.logical_and(bg_mask, local_mask) > 0]
        
        if local_glass_pixels.size > 5 and local_bg_pixels.size > 5:
            grad = np.mean(local_glass_pixels) - np.mean(local_bg_pixels)
            gradients.append(grad)
        else:
            gradients.append(-1) # 无效梯度

    # 5. 寻找梯度异常低的区域
    if not any(g > 0 for g in gradients): return []
    avg_gradient = np.mean([g for g in gradients if g > 0])
    threshold = avg_gradient * p["CHIPPING_GRADIENT_SENSITIVITY"]
    
    in_defect = False
    defect_start_idx = -1
    for i, grad in enumerate(gradients):
        if grad >= 0 and grad < threshold and not in_defect:
            in_defect = True
            defect_start_idx = i
        elif (grad < 0 or grad >= threshold) and in_defect:
            in_defect = False
            # 检查缺陷长度
            if (i - defect_start_idx) * 5 > p["CHIPPING_MIN_LENGTH"]:
                defect_points = points_along_edge[defect_start_idx:i]
                defects_found.append({
                    "type": "B", # (规则2) 统一标记为崩边'B'
                    "points": defect_points.astype(np.int32)
                })

    return defects_found

def scan_edge_for_luminosity_defects(roi_gray, edge, params):
    # (此函数无变化)
    p = params["DEFECT_DETECTION"]; defects_found = []
    x1, y1, x2, y2 = map(int, edge); scan_mask = np.zeros_like(roi_gray)
    cv2.line(scan_mask, (x1, y1), (x2, y2), 255, thickness=p["SCAN_WIDTH"])
    pixels_in_band = roi_gray[scan_mask == 255]
    if len(pixels_in_band) < 100: return []
    median = np.median(pixels_in_band); q75, q25 = np.percentile(pixels_in_band, [75, 25])
    iqr = q75 - q25;
    if iqr < 5: return []
    threshold_low = median - p["IQR_MULTIPLIER"] * iqr
    potential_defects = (roi_gray < max(0, threshold_low)).astype(np.uint8) * 255
    defect_mask = cv2.bitwise_and(potential_defects, scan_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask)
    if num_labels > 1:
        edge_angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1)) % 180
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > p["MIN_DEFECT_AREA"]:
                component_mask = (labels == i).astype(np.uint8)
                points = np.argwhere(component_mask > 0)[:, [1, 0]]
                rect = cv2.minAreaRect(points.astype(np.float32))
                defect_angle = rect[2]; (w_px, h_px) = rect[1]
                if w_px < h_px: defect_angle += 90
                angle_diff = abs((defect_angle % 180) - edge_angle); angle_diff = min(angle_diff, 180 - angle_diff)
                defect_type = "B"
                if angle_diff > (90 - p["ORIENTATION_PARALLEL_THRESHOLD"]): defect_type = "L"
                elif angle_diff < p["ORIENTATION_PARALLEL_THRESHOLD"]: defect_type = "B"
                defects_found.append({"type": defect_type, "center": tuple(map(int, centroids[i])), "mask": component_mask})
    return defects_found

# ====================================================================================
# --- 核心处理流程 ---
# ====================================================================================

def load_rois_from_file(roi_file_path):
    # ... (无变化)
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
    # ... (无变化)
    p = params["PREPROCESSING"]
    blurred = cv2.medianBlur(roi_gray, p["MEDIAN_BLUR_KSIZE"])
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP_LIMIT"], tileGridSize=p["CLAHE_GRID_SIZE"])
    enhanced_contrast = clahe.apply(blurred)
    edges = cv2.Canny(enhanced_contrast, p["CANNY_THRESHOLD_LOW"], p["CANNY_THRESHOLD_HIGH"])
    return edges

def merge_lines_and_get_main_edges(lines, params):
    # ... (无变化)
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
        proximity_groups = []
        if segments: proximity_groups.append([segments.pop(0)])
        for segment in segments:
            mid_point = np.array([(segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2])
            placed = False
            for group in proximity_groups:
                ref_line = group[0]; p1, p2 = ref_line[0:2], ref_line[2:4]
                vec_line, vec_point_to_line = p2 - p1, p1 - mid_point
                cross_product_2d = vec_line[0] * vec_point_to_line[1] - vec_line[1] * vec_point_to_line[0]
                line_length = np.linalg.norm(vec_line)
                if line_length > 1e-6 and np.abs(cross_product_2d) / line_length < p["MAX_LATERAL_DISTANCE"]:
                    group.append(segment); placed = True; break
            if not placed: proximity_groups.append([segment])
        final_line_groups.extend(proximity_groups)
    merged_lines_with_scores = []
    for group in final_line_groups:
        support_score = sum(np.linalg.norm(l[2:4] - l[0:2]) for l in group)
        points = np.array([pt for line in group for pt in ((line[0], line[1]), (line[2], line[3]))], dtype=np.float32)
        if len(points) < 2: continue
        line_params = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()
        projected = (points[:, 0] - x0) * vx + (points[:, 1] - y0) * vy
        pt1, pt2 = points[np.argmin(projected)], points[np.argmax(projected)]
        final_merged_line = np.array([pt1[0], pt1[1], pt2[0], pt2[1]])
        merged_lines_with_scores.append({'line': final_merged_line, 'score': support_score})
    merged_lines_with_scores.sort(key=lambda item: item['score'], reverse=True)
    final_best_lines = [item['line'] for item in merged_lines_with_scores]
    return final_best_lines[:p["TOP_N_EDGES"]]

def find_and_analyze_corners(edges, roi_gray, params):
    """(已修改) 寻找角点并分析缺陷，使用“一个端点只配对一次”的规则。"""
    p_defect = params["DEFECT_DETECTION"]; num_edges = len(edges)
    endpoint_paired_status = {i: [False, False] for i in range(num_edges)}
    all_defects = []
    
    potential_pairs = sorted([(i, j) for i, j in combinations(range(num_edges), 2)],
        key=lambda p: min(np.linalg.norm(edges[p[0]][:2] - edges[p[1]][:2]), np.linalg.norm(edges[p[0]][:2] - edges[p[1]][2:]),
                          np.linalg.norm(edges[p[0]][2:] - edges[p[1]][:2]), np.linalg.norm(edges[p[0]][2:] - edges[p[1]][2:])))

    for i, j in potential_pairs:
        if all(endpoint_paired_status[i]) and all(endpoint_paired_status[j]): continue
        line1, line2 = edges[i], edges[j]
        intersection = find_line_intersection(line1, line2)
        if intersection is None: continue
        
        dists_i = [np.linalg.norm(intersection - line1[:2]), np.linalg.norm(intersection - line1[2:])]
        dists_j = [np.linalg.norm(intersection - line2[:2]), np.linalg.norm(intersection - line2[2:])]
        endpoint_idx_i, endpoint_idx_j = np.argmin(dists_i), np.argmin(dists_j)

        if endpoint_paired_status[i][endpoint_idx_i] or endpoint_paired_status[j][endpoint_idx_j]: continue
        
        is_physical = dists_i[endpoint_idx_i] < p_defect["CORNER_MAX_PHYSICAL_GAP"] and \
                      dists_j[endpoint_idx_j] < p_defect["CORNER_MAX_PHYSICAL_GAP"]
        is_valid_virtual = dists_i[endpoint_idx_i] < p_defect["CORNER_MAX_EXTENSION_DIST"] and \
                           dists_j[endpoint_idx_j] < p_defect["CORNER_MAX_EXTENSION_DIST"]

        if is_physical or is_valid_virtual:
            endpoint_paired_status[i][endpoint_idx_i] = True
            endpoint_paired_status[j][endpoint_idx_j] = True
            
            p1 = line1[2:] if endpoint_idx_i == 0 else line1[:2]
            p2 = line2[2:] if endpoint_idx_j == 0 else line2[:2]
            
            # (规则1) 回归窗口分析法
            if analyze_corner_for_chipping(roi_gray, intersection, params):
                # 为扇形高亮准备向量
                vec1 = p1 - intersection; vec2 = p2 - intersection
                all_defects.append({"type": "Q", "center": tuple(map(int, intersection)), "vectors": (vec1, vec2)})
            else:
                angle = calculate_vertex_angle(p1, intersection, p2)
                deviation = abs(angle - 90.0)
                if deviation > p_defect["ANGLE_BEVEL_TOLERANCE"]:
                    all_defects.append({"type": "Q", "center": tuple(map(int, intersection)), "angle": angle})
                elif deviation > p_defect["ANGLE_NORMAL_TOLERANCE"]:
                    all_defects.append({"type": "X", "center": tuple(map(int, intersection)), "angle": angle})
    return all_defects

def process_roi_hough_based(roi_gray, params):
    """(最终智能版) 包含基于梯度分析的崩边检测。"""
    p_hough = params["HOUGH_TRANSFORM"]
    
    # 1. 边缘检测与合并
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    min_len_pixels = roi_gray.shape[1] * p_hough["MIN_LINE_LENGTH_RATIO"]
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=p_hough["MAX_LINE_GAP"])
    main_edges = merge_lines_and_get_main_edges(raw_lines, params)
    
    # 2. 缺陷检测
    corner_defects = find_and_analyze_corners(main_edges, roi_gray, params)
    chipping_defects = []
    for edge in main_edges:
        # 调用新的崩边检测函数
        chipping_defects.extend(scan_edge_for_chipping_defects(roi_gray, edge, params))
        
    all_defects = corner_defects + chipping_defects
        
    # --- 3. 可视化 ---
    roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    # (规则2) B现在代表崩边，使用橙色
    DEFECT_COLORS = {'Q': (0, 0, 255), 'X': (0, 255, 255), 'B': (0, 165, 255)}
    
    for edge in main_edges:
        cv2.line(roi_color, tuple(map(int, edge[:2])), tuple(map(int, edge[2:])), (0, 255, 0), 2, cv2.LINE_AA)
        
    for defect in all_defects:
        color = DEFECT_COLORS.get(defect["type"], (255, 255, 255))
        
        if defect["type"] == "Q" and "vectors" in defect:
            # ... (扇形绘制部分无变化)
            center = defect.get("center"); intersection_point = np.array(center)
            vec1_outward, vec2_outward = defect["vectors"]
            radius = params["DEFECT_DETECTION"]["CORNER_SECTOR_RADIUS"]
            norm_v1, norm_v2 = np.linalg.norm(vec1_outward), np.linalg.norm(vec2_outward)
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                unit_vec1, unit_vec2 = vec1_outward / norm_v1, vec2_outward / norm_v2
                point1_on_line = intersection_point + unit_vec1 * radius
                point2_on_line = intersection_point + unit_vec2 * radius
                triangle_vertices = np.array([intersection_point, point1_on_line, point2_on_line], dtype=np.int32)
                highlight_mask = np.zeros_like(roi_gray); cv2.fillPoly(highlight_mask, [triangle_vertices], 255)
                overlay = roi_color.copy(); overlay[highlight_mask == 255] = color
                cv2.addWeighted(overlay, 0.5, roi_color, 0.5, 0, roi_color)
        elif defect["type"] == "X" and "center" in defect:
            cv2.circle(roi_color, defect["center"], 15, color, 2)
        
        # (全新) 崩边'B'类缺陷的可视化
        elif defect["type"] == "B" and "points" in defect:
            # 将崩边区域加粗绘制以高亮
            points = defect["points"].reshape(-1, 1, 2)
            cv2.polylines(roi_color, [points], isClosed=False, color=color, thickness=4)
            
    return roi_color

# (process_video_with_hough 和 main 部分保持不变)
def process_video_with_hough(video_path, roi_file_path, output_path, params):
    rois = load_rois_from_file(roi_file_path)
    if not rois: return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    print(f"开始使用最终精简算法处理视频: {video_path}")
    with tqdm(total=total_frames, desc="处理中") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for roi_def in rois:
                x, y, w, h = [int(roi_def.get(k, 0)) for k in ('x', 'y', 'width', 'height')]
                if w > 0 and h > 0:
                    x, y = max(0, x), max(0, y); w, h = min(w, fw - x), min(h, fh - y)
                    if w > 0 and h > 0:
                        roi_gray = gray_frame[y:y+h, x:x+w]
                        processed_roi = process_roi_hough_based(roi_gray, params)
                        frame[y:y+h, x:x+w] = processed_roi
            out.write(frame)
            pbar.update(1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成！视频已保存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用精简后的几何分析算法在视频ROI内检测玻璃缺陷。")
    parser.add_argument('--video-in', type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument('--roi-file', type=str, default='cam5_roi_averaged_by_group.json', help="包含ROI定义的JSON文件路径。")
    parser.add_argument('--video-out', type=str, default='output_hough_refined.mp4', help="处理后输出的视频文件路径。")
    args = parser.parse_args()
    process_video_with_hough(args.video_in, args.roi_file, args.video_out, TUNABLE_PARAMETERS)