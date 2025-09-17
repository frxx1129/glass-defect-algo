import cv2
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

# ====================================================================================
# --- 算法核心参数调节区 ---
# (已简化，暂时移除了分类和缺陷扫描的参数)
# ====================================================================================
TUNABLE_PARAMETERS = {
    "PREPROCESSING": {
        "MEDIAN_BLUR_KSIZE": 3,
        "CLAHE_CLIP_LIMIT": 2.0,
        "CLAHE_GRID_SIZE": (8, 8),
        "CANNY_THRESHOLD_LOW": 30,
        "CANNY_THRESHOLD_HIGH": 90,
    },
    "HOUGH_TRANSFORM": {
        "THRESHOLD": 40,
        "MIN_LINE_LENGTH_RATIO": 0.05,
        "MAX_LINE_GAP": 25,
    },
    "LINE_MERGING": {
        "ANGLE_TOLERANCE": 5.0,
        
        # (新增) 最大横向距离（像素）。
        # 只有当两条平行线的垂直距离小于此阈值时，它们才会被考虑合并。
        # 这是防止合并ROI两侧边缘的核心参数。
        "MAX_LATERAL_DISTANCE": 25, # A good starting value is 25 pixels

        "TOP_N_EDGES": 8, 
    },
}
# ====================================================================================
# --- 算法主代码 ---
# ====================================================================================

def load_rois_from_file(roi_file_path):
    """从指定的JSON文件中加载ROI区域。"""
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
    """预处理函数，为 HoughLinesP 生成高质量的二值边缘图。"""
    p = params["PREPROCESSING"]
    blurred = cv2.medianBlur(roi_gray, p["MEDIAN_BLUR_KSIZE"])
    clahe = cv2.createCLAHE(clipLimit=p["CLAHE_CLIP_LIMIT"], tileGridSize=p["CLAHE_GRID_SIZE"])
    enhanced_contrast = clahe.apply(blurred)
    edges = cv2.Canny(enhanced_contrast, p["CANNY_THRESHOLD_LOW"], p["CANNY_THRESHOLD_HIGH"])
    return edges

def merge_lines_and_get_main_edges(lines, params):
    """
    (最终修正版) 基于“角度优先，邻近度次之”的原则合并线段。
    使用手动2D叉积计算，以消除NumPy警告并确保未来兼容性。
    """
    if lines is None or len(lines) < 1: return []
    
    p = params["LINE_MERGING"]
    lines_np = np.array(lines).reshape(-int(len(lines)), 4)

    # --- 1. 按角度聚类 (粗分类) ---
    angles = np.rad2deg(np.arctan2(lines_np[:, 3] - lines_np[:, 1], lines_np[:, 2] - lines_np[:, 0]))
    angles[angles < 0] += 180
    
    angle_clusters = {}
    for i, angle in enumerate(angles):
        placed = False
        for cluster_angle in angle_clusters:
            diff = abs(angle - cluster_angle)
            if min(diff, 180 - diff) < p["ANGLE_TOLERANCE"]:
                angle_clusters[cluster_angle].append(lines_np[i])
                placed = True
                break
        if not placed: angle_clusters[angle] = [lines_np[i]]

    # --- 2. 在每个角度聚类内部，再按邻近度进行子聚类 (精分类) ---
    final_line_groups = []
    for angle, segments in angle_clusters.items():
        if not segments: continue
        
        segments.sort(key=lambda s: np.linalg.norm(s[2:4] - s[0:2]), reverse=True)
        
        proximity_groups = []
        if segments:
            proximity_groups.append([segments.pop(0)])
        
        for segment in segments:
            mid_point = np.array([(segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2])
            placed = False
            for group in proximity_groups:
                ref_line = group[0]
                
                # --- (最终修正) ---
                # 用直接的数学公式替换 np.cross，彻底解决警告问题
                p1 = ref_line[0:2]
                p2 = ref_line[2:4]
                vec_line = p2 - p1
                vec_point_to_line = p1 - mid_point
                
                # 手动计算2D叉积 (行列式)
                cross_product_2d = vec_line[0] * vec_point_to_line[1] - vec_line[1] * vec_point_to_line[0]
                
                # 距离 = |叉积| / |线段向量长度|
                line_length = np.linalg.norm(vec_line)
                if line_length < 1e-6: continue # 避免除以零
                dist = np.abs(cross_product_2d) / line_length

                if dist < p["MAX_LATERAL_DISTANCE"]:
                    group.append(segment)
                    placed = True
                    break
            if not placed:
                proximity_groups.append([segment])
        
        final_line_groups.extend(proximity_groups)

    # --- 3. 合并每个最终的精细分组，并计算证据得分 ---
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

    # --- 4. 按证据得分排序并返回最佳结果 ---
    merged_lines_with_scores.sort(key=lambda item: item['score'], reverse=True)
    final_best_lines = [item['line'] for item in merged_lines_with_scores]
    return final_best_lines[:p["TOP_N_EDGES"]]

def process_roi_hough_based(roi_gray, params):
    """(已简化) 处理单个ROI的核心函数，只专注于合并和显示主边缘。"""
    p_hough = params["HOUGH_TRANSFORM"]
    roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # 1. 预处理 + Hough变换
    binary_edges = preprocess_for_hough_enhanced(roi_gray, params)
    min_len_pixels = roi_gray.shape[1] * p_hough["MIN_LINE_LENGTH_RATIO"]
    raw_lines = cv2.HoughLinesP(binary_edges, 1, np.pi / 180, p_hough["THRESHOLD"], minLineLength=min_len_pixels, maxLineGap=p_hough["MAX_LINE_GAP"])
    
    # 2. (核心) 将所有检测到的原始线段直接送入合并函数
    main_edges = merge_lines_and_get_main_edges(raw_lines, params)
    
    # 3. (简化) 只绘制最终合并成的主边缘 (绿色)
    for edge in main_edges:
        x1, y1, x2, y2 = map(int, edge)
        cv2.line(roi_color, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    return roi_color

def process_video_with_hough(video_path, roi_file_path, output_path, params):
    """主函数：读取视频，在ROI内应用最终的合并算法，并保存结果。"""
    rois = load_rois_from_file(roi_file_path)
    if not rois: return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
    print(f"开始使用最终合并算法处理视频: {video_path}")
    with tqdm(total=total_frames, desc="处理中") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for roi_def in rois:
                x, y, w, h = [int(roi_def.get(k, 0)) for k in ('x', 'y', 'width', 'height')]
                if w > 0 and h > 0:
                    # 边界检查
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, fw - x), min(h, fh - y)
                    roi_gray = gray_frame[y:y+h, x:x+w]
                    if roi_gray.size > 0:
                        processed_roi = process_roi_hough_based(roi_gray, params)
                        frame[y:y+h, x:x+w] = processed_roi
            out.write(frame)
            pbar.update(1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成！视频已保存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用优化的HoughLinesP和高级合并算法在视频ROI内检测玻璃主边缘。")
    parser.add_argument('--video-in', type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument('--roi-file', type=str, default='cam5_roi_averaged_by_group.json', help="包含ROI定义的JSON文件路径。")
    parser.add_argument('--video-out', type=str, default='output_hough_merged.mp4', help="处理后输出的视频文件路径。")
    args = parser.parse_args()
    process_video_with_hough(args.video_in, args.roi_file, args.video_out, TUNABLE_PARAMETERS)