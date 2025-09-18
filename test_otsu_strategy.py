import cv2
import numpy as np
import json
import os

# --- 配置区 ---
INPUT_IMAGE_FOLDER = 'valid'
ROI_FILE = 'dark_ROI.json'
OUTPUT_FOLDER = 'merged_line_results' # 新建输出文件夹

# --- FIX: 降低霍夫变换的敏感度，以便检测到更短的线段 ---
HOUGH_THRESHOLD = 30       # 霍夫投票阈值
HOUGH_MIN_LINE_LENGTH = 25 # [关键改动] 显著降低最小线长，让 merging 算法有材料可用
HOUGH_MAX_LINE_GAP = 20      # 最大线间距

# --- FIX: 从主算法中引入 line merging 的参数 ---
LINE_MERGING_PARAMS = {
    "ANGLE_TOLERANCE": 5.0,         # 合并线段时允许的最大角度差
    "MAX_LATERAL_DISTANCE": 30,     # 判断线段是否属于同一组的最大横向距离
    "TOP_N_EDGES": 8,               # 最终保留得分最高的N条边
}

# --- 核心算法函数 (从 image_processor_hough.py 移植) ---

def merge_lines_and_get_main_edges(lines, params):
    """
    接收来自霍夫变换的原始、碎片化直线，并将它们合并成更长、更连贯的主边缘。
    """
    if lines is None or len(lines) < 1: return []
    # 从参数字典中获取配置
    p_angle_tolerance = params.get("ANGLE_TOLERANCE", 5.0)
    p_max_lateral_dist = params.get("MAX_LATERAL_DISTANCE", 30)
    p_top_n_edges = params.get("TOP_N_EDGES", 8)
    
    lines_np = np.array(lines).reshape(-int(len(lines)), 4)
    angles = np.rad2deg(np.arctan2(lines_np[:, 3] - lines_np[:, 1], lines_np[:, 2] - lines_np[:, 0]))
    angles[angles < 0] += 180
    
    # 1. 根据角度聚类
    angle_clusters = {}
    for i, angle in enumerate(angles):
        placed = False
        for cluster_angle in angle_clusters:
            if min(abs(angle - cluster_angle), 180 - abs(angle - cluster_angle)) < p_angle_tolerance:
                angle_clusters[cluster_angle].append(lines_np[i]); placed = True; break
        if not placed: angle_clusters[angle] = [lines_np[i]]
        
    # 2. 在每个角度簇内，根据邻近度再次聚类
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
                    if np.abs(cross_product_2d) / line_length < p_max_lateral_dist:
                        group.append(segment); placed = True; break
            if not placed: proximity_groups.append([segment])
        final_line_groups.extend(proximity_groups)
        
    # 3. 对每个最终组进行直线拟合，并计算得分
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
        
    # 4. 返回得分最高的 N 条直线
    merged_lines_with_scores.sort(key=lambda item: item['score'], reverse=True)
    return [item['line'] for item in merged_lines_with_scores[:p_top_n_edges]]

def load_rois(file_path):
    """从JSON文件加载ROI定义"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        first_key = next(iter(data))
        return data[first_key].get('averaged_rois', [])
    except Exception as e:
        print(f"错误: 无法从 {file_path} 加载或解析ROI: {e}")
        return None

def main():
    print("--- 开始测试带有直线合并的最终流程 ---")
    
    rois = load_rois(ROI_FILE)
    if not rois:
        print("ROI文件加载失败，脚本终止。")
        return
        
    if not os.path.exists(INPUT_IMAGE_FOLDER):
        print(f"错误: 输入文件夹 '{INPUT_IMAGE_FOLDER}' 不存在。")
        return
        
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"结果将保存在: '{OUTPUT_FOLDER}'")

    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith('.bmp')]

    for filename in image_files:
        try:
            image_path = os.path.join(INPUT_IMAGE_FOLDER, filename)
            print(f"\n处理图片: {filename}")
            
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            visualization_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            for i, roi_def in enumerate(rois):
                x, y, w, h = [int(roi_def.get(k, 0)) for k in ('x', 'y', 'width', 'height')]
                roi_gray = gray_image[y:y+h, x:x+w]
                
                _, otsu_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                kernel_open = np.ones((21,21), np.uint8) 
                cleaned_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel_open)
                
                kernel_gradient = np.ones((3,3), np.uint8)
                gradient_image = cv2.morphologyEx(cleaned_mask, cv2.MORPH_GRADIENT, kernel_gradient)
                
                # 步骤 a: 使用更敏感的参数进行霍夫变换，获取原始线段
                raw_lines = cv2.HoughLinesP(
                    gradient_image, 
                    rho=1, theta=np.pi / 180, 
                    threshold=HOUGH_THRESHOLD, 
                    minLineLength=HOUGH_MIN_LINE_LENGTH, 
                    maxLineGap=HOUGH_MAX_LINE_GAP
                )
                
                # --- FIX: 步骤 b: 对原始线段进行合并 ---
                merged_lines = merge_lines_and_get_main_edges(raw_lines, LINE_MERGING_PARAMS)
                
                # 可视化
                roi_color_to_draw_on = visualization_image[y:y+h, x:x+w]
                if merged_lines:
                    raw_count = len(raw_lines) if raw_lines is not None else 0
                    print(f"  -> ROI #{i+1}: 原始线段 {raw_count} -> 合并后 {len(merged_lines)}")
                    for line in merged_lines:
                        # 合并后的line是 [x1, y1, x2, y2] 格式
                        x1, y1, x2, y2 = map(int, line)
                        cv2.line(roi_color_to_draw_on, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    print(f"  -> 在 ROI #{i+1} 中未检测到合并后的直线。")

            # 保存结果
            output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_merged_lines.jpg")
            cv2.imwrite(output_path, visualization_image)

        except Exception as e:
            print(f"处理图片 {filename} 时发生意外错误: {e}")
            
    print("\n--- 所有图片处理完成 ---")

if __name__ == '__main__':
    main()