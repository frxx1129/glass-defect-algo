import cv2
import numpy as np
import json
import os
import math
import numba
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from collections import deque
import time
from itertools import combinations

# OpenCV: 避免与 ThreadPoolExecutor 过度竞争
cv2.setUseOptimized(True)
try:
    # 线程数可在 processing_worker 中设置；此处仅作为保守缺省
    cv2.setNumThreads(1)
except Exception:
    pass

_KERNEL_CACHE = {}
def _get_kernel(k: int):
    if k <= 1:
        return None
    k = int(k)
    hit = _KERNEL_CACHE.get(k)  # 注意：IDE 自动修正成 _KERNEL_CACHE
    if hit is None:
        hit = np.ones((k, k), np.uint8)
        _KERNEL_CACHE[k] = hit
    return hit
# =====================================================================
# --- 辅助函数 (保持不变) ---
# =====================================================================
@numba.jit(nopython=True)
def normalize_angle_numba(angle):
    return angle % 180

@numba.jit(nopython=True)
def calculate_angle_difference_numba(angle1, angle2):
    diff = abs(angle1 - angle2)
    return min(diff, 180 - diff)

@numba.jit(nopython=True)
def find_closest_edge_numba(polygon_contour, point_x, point_y):
    if len(polygon_contour) < 2: return 0.0, float('inf')
    min_dist_sq = float('inf')
    edge_angle = 0.0
    found = False
    best_p1 = np.zeros(2, dtype=np.float64)
    best_p2 = np.zeros(2, dtype=np.float64)
    point_np = np.array([float(point_x), float(point_y)], dtype=np.float64)
    for i in range(len(polygon_contour)):
        p1, p2 = polygon_contour[i], polygon_contour[(i + 1) % len(polygon_contour)]
        line_vec, point_vec = p2 - p1, point_np - p1
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        if line_len_sq < 1e-10:
            dist_sq = point_vec[0]**2 + point_vec[1]**2
        else:
            dot_product = point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]
            t = max(0.0, min(1.0, dot_product / line_len_sq))
            projection = p1 + t * line_vec
            dist_sq = (point_np[0] - projection[0])**2 + (point_np[1] - projection[1])**2
        if dist_sq < min_dist_sq:
            min_dist_sq, best_p1, best_p2, found = dist_sq, p1, p2, True
    if found:
        dx, dy = best_p2[0] - best_p1[0], best_p2[1] - best_p1[1]
        edge_angle = np.degrees(np.arctan2(dy, dx)) % 180
        return edge_angle, np.sqrt(min_dist_sq)
    return 0.0, float('inf')

@numba.jit(nopython=True)
def is_near_boundary_numba(v_x, v_y, roi_w, roi_h, threshold): 
    return (v_x < threshold or v_x > roi_w - threshold or v_y < threshold or v_y > roi_h - threshold)

@numba.jit(nopython=True)
def calculate_vertex_angle(p_prev, p_curr, p_next):
    v1 = (p_prev - p_curr).astype(np.float64)
    v2 = (p_next - p_curr).astype(np.float64)
    v1_mag, v2_mag = np.sqrt(v1[0]**2 + v1[1]**2), np.sqrt(v2[0]**2 + v2[1]**2)
    if v1_mag < 1e-10 or v2_mag < 1e-10: return 0.0
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    cosine_angle = min(1.0, max(-1.0, dot_product / (v1_mag * v2_mag)))
    return np.degrees(np.arccos(cosine_angle))

def _force_simplify(contour, max_vertices):
    if len(contour) <= max_vertices: return contour
    epsilon = 0.01 * cv2.arcLength(contour, True)
    for _ in range(30):
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) <= max_vertices: return simplified
        epsilon *= 1.5
    return simplified

def _find_farthest_pair(points):
    """(新增) 辅助函数：从点集中找到欧氏距离最远的两个点。"""
    max_dist_sq = -1
    farthest_pair = (None, None)
    if len(points) < 2: return farthest_pair
    
    # Brute-force is fine for a small number of border points
    for p1, p2 in combinations(points, 2):
        dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq
            farthest_pair = (p1, p2)
    return farthest_pair

def _find_closest_pairs_between_sets(set1, set2):
    """(新增) 辅助函数：在两个点集之间找到距离最近的点对。"""
    if not set1 or not set2: return []
    
    pairs = []
    # For simplicity with small numbers of points, we use a greedy approach.
    # A more complex assignment algorithm (like Hungarian) could be used for larger sets.
    
    # Create copies to modify
    remaining1 = list(set1)
    remaining2 = list(set2)
    
    while remaining1 and remaining2:
        best_dist_sq = float('inf')
        best_pair = (None, None)
        best_indices = (-1, -1)
        
        for i, p1 in enumerate(remaining1):
            for j, p2 in enumerate(remaining2):
                dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_pair = (p1, p2)
                    best_indices = (i, j)
        
        if best_pair[0]:
            pairs.append(best_pair)
            # Remove the found points to find the next closest pair
            remaining1.pop(best_indices[0])
            remaining2.pop(best_indices[1])
        else:
            break # No more pairs can be found
            
    return pairs

_CLAHE_CACHE = {}

def _get_clahe(w, h, p_cfg):
    # 允许固定网格：例如 {"CLAHE_GRID": [4,4]}
    grid_cfg = p_cfg.get('CLAHE_GRID', None)
    clip = float(p_cfg.get('CLAHE_CLIP_LIMIT', 3.0))
    if grid_cfg and len(grid_cfg) == 2:
        grid_x, grid_y = int(grid_cfg[0]), int(grid_cfg[1])
        # 防止非法网格（如0或负数）
        grid_x = max(2, grid_x)
        grid_y = max(2, grid_y)
    else:
        # 目标每块像素边长（64–128较稳），可在配置里调
        target_px = int(p_cfg.get('CLAHE_TILE_TARGET_PX', 128))
        grid_x = int(np.clip(round(w / max(1, target_px)), 2, 12))
        grid_y = int(np.clip(round(h / max(1, target_px)), 2, 12))
    key = (grid_x, grid_y, clip)
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid_x, grid_y))
        _CLAHE_CACHE[key] = clahe
    return clahe

def preprocess_roi_enhanced(roi_image, p_cfg):
    APPLY_BILATERAL, BILATERAL_D, SIGMA_COLOR, SIGMA_SPACE = True, 5, 50, 50
    # 基础健壮性：空/多通道/非uint8 保护
    if roi_image is None or getattr(roi_image, 'size', 0) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    processed_roi = roi_image
    if processed_roi.ndim == 3 and processed_roi.shape[2] >= 3:
        processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2GRAY)
    if processed_roi.dtype != np.uint8:
        # 直接截断到 0..255 并转为 uint8，避免 CLAHE 崩溃
        processed_roi = np.clip(processed_roi, 0, 255).astype(np.uint8, copy=False)
    if processed_roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    # 对极小 ROI 或非常窄/高的 ROI，跳过 CLAHE，避免内部 ROI 计算越界
    h, w = processed_roi.shape[:2]
    if h < 2 or w < 2:
        return processed_roi.copy()
    # 保证内存连续，避免某些 OpenCV 算法对步长敏感
    if not processed_roi.flags.c_contiguous:
        processed_roi = np.ascontiguousarray(processed_roi)
    # 自适应 CLAHE
    if float(p_cfg.get('CLAHE_CLIP_LIMIT', 3.0)) > 0:
        h, w = processed_roi.shape[:2]
        clahe = _get_clahe(w, h, p_cfg)
        processed_roi = clahe.apply(processed_roi)
    if p_cfg.get('GAMMA_VALUE', 1.0) != 1.0:
        inv_gamma = 1.0 / float(p_cfg['GAMMA_VALUE'])
        table = (np.arange(256) / 255.0) ** inv_gamma * 255.0
        processed_roi = cv2.LUT(processed_roi, table.astype("uint8"))
    if APPLY_BILATERAL:
        processed_roi = cv2.bilateralFilter(processed_roi, BILATERAL_D, SIGMA_COLOR, SIGMA_SPACE)
    return processed_roi

def cluster_points_with_dbscan(points_xy, eps, min_samples, min_points_per_cluster):
    if points_xy.shape[0] < min_points_per_cluster: return []
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
    labels = db.labels_
    unique_labels = set(labels)
    unique_labels.discard(-1)
    valid_clusters = []
    for label in unique_labels:
        cluster_points = points_xy[labels == label]
        if cluster_points.shape[0] >= min_points_per_cluster:
            valid_clusters.append(cluster_points)
    return valid_clusters

def _get_edge_and_corner_info(p, w, h):
    """(已恢复使用) 辅助函数：判断一个点在哪条边上，并返回相邻的角点。"""
    x, y = p
    if y == 0: return "top", (0, 0), (w - 1, 0)
    if y == h - 1: return "bottom", (0, h - 1), (w - 1, h - 1)
    if x == 0: return "left", (0, 0), (0, h - 1)
    if x == w - 1: return "right", (w - 1, 0), (w - 1, h - 1)
    return None, None, None

# _get_edge 函数依然需要，被 _connect_points_along_border 使用
def _get_edge(p, w, h):
    """辅助函数：判断一个点在哪条边上。"""
    x, y = p
    if y == 0: return "top"
    if x == w - 1: return "right"
    if y == h - 1: return "bottom"
    if x == 0: return "left"
    return None

# 文件：image_processor_optimized.py

def _connect_points_along_border(image, p_start, p_end, direction, h, w):
    """
    (已优化) 智能辅助函数：根据点的位置，选择最优路径进行沿边框连接。
    - 对于邻边点，自动走最短路径（经过1个角点）。
    - 对于对边点，严格遵循传入的direction方向（经过2个角点）。
    """
    corners_cw = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)] # TL, TR, BR, BL
    edge_order_cw = {"top": 0, "right": 1, "bottom": 2, "left": 3}

    start_edge = _get_edge(p_start, w, h)
    end_edge = _get_edge(p_end, w, h)

    if not start_edge or not end_edge:
        cv2.line(image, p_start, p_end, 255, 1)
        return

    start_idx = edge_order_cw[start_edge]
    end_idx = edge_order_cw[end_edge]
    
    path = [p_start]
    
    # --- 核心优化逻辑 ---
    
    # 计算顺时针和逆时针走的步数
    steps_cw = (end_idx - start_idx + 4) % 4
    steps_ccw = (start_idx - end_idx + 4) % 4

    # 判断是否为邻边
    is_adjacent = (steps_cw == 1 or steps_ccw == 1)

    # 决策：
    # 1. 如果是邻边，则永远走最短路径（1步）
    # 2. 如果是对边，则遵循传入的 direction
    final_direction = direction
    if is_adjacent and steps_ccw == 1:
        final_direction = 'ccw' # 邻边强制走最短路径
    elif is_adjacent and steps_cw == 1:
        final_direction = 'cw' # 邻边强制走最短路径

    # --- 路径计算 (与之前相同，但现在基于更智能的 final_direction) ---
    current_idx = start_idx
    if final_direction == 'cw':
        while current_idx != end_idx:
            next_corner_idx = (current_idx + 1) % 4
            path.append(corners_cw[next_corner_idx])
            current_idx = next_corner_idx
    else: # ccw
        while current_idx != end_idx:
            path.append(corners_cw[current_idx])
            current_idx = (current_idx - 1 + 4) % 4
    
    path.append(p_end)

    # --- 绘图 (与之前相同) ---
    for i in range(len(path) - 1):
        p1 = (int(path[i][0]), int(path[i][1]))
        p2 = (int(path[i+1][0]), int(path[i+1][1])) 
        cv2.line(image, p1, p2, 255, 1)
        
def _dist_point_to_segment(p, a, b):
    """(新增) 辅助函数：计算点p到一个线段(a,b)的最短距离。"""
    p, a, b = np.array(p), np.array(a), np.array(b)
    line_vec = b - a
    point_vec = p - a
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-10:
        return np.linalg.norm(point_vec)
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0.0, 1.0)
    projection = a + t * line_vec
    return np.linalg.norm(p - projection)

def _connect_intra_edge_smart(image, points):
    """
    (新增) 智能同边连接函数，实现了“跳跃连接”和“在线验证”的逻辑。
    """
    if len(points) < 2:
        return

    i = 0
    while i < len(points) - 1:
        p1 = points[i]
        p2 = points[i+1]
        
        partner_found = False
        
        # 优先尝试“跳跃连接” p1 和 p3
        if i + 2 < len(points):
            p3 = points[i+2]
            # 检查 p1-p3 的距离是否合格
            if np.linalg.norm(np.array(p1) - np.array(p3)) >= 25:
                # 检查被跳过的 p2 是否在线上
                if _dist_point_to_segment(p2, p1, p3) < 3.0: # 3像素的容忍度
                    cv2.line(image, p1, p3, 255, 1)
                    i += 3 # 成功连接，跳过 p1, p2, p3
                    partner_found = True

        # 如果“跳跃连接”不成功，再尝试常规的近邻连接 p1 和 p2
        if not partner_found:
            if np.linalg.norm(np.array(p1) - np.array(p2)) >= 25:
                cv2.line(image, p1, p2, 255, 1)
                i += 2 # 成功连接，跳过 p1, p2
            else:
                # p1 找不到任何伙伴，将 p1 视为落单点，从 p2 开始继续
                i += 1
                
def _find_closest_pairs_between_sets(set1, set2):
    """(新增) 辅助函数：在两个点集之间贪心匹配距离最近的点对。"""
    if not set1 or not set2: return []
    
    pairs = []
    remaining1 = list(set1)
    remaining2 = list(set2)
    
    while remaining1 and remaining2:
        best_dist_sq = float('inf')
        best_pair = (None, None)
        best_indices = (-1, -1)
        
        for i, p1 in enumerate(remaining1):
            for j, p2 in enumerate(remaining2):
                dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_pair = (p1, p2)
                    best_indices = (i, j)
        
        if best_pair[0] is not None:
            pairs.append(best_pair)
            remaining1.pop(best_indices[0])
            remaining2.pop(best_indices[1])
        else:
            break
            
    return pairs

def _is_parallel_curve_segment(points, max_dist=10.0):
    """
    (新增) 辅助函数：根据您的精确定义，判断一条边上的点是否构成“平行曲线”片段。
    条件：点数为偶数 (>=2)，且所有配对点的距离都小于 max_dist。
    """
    if len(points) < 2 or len(points) % 2 != 0:
        return False
    for i in range(0, len(points), 2):
        p1 = np.array(points[i])
        p2 = np.array(points[i+1])
        if np.linalg.norm(p1 - p2) >= max_dist:
            return False
    return True

def _get_max_dist_in_set(points):
    """(新增) 辅助函数：计算一个点集内部任意两点间的最大距离。"""
    if len(points) < 2:
        return 0.0
    # _find_farthest_pair 返回的是点对，我们需要计算其实际距离
    p1, p2 = _find_farthest_pair(points)
    return np.linalg.norm(np.array(p1) - np.array(p2))

def connect_border_points_advanced(edges_image, direction='cw'):
    """
    (最终完备修正版) 基于最终分层决策树的高级边界点连接函数。
    - 完整实现了所有分层规则和距离门槛 (10px 和 25px)。
    - 补全并修正了对边情况的三种子模式判断。
    """
    h, w = edges_image.shape
    closed_edges_image = edges_image.copy()

    # 1. 获取所有边界点 (原始点集)
    point_lists = {
        "top": sorted([(x, 0) for x in range(w) if edges_image[0, x] > 0], key=lambda p: p[0]),
        "bottom": sorted([(x, h-1) for x in range(w) if edges_image[h-1, x] > 0], key=lambda p: p[0]),
        "left": sorted([(0, y) for y in range(h) if edges_image[y, 0] > 0], key=lambda p: p[1]),
        "right": sorted([(w-1, y) for y in range(h) if edges_image[y, w-1] > 0], key=lambda p: p[1])
    }
    # 保留原始点集合副本（便于调试或未来可视化）
    _original_point_lists_for_debug = {k: v[:] for k, v in point_lists.items()}

    # 1.1 连接阶段点合并：对所有边界点做半径聚类 (r<10) 以减少密集重复点，仅影响连接推理，不改动原图
    merge_thresh = 10.0
    raw_points_all = point_lists["top"] + point_lists["bottom"] + point_lists["left"] + point_lists["right"]
    if len(raw_points_all) > 1:
        # 角点 (保持原样不参与合并)
        corner_points = {(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)}
        non_corner_points = [p for p in raw_points_all if p not in corner_points]
        r2 = merge_thresh * merge_thresh
        clusters = []  # 每簇: [cx, cy, n]
        for (px, py) in non_corner_points:
            placed = False
            for c in clusters:
                cx, cy, n = c
                if (px - cx)**2 + (py - cy)**2 <= r2:
                    n_new = n + 1
                    c[0] = cx + (px - cx) / n_new  # 增量均值
                    c[1] = cy + (py - cy) / n_new
                    c[2] = n_new
                    placed = True
                    break
            if not placed:
                clusters.append([float(px), float(py), 1])
        merged_points = [(int(round(c[0])), int(round(c[1]))) for c in clusters]
        # 合并结果 + 原始角点
        merged_points.extend(list(corner_points & set(raw_points_all)))

        reassigned = {"top": [], "bottom": [], "left": [], "right": []}
        for (mx, my) in merged_points:
            if my == 0: reassigned["top"].append((mx, my))
            elif my == h - 1: reassigned["bottom"].append((mx, my))
            elif mx == 0: reassigned["left"].append((mx, my))
            elif mx == w - 1: reassigned["right"].append((mx, my))
        point_lists = {
            "top": sorted(set(reassigned["top"]), key=lambda p: p[0]),
            "bottom": sorted(set(reassigned["bottom"]), key=lambda p: p[0]),
            "left": sorted(set(reassigned["left"]), key=lambda p: p[1]),
            "right": sorted(set(reassigned["right"]), key=lambda p: p[1])
        }

    all_border_points = point_lists["top"] + point_lists["bottom"] + point_lists["left"] + point_lists["right"]
    if len(all_border_points) < 2:
        return closed_edges_image

    # 2. 按规则进行决策
    edges_present = {name for name, points in point_lists.items() if points}
    
    # --- 规则 for 2条对边 (最高优先级) ---
    is_top_bottom = edges_present == {'top', 'bottom'}
    is_left_right = edges_present == {'left', 'right'}
    if is_top_bottom or is_left_right:
        side1_pts = point_lists["top"] if is_top_bottom else point_lists["left"]
        side2_pts = point_lists["bottom"] if is_top_bottom else point_lists["right"]

        # 计算同边最大距离
        max_dist_side1 = _get_max_dist_in_set(side1_pts)
        max_dist_side2 = _get_max_dist_in_set(side2_pts)
        max_intra_edge_dist = max(max_dist_side1, max_dist_side2)

        # --- 根据距离门槛，进入三层决策 ---
        if max_intra_edge_dist < 10:
            # --- 情况A: 平行曲线 -> 连接一次 ---
            p_start, p_end = _find_farthest_pair(all_border_points)
            if p_start:
                start_edge, end_edge = _get_edge(p_start, w, h), _get_edge(p_end, w, h)
                # 严格执行起点规则
                if is_top_bottom and start_edge == "bottom": p_start, p_end = p_end, p_start
                elif is_left_right and start_edge == "left": p_start, p_end = p_end, p_start
                _connect_points_along_border(closed_edges_image, p_start, p_end, direction, h, w)

        elif 10 <= max_intra_edge_dist < 25:
            # --- 情况B: 特殊情况 -> 两次连接，经过所有角 ---
            if len(side1_pts) >= 2 and len(side2_pts) >= 2:
                pairs = _find_closest_pairs_between_sets(side1_pts, side2_pts)
                # 排序以确保连接路径不交叉
                sort_key = lambda p: (p[0][0] + p[1][0]) if is_top_bottom else (p[0][1] + p[1][1])
                pairs.sort(key=sort_key)
                if len(pairs) >= 2:
                    p1_start, p1_end = pairs[0]
                    p2_start, p2_end = pairs[1]
                    # 确保起点正确
                    if is_left_right: p1_start, p1_end = p1_end, p1_start
                    if is_left_right: p2_start, p2_end = p2_end, p2_start
                    
                    _connect_points_along_border(closed_edges_image, p1_start, p1_end, 'ccw', h, w)
                    _connect_points_along_border(closed_edges_image, p2_start, p2_end, 'cw', h, w)
        else: # max_intra_edge_dist >= 25
            # --- 情况C: 一般情况 -> 只连接同边 ---
            for points in [side1_pts, side2_pts]:
                if len(points) >= 2:
                    for i in range(0, len(points) // 2 * 2, 2):
                        cv2.line(closed_edges_image, points[i], points[i+1], 255, 1)
        return closed_edges_image

    # --- 规则 for 3条边 ---
    elif len(edges_present) == 3:
        middle_edge, side_A, side_B = None, None, None
        if edges_present == {'top', 'right', 'bottom'}: middle_edge, side_A, side_B = "right", "top", "bottom"
        elif edges_present == {'right', 'bottom', 'left'}: middle_edge, side_A, side_B = "bottom", "right", "left"
        elif edges_present == {'bottom', 'left', 'top'}: middle_edge, side_A, side_B = "left", "bottom", "top"
        elif edges_present == {'left', 'top', 'right'}: middle_edge, side_A, side_B = "top", "left", "right"
        
        if middle_edge and point_lists[middle_edge] and point_lists[side_A] and point_lists[side_B]:
            p_mid_A = point_lists[middle_edge][0] if middle_edge in ["right", "bottom"] else point_lists[middle_edge][-1]
            _, p_side_A = _find_farthest_pair([p_mid_A] + point_lists[side_A])
            _connect_points_along_border(closed_edges_image, p_mid_A, p_side_A, direction, h, w)
            p_mid_B = point_lists[middle_edge][-1] if middle_edge in ["right", "bottom"] else point_lists[middle_edge][0]
            _, p_side_B = _find_farthest_pair([p_mid_B] + point_lists[side_B])
            _connect_points_along_border(closed_edges_image, p_mid_B, p_side_B, direction, h, w)
            return closed_edges_image

    # --- 规则 for 2条邻边 ---
    elif len(edges_present) == 2:
        p_start, p_end = _find_farthest_pair(all_border_points)
        if p_start:
            _connect_points_along_border(closed_edges_image, p_start, p_end, direction, h, w)
        return closed_edges_image
            
    # --- 规则 for 1条边 ---
    elif len(edges_present) == 1:
        edge_name = list(edges_present)[0]
        points = point_lists[edge_name]
        if len(points) >= 2:
            for i in range(0, len(points) // 2 * 2, 2):
                p1, p2 = points[i], points[i+1]
                if np.linalg.norm(np.array(p1) - np.array(p2)) >= 25:
                    cv2.line(closed_edges_image, p1, p2, 255, 1)
        return closed_edges_image

    # --- 如果不满足任何明确规则，则不进行连接 ---
    return edges_image.copy()

def _generate_candidate_spectrum(base_contour, targets=[6, 5, 4, 3]):
    """
    (已更新) 通过在 epsilon 空间进行更密集的搜索，生成一系列候选轮廓。
    修正了提前退出的逻辑，确保所有目标候选都能被搜索。
    """
    candidates = {}
    arc_length = cv2.arcLength(base_contour, True)
    if arc_length < 1e-6: return {}

    base_vertices = len(base_contour)
    if base_vertices in targets:
        candidates[base_vertices] = base_contour

    # 在一个更宽、更密集的 epsilon 区间内搜索
    for epsilon_ratio in np.logspace(-4, -1, 30):
        epsilon = epsilon_ratio * arc_length
        simplified = cv2.approxPolyDP(base_contour, epsilon, True)
        num_vertices = len(simplified)

        if num_vertices in targets and num_vertices not in candidates:
            candidates[num_vertices] = simplified
        
        # --- 核心修改 ---
        # 退出条件改为当顶点数少于我们关心的最小值(3)时才退出
        if num_vertices < 3:
            break
            
        # 如果所有目标都已找到，也可以提前退出
        if all(t in candidates for t in targets):
            break

    return candidates

def _determine_connection_direction(roi_original):
    """
    (最终版) 通过将ROI十字切分为四个象限，对比其平均亮度来决定方向。
    严格遵循“背景在左或上时，为逆时針”的规则。
    """
    h, w = roi_original.shape
    if h < 10 or w < 10: # 确保ROI足够大以进行有意义的切分
        return 'cw'

    # 1. 精确地将ROI十字切分为四个象限
    mid_x, mid_y = w // 2, h // 2
    
    top_left_quad = roi_original[0:mid_y, 0:mid_x]
    top_right_quad = roi_original[0:mid_y, mid_x:w]
    bottom_left_quad = roi_original[mid_y:h, 0:mid_x]
    bottom_right_quad = roi_original[mid_y:h, mid_x:w]

    # 2. 计算每个象限的平均亮度
    mean_tl = np.mean(top_left_quad) if top_left_quad.size > 0 else 0
    mean_tr = np.mean(top_right_quad) if top_right_quad.size > 0 else 0
    mean_bl = np.mean(bottom_left_quad) if bottom_left_quad.size > 0 else 0
    mean_br = np.mean(bottom_right_quad) if bottom_right_quad.size > 0 else 0

    # 3. 进行宏观的左右和上下对比
    mean_left_half = (mean_tl + mean_bl) / 2
    mean_right_half = (mean_tr + mean_br) / 2
    
    mean_top_half = (mean_tl + mean_tr) / 2
    mean_bottom_half = (mean_bl + mean_br) / 2
    
    diff_horizontal = mean_right_half - mean_left_half # 正数表示右边更亮
    diff_vertical = mean_bottom_half - mean_top_half   # 正数表示下边更亮

    # 4. 决策：哪个方向的差异更显著，就以哪个方向为准
    if abs(diff_horizontal) > abs(diff_vertical):
        # 水平差异是主导
        if diff_horizontal < 0: # 左半边更亮 -> 背景在左侧
            return 'cw'  # 规则：左是顺时针
        else: # 右半边更亮 -> 背景在右侧
            return 'ccw'
    else:
        # 垂直差异是主导
        if diff_vertical < 0: # 上半边更亮 -> 背景在顶部
            return 'cw'  # 规则：上是顺时针
        else: # 下半边更亮 -> 背景在底部
            return 'ccw'


def _calculate_fidelity_distance_based(contour, original_contour):
    """
    (新) 基于简化轮廓顶点到原始轮廓的平均距离来计算保真度分数。
    这个方法比 matchShapes 更直观地反映“贴合程度”。

    :param contour: 简化后的轮廓。
    :param original_contour: 原始轮廓。
    :return: 保真度分数（越高越好）。
    """
    if len(contour) == 0:
        return 0

    total_distance = 0
    # 遍历简化轮廓的每一个顶点
    for point in contour:
        # point[0] 是顶点的 (x, y) 坐标
        # cv2.pointPolygonTest 计算点到轮廓的最短距离
        #   - 结果 > 0: 点在轮廓内
        #   - 结果 < 0: 点在轮廓外
        #   - 结果 = 0: 点在轮廓上
        # 我们取绝对值，因为我们只关心距离
        distance = cv2.pointPolygonTest(original_contour, tuple(point[0]), True)
        total_distance += abs(distance)
    
    # 计算平均距离（误差）
    average_distance_error = total_distance / len(contour)

    # 将误差转换为分数（误差越小，分数越高）
    # + 0.1 是为了防止除以零，并对微小误差不给予过高的分数
    fidelity_score = 1.0 / (average_distance_error + 0.1)
    
    # 分数乘以一个系数，使其量级与之前的 matchShapes 分数大致可比
    return fidelity_score * 100

# 确保 _calculate_fidelity_distance_based 函数也存在
def _calculate_fidelity_distance_based(contour, original_contour):
    if len(contour) == 0: return 0
    total_distance = sum(abs(cv2.pointPolygonTest(original_contour, (float(p[0][0]), float(p[0][1])), True)) for p in contour)
    average_distance_error = total_distance / len(contour)
    return 100.0 / (average_distance_error + 0.1)

def _is_polygon_credible(contour, min_edge_length):
    """
    (新增) 检查一个多边形的所有边长是否都大于指定的最小长度。

    :param contour: 待检查的多边形轮廓。
    :param min_edge_length: 最小允许的边长（像素）。
    :return: 如果所有边都合格，则返回 True，否则返回 False。
    """
    # 遍历多边形的每条边
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        
        # 计算边长
        edge_length = np.linalg.norm(p1 - p2)
        
        # 如果发现任何一条边小于阈值，则立即判定为不可信
        if edge_length < min_edge_length:
            return False
            
    # 如果所有边都检查合格，则返回 True
    return True

def _calculate_score_advanced(
    contour, 
    original_contour,
    base_w_fidelity=0.4,
    base_w_simplicity=0.6
):
    """
    (已更新) 高级评分函数。
    加大了动态权重的调整幅度，使保真度对多边形的影响更大。
    """
    num_vertices = len(contour)
    simplicity_score_map = {3: 0.3, 4: 1.0, 5: 0.8, 6: 0.8, 7:0.3, 8:0.3}
    score_simplicity = simplicity_score_map.get(num_vertices, 0.8 * (6 / max(num_vertices, 1)))

    min_v, max_v = 3, 8
    complexity_factor = np.clip((num_vertices - min_v) / (max_v - min_v), 0.0, 1.0)
    
    # --- 核心改动：加大权重调整幅度从 0.2 -> 0.4 ---
    w_simp_dyn = base_w_simplicity + (1 - complexity_factor) * 0.4 
    w_fid_dyn = base_w_fidelity + complexity_factor * 0.4
    
    total_w = w_fid_dyn + w_simp_dyn
    w_fid_dyn /= total_w
    w_simp_dyn /= total_w
    
    score_fidelity = _calculate_fidelity_distance_based(contour, original_contour)
    
    final_score = (w_fid_dyn * score_fidelity) + (w_simp_dyn * score_simplicity)
    
    return {
        "v": num_vertices, "s_simp": score_simplicity, "s_fid": score_fidelity,
        "w_simp": w_simp_dyn, "w_fid": w_fid_dyn, "total": final_score
    }

def find_best_fit_polygon(original_contour, min_edge_length_px=15):
    """
    (已更新) 寻找最佳拟合多边形，核心逻辑更新：
    1. 将三角形(3个顶点)纳入最终候选池。
    2. 确保降级逻辑的稳健性。
    """
    if original_contour is None or len(original_contour) < 3:
        return np.array([[]], dtype=np.int32)

    hull = cv2.convexHull(original_contour)
    # 将3加入目标列表
    targets = [6, 5, 4, 3]

    candidates_from_original = _generate_candidate_spectrum(original_contour, targets=targets)
    candidates_from_hull = _generate_candidate_spectrum(hull, targets=targets)
    
    unique_candidates = candidates_from_original.copy()
    for v, c in candidates_from_hull.items():
        if v not in unique_candidates:
            unique_candidates[v] = c

    if not unique_candidates:
        return _force_simplify(hull, 4)

    credible_candidates = []
    # --- 核心修改：将循环检查的范围从 4 <= v <= 6 改为 3 <= v <= 6 ---
    for v in sorted(unique_candidates.keys(), reverse=True):
        candidate = unique_candidates[v]
        if 3 <= v <= 6: # <--- 修改点在这里
            if _is_polygon_credible(candidate, min_edge_length=min_edge_length_px):
                scores = _calculate_score_advanced(candidate, original_contour)
                credible_candidates.append({'contour': candidate, 'scores': scores})

    if credible_candidates:
        best_candidate = max(credible_candidates, key=lambda x: x['scores']['total'])
        return best_candidate['contour']
    else:
        if 4 in unique_candidates:
            return unique_candidates[4]
        elif 5 in unique_candidates: # 增加降级逻辑
            return unique_candidates[5]
        elif 6 in unique_candidates:
            return unique_candidates[6]
        elif 3 in unique_candidates:
            return unique_candidates[3]
        else:
            return _force_simplify(hull, 4)

def _correct_right_angle_numeric(angle_deg, d_cfg):
    """
    只对数值结果做后处理：
    - ANGLE_BIAS_DEG: 常量偏置（度），修正系统性偏差；正值使角度增大。
    - ANGLE_SNAP_TO_90_DEADBAND_DEG: 死区吸附阈值，落入±阈值则直接置为90°。
    - ANGLE_SOFT_SNAP_GAIN: 软吸附增益(0..1)，将偏差按(1-gain)缩小；
      仅在 |角度-90| ≤ ANGLE_SOFT_SNAP_MAX_DEVIATION 时生效。
    - ANGLE_SOFT_SNAP_MAX_DEVIATION: 软吸附的最大允许偏差范围（度）。
    - ANGLE_REPORT_ROUND_TO: 可选数值报告四舍五入步长（度），0表示不处理。
    默认均为0或关闭，不改变原逻辑。
    """
    try:
        a = float(angle_deg)
        # 1) 常量偏置
        bias = float(d_cfg.get('ANGLE_BIAS_DEG', 0.0))
        a = a + bias

        # 2) 死区吸附到 90°
        deadband = float(d_cfg.get('ANGLE_SNAP_TO_90_DEADBAND_DEG', 0.0))
        if deadband > 0.0 and abs(a - 90.0) <= deadband:
            a = 90.0
        else:
            # 3) 软吸附（仅在小偏差范围内）
            gain = float(d_cfg.get('ANGLE_SOFT_SNAP_GAIN', 0.5))
            max_dev = float(d_cfg.get('ANGLE_SOFT_SNAP_MAX_DEVIATION', 5.0))
            if gain > 0.0 and max_dev > 0.0 and abs(a - 90.0) <= max_dev:
                dev = a - 90.0
                a = 90.0 + dev * (1.0 - max(0.0, min(1.0, gain)))

        # 4) 可选数值报告四舍五入
        round_to = float(d_cfg.get('ANGLE_REPORT_ROUND_TO', 0.0))
        if round_to > 0.0:
            a = round(a / round_to) * round_to

        # 5) 限幅到 [0, 180]
        if a < 0.0:
            a = 0.0
        elif a > 180.0:
            a = 180.0
        return float(a)
    except Exception:
        return float(angle_deg)
        
def process_roi_with_defect_detection(roi_idx, roi_template, image_gray, config, draw_main_contour=True):
    # --- 1. 初始化 ---
    # 兼容两种配置结构：顶层平铺 / defect_detection_params 嵌套
    dd_cfg = config.get('defect_detection_params', {}) if isinstance(config, dict) else {}
    p_cfg = config.get('preprocess_params') or dd_cfg.get('preprocess_params', {})
    dbscan_cfg = config.get('dbscan_params') or dd_cfg.get('dbscan_params', {})
    contour_cfg = config.get('contour_params') or dd_cfg.get('contour_params', {})
    d_cfg = config.get('defect_params') or dd_cfg.get('defect_params', {})
    dr_cfg = config.get('drawing_params') or dd_cfg.get('drawing_params', {})

    # 预取常量，避免循环中反复 dict 查找
    boundary_threshold = dr_cfg.get('BOUNDARY_THRESHOLD', 10)
    raw_contour_color = tuple(dr_cfg.get('RAW_CONTOUR_COLOR', [0, 255, 0]))  # 绿色更醒目
    raw_contour_thickness = int(dr_cfg.get('RAW_CONTOUR_THICKNESS', 2))
    simplified_color = tuple(dr_cfg.get('SIMPLIFIED_CONTOUR_COLOR', [0, 0, 255]))
    simplified_thickness = int(dr_cfg.get('SIMPLIFIED_CONTOUR_THICKNESS', 1))
    mask_line_thickness = int(dr_cfg.get('DEFECT_CONTOUR_MASK_THICKNESS', 3))
    debug_trace = bool(dr_cfg.get('DEBUG_TRACE', False))
    angle_min = d_cfg.get('ANGLE_MIN_THRESHOLD', 10.0)
    angle_max = d_cfg.get('ANGLE_MAX_THRESHOLD', 170.0)
    bevel_tol = d_cfg.get('ANGLE_BEVEL_TOLERANCE', 2.5)
    normal_tol = d_cfg.get('ANGLE_NORMAL_TOLERANCE', 2.0)
    ppmm = float(d_cfg.get('PIXELS_PER_MM', 2.448))
    min_defect_area_px = float(d_cfg.get('MIN_DEFECT_AREA_MM2', 4.0)) * float(d_cfg.get('PIXELS_PER_MM', 2.735))**2
    merge_kernel_size = int(d_cfg.get('DEFECT_MERGE_KERNEL_SIZE', 3))
    merge_kernel = _get_kernel(merge_kernel_size)

    # 基础健壮性：确保输入为灰度 uint8
    if image_gray is None or getattr(image_gray, 'size', 0) == 0:
        return {"polygons_in_roi": 0, "defects": []}, np.zeros((1, 1, 3), dtype=np.uint8)
    if image_gray.ndim == 3 and image_gray.shape[2] >= 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
    if image_gray.dtype != np.uint8:
        image_gray = np.clip(image_gray, 0, 255).astype(np.uint8, copy=False)

    H, W = image_gray.shape[:2]
    x, y, w, h = int(roi_template['x']), int(roi_template['y']), int(roi_template['width']), int(roi_template['height'])
    if w <= 0 or h <= 0:
        return {"polygons_in_roi": 0, "defects": []}, np.zeros((1, 1, 3), dtype=np.uint8)
    # 裁剪到图像范围，避免越界
    x = max(0, min(x, W - 1)) if W > 0 else 0
    y = max(0, min(y, H - 1)) if H > 0 else 0
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    if w <= 0 or h <= 0:
        return {"polygons_in_roi": 0, "defects": []}, np.zeros((1, 1, 3), dtype=np.uint8)

    roi_original = image_gray[y:y+h, x:x+w]
    if roi_original is None or roi_original.size == 0:
        return {"polygons_in_roi": 0, "defects": []}, np.zeros((1, 1, 3), dtype=np.uint8)
    roi_report = { "roi_idx": roi_idx, "x": x, "y": y, "w": w, "h": h, "polygons_in_roi": 0, "defects": [] }

    # --- 2. 轮廓提取 ---
    roi_preprocessed = preprocess_roi_enhanced(roi_original, p_cfg)
    v = float(np.median(roi_preprocessed))
    sigma = float(p_cfg.get('CANNY_SIGMA', 0.2))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny_edges = cv2.Canny(roi_preprocessed, lower, upper)

    cleaned_edges = np.zeros_like(canny_edges)
    canny_points_yx = np.argwhere(canny_edges == 255)
    canny_points_count = int(canny_points_yx.shape[0])
    if canny_points_yx.shape[0] >= dbscan_cfg.get('min_points_per_cluster', 100):
        # 限流/下采样，避免 DBSCAN 过慢
        max_points = int(dbscan_cfg.get('max_points', 8000))
        if canny_points_yx.shape[0] > max_points:
            step = int(np.ceil(canny_points_yx.shape[0] / max_points))
            canny_points_yx = canny_points_yx[::step]

        canny_points_xy = canny_points_yx[:, [1, 0]]
        cleaned_point_clusters = cluster_points_with_dbscan(canny_points_xy, **dbscan_cfg)
        if cleaned_point_clusters:
            all_cleaned_points = np.vstack(cleaned_point_clusters)
            # 矢量化一次性赋值，替代逐点 for 写入
            cleaned_edges[all_cleaned_points[:, 1], all_cleaned_points[:, 0]] = 255
            cleaned_points_count = int(all_cleaned_points.shape[0])
        else:
            cleaned_points_count = 0
    else:
        cleaned_points_count = 0

    connection_direction = _determine_connection_direction(roi_original)

    closed_wireframe = connect_border_points_advanced(cleaned_edges, direction=connection_direction)
    contours, _ = cv2.findContours(closed_wireframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area_cfg = contour_cfg.get('min_area', 5000)
    valid_polygons_raw = [p for p in contours if cv2.contourArea(p) > min_area_cfg] if contours else []

    # --- Fallback: 若按原规则仍无轮廓，尝试一次“每条边连接最远点” ---
    if not valid_polygons_raw:
        try:
            # 第二次连接才使用最远点策略；若 cleaned_edges 为空则不启用新策略
            if not cleaned_edges.any():
                raise RuntimeError("skip new connection strategy: no cleaned edges")
            edge_src = cleaned_edges
            h_loc, w_loc = edge_src.shape[:2]
            fallback_wire = np.zeros_like(edge_src)
            # 收集四边点 (局部 ROI 坐标)
            # 收集四边点（排除四个角点）
            corners = {(0,0), (w_loc-1,0), (0,h_loc-1), (w_loc-1,h_loc-1)}
            top_pts = [(int(xx), 0) for xx in np.where(edge_src[0, :] == 255)[0]]
            bot_pts = [(int(xx), h_loc - 1) for xx in np.where(edge_src[h_loc - 1, :] == 255)[0]] if h_loc > 1 else []
            left_pts = [(0, int(yy)) for yy in np.where(edge_src[:, 0] == 255)[0]]
            right_pts = [(w_loc - 1, int(yy)) for yy in np.where(edge_src[:, w_loc - 1] == 255)[0]] if w_loc > 1 else []
            def _exclude_corners(pts):
                return [p for p in pts if p not in corners]
            top_pts = _exclude_corners(top_pts)
            bot_pts = _exclude_corners(bot_pts)
            left_pts = _exclude_corners(left_pts)
            right_pts = _exclude_corners(right_pts)

            def _connect_farthest(pts):
                if len(pts) < 2:
                    return
                # 由于这些点都在同一条直边上，最远的就是 min / max 端点
                # 但为通用性，再线性扫描一次（点数通常不大）
                max_d = -1; p1 = p2 = None
                for i in range(len(pts)):
                    xi, yi = pts[i]
                    for j in range(i + 1, len(pts)):
                        xj, yj = pts[j]
                        d = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
                        if d > max_d:
                            max_d = d; p1, p2 = pts[i], pts[j]
                if p1 and p2:
                    cv2.line(fallback_wire, p1, p2, 255, 1)

            _connect_farthest(top_pts)
            _connect_farthest(bot_pts)
            _connect_farthest(left_pts)
            _connect_farthest(right_pts)

            if fallback_wire.any():
                contours2, _ = cv2.findContours(fallback_wire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_polygons_raw = [p for p in contours2 if cv2.contourArea(p) > min_area_cfg] if contours2 else []
                if valid_polygons_raw:
                    closed_wireframe = fallback_wire  # 使用回退结果
        except Exception:
            pass

    final_simplified_polygons = [find_best_fit_polygon(c) for c in valid_polygons_raw]

    # --- 3. 缺陷检测与标注 ---
    roi_color = cv2.cvtColor(roi_original, cv2.COLOR_GRAY2BGR)
    DEFECT_COLORS = { 'Q': (0, 0, 255), 'X': (0, 255, 255), 'L': (255, 0, 255), 'B': (0, 165, 255) }

    for raw_contour, simplified_contour in zip(valid_polygons_raw, final_simplified_polygons):
        roi_report["polygons_in_roi"] += 1
        if draw_main_contour:
            cv2.polylines(roi_color, [raw_contour], isClosed=True, color=raw_contour_color, thickness=raw_contour_thickness)
            if simplified_contour is not None and len(simplified_contour) >= 3:
                cv2.polylines(roi_color, [simplified_contour], isClosed=True, color=simplified_color, thickness=simplified_thickness)

        vertices_f32 = simplified_contour.reshape(-1, 2).astype(np.float32, copy=False)
        n = len(vertices_f32)
        if n > 2:
            for i in range(n):
                p_curr = vertices_f32[i]
                if is_near_boundary_numba(float(p_curr[0]), float(p_curr[1]), w, h, boundary_threshold): 
                    continue
                p_prev = vertices_f32[(i - 1 + n) % n]
                p_next = vertices_f32[(i + 1) % n]
                angle_raw = float(calculate_vertex_angle(p_prev, p_curr, p_next))
                angle = _correct_right_angle_numeric(angle_raw, d_cfg)
                if not (angle_min <= angle <= angle_max):
                    continue
                deviation = abs(angle - 90.0)
                defect_type = "Q" if deviation > bevel_tol else ("X" if deviation > normal_tol else None)
                if defect_type:
                    roi_report["defects"].append({"type": defect_type, "location": {"x": int(p_curr[0] + x), "y": int(p_curr[1] + y) , "angle": angle}})
                    cv2.circle(roi_color, (int(p_curr[0]), int(p_curr[1])), 12, DEFECT_COLORS[defect_type], 2)

        mask = np.zeros(roi_original.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [raw_contour], 255)
        mean, std_dev = cv2.meanStdDev(roi_original, mask=mask)
        mean, std_dev = float(mean[0][0]), float(std_dev[0][0])

        if std_dev > 1.0:
            lower_bound = mean - d_cfg.get('DEFECT_STD_DEV_THRESHOLD', 3.0) * std_dev
            upper_bound = mean + d_cfg.get('DEFECT_STD_DEV_THRESHOLD', 3.0) * std_dev
            anomaly_mask = ((roi_original < lower_bound) | (roi_original > upper_bound)).astype(np.uint8) * 255
            anomaly_mask = cv2.bitwise_and(anomaly_mask, mask)

            contour_line_mask = np.zeros(roi_original.shape, dtype=np.uint8)
            cv2.polylines(contour_line_mask, [raw_contour], isClosed=True, color=255, thickness=mask_line_thickness)
            anomaly_mask = cv2.subtract(anomaly_mask, contour_line_mask)

            if merge_kernel is not None:
                anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_CLOSE, merge_kernel, iterations=2)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(anomaly_mask, 8, cv2.CV_32S)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_defect_area_px:
                    continue

                component_points = np.argwhere(labels == i)[:, ::-1]
                if component_points.shape[0] < 5:
                    continue

                rotated_rect = cv2.minAreaRect(component_points.astype(np.float32))
                defect_center = rotated_rect[0]
                if is_near_boundary_numba(defect_center[0], defect_center[1], w, h, boundary_threshold):
                    continue

                (w_px, h_px) = rotated_rect[1]
                length_px, width_px = (w_px, h_px) if w_px >= h_px else (h_px, w_px)
                length_mm, width_mm = length_px / ppmm, width_px / ppmm
                if length_mm < d_cfg.get('MIN_DEFECT_DIMENSION_MM', 2.0) or width_mm < d_cfg.get('MIN_DEFECT_WIDTH_MM', 2.0):
                    continue

                rect_angle = rotated_rect[2]
                defect_long_axis_angle = rect_angle if w_px >= h_px else rect_angle + 90

                simplified_vertices_float = vertices_f32.astype(np.float64, copy=False)
                local_edge_angle, _ = find_closest_edge_numba(simplified_vertices_float, defect_center[0], defect_center[1])

                angle_diff = calculate_angle_difference_numba(
                    normalize_angle_numba(defect_long_axis_angle),
                    normalize_angle_numba(local_edge_angle)
                )

                parallel_threshold = d_cfg.get('ORIENTATION_PARALLEL_THRESHOLD', 20.0)
                perpendicular_threshold = 90.0 - parallel_threshold

                defect_type = "B" if angle_diff <= parallel_threshold else ("L" if angle_diff >= perpendicular_threshold else "B")

                box = cv2.boxPoints(rotated_rect)
                box = np.intp(box)
                cv2.drawContours(roi_color, [box], 0, DEFECT_COLORS.get(defect_type, (255,255,255)), 2)

                roi_report["defects"].append({
                    "type": defect_type,
                    "location": { "x": int(defect_center[0] + x), "y": int(defect_center[1] + y),
                                  "length_mm": round(length_mm, 2), "width_mm": round(width_mm, 2) }
                })

    if debug_trace:
        try:
            roi_report['debug'] = {
                'canny_points': canny_points_count,
                'cleaned_points': cleaned_points_count,
                'contours_found': int(len(contours) if contours is not None else 0),
                'valid_polygons': int(len(valid_polygons_raw)),
                'simplified_polygons': int(len(final_simplified_polygons)),
            }
        except Exception:
            pass
    return roi_report, roi_color

def process_image_from_memory_serial(image_gray, template_rois, config, draw_contours=False):
    """
    (新增) 按顺序串行处理图像中的所有ROI。
    """
    final_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    report = {"image_status": "OK", "defects": [], "rois": []}
    results = []

    # 使用简单的 for 循环代替线程池
    H, W = image_gray.shape[:2]
    for i, r in enumerate(template_rois):
        try:
            result = process_roi_with_defect_detection(i, r, image_gray, config, draw_contours)
        except Exception:
            x = int(r.get('x', 0)); y = int(r.get('y', 0))
            w = int(r.get('width', 0)); h = int(r.get('height', 0))
            x = max(0, min(x, max(0, W - 1))); y = max(0, min(y, max(0, H - 1)))
            w = max(0, min(w, max(0, W - x))); h = max(0, min(h, max(0, H - y)))
            safe_roi = image_gray[y:y+h, x:x+w] if (w > 0 and h > 0) else np.zeros((1,1), dtype=np.uint8)
            try:
                roi_bgr = cv2.cvtColor(safe_roi, cv2.COLOR_GRAY2BGR)
            except Exception:
                roi_bgr = np.zeros((max(1,h), max(1,w), 3), dtype=np.uint8)
            result = ({"roi_idx": i, "x": x, "y": y, "w": w, "h": h, "polygons_in_roi": 0, "defects": []}, roi_bgr)
        results.append(result)

    # 后续的结果聚合逻辑与并行版本完全相同
    max_polygons_in_any_roi = 0
    for roi_report, roi_color in results:
        if "x" not in roi_report: continue
        x, y, w, h = roi_report["x"], roi_report["y"], roi_report["w"], roi_report["h"]
        if w > 0 and h > 0: final_image[y:y+h, x:x+w] = roi_color
        max_polygons_in_any_roi = max(max_polygons_in_any_roi, roi_report.get("polygons_in_roi", 0))
        if roi_report.get("defects"):
            report["image_status"] = "NG"
            report["defects"].extend(roi_report["defects"])
        slim = {k: roi_report.get(k) for k in ("roi_idx","x","y","w","h","polygons_in_roi")}
        if 'debug' in roi_report:
            slim['debug'] = roi_report['debug']
        report['rois'].append(slim)
    report["state_code"] = min(max_polygons_in_any_roi, 2)
    print(report["state_code"])
    return report, final_image

def process_image_from_memory_parallel(image_gray, template_rois, config, draw_contours=True):
    # (新) 增加了 draw_contours 参数传递
    final_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    report = {"image_status": "OK", "defects": [] , "state_code": 0, "rois": []}

    sys_params = config.get('system_params', {})
    # 新参数 roi_threads 控制每帧 ROI 并行度；旧 max_roi_workers 兼容作为回退
    try:
        roi_threads_cfg = int(sys_params.get('roi_threads', 0) or 0)
    except Exception:
        roi_threads_cfg = 0
    if roi_threads_cfg <= 0:
        try:
            roi_threads_cfg = int(sys_params.get('max_roi_workers', 0) or 0)
        except Exception:
            roi_threads_cfg = 0
    cpu_workers = multiprocessing.cpu_count()
    auto_workers = min(cpu_workers, len(template_rois))
    num_workers = auto_workers if roi_threads_cfg <= 0 else max(1, min(roi_threads_cfg, len(template_rois)))
    if roi_threads_cfg > 0 and roi_threads_cfg != num_workers:
        # 被 ROI 数裁剪
        pass
    
    H, W = image_gray.shape[:2]

    def _safe_roi(i, r):
        try:
            return process_roi_with_defect_detection(i, r, image_gray, config, draw_contours)
        except Exception:
            # 安全回退：该 ROI 失败则返回“原始ROI的BGR拷贝”，避免黑屏遮盖
            x = int(r.get('x', 0)); y = int(r.get('y', 0))
            w = int(r.get('width', 0)); h = int(r.get('height', 0))
            x = max(0, min(x, max(0, W - 1))); y = max(0, min(y, max(0, H - 1)))
            w = max(0, min(w, max(0, W - x))); h = max(0, min(h, max(0, H - y)))
            safe_roi = image_gray[y:y+h, x:x+w] if (w > 0 and h > 0) else np.zeros((1,1), dtype=np.uint8)
            try:
                roi_bgr = cv2.cvtColor(safe_roi, cv2.COLOR_GRAY2BGR)
            except Exception:
                roi_bgr = np.zeros((max(1,h), max(1,w), 3), dtype=np.uint8)
            return ({"roi_idx": i, "x": x, "y": y, "w": w, "h": h,
                     "polygons_in_roi": 0, "defects": []}, roi_bgr)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_safe_roi, i, r) for i, r in enumerate(template_rois)]
        results = [future.result() for future in futures]

    max_polygons_in_any_roi = 0
    for roi_report, roi_color in results:
        if "x" not in roi_report: continue
        x, y, w, h = roi_report["x"], roi_report["y"], roi_report["w"], roi_report["h"]
        if w > 0 and h > 0: final_image[y:y+h, x:x+w] = roi_color
        max_polygons_in_any_roi = max(max_polygons_in_any_roi, roi_report.get("polygons_in_roi", 0))
        if roi_report.get("defects"):
            report["image_status"] = "NG"
            report["defects"].extend(roi_report["defects"])
        slim = {k: roi_report.get(k) for k in ("roi_idx","x","y","w","h","polygons_in_roi")}
        if 'debug' in roi_report:
            slim['debug'] = roi_report['debug']
        report['rois'].append(slim)

    report["state_code"] = min(max_polygons_in_any_roi, 2)
    return report, final_image

def process_image(image_path, template_rois, output_folder, config, draw_contours=False, use_parallel=True):
    """
    (已修改) 顶层处理函数，可选择并行或串行处理。
    """
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None: 
        print(f"警告: 无法读取图像 {image_path}")
        return None

    # 根据 use_parallel 参数选择处理函数
    if use_parallel:
        report, final_image = process_image_from_memory_parallel(image_gray, template_rois, config, draw_contours)
    else:
        report, final_image = process_image_from_memory_serial(image_gray, template_rois, config, draw_contours)

    if output_folder:
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path = os.path.join(output_folder, f"{base_filename}_processed.jpg")
        output_json_path = os.path.join(output_folder, f"{base_filename}_report.json")
        cv2.imwrite(output_image_path, final_image)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
    return report