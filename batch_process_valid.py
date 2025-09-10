import os
import json
import cv2
import time
import argparse
import image_processor_optimized
import numpy as np

# 批处理 valid 文件夹下图片，复用现有算法/参数

def load_config(cfg_path: str = 'config.json'):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_template_rois(config, cam_idx: int = 0):
    camera_rois_cfg = config.get('camera_rois', {})
    roi_path = None
    if isinstance(camera_rois_cfg, dict):
        roi_path = camera_rois_cfg.get(str(cam_idx)) or camera_rois_cfg.get(cam_idx)
    elif isinstance(camera_rois_cfg, list) and cam_idx < len(camera_rois_cfg):
        roi_path = camera_rois_cfg[cam_idx]
    if not roi_path:
        roi_path = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
    try:
        with open(roi_path, 'r', encoding='utf-8') as f:
            averaged_data = json.load(f)
        # 取 source_image_count 最大的组
        best_group_key = max(averaged_data, key=lambda k: averaged_data[k].get('source_image_count', 0))
        rois = averaged_data[best_group_key]['averaged_rois']
        return rois
    except Exception as e:
        print(f"[批处理]: 加载ROI失败: {e}")
        return []

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def process_single_image(img_path, template_rois, config, out_dir, draw_contours=True, debug_intermediates=False):
    # 直接调用算法顶层接口，自动处理读图/输出
    t0 = time.time()
    report = image_processor_optimized.process_image(
        img_path,
        template_rois,
        out_dir,
        config,
        draw_contours=draw_contours,
        use_parallel=True
    )
    dt = (time.time() - t0) * 1000.0
    base = os.path.splitext(os.path.basename(img_path))[0]
    if report is not None:
        print(f"[完成]: {base} 状态={report.get('image_status')} 缺陷数={len(report.get('defects', []))} 耗时={dt:.1f}ms")
    else:
        print(f"[跳过]: 无法处理 {img_path}")

    if debug_intermediates:
        try:
            _export_intermediates(img_path, template_rois, config, out_dir)
        except Exception as e:
            print(f"[调试导出失败]: {base}: {e}")

def _export_intermediates(img_path, template_rois, config, out_dir):
    """生成边框连接结果 + 多边形拟合轮廓（亮度检测前阶段）。"""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"[调试]: 无法读取图像 {img_path}")
        return
    H, W = img_gray.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    inter_dir = os.path.join(out_dir, 'intermediates')
    os.makedirs(inter_dir, exist_ok=True)

    # 画布：边框补全 + 多边形拟合可视化
    border_canvas = np.zeros((H, W), dtype=np.uint8)
    poly_canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # 取配置片段
    p_cfg = config.get('defect_detection_params', {}).get('preprocess_params', {})
    dbscan_cfg = config.get('defect_detection_params', {}).get('dbscan_params', {})
    contour_cfg = config.get('defect_detection_params', {}).get('contour_params', {})

    # 函数引用（即便是下划线也可访问）
    preprocess_roi = image_processor_optimized.preprocess_roi_enhanced
    cluster_fn = image_processor_optimized.cluster_points_with_dbscan
    connect_fn = image_processor_optimized.connect_border_points_advanced
    simplify_fn = image_processor_optimized.find_best_fit_polygon
    direction_fn = image_processor_optimized._determine_connection_direction  # noqa: protected-access

    for r in template_rois:
        try:
            x = int(r.get('x', 0)); y = int(r.get('y', 0))
            w = int(r.get('width', 0)); h = int(r.get('height', 0))
            if w <= 0 or h <= 0: continue
            x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
            w = max(0, min(w, W - x)); h = max(0, min(h, H - y))
            roi_raw = img_gray[y:y+h, x:x+w]
            if roi_raw.size == 0: continue

            roi_pre = preprocess_roi(roi_raw, p_cfg)
            v = float(np.median(roi_pre))
            sigma = float(p_cfg.get('CANNY_SIGMA', 0.2))
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            canny_edges = cv2.Canny(roi_pre, lower, upper)
            # 不再单独导出 Canny，全局不存储

            # DBSCAN 清理
            cleaned_edges = np.zeros_like(canny_edges)
            pts_yx = np.argwhere(canny_edges == 255)
            if pts_yx.shape[0] >= dbscan_cfg.get('min_points_per_cluster', 100):
                max_points = int(dbscan_cfg.get('max_points', 8000))
                if pts_yx.shape[0] > max_points:
                    step = int(np.ceil(pts_yx.shape[0] / max_points))
                    pts_yx = pts_yx[::step]
                pts_xy = pts_yx[:, [1, 0]]
                clusters = cluster_fn(pts_xy, **dbscan_cfg)
                if clusters:
                    all_pts = np.vstack(clusters)
                    cleaned_edges[all_pts[:,1], all_pts[:,0]] = 255

            # 若清理后为空，降级使用原 canny
            edge_basis = cleaned_edges if cleaned_edges.any() else canny_edges
            direction = direction_fn(roi_raw)
            wireframe = connect_fn(edge_basis, direction=direction)
            border_canvas[y:y+h, x:x+w] = np.maximum(border_canvas[y:y+h, x:x+w], wireframe)
            # 轮廓 -> 过滤 -> 多边形拟合（在亮度检测前阶段停止）
            contours, _ = cv2.findContours(wireframe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                min_area = contour_cfg.get('min_area', 5000)
                kept = [c for c in contours if cv2.contourArea(c) > min_area]
                if not kept and edge_basis.any():
                    # 尝试回退一次：使用 cleaned 边缘集合在四边最远点连接（与主流程简化版）
                    h_loc, w_loc = edge_basis.shape[:2]
                    fb = np.zeros_like(edge_basis)
                    def _connect_line(points):
                        if len(points) < 2: return
                        cv2.line(fb, points[0], points[-1], 255, 1)
                    if edge_basis[0, :].any():
                        xs = np.where(edge_basis[0, :] == 255)[0]
                        _connect_line([(int(xs[0]),0),(int(xs[-1]),0)])
                    if edge_basis[h_loc-1, :].any():
                        xs = np.where(edge_basis[h_loc-1, :] == 255)[0]
                        _connect_line([(int(xs[0]),h_loc-1),(int(xs[-1]),h_loc-1)])
                    if edge_basis[:,0].any():
                        ys = np.where(edge_basis[:,0] == 255)[0]
                        _connect_line([(0,int(ys[0])),(0,int(ys[-1]))])
                    if edge_basis[:,w_loc-1].any():
                        ys = np.where(edge_basis[:,w_loc-1] == 255)[0]
                        _connect_line([(w_loc-1,int(ys[0])),(w_loc-1,int(ys[-1]))])
                    if fb.any():
                        c2,_ = cv2.findContours(fb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        kept = [c for c in c2 if cv2.contourArea(c) > min_area]
                        if kept:
                            border_canvas[y:y+h, x:x+w] = np.maximum(border_canvas[y:y+h, x:x+w], fb)
                for c in kept:
                    poly = simplify_fn(c)
                    if poly is not None and len(poly) >= 3:
                        poly_off = poly + np.array([[x, y]])
                        cv2.polylines(poly_canvas, [poly_off], True, (0,255,0), 1, lineType=cv2.LINE_AA)
        except Exception as _e:
            print(f"[调试-ROI失败]: {os.path.basename(img_path)} ROI跳过 {_e}")

    # 保存
    cv2.imwrite(os.path.join(inter_dir, f"{base}_border.png"), border_canvas)
    cv2.imwrite(os.path.join(inter_dir, f"{base}_poly.png"), poly_canvas)
    print(f"[调试输出]: {base} -> border / poly 已生成")


def main():
    parser = argparse.ArgumentParser(description='批量处理 valid 目录下图片')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--input', default='valid', help='输入图片目录')
    parser.add_argument('--output', default='valid_results', help='输出结果目录')
    parser.add_argument('--ext', nargs='*', default=['.jpg', '.png', '.bmp'], help='允许的扩展名')
    parser.add_argument('--draw-contours', action='store_true', help='在最终结果图中绘制原始轮廓')
    parser.add_argument('--debug-intermediates', action='store_true', help='输出边框补全二值结果')
    args = parser.parse_args()

    cfg = load_config(args.config)
    template_rois = load_template_rois(cfg, cam_idx=0)
    if not template_rois:
        print('[警告]: 未加载到任何ROI，将继续处理但结果可能全部为OK。')

    ensure_dir(args.output)

    # 收集文件
    exts = set([e.lower() for e in args.ext])
    all_files = []
    for name in os.listdir(args.input):
        p = os.path.join(args.input, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            all_files.append(p)
    if not all_files:
        print('[提示]: 未找到匹配图片。')
        return

    print(f'[批处理]: 发现 {len(all_files)} 张图片，开始处理...')
    for fp in all_files:
        process_single_image(fp, template_rois, cfg, args.output,
                              draw_contours=args.draw_contours,
                              debug_intermediates=args.debug_intermediates)

if __name__ == '__main__':
    main()
