import cv2
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 采集器封装为对主算法的薄层：按 algo_mode 选择明/暗参数，调用主实现的 ROI 处理函数
from image_processor_hough import process_roi_hough_based

def process_frame(image_gray, template_rois, config, thread_workers: int | None = None):
    """运行 Hough 检测并返回 (report, final_image, per_roi_images)。
    - 自动根据 system_params.algo_mode 选择 hough_inspector_params 或 hough_inspector_dark_params
    - 复用主实现 logic，确保明/暗一致
    """
    final_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    report = {"image_status": "OK", "defects": [] , "state_code": 0, "rois": []}

    sys_params = config.get('system_params', {})
    pixels_per_mm = float(sys_params.get('pixels_per_mm', 1.0))
    algo_mode = (sys_params.get('algo_mode') or 'bright').lower()

    if algo_mode == 'dark':
        hough_params = config.get('hough_inspector_dark_params')
        if not hough_params:
            raise ValueError("Configuration error: 'hough_inspector_dark_params' not found.")
    else:
        hough_params = config.get('hough_inspector_params')
        if not hough_params:
            raise ValueError("Configuration error: 'hough_inspector_params' not found.")

    # 线程数选择: 优先 thread_workers 其后 system_params.roi_threads, 再自动
    try:
        roi_threads_cfg = int(sys_params.get('roi_threads', 0) or 0)
    except Exception:
        roi_threads_cfg = 0
    cpu_workers = multiprocessing.cpu_count()
    auto_workers = min(cpu_workers, len(template_rois))
    num_workers = thread_workers or (auto_workers if roi_threads_cfg <= 0 else max(1, min(roi_threads_cfg, len(template_rois))))

    def _safe_roi(i, r):
        try:
            return process_roi_hough_based(i, r, image_gray, hough_params, pixels_per_mm)
        except Exception as e:
            print(f"[collector_hough] ROI {i} 处理异常: {e}")
            x, y = int(r.get('x',0)), int(r.get('y',0))
            w, h = int(r.get('width', r.get('w',0))), int(r.get('height', r.get('h',0)))
            roi_bgr = np.zeros((max(1,h), max(1,w), 3), dtype=np.uint8)
            try:
                roi_gray_crop = image_gray[y:y+h, x:x+w]
                if roi_gray_crop.size>0:
                    roi_bgr = cv2.cvtColor(roi_gray_crop, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass
            return ({"roi_idx": i, "x": x, "y": y, "w": w, "h": h, "defects": [], "edges_found": 0}, roi_bgr)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_safe_roi, i, r) for i, r in enumerate(template_rois)]
        results = [future.result() for future in futures]

    per_roi_images = []
    total_defects = 0
    for roi_report, roi_color in results:
        if "x" not in roi_report:
            continue
        x, y, w, h = roi_report["x"], roi_report["y"], roi_report["w"], roi_report["h"]
        if w > 0 and h > 0:
            final_image[y:y+h, x:x+w] = roi_color
        defects_list = roi_report.get("defects", [])
        if defects_list:
            report["image_status"] = "NG"
            report["defects"].extend(defects_list)
            total_defects += len(defects_list)
        slim_report = {k: roi_report.get(k) for k in ("roi_idx","x","y","w","h","edges_found")}
        report['rois'].append(slim_report)
        per_roi_images.append((roi_report['roi_idx'], roi_report, roi_color))

    report["state_code"] = 1 if total_defects > 0 else 0
    return report, final_image, per_roi_images

__all__ = ['process_frame']