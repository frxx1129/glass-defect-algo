import cv2
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 暗场算法改为对主算法的薄封装：所有几何/筛选逻辑复用 image_processor_hough
from image_processor_hough import process_roi_hough_based

def process_image_from_memory_parallel(image_gray, template_rois, config):
    """暗场检测：使用 hough_inspector_dark_params 作为参数，但算法实现复用明场主实现。
    返回 (report, final_image) 与明场一致。
    """
    final_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    report = {"image_status": "OK", "defects": [] , "state_code": 0, "rois": []}

    hough_params = config.get('hough_inspector_dark_params')
    if not hough_params:
        raise ValueError("Configuration error: 'hough_inspector_dark_params' section not found in the config file.")

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
            print(f"[dark] Error processing ROI {i}: {e}")
            x, y, w, h = int(r.get('x',0)), int(r.get('y',0)), int(r.get('width',0)), int(r.get('height',0))
            roi_bgr = np.zeros((max(1,h), max(1,w), 3), dtype=np.uint8)
            try:
                roi_gray_crop = image_gray[y:y+h, x:x+w]
                if roi_gray_crop.size>0:
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