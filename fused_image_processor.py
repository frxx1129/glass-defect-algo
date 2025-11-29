"""统一调度浅色(算法模式1) 与 深色(算法模式2) 玻璃检测逻辑的适配层。

外部仅调用:
    process_image(image_gray, rois, full_config, mode)

mode: 1=浅色 => 使用 full_config['hough_inspector_params'] + 浅色流程
      2=深色 => 使用 full_config['hough_inspector_dark_params'] + 深色流程(若缺失则回退1)

保证接口稳定，内部捕获异常返回 (report, image_bgr)，避免 worker 崩溃。
"""
from __future__ import annotations
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import List, Tuple, Dict, Any

# 直接复用现有两个实现模块(浅色/深色), 若深色模块结构不完整则做最小安全封装
import copy
import image_processor_hough as light_impl
import image_processor_hough_dark as dark_impl

def _run_light(image_gray, rois, full_config):
    # 仅使用浅色参数集
    params = full_config.get('hough_inspector_params', full_config)
    return light_impl.process_image_from_memory_parallel(image_gray, rois, params)


def _scale_numeric(value, scale):
    try:
        if isinstance(value, (int, float)):
            return type(value)(value * scale)
        if isinstance(value, (list, tuple)):
            return type(value)([_scale_numeric(v, scale) for v in value])
    except Exception:
        pass
    return value


def _adjust_params_for_dark(params: Dict[str, Any]) -> Dict[str, Any]:

    p = copy.deepcopy(params) if isinstance(params, dict) else {}

    # 1) Canny 阈值提升至 36/96（严格键名）
    try:
        pre = p.setdefault('PREPROCESSING', {})
        pre['CANNY_THRESHOLD_LOW'] = 36
        pre['CANNY_THRESHOLD_HIGH'] = 96
    except Exception:
        pass

    # 2) 获取 DEFECT_DETECTION
    dd = p.setdefault('DEFECT_DETECTION', {})

    # 2.1 缺角/Chipping 过滤参数 ×0.25（仅保留算法中仍使用的键）
    q_keys = [
        'Q_CORNER_CONTOUR_MIN_DIST_PX',
        'Q_CANNY_STRIPE_HALF_WIDTH_PX',
        'Q_CORNER_ALIGNMENT_TOL_PX',
        'Q_TRIANGLE_MIN_AREA_MM2',
        # 平行四边形排除/聚类过滤相关：
        'Q_PARALLELOGRAM_EXCLUDE_STRIPE_HALF_PX',
        'Q_PARALLELOGRAM_USE_DILATE',
        'Q_PARALLELOGRAM_MIN_EDGE_PIXELS',
        'Q_PARALLELOGRAM_MIN_SPAN_FRAC',
    ]
    for k in q_keys:
        if k in dd and isinstance(dd[k], (int, float)):
            dd[k] = _scale_numeric(dd[k], 0.25)

    # 2.2 崩边(B)过滤参数 ×1.5（仅保留算法中仍使用的键；不含 B->L 重分类）
    b_keys = [
        'B_MAX_DISTANCE_TO_EDGE_MM',
        'B_FILTER_PARALLEL_TOLERANCE_DEG',
        'B_FILTER_PARALLEL_AR_MIN',
        'B_FILTER_PARALLEL_MIN_SIDE_MM',
    ]
    for k in b_keys:
        if k in dd and isinstance(dd[k], (int, float)):
            dd[k] = _scale_numeric(dd[k], 1.5)

    # 2.3 不调整 B->L 重分类参数（按要求不考虑该类参数）

    p['DEFECT_DETECTION'] = dd
    return p

def _run_dark(image_gray, rois, full_config):
    """深色玻璃：仅基于浅色参数集做运行时调整，不再读取独立 dark 参数。"""
    try:
        base_params = full_config.get('hough_inspector_params', full_config)
        dark_params = _adjust_params_for_dark(base_params)
        return light_impl.process_image_from_memory_parallel(image_gray, rois, dark_params)
    except Exception as e:
        print(f"[fused_image_processor] 深色模式执行异常, 回退浅色: {e}")
        return _run_light(image_gray, rois, full_config)

def process_image(image_gray, rois, full_config: Dict[str, Any], mode: int):
    """统一入口.
    image_gray: np.ndarray 灰度图
    rois: list ROI 配置
    full_config: 全量 config.json 解析后的 dict
    mode: 1(浅) / 2(深)
    """
    try:
        if mode == 2:
            return _run_dark(image_gray, rois, full_config)
        return _run_light(image_gray, rois, full_config)
    except Exception as e:
        # 兜底: 返回空报告，避免 worker 中断
        h, w = image_gray.shape[:2]
        empty = np.zeros((h, w, 3), dtype=np.uint8)
        report = {"image_status": "OK", "defects": [], "state_code": 0, "rois": []}
        report["state_code"] = 0
        return report, empty

# =============================================================
# ROI 图像保存辅助
# =============================================================
def save_roi_crops(image_gray: np.ndarray, annotated_bgr: np.ndarray, report: Dict[str, Any], rois: list,
                   output_root: str, cam_idx: int, frame_ts: float, save_original: bool, save_ng: bool,
                   per_pane_folder: bool = False):
    """保存 ROI 裁剪：
    - 原始灰度 ROI (save_original=True 且 state_code>0 认为有玻璃) 目录: original/
    - 标注 NG ROI (save_ng=True 且 image_status=NG) 目录: ng/
    文件命名: cam{cam_idx}_ts{ms}_{roi_idx}.jpg
    只在 rois 与 report['rois'] 对齐时使用。"""
    try:
        import os, cv2, time
        if not rois or 'rois' not in report:
            return
        millis = int(frame_ts * 1000)
        day_dir = time.strftime('%Y%m%d', time.localtime(frame_ts))
        # 如果按玻璃单独目录:  base/日期/pane_cam{idx}_{millis}/{original,ng}
        if per_pane_folder:
            pane_root = os.path.join(output_root, day_dir, f"pane_cam{cam_idx}_{millis}")
            orig_dir = os.path.join(pane_root, 'original')
            ng_dir = os.path.join(pane_root, 'ng')
        else:
            base_dir = os.path.join(output_root, day_dir)
            orig_dir = os.path.join(base_dir, 'original')
            ng_dir = os.path.join(base_dir, 'ng')
        if save_original:
            os.makedirs(orig_dir, exist_ok=True)
        if save_ng and report.get('image_status') == 'NG':
            os.makedirs(ng_dir, exist_ok=True)
        # 遍历 ROI
        for roi_idx, roi in enumerate(rois):
            try:
                x = int(roi.get('x',0)); y = int(roi.get('y',0))
                w = int(roi.get('width',0)); h = int(roi.get('height',0))
                if w <=0 or h<=0: continue
                crop_gray = image_gray[y:y+h, x:x+w]
                if crop_gray is None or crop_gray.size == 0: continue
                filename = f"cam{cam_idx}_ts{millis}_roi{roi_idx}.jpg"
                if save_original and report.get('state_code',0) > 0:
                    cv2.imwrite(os.path.join(orig_dir, filename), crop_gray)
                if save_ng and report.get('image_status') == 'NG':
                    crop_anno = annotated_bgr[y:y+h, x:x+w]
                    if crop_anno is not None and crop_anno.size>0:
                        cv2.imwrite(os.path.join(ng_dir, filename), crop_anno)
            except Exception:
                continue
    except Exception:
        pass
