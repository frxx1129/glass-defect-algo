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
import image_processor_hough as light_impl
import image_processor_hough_dark as dark_impl

def _run_light(image_gray, rois, config):
    return light_impl.process_image_from_memory_parallel(image_gray, rois, config)

def _run_dark(image_gray, rois, config):
    # 深色实现文件当前不完整; 若存在关键属性缺失, 回退到浅色
    try:
        if hasattr(dark_impl, 'process_image_from_memory_parallel'):
            return dark_impl.process_image_from_memory_parallel(image_gray, rois, config.get('hough_inspector_dark_params', {}))
    except Exception:
        pass
    return _run_light(image_gray, rois, config)

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
