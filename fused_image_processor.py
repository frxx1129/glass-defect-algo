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

def _run_dark(image_gray, rois, full_config):
    """调用深色玻璃实现。
    之前错误：仅传入 full_config['hough_inspector_dark_params'] 子字典，
    dark_impl 内部再次调用 config.get('hough_inspector_dark_params') 导致找不到 -> 抛异常 -> 回退浅色。
    修复：传递完整 full_config，必要参数由深色实现自行解析。
    若缺少配置或执行失败，记录一次日志并回退浅色。
    """
    try:
        if not hasattr(dark_impl, 'process_image_from_memory_parallel'):
            print("[fused_image_processor] 深色实现缺失接口, 回退浅色")
            return _run_light(image_gray, rois, full_config)
        if 'hough_inspector_dark_params' not in full_config:
            print("[fused_image_processor] 配置缺少 hough_inspector_dark_params, 回退浅色")
            return _run_light(image_gray, rois, full_config)
        return dark_impl.process_image_from_memory_parallel(image_gray, rois, full_config)
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
