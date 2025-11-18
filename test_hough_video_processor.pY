import cv2
import json
import yaml
import time
import os
import argparse
import math
from datetime import datetime
import base64
import numpy as np
# 假设 image_processor_hough.py 在同一目录下
import image_processor_hough as image_processor

# --- 模拟 SharedSettings 类 (来自 glass_inspector 项目，简化用于测试) ---
class MockSharedSettings:
    def __init__(self):
        self.max_defect_size_mm = 20 # 示例值，实际应从配置读取

# --- 模拟 should_reject_pane 函数 (来自 processing_worker.py，简化用于测试) ---
def should_reject_pane(pane_json, shared_settings, pixels_per_mm):
    """
    模拟 glass_inspector 中用于判断是否拒绝玻璃板的逻辑。
    这里简化为只要有缺陷就拒绝。
    """
    if pane_json['image_status'] == 'OK':
        return False
    # 只要检测到任何缺陷，就视为需要拒绝
    if pane_json.get('defects', []):
        return True
    return False

# --- 模拟工作进程的实际处理逻辑（整合到主脚本，不创建单独的进程） ---
# 在实际生产环境中，这会是 processing_worker.py 的核心逻辑
def process_frame_task(frame_data_gray, cam_index, config, shared_settings, roi_cache):
    """
    处理单帧图像，模拟 processing_worker.py 的行为。
    """
    PIXELS_PER_MM = config['system_params']['pixels_per_mm']

    if cam_index not in roi_cache:
        # 实际项目中，ROI加载在worker启动时完成
        # 这里为测试方便，如果未加载，则尝试加载
        roi_path = None
        camera_rois_cfg = config.get('camera_rois', {})
        if isinstance(camera_rois_cfg, dict):
            roi_path = camera_rois_cfg.get(str(cam_index)) or camera_rois_cfg.get(cam_index)
        elif isinstance(camera_rois_cfg, list) and cam_index < len(camera_rois_cfg):
            roi_path = camera_rois_cfg[cam_index]
        if not roi_path:
            roi_path = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
        
        try:
            with open(roi_path, 'r', encoding='utf-8') as f:
                averaged_data = json.load(f)
            # 假设我们总是选择包含最多源图像的ROI组
            best_group_key = max(averaged_data, key=lambda k: averaged_data[k].get('source_image_count', 0))
            roi_cache[cam_index] = averaged_data[best_group_key]['averaged_rois']
            print(f"Cam{cam_index} 载入ROI: '{roi_path}', 组 '{best_group_key}'")
        except Exception as e:
            print(f"Cam{cam_index} 加载ROI失败: {e}")
            roi_cache[cam_index] = [] # 加载失败，ROI为空列表

    pane_json, annotated_image = image_processor.process_image_from_memory_parallel(
        frame_data_gray, roi_cache[cam_index], config)

    should_reject_overall = should_reject_pane(pane_json, shared_settings, PIXELS_PER_MM)

    result = {
        "camera_index": cam_index,
        "timestamp": datetime.now().isoformat(),
        "image_status": pane_json.get('image_status', 'OK'),
        "defects": pane_json.get('defects', []),
        "state_code": pane_json.get('state_code', 0),
        "should_reject": should_reject_overall,
    }
    # --- FIX: 返回原始标注图像 (numpy array) 和结果字典 ---
    return result, annotated_image

# （已移除 Canny 图片导出逻辑）

def main():
    # --- 使用 argparse 处理命令行参数 ---
    parser = argparse.ArgumentParser(description="处理视频文件或图片文件夹中的玻璃缺陷。")
    parser.add_argument('-i', '--input', required=True, help="输入视频文件或图片文件夹的路径。")
    parser.add_argument('-o', '--output', default='hough_results', help="保存结果的输出文件夹路径。")
    default_cfg = 'config.yaml' if os.path.exists('config.yaml') else 'config.json'
    parser.add_argument('--config', default=default_cfg, help="配置文件的路径（支持 YAML/JSON，默认优先读取 config.yaml）。")
    parser.add_argument('--output-fps', type=float, default=None, help='覆盖输出视频的帧率；不指定则尽量使用输入视频FPS，失败回退到30.0')
    args = parser.parse_args()

    # 1. 规范化输入路径（去掉可能的反引号/引号/空格），然后检查是否存在
    input_path = args.input.strip().strip('`"\'')
    if not os.path.exists(input_path):
        print(f"错误: 输入路径 '{args.input}' 不存在。")
        return
        
    os.makedirs(args.output, exist_ok=True)

    # 2. 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            if args.config.endswith(('.yml', '.yaml')):
                config = yaml.safe_load(f)
            else:
                try:
                    config = yaml.safe_load(f)  # YAML 解析器可兼容 JSON 子集
                except Exception:
                    f.seek(0)
                    config = json.load(f)
        print(f"成功加载配置: {args.config}")
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        return
    except Exception as e:
        print(f"错误: 解析配置文件 '{args.config}' 失败: {e}")
        return

    # 3. 模拟 SharedSettings 和 ROI 缓存
    shared_settings = MockSharedSettings()
    roi_cache = {} 
    
    # 模拟单路相机，相机索引为0
    cam_index = 0 
    
    start_time = time.time()
    total_items = 0

    # --- 根据输入路径是文件夹还是文件，执行不同逻辑 ---
    if os.path.isdir(input_path):
        print(f"\n--- 开始处理文件夹: {args.input} ---")
        
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = sorted([f for f in os.listdir(args.input) if f.lower().endswith(supported_extensions)])
        
        if not image_files:
            print("错误: 在指定文件夹中未找到支持的图片文件。")
            return
        
        total_items = len(image_files)

        for i, filename in enumerate(image_files):
            print(f"处理图片 {i+1}/{total_items}: {filename}...")
            image_path = os.path.join(args.input, filename)
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"  警告: 无法读取图片 {filename}, 跳过。")
                continue
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 注意：此处返回的 annotated_image 是用于保存的，不是用于视频流
            result, annotated_image = process_frame_task(gray_frame, cam_index, config, shared_settings, roi_cache)

            # 不再输出 Canny 图片
            
            print(f"  报告: 状态={result['image_status']}, state_code={result['state_code']}, 缺陷数={len(result['defects'])}")
            
            base_filename = os.path.splitext(filename)[0]
            output_image_path = os.path.join(args.output, f"{base_filename}_annotated.jpg")
            report_output_path = os.path.join(args.output, f"{base_filename}_report.json")
            
            # 保存结果
            cv2.imwrite(output_image_path, annotated_image)
            with open(report_output_path, 'w', encoding='utf-8') as f: json.dump(result, f, indent=2, ensure_ascii=False)

        # 文件夹模式下无需视频写入，直接完成

    elif os.path.isfile(input_path):
        # 解析扩展名，区分“单张图片”与“视频文件”
        base_filename, input_ext = os.path.splitext(os.path.basename(input_path))
        input_ext = input_ext.strip().strip('`"\'')
        ext_lower = input_ext.lower()
        image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        if ext_lower in image_exts:
            # --- 单张图片模式：直接按原格式保存标注结果 ---
            print(f"\n--- 开始处理图片: {args.input} ---")
            frame = cv2.imread(input_path)
            if frame is None:
                print(f"错误: 无法读取图片 '{args.input}'。")
                return
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result, annotated_image = process_frame_task(gray_frame, cam_index, config, shared_settings, roi_cache)

            print(f"  报告: 状态={result['image_status']}, state_code={result['state_code']}, 缺陷数={len(result['defects'])}")

            # 输出文件名：原名 + _annotated + 原扩展名
            out_base = base_filename
            output_image_path = os.path.join(args.output, f"{out_base}_annotated{input_ext}")
            report_output_path = os.path.join(args.output, f"{out_base}_report.json")

            cv2.imwrite(output_image_path, annotated_image)
            with open(report_output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            total_items = 1

        else:
            print(f"\n--- 开始处理视频: {args.input} ---")
            # 预先解析扩展名以判断是否为 WebM，并尽量使用 FFMPEG 后端以提升兼容性
            is_webm = (ext_lower == '.webm') or base_filename.lower().endswith('.webm')

            if is_webm:
                cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    # 回退到默认后端再试一次
                    cap = cv2.VideoCapture(input_path)
            else:
                cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 '{args.input}'。")
            return

        # 先读取第一帧以确定尺寸（WebM 上部分属性可能为 0）
        first_ok, first_frame = cap.read()
        if not first_ok or first_frame is None:
            print("错误: 读取视频第一帧失败，无法确定尺寸。")
            cap.release()
            return
        height, width = first_frame.shape[:2]

        # 目标 FPS：优先命令行；否则尽量沿用输入FPS；失败回退到 30.0
        if args.output_fps and args.output_fps > 0:
            fps = float(args.output_fps)
            fps = max(1.0, min(120.0, fps))
            print(f"指定输出FPS: {fps}")
        else:
            in_fps = cap.get(cv2.CAP_PROP_FPS)
            try:
                if not in_fps or in_fps <= 0 or (isinstance(in_fps, float) and math.isnan(in_fps)):
                    fps = 30.0
                    print("输入FPS不可用，回退到 30.0")
                else:
                    # 限定合理范围，避免异常值
                    fps = max(1.0, min(120.0, float(in_fps)))
                    print(f"沿用输入FPS: {in_fps} -> 输出 {fps}")
            except Exception:
                fps = 30.0
                print("读取输入FPS异常，回退到 30.0")

        # --- 设置视频编写器 (VideoWriter) ---
        # 按你的需求，强制输出为 AVI 容器，优先 MJPG 编码，失败再尝试 XVID/DIVX
        fourcc_candidates = [
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
            ('DIVX', '.avi'),
        ]
        out_video = None
        chosen = None
        for cc, ext in fourcc_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*cc)
                output_video_path = os.path.join(args.output, f"{base_filename}_annotated{ext}")
                out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                if out_video.isOpened():
                    chosen = (cc, ext)
                    break
            except Exception:
                pass
        if not out_video.isOpened():
            print("错误: 无法创建可写的 AVI 视频文件（尝试 MJPG/XVID/DIVX 失败）。")
            cap.release()
            return
        print(f"输出视频将保存至: {output_video_path} (编码器: {chosen[0] if chosen else '未知'})")

        # 某些容器（尤其是 WebM）上，CAP_PROP_FRAME_COUNT 可能严重不准。
        cap_reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        show_total = (cap_reported_total > 0 and not is_webm)
        if show_total:
            total_items = cap_reported_total
        else:
            total_items = 0

        # 处理第一帧（已读取）
        frame_count = 1
        print(f"处理帧 {frame_count}/{cap_reported_total}..." if show_total else f"处理帧 {frame_count}...")
        gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        result, annotated_image = process_frame_task(gray_frame, cam_index, config, shared_settings, roi_cache)
        if annotated_image is None:
            annotated_image = first_frame
        if annotated_image.ndim == 2:
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
        h2, w2 = annotated_image.shape[:2]
        if (w2 != width) or (h2 != height):
            annotated_image = cv2.resize(annotated_image, (width, height), interpolation=cv2.INTER_AREA)
        out_video.write(annotated_image)
        print(f"  报告: 状态={result['image_status']}, state_code={result['state_code']}, 缺陷数={len(result['defects'])}")
        report_output_path = os.path.join(args.output, f"frame_{frame_count:05d}_report.json")
        with open(report_output_path, 'w', encoding='utf-8') as f: json.dump(result, f, indent=2, ensure_ascii=False)

        # 继续处理剩余帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"处理帧 {frame_count}/{cap_reported_total}..." if show_total else f"处理帧 {frame_count}...")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result, annotated_image = process_frame_task(gray_frame, cam_index, config, shared_settings, roi_cache)
            if annotated_image is None:
                annotated_image = frame
            if annotated_image.ndim == 2:
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
            h2, w2 = annotated_image.shape[:2]
            if (w2 != width) or (h2 != height):
                annotated_image = cv2.resize(annotated_image, (width, height), interpolation=cv2.INTER_AREA)
            out_video.write(annotated_image)
            print(f"  报告: 状态={result['image_status']}, state_code={result['state_code']}, 缺陷数={len(result['defects'])}")
            report_output_path = os.path.join(args.output, f"frame_{frame_count:05d}_report.json")
            with open(report_output_path, 'w', encoding='utf-8') as f: json.dump(result, f, indent=2, ensure_ascii=False)

        cap.release()
        out_video.release()
        total_items = frame_count
        print("\n视频处理完成。")
    
    else:
        print(f"错误: 输入路径 '{args.input}' 既不是文件也不是文件夹。")
        return

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- 整体处理完成 ---")
    print(f"总处理项目数: {total_items}")
    print(f"总耗时: {total_time:.2f} 秒")
    if total_items > 0:
        print(f"平均每项处理时间: {total_time / total_items:.4f} 秒")

if __name__ == "__main__":
    main()