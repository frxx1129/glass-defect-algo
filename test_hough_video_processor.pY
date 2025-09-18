import cv2
import json
import time
import os
import argparse
from datetime import datetime
import base64
import numpy as np

# 假设 image_processor_hough.py 在同一目录下
import image_processor_hough as image_processor

# --- 模拟 SharedSettings 类 (来自 glass_inspector 项目，简化用于测试) ---
class MockSharedSettings:
    def __init__(self):
        self.max_defect_size_mm = 20 # 示例值，实际应从 config.json 获取

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

    # 编码为主图
    jpg_q_main = int(config.get('system_params', {}).get('jpeg_quality_main', 85) or 85)
    success_original, original_buffer_encoded = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_main])
    
    # 编码为预览图
    PREVIEW_WIDTH = int(config.get('system_params', {}).get('preview_width', 800) or 800)
    height, width, _ = annotated_image.shape
    scale = PREVIEW_WIDTH / width
    preview_image = cv2.resize(annotated_image, (PREVIEW_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA)
    jpg_q_preview = int(config.get('system_params', {}).get('jpeg_quality_preview', 75) or 75)
    success_preview, preview_buffer_encoded = cv2.imencode('.jpg', preview_image, [cv2.IMWRITE_JPEG_QUALITY, jpg_q_preview])

    base64_image_string = base64.b64encode(preview_buffer_encoded if success_preview else original_buffer_encoded).decode('utf-8')

    result = {
        "camera_index": cam_index,
        "timestamp": datetime.now().isoformat(),
        "image": f"data:image/jpeg;base64,{base64_image_string}", # 预览图
        "image_status": pane_json.get('image_status', 'OK'),
        "defects": pane_json.get('defects', []),
        "state_code": pane_json.get('state_code', 0),
        "should_reject": should_reject_overall,
        "annotated_image_buffer_raw": original_buffer_encoded.tobytes(), # 原始尺寸标注图
    }
    return result

def main():
    # --- FIX: 使用 argparse 处理命令行参数 ---
    parser = argparse.ArgumentParser(description="处理视频文件或图片文件夹中的玻璃缺陷。")
    parser.add_argument('-i', '--input', required=True, help="输入视频文件或图片文件夹的路径。")
    parser.add_argument('-o', '--output', default='hough_results', help="保存结果的输出文件夹路径。")
    parser.add_argument('--config', default='config.json', help="配置文件的路径。")
    args = parser.parse_args()

    # 1. 检查输入路径是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入路径 '{args.input}' 不存在。")
        return
        
    os.makedirs(args.output, exist_ok=True)

    # 2. 加载配置文件
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"成功加载配置: {args.config}")
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 解析配置文件 '{args.config}' 失败: {e}")
        return

    # 3. 模拟 SharedSettings 和 ROI 缓存
    shared_settings = MockSharedSettings()
    roi_cache = {} 
    
    # 模拟单路相机，相机索引为0
    cam_index = 0 
    
    start_time = time.time()
    total_items = 0

    # --- FIX: 根据输入路径是文件夹还是文件，执行不同逻辑 ---
    if os.path.isdir(args.input):
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
            result = process_frame_task(gray_frame, cam_index, config, shared_settings, roi_cache)
            
            print(f"  报告: 状态={result['image_status']}, state_code={result['state_code']}, 缺陷数={len(result['defects'])}")
            
            base_filename = os.path.splitext(filename)[0]
            output_image_path = os.path.join(args.output, f"{base_filename}_annotated.jpg")
            report_output_path = os.path.join(args.output, f"{base_filename}_report.json")
            
            # 保存结果
            with open(output_image_path, 'wb') as f: f.write(result['annotated_image_buffer_raw'])
            report_to_save = result.copy()
            report_to_save.pop('image', None)
            report_to_save.pop('annotated_image_buffer_raw', None)
            with open(report_output_path, 'w', encoding='utf-8') as f: json.dump(report_to_save, f, indent=2, ensure_ascii=False)

    elif os.path.isfile(args.input):
        print(f"\n--- 开始处理视频: {args.input} ---")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 '{args.input}'。")
            return
        
        total_items = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频处理完成或读取失败。")
                break

            frame_count += 1
            print(f"处理帧 {frame_count}/{total_items}...")
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = process_frame_task(gray_frame, cam_index, config, shared_settings, roi_cache)
            
            print(f"  报告: 状态={result['image_status']}, state_code={result['state_code']}, 缺陷数={len(result['defects'])}")
            
            output_image_path = os.path.join(args.output, f"frame_{frame_count:05d}_annotated.jpg")
            report_output_path = os.path.join(args.output, f"frame_{frame_count:05d}_report.json")

            with open(output_image_path, 'wb') as f: f.write(result['annotated_image_buffer_raw'])
            report_to_save = result.copy()
            report_to_save.pop('image', None)
            report_to_save.pop('annotated_image_buffer_raw', None)
            with open(report_output_path, 'w', encoding='utf-8') as f: json.dump(report_to_save, f, indent=2, ensure_ascii=False)
        
        cap.release()
    
    else:
        print(f"错误: 输入路径 '{args.input}' 既不是文件也不是文件夹。")
        return

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- 处理完成 ---")
    print(f"总处理项目数: {total_items}")
    print(f"总耗时: {total_time:.2f} 秒")
    if total_items > 0:
        print(f"平均每项处理时间: {total_time / total_items:.4f} 秒")

if __name__ == "__main__":
    main()