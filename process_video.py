# process_video.py (Updated Version)

import cv2
import json
import os
import argparse
from tqdm import tqdm

# From your algorithm file, import the core processing function
# Ensure image_processor_optimized.py is in the same folder
try:
    from image_processor_optimized import process_image_from_memory_parallel, process_image_from_memory_serial
except ImportError:
    print("错误: 无法找到 'image_processor_optimized.py'。")
    print("请确保该文件与本脚本在同一个目录下。")
    exit()

def load_rois_from_file(roi_file_path):
    """
    Loads ROIs from the specific JSON structure provided by the user.
    It looks for the 'averaged_rois' key within the nested dictionary.
    """
    try:
        with open(roi_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # The algorithm expects a list of ROI dictionaries.
        # We need to extract this list from the loaded JSON structure.
        rois_list = []
        if isinstance(data, dict):
            # Iterate through the top-level keys (e.g., "group_with_2_rois")
            for group_key in data:
                group_data = data[group_key]
                # Check if the 'averaged_rois' key exists and is a list
                if isinstance(group_data, dict) and 'averaged_rois' in group_data:
                    rois_list.extend(group_data['averaged_rois'])
        
        if not rois_list:
            print(f"错误: 在ROI文件 '{roi_file_path}' 中未能找到有效的ROI列表。")
            print("请确保文件包含一个名为 'averaged_rois' 的键，其值为一个ROI列表。")
            return None
            
        return rois_list

    except json.JSONDecodeError:
        print(f"错误: 无法解析ROI文件: {roi_file_path}。请检查是否为有效的JSON格式。")
        return None
    except Exception as e:
        print(f"读取ROI文件时发生未知错误: {e}")
        return None

def process_video(video_path, roi_file_path, output_path, use_parallel=True):
    """
    Uses the specified ROI and algorithm to process an input video and save the output.
    """
    # --- 1. Validate input files ---
    if not os.path.exists(video_path):
        print(f"错误: 输入视频文件未找到: {video_path}")
        return
    if not os.path.exists(roi_file_path):
        print(f"错误: ROI文件未找到: {roi_file_path}")
        return

    # --- 2. Load ROIs and Config ---
    rois = load_rois_from_file(roi_file_path)
    if rois is None:
        return # Stop execution if ROIs could not be loaded

    # Define an empty config dictionary. Your algorithm will use its internal defaults.
    config = {}

    # --- 3. Initialize Video Reader and Writer ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter object for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"开始处理视频: {video_path}")
    print(f" - 分辨率: {frame_width}x{frame_height}, 帧率: {fps:.2f}, 总帧数: {total_frames}")
    print(f" - ROI配置: {roi_file_path} (加载了 {len(rois)} 个ROI)")
    print(f" - 输出文件: {output_path}")

    # --- 4. Process Video Frame by Frame ---
    with tqdm(total=total_frames, desc="处理中") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Your algorithm requires a grayscale image as input
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Choose processing function (parallel or serial)
            process_func = process_image_from_memory_parallel if use_parallel else process_image_from_memory_serial
            
            report, processed_frame = process_func(
                image_gray=gray_frame,
                template_rois=rois,
                config=config,
                draw_contours=True  # Set to True to draw detection results in the video
            )
            
            # Write the processed frame to the output video
            out.write(processed_frame)
            
            # Update progress bar
            pbar.update(1)

    # --- 5. Release Resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n处理完成！视频已保存至: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用自定义算法处理视频文件。")
    parser.add_argument(
        '--video-in', 
        type=str, 
        required=True, 
        help="输入视频文件的路径。"
    )
    parser.add_argument(
        '--roi-file', 
        type=str, 
        default='cam5_roi_averaged_by_group.json',
        help="包含ROI定义的JSON文件路径。默认为 'cam5_roi_averaged_by_group.json'。"
    )
    parser.add_argument(
        '--video-out', 
        type=str, 
        default='output_processed.mp4',
        help="处理后输出的视频文件路径。默认为 'output_processed.mp4'。"
    )
    parser.add_argument(
        '--serial',
        action='store_true',
        help="使用串行模式处理。如果未指定，则默认使用更快的并行模式。"
    )

    args = parser.parse_args()

    use_parallel_processing = not args.serial

    process_video(
        video_path=args.video_in,
        roi_file_path=args.roi_file,
        output_path=args.video_out,
        use_parallel=use_parallel_processing
    )