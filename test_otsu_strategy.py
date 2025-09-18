import cv2
import numpy as np
import json
import os

# --- 配置区 ---
INPUT_IMAGE_FOLDER = 'valid'
ROI_FILE = 'dark_ROI.json'
OUTPUT_FOLDER = 'otsu_to_hough_results' # 新建输出文件夹

# Canny 和 Hough 的参数 (Canny不再使用, 保留Hough参数)
HOUGH_THRESHOLD = 50       # 霍夫变换阈值，可能需要调高一些，因为轮廓线很清晰
HOUGH_MIN_LINE_LENGTH = 50 # 最小线长，用于过滤掉短的噪声线
HOUGH_MAX_LINE_GAP = 20      # 最大线间距

# --- 脚本主逻辑 ---

def load_rois(file_path):
    """从JSON文件加载ROI定义"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 假设我们使用文件中找到的第一个ROI组
        first_key = next(iter(data))
        return data[first_key].get('averaged_rois', [])
    except Exception as e:
        print(f"错误: 无法从 {file_path} 加载或解析ROI: {e}")
        return None

def main():
    print("--- 开始测试 Otsu -> Hough 变换策略 ---")
    
    # 1. 准备工作
    rois = load_rois(ROI_FILE)
    if not rois:
        print("ROI文件加载失败，脚本终止。")
        return
        
    if not os.path.exists(INPUT_IMAGE_FOLDER):
        print(f"错误: 输入文件夹 '{INPUT_IMAGE_FOLDER}' 不存在。")
        return
        
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"结果将保存在: '{OUTPUT_FOLDER}'")

    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if f.lower().endswith('.bmp')]
    if not image_files:
        print(f"在 '{INPUT_IMAGE_FOLDER}' 中未找到 .bmp 图片。")
        return

    # 2. 循环处理每张图片
    for filename in image_files:
        try:
            image_path = os.path.join(INPUT_IMAGE_FOLDER, filename)
            print(f"\n处理图片: {filename}")
            
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_image is None:
                print("  -> 无法读取图片，跳过。")
                continue
            
            # 创建一个彩色副本用于绘制最终结果
            visualization_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # 3. 对每个ROI应用新策略
            for i, roi_def in enumerate(rois):
                x, y, w, h = [int(roi_def.get(k, 0)) for k in ('x', 'y', 'width', 'height')]
                
                # 裁剪ROI
                roi_gray = gray_image[y:y+h, x:x+w]
                
                # --- 核心策略实现 ---
                # 步骤 a: 使用Otsu算法创建玻璃区域的掩膜
                _, otsu_mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 步骤 b: 直接在Otsu掩膜上进行霍夫变换
                lines = cv2.HoughLinesP(
                    otsu_mask, # 直接使用Otsu的结果作为输入
                    rho=1, 
                    theta=np.pi / 180, 
                    threshold=HOUGH_THRESHOLD, 
                    minLineLength=HOUGH_MIN_LINE_LENGTH, 
                    maxLineGap=HOUGH_MAX_LINE_GAP
                )
                
                # --- 结果可视化 ---
                # 在彩色副本的对应ROI区域上绘制检测到的直线
                roi_color_to_draw_on = visualization_image[y:y+h, x:x+w]
                if lines is not None:
                    print(f"  -> 在 ROI #{i+1} 中检测到 {len(lines)} 条直线。")
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(roi_color_to_draw_on, (x1, y1), (x2, y2), (0, 255, 0), 2) # 用绿色粗线条绘制
                else:
                    print(f"  -> 在 ROI #{i+1} 中未检测到直线。")

            # 4. 保存结果图片
            output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_otsu_to_hough.jpg")
            cv2.imwrite(output_path, visualization_image)

        except Exception as e:
            print(f"处理图片 {filename} 时发生意外错误: {e}")
            
    print("\n--- 所有图片处理完成 ---")

if __name__ == '__main__':
    main()