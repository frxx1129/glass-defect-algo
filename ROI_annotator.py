# merged_roi_tool.py (v2 - 已修复窗口大小问题)
# 导入必要的库
import cv2
import numpy as np
import glob
import os
import json
from collections import defaultdict

# --- 全局变量 ---
ref_point_start = None
ref_point_end = None
cropping = False
rois_for_current_image = []
image_clone = None
scale_factor = 1.0  # ---【修改1：新增全局变量，用于存储图像缩放比例】---

# ---【修改2：新增常量，定义显示窗口的最大宽度】---
MAX_DISPLAY_WIDTH = 1280

def display_help_text(image):
    """在图像的左上角显示帮助文本。"""
    help_text = "Keys: (N)ext | (B)ack | (R)eset | (Q)uit & Process"
    cv2.putText(image, help_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return image

def draw_rois(image, rois):
    """在图像上绘制所有已选定的 ROI。"""
    for (x, y, w, h) in rois:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于处理鼠标事件以绘制矩形。"""
    global ref_point_start, ref_point_end, cropping, rois_for_current_image, scale_factor

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point_start = (x, y)
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        ref_point_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point_end = (x, y)
        cropping = False
        if ref_point_start and ref_point_end and ref_point_start != ref_point_end:
            x1, y1 = ref_point_start
            x2, y2 = ref_point_end
            start_x, start_y = min(x1, x2), min(y1, y2)
            end_x, end_y = max(x1, x2), max(y1, y2)

            # ---【修改3：坐标换算，将标注坐标乘以缩放比例，还原为原始坐标】---
            # 将在缩放后图像上绘制的坐标，等比例换算回原始图像的坐标
            original_x = int(start_x * scale_factor)
            original_y = int(start_y * scale_factor)
            original_w = int((end_x - start_x) * scale_factor)
            original_h = int((end_y - start_y) * scale_factor)
            
            roi = (original_x, original_y, original_w, original_h)
            rois_for_current_image.append(roi)
            
        ref_point_start = None
        ref_point_end = None

def run_annotator(image_folder_path='valid'):
    """
    启动用于在图像上标注 ROI 的图形用户界面。
    返回所有已标注的 ROI 数据。
    """
    global rois_for_current_image, image_clone, ref_point_start, ref_point_end, scale_factor

    if not os.path.isdir(image_folder_path):
        print(f"图像文件夹 '{image_folder_path}' 不存在。")
        os.makedirs(image_folder_path)
        print(f"已创建文件夹 '{image_folder_path}'。请将您的 .bmp 图像添加到该文件夹并重新运行脚本。")
        return None

    image_paths = sorted(glob.glob(os.path.join(image_folder_path, '*.bmp')))
    if not image_paths:
        print(f"警告: 在文件夹 '{image_folder_path}' 中未找到 .bmp 图像。")
        return None

    print("--- ROI 标注工具 ---")
    print("操作指南:")
    print(" - 按住鼠标左键并拖动以绘制一个矩形的 ROI。")
    print(" - 您可以在一张图像上绘制多个 ROI。")
    print(" - 请参考图像窗口中的快捷键提示。")
    print("-" * 30)

    # ---【修改4：让窗口可以被用户自由调整大小】---
    cv2.namedWindow("ROI Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Annotator", mouse_callback)

    all_rois_data = []
    annotated_filenames = set()

    index = 0
    while 0 <= index < len(image_paths):
        image_path = image_paths[index]
        image_filename = os.path.basename(image_path)
        
        original_image = cv2.imread(image_path) # 读取原始图像
        if original_image is None:
            print(f"警告: 无法读取图像 {image_path}，已跳过。")
            index += 1
            continue
        
        # ---【修改5：图像自动缩放逻辑】---
        original_h, original_w = original_image.shape[:2]
        
        # 如果图像宽度超过最大限制，则进行等比例缩小
        if original_w > MAX_DISPLAY_WIDTH:
            scale_factor = original_w / MAX_DISPLAY_WIDTH
            display_w = MAX_DISPLAY_WIDTH
            display_h = int(original_h / scale_factor)
            display_image = cv2.resize(original_image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        else:
            # 如果图像本身不大，则直接显示，缩放比例为1
            scale_factor = 1.0
            display_image = original_image
        
        image_clone = display_image.copy() # 操作的对象是缩放后的图像
        
        # 加载已标注的ROI时，需要将其从原始坐标换算为显示坐标
        temp_rois_for_display = []
        if image_filename in annotated_filenames:
            for item in all_rois_data:
                if item["filename"] == image_filename:
                    # 将存储的原始坐标换算成显示坐标
                    rois_for_current_image = item["rois"]
                    for (ox, oy, ow, oh) in rois_for_current_image:
                        dx = int(ox / scale_factor)
                        dy = int(oy / scale_factor)
                        dw = int(ow / scale_factor)
                        dh = int(oh / scale_factor)
                        temp_rois_for_display.append((dx, dy, dw, dh))
                    break
        else:
            rois_for_current_image = []

        while True:
            current_display_image = image_clone.copy()
            
            # 绘制时，使用换算后的显示坐标
            if image_filename in annotated_filenames:
                 current_display_image = draw_rois(current_display_image, temp_rois_for_display)
            else:
                # 实时绘制新的ROI
                display_rois = [(int(r[0]/scale_factor), int(r[1]/scale_factor), int(r[2]/scale_factor), int(r[3]/scale_factor)) for r in rois_for_current_image]
                current_display_image = draw_rois(current_display_image, display_rois)

            if cropping and ref_point_start and ref_point_end:
                cv2.rectangle(current_display_image, ref_point_start, ref_point_end, (0, 255, 255), 2)
            
            current_display_image = display_help_text(current_display_image)
            cv2.imshow("ROI Annotator", current_display_image)
            
            key = cv2.waitKey(1) & 0xFF

            def save_current_rois():
                nonlocal all_rois_data
                all_rois_data = [d for d in all_rois_data if d["filename"] != image_filename]

                # 保存到 all_rois_data 的是原始坐标，这是正确的
                if rois_for_current_image:
                    print(f"已保存 {len(rois_for_current_image)} 个 ROI 来自: {image_filename} (坐标已换算为原始尺寸)")
                    all_rois_data.append({
                        "filename": image_filename,
                        "rois": rois_for_current_image
                    })
                    annotated_filenames.add(image_filename)

            if key == ord('r'):
                rois_for_current_image = []
                temp_rois_for_display = [] # 同时清空显示用的ROI
                print(f"已为: {image_filename} 重置 ROIs")

            elif key == ord('n'):
                save_current_rois()
                index += 1
                break

            elif key == ord('b'):
                save_current_rois()
                if index > 0:
                    index -= 1
                else:
                    print("已经是第一张图像。")
                break

            elif key == ord('q'):
                save_current_rois()
                index = len(image_paths)
                break
    
    cv2.destroyAllWindows()
    return all_rois_data

# 后续的处理函数无需任何修改，因为它们操作的已经是原始坐标
def process_and_save_results(all_rois_data, stats_filename='rois_statistics.json', grouped_filename='roi_averaged_by_group_CORRECTED.json'):
    """
    对所有标注数据进行最终处理，包括计算总体统计和经过排序修正的分组平均，并保存结果。
    """
    if not all_rois_data:
        print("\n未标注任何 ROI，无需处理。")
        return

    # --- 第 1 部分: 计算总体统计数据并保存 ---
    print("\n--- 1. 计算总体统计数据 ---")
    all_rois_flat_list = [roi for entry in all_rois_data for roi in entry['rois']]
    total_rois = len(all_rois_flat_list)
    
    if total_rois > 0:
        rois_array = np.array(all_rois_flat_list)
        avg_roi_values = np.mean(rois_array, axis=0)
        avg_roi = {
            "x": int(round(avg_roi_values[0])), "y": int(round(avg_roi_values[1])),
            "width": int(round(avg_roi_values[2])), "height": int(round(avg_roi_values[3]))
        }
        
        print("计算出的总体平均 ROI:")
        print(f"  - 平均 (x, y, w, h): ({avg_roi['x']}, {avg_roi['y']}, {avg_roi['width']}, {avg_roi['height']})")

        output_data = {
            "average_roi_overall": avg_roi,
            "total_images_annotated": len(all_rois_data),
            "total_rois_annotated": total_rois,
            "detailed_data": all_rois_data
        }
        
        try:
            with open(stats_filename, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"详细统计数据已成功保存到: {stats_filename}")
        except Exception as e:
            print(f"\n错误: 无法将详细结果写入文件。 {e}")

    # --- 第 2 部分: 按组处理 ROI (已加入排序修正) ---
    print("\n--- 2. 按组处理 ROI (已修正) ---")
    grouped_by_roi_count = defaultdict(list)
    for image_data in all_rois_data:
        roi_count = len(image_data.get("rois", []))
        if roi_count > 0:
            grouped_by_roi_count[roi_count].append(image_data)
    
    if not grouped_by_roi_count:
        print("未找到有效的 ROI 分组进行处理。")
        return

    final_results = {}
    for count, items in grouped_by_roi_count.items():
        print(f" - 正在处理包含 {count} 个ROI的组，共 {len(items)} 张图片。")
        
        try:
            sorted_rois_per_image = [sorted(img['rois'], key=lambda r: r[1]) for img in items]
        except IndexError:
            print(f"错误: ROI数据格式不正确，无法排序。请确保每个ROI是 [x, y, w, h] 格式。")
            continue
            
        stacked_rois = np.array(sorted_rois_per_image)
        mean_values = np.mean(stacked_rois, axis=0)
        
        averaged_rois = [{
            "x": int(round(roi[0])), "y": int(round(roi[1])),
            "width": int(round(roi[2])), "height": int(round(roi[3]))
        } for roi in mean_values]

        group_key = f"group_with_{count}_rois"
        final_results[group_key] = {
            "source_image_count": len(items),
            "averaged_rois": averaged_rois
        }

    try:
        with open(grouped_filename, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"修正后的分组平均结果已成功保存到: {grouped_filename}")
    except Exception as e:
        print(f"\n错误: 无法将分组结果写入文件。 {e}")

def main():
    """主函数，用于执行整个工作流程。"""
    all_rois_data = run_annotator()

    if all_rois_data:
        print("\n--- 标注完成 ---")
        print(f"总共处理了 {len(all_rois_data)} 张图像。")
        process_and_save_results(all_rois_data)
        print("\n--- 所有处理完成 ---")
    else:
        print("\n未标注任何 ROI 或未找到图像。程序已结束。")

if __name__ == '__main__':
    main()