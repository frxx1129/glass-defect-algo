import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

class ROIAnnotator:
    def __init__(self):
        self.ref_point_start = None
        self.ref_point_end = None
        self.cropping = False
        self.rois_for_current_image = []
        self.image_clone = None
        self.scale_factor = 1.0
        self.MAX_DISPLAY_WIDTH = 1280

    def display_help_text(self, image):
        help_text = "Keys: (N)ext | (B)ack | (R)eset | (Q)uit & Process"
        cv2.putText(image, help_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def draw_rois(self, image, rois):
        for (x, y, w, h) in rois:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point_start = (x, y)
            self.cropping = True
        elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
            self.ref_point_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point_end = (x, y)
            self.cropping = False
            if self.ref_point_start and self.ref_point_end and self.ref_point_start != self.ref_point_end:
                x1, y1 = self.ref_point_start
                x2, y2 = self.ref_point_end
                start_x, start_y = min(x1, x2), min(y1, y2)
                end_x, end_y = max(x1, x2), max(y1, y2)
                original_x = int(start_x * self.scale_factor)
                original_y = int(start_y * self.scale_factor)
                original_w = int((end_x - start_x) * self.scale_factor)
                original_h = int((end_y - start_y) * self.scale_factor)
                self.rois_for_current_image.append((original_x, original_y, original_w, original_h))
            self.ref_point_start = None
            self.ref_point_end = None

    def run_annotator(self, image_path):
        if not os.path.isfile(image_path) or not image_path.lower().endswith(('.bmp','.jpg','.jpeg','.png')):
            print(f"无效的图片路径: {image_path}")
            return None

        print("--- ROI 标注工具 ---")
        print("操作指南:")
        print(" - 按住鼠标左键并拖动以绘制矩形 ROI。")
        print(" - 快捷键: N(下一张), B(上一张), R(重置), Q(退出)")
        print("-" * 30)

        cv2.namedWindow("ROI Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI Annotator", self.mouse_callback)

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"无法读取图像: {image_path}")
            return None

        original_h, original_w = original_image.shape[:2]
        if original_w > self.MAX_DISPLAY_WIDTH:
            self.scale_factor = original_w / self.MAX_DISPLAY_WIDTH
            display_w = self.MAX_DISPLAY_WIDTH
            display_h = int(original_h / self.scale_factor)
            display_image = cv2.resize(original_image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        else:
            self.scale_factor = 1.0
            display_image = original_image

        self.image_clone = display_image.copy()
        image_filename = os.path.basename(image_path)
        all_rois_data = []
        annotated_filenames = set()

        while True:
            current_display_image = self.image_clone.copy()
            display_rois = [(int(r[0]/self.scale_factor), int(r[1]/self.scale_factor), int(r[2]/self.scale_factor), int(r[3]/self.scale_factor)) for r in self.rois_for_current_image]
            current_display_image = self.draw_rois(current_display_image, display_rois)

            if self.cropping and self.ref_point_start and self.ref_point_end:
                cv2.rectangle(current_display_image, self.ref_point_start, self.ref_point_end, (0, 255, 255), 2)

            current_display_image = self.display_help_text(current_display_image)
            cv2.imshow("ROI Annotator", current_display_image)

            key = cv2.waitKey(10) & 0xFF

            if key in [ord('r'), ord('R')]:
                self.rois_for_current_image = []
                print(f"已重置 {image_filename} 的 ROIs")
            elif key in [ord('n'), ord('N')]:
                if self.rois_for_current_image:
                    all_rois_data.append({"filename": image_filename, "rois": self.rois_for_current_image})
                    annotated_filenames.add(image_filename)
                    print(f"已保存 {len(self.rois_for_current_image)} 个 ROI")
                break
            elif key in [ord('b'), ord('B')]:
                print("已经是第一张图像。")
                break
            elif key in [ord('q'), ord('Q')]:
                if self.rois_for_current_image:
                    all_rois_data.append({"filename": image_filename, "rois": self.rois_for_current_image})
                    annotated_filenames.add(image_filename)
                break

        cv2.destroyAllWindows()
        return all_rois_data

    def process_and_save_results(self, all_rois_data, image_path):
        if not all_rois_data:
            print("未标注任何 ROI。")
            return

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        stats_filename = f"{base_name}_rois_statistics.json"
        grouped_filename = f"{base_name}_roi_averaged_by_group.json"

        all_rois_flat = [roi for entry in all_rois_data for roi in entry['rois']]
        total_rois = len(all_rois_flat)
        if total_rois > 0:
            avg_roi = np.mean(np.array(all_rois_flat), axis=0)
            avg_roi_dict = {"x": int(avg_roi[0]), "y": int(avg_roi[1]), "width": int(avg_roi[2]), "height": int(avg_roi[3])}
            output_data = {
                "average_roi_overall": avg_roi_dict,
                "total_images_annotated": len(all_rois_data),
                "total_rois_annotated": total_rois,
                "detailed_data": all_rois_data
            }
            with open(stats_filename, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"统计数据保存到: {stats_filename}")

        grouped = defaultdict(list)
        for data in all_rois_data:
            count = len(data.get("rois", []))
            if count > 0:
                grouped[count].append(data)

        final_results = {}
        for count, items in grouped.items():
            sorted_rois = [sorted(img['rois'], key=lambda r: r[1]) for img in items]
            mean_values = np.mean(np.array(sorted_rois), axis=0)
            averaged_rois = [{"x": int(roi[0]), "y": int(roi[1]), "width": int(roi[2]), "height": int(roi[3])} for roi in mean_values]
            final_results[f"group_with_{count}_rois"] = {"source_image_count": len(items), "averaged_rois": averaged_rois}

        with open(grouped_filename, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"分组结果保存到: {grouped_filename}")

def main():
    print("程序开始执行")
    try:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(title="选择要标注的图片", filetypes=[("BMP files", "*.bmp"), ("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if not image_path:
            print("未选择文件，程序退出。")
            return

        annotator = ROIAnnotator()
        all_rois_data = annotator.run_annotator(image_path)
        if all_rois_data:
            annotator.process_and_save_results(all_rois_data, image_path)
            print("标注完成。")
        else:
            print("未标注任何 ROI。")
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == '__main__':
    main()