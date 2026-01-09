"""
可视化简化后的 exclusion zones
"""

import cv2
import numpy as np
import os

# Line2 cam3 exclusion zones - 简化版本 (4个框架，每框架4边)
LINE2_CAM3_EXCLUSION_ZONES = [
    # ===== 左上框架 =====
    {"x": 822, "y": 462, "width": 295, "height": 61},   # 顶部边
    {"x": 808, "y": 655, "width": 310, "height": 83},   # 底部边
    {"x": 819, "y": 469, "width": 49, "height": 264},   # 左边
    {"x": 1071, "y": 452, "width": 44, "height": 290},  # 右边
    
    # ===== 右上框架 =====
    {"x": 1512, "y": 460, "width": 308, "height": 54},  # 顶部边
    {"x": 1506, "y": 667, "width": 305, "height": 71},  # 底部边
    {"x": 1498, "y": 462, "width": 46, "height": 293},  # 左边
    {"x": 1742, "y": 465, "width": 78, "height": 271},  # 右边
    
    # ===== 左下框架 =====
    {"x": 820, "y": 989, "width": 317, "height": 68},   # 顶部边
    {"x": 826, "y": 1187, "width": 286, "height": 68},  # 底部边
    {"x": 824, "y": 1001, "width": 73, "height": 225},  # 左边
    {"x": 1058, "y": 979, "width": 46, "height": 274},  # 右边
    
    # ===== 右下框架 =====
    {"x": 1536, "y": 972, "width": 274, "height": 71},  # 顶部边
    {"x": 1544, "y": 1165, "width": 264, "height": 69}, # 底部边
    {"x": 1500, "y": 984, "width": 63, "height": 252},  # 左边
    {"x": 1754, "y": 1027, "width": 51, "height": 203}, # 右边
]

# Line3 cam3 exclusion zones - 简化版本 (4个框架，每框架4边)
LINE3_CAM3_EXCLUSION_ZONES = [
    # ===== 左上框架 =====
    {"x": 745, "y": 409, "width": 259, "height": 22},   # 顶部边
    {"x": 754, "y": 552, "width": 233, "height": 49},   # 底部边
    {"x": 744, "y": 419, "width": 69, "height": 175},   # 左边
    {"x": 950, "y": 411, "width": 47, "height": 204},   # 右边
    
    # ===== 右上框架 =====
    {"x": 1406, "y": 382, "width": 261, "height": 32},  # 顶部边
    {"x": 1387, "y": 537, "width": 266, "height": 63},  # 底部边
    {"x": 1394, "y": 380, "width": 37, "height": 213},  # 左边
    {"x": 1619, "y": 368, "width": 34, "height": 215},  # 右边
    
    # ===== 左下框架 =====
    {"x": 742, "y": 890, "width": 281, "height": 80},   # 顶部边
    {"x": 749, "y": 1064, "width": 269, "height": 78},  # 底部边
    {"x": 737, "y": 906, "width": 63, "height": 216},   # 左边
    {"x": 948, "y": 870, "width": 80, "height": 268},   # 右边
    
    # ===== 右下框架 =====
    {"x": 1414, "y": 851, "width": 278, "height": 46},  # 顶部边
    {"x": 1394, "y": 1054, "width": 298, "height": 71}, # 底部边
    {"x": 1406, "y": 873, "width": 51, "height": 252},  # 左边
    {"x": 1616, "y": 834, "width": 56, "height": 302},  # 右边
]


def draw_zones(image, zones, color=(0, 255, 0), alpha=0.3):
    """在图片上绘制 exclusion zones"""
    overlay = image.copy()
    
    for i, zone in enumerate(zones):
        x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
        
        # 绘制半透明填充
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        
    # 应用透明度
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 绘制边框和标签
    for i, zone in enumerate(zones):
        x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
        
        # 绘制边框
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # 绘制区域编号
        label = f"Zone {i+1}"
        cv2.putText(image, label, (x + 5, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制尺寸
        size_text = f"{w}x{h}"
        cv2.putText(image, size_text, (x + 5, y + h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "sim_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理 Line2 cam3
    line2_path = os.path.join(script_dir, "line2cam3.BMP")
    if os.path.exists(line2_path):
        print(f"处理 Line2 cam3...")
        img = cv2.imread(line2_path)
        img = draw_zones(img, LINE2_CAM3_EXCLUSION_ZONES)
        
        # 添加标题
        cv2.putText(img, "Line2 Cam3 - Simplified Exclusion Zones (6 zones)", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        output_path = os.path.join(output_dir, "line2cam3_simplified_zones.png")
        cv2.imwrite(output_path, img)
        print(f"  保存到: {output_path}")
    else:
        print(f"未找到: {line2_path}")
    
    # 处理 Line3 cam3
    line3_path = os.path.join(script_dir, "line3cam3.BMP")
    if os.path.exists(line3_path):
        print(f"处理 Line3 cam3...")
        img = cv2.imread(line3_path)
        img = draw_zones(img, LINE3_CAM3_EXCLUSION_ZONES)
        
        # 添加标题
        cv2.putText(img, "Line3 Cam3 - Simplified Exclusion Zones (6 zones)", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        output_path = os.path.join(output_dir, "line3cam3_simplified_zones.png")
        cv2.imwrite(output_path, img)
        print(f"  保存到: {output_path}")
    else:
        print(f"未找到: {line3_path}")
    
    print("\n完成！请查看 sim_output 文件夹中的图片。")


if __name__ == '__main__':
    main()
