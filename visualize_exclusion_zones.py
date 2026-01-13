"""
可视化简化后的 exclusion zones
直接读取 image_processor_hough.py 源文件，完全绕过 Python 缓存
"""

import cv2
import numpy as np
import os
import re
import sys

def load_exclusion_zones_from_source():
    """直接从源文件读取 exclusion zones，完全绕过 .pyc 缓存"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_file = os.path.join(script_dir, "image_processor_hough.py")
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取 LINE2_CAM3_EXCLUSION_ZONES
    line2_match = re.search(
        r'LINE2_CAM3_EXCLUSION_ZONES\s*=\s*\[(.*?)\n\]',
        content,
        re.DOTALL
    )
    
    # 使用正则表达式提取 LINE3_CAM3_EXCLUSION_ZONES
    line3_match = re.search(
        r'LINE3_CAM3_EXCLUSION_ZONES\s*=\s*\[(.*?)\n\]',
        content,
        re.DOTALL
    )
    
    line2_zones = []
    line3_zones = []
    
    # 解析每个 zone 字典
    zone_pattern = re.compile(r'\{[^}]+\}')
    
    if line2_match:
        for zone_str in zone_pattern.findall(line2_match.group(1)):
            try:
                # 安全解析字典
                zone = eval(zone_str)
                if isinstance(zone, dict) and all(k in zone for k in ['x', 'y', 'width', 'height']):
                    line2_zones.append(zone)
            except:
                pass
    
    if line3_match:
        for zone_str in zone_pattern.findall(line3_match.group(1)):
            try:
                zone = eval(zone_str)
                if isinstance(zone, dict) and all(k in zone for k in ['x', 'y', 'width', 'height']):
                    line3_zones.append(zone)
            except:
                pass
    
    print(f"从源文件加载: LINE2_CAM3_EXCLUSION_ZONES = {len(line2_zones)} 个区域")
    print(f"从源文件加载: LINE3_CAM3_EXCLUSION_ZONES = {len(line3_zones)} 个区域")
    
    # 打印第一个 zone 以验证
    if line3_zones:
        print(f"  Line3 Zone 1: {line3_zones[0]}")
    
    return line2_zones, line3_zones


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
    
    # 直接从源文件加载配置，完全绕过缓存
    LINE2_CAM3_EXCLUSION_ZONES, LINE3_CAM3_EXCLUSION_ZONES = load_exclusion_zones_from_source()
    
    # 处理 Line2 cam3
    line2_path = os.path.join(script_dir, "line2cam3.BMP")
    if os.path.exists(line2_path):
        print(f"处理 Line2 cam3...")
        img = cv2.imread(line2_path)
        img = draw_zones(img, LINE2_CAM3_EXCLUSION_ZONES)
        
        # 添加标题
        cv2.putText(img, f"Line2 Cam3 - Exclusion Zones ({len(LINE2_CAM3_EXCLUSION_ZONES)} zones)", 
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
        cv2.putText(img, f"Line3 Cam3 - Exclusion Zones ({len(LINE3_CAM3_EXCLUSION_ZONES)} zones)", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        output_path = os.path.join(output_dir, "line3cam3_simplified_zones.png")
        cv2.imwrite(output_path, img)
        print(f"  保存到: {output_path}")
    else:
        print(f"未找到: {line3_path}")
    
    print("\n完成！请查看 sim_output 文件夹中的图片。")


if __name__ == '__main__':
    main()
