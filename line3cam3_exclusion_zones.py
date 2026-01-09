"""
Auto-generated exclusion zone definition (simplified)
Source image: line3cam3.BMP
Original: 18 zones -> Simplified: 16 zones (4个框架 x 4边)
"""

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
    {"x": 742, "y": 890, "width": 281, "height": 80},   # 顶部边 (向下拓展) (合并了重叠区域)
    {"x": 749, "y": 1064, "width": 269, "height": 78},  # 底部边
    {"x": 737, "y": 906, "width": 63, "height": 216},   # 左边 (合并了重叠区域)
    {"x": 948, "y": 870, "width": 80, "height": 268},   # 右边 (合并了两个相邻区域)
    
    # ===== 右下框架 =====
    {"x": 1414, "y": 851, "width": 278, "height": 46},  # 顶部边
    {"x": 1394, "y": 1054, "width": 298, "height": 71}, # 底部边
    {"x": 1406, "y": 873, "width": 51, "height": 252},  # 左边
    {"x": 1616, "y": 834, "width": 56, "height": 302},  # 右边
]
