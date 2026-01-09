"""
Auto-generated exclusion zone definition (simplified)
Source image: line2cam3.BMP
Original: 18 zones -> Simplified: 16 zones (4个框架 x 4边)
"""

# Line2 cam3 exclusion zones - 简化版本 (4个框架，每框架4边)
LINE2_CAM3_EXCLUSION_ZONES = [
    # ===== 左上框架 =====
    {"x": 822, "y": 462, "width": 295, "height": 61},   # 顶部边
    {"x": 808, "y": 655, "width": 310, "height": 83},   # 底部边
    {"x": 819, "y": 469, "width": 49, "height": 264},   # 左边
    {"x": 1071, "y": 452, "width": 44, "height": 290},  # 右边
    
    # ===== 右上框架 =====
    {"x": 1512, "y": 460, "width": 308, "height": 54},  # 顶部边
    {"x": 1506, "y": 667, "width": 305, "height": 71},  # 底部边 (合并了重叠区域)
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
    {"x": 1500, "y": 984, "width": 63, "height": 252},  # 左边 (合并了两个左侧区域)
    {"x": 1754, "y": 1027, "width": 51, "height": 203}, # 右边
]
