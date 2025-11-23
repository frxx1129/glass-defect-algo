import cv2
import numpy as np
import json
import os
import time

# 添加全局变量来跟踪大矩形的位置和状态
_current_y_position = -4096  # 起始位置在图像外上方
_move_step = 230  # 每次调用向下移动的像素数
_defect_positions = []  # 存储缺陷位置信息，保持缺陷相对位置不变
_current_camera = 0  # 当前相机索引 (0-4)
_camera_count = 5  # 总相机数量
_waiting_until = 0  # 等待时间戳，0表示不等待
_is_waiting = False  # 是否处于等待状态

def generate_test_image():
    global _current_y_position, _defect_positions, _current_camera, _waiting_until, _is_waiting
    
    # 生成一张纯灰度图（单通道 uint8）
    height, width = 2048, 2448
    test_image = np.zeros((height, width), dtype=np.uint8)
    
    # 添加随机噪点（非常小的噪点）
    noise_mask = np.random.random((height, width)) < 0.05  # 5%的像素有噪点
    noise_values = np.random.randint(5, 20, size=(height, width), dtype=np.uint8)  # 噪点灰度值5-20
    test_image[noise_mask] = noise_values[noise_mask]

    # 导入ROI区域（如果有）
    roi_template_path = "roi_averaged_by_group_CORRECTED.json"
    rois = []
    if os.path.exists(roi_template_path):
        try:
            with open(roi_template_path, "r", encoding="utf-8") as f:
                roi_data = json.load(f)

            if isinstance(roi_data, dict):
                if "rois" in roi_data and isinstance(roi_data["rois"], list):
                    rois = roi_data["rois"]
                else:
                    # 查找含 averaged_rois 的组，选择 source_image_count 最大的组
                    best_key = None
                    best_count = -1
                    for k, v in roi_data.items():
                        if isinstance(v, dict) and "averaged_rois" in v and isinstance(v["averaged_rois"], list):
                            count = v.get("source_image_count", len(v["averaged_rois"]))
                            if count > best_count:
                                best_count = count
                                best_key = k
                    if best_key is not None:
                        rois = roi_data[best_key]["averaged_rois"]
                    else:
                        if set(("x","y","width","height")).issubset(roi_data.keys()):
                            rois = [roi_data]
            elif isinstance(roi_data, list):
                rois = roi_data
        except Exception:
            rois = []
    
    # 如果没有找到有效的ROI，创建一个测试用的ROI
    if not rois:
        print("未找到有效的ROI，创建测试用ROI")
        rois = [{"x": 100, "y": 100, "width": 1000, "height": 800}]

    # 为ROI区域填充白色
    for roi in rois:
        try:
            x = int(max(0, roi.get("x", 0)))
            y = int(max(0, roi.get("y", 0)))
            w = int(max(0, roi.get("width", roi.get("w", 0))))
            h = int(max(0, roi.get("height", roi.get("h", 0))))
            if w <= 0 or h <= 0:
                continue
            x2 = min(x + w - 1, width - 1)
            y2 = min(y + h - 1, height - 1)

            # 在ROI区域填充白色（158）
            cv2.rectangle(test_image, (x, y), (x2, y2), 158, -1)
        except Exception as e:
            print(f"处理ROI时出错: {e}")
            continue

    # 定义超大矩形的参数
    full_rect_width = 11000  # 超大矩形的完整宽度
    rect_height = 4096  # 矩形高度是图像高度的两倍
    viewport_count = _camera_count  # 水平视口数量等于相机数量
    viewport_width = width  # 每个视口的宽度
    
    # 检查是否处于等待状态
    current_time = time.time()
    if _is_waiting:
        if current_time >= _waiting_until:
            # 等待时间结束，重置矩形位置到上方
            _current_y_position = -rect_height
            _is_waiting = False
            _current_camera = 0  # 重置为第一个相机
            
            # 重新生成缺陷，但概率更小
            _defect_positions = []
            # 减少缺陷数量，增加靠近边缘的概率
            if np.random.random() < 0.7:  # 只有70%的机会生成缺陷
                defect_count = np.random.randint(1, 5)  # 减少到1-4个缺陷
                
                # 生成偏向边缘的缺陷
                for _ in range(defect_count):
                    # 决定缺陷位于哪个边缘区域，增加边缘概率
                    edge_type = np.random.choice(['top', 'bottom', 'left', 'right'], 
                                               p=[0.3, 0.3, 0.2, 0.2])
                    
                    # 计算这个缺陷在全局矩形中的相对位置
                    if edge_type == 'top':
                        rel_x = np.random.randint(100, full_rect_width - 200)
                        rel_y = np.random.randint(50, rect_height // 8)  # 更靠近上边缘
                    elif edge_type == 'bottom':
                        rel_x = np.random.randint(100, full_rect_width - 200)
                        rel_y = np.random.randint(rect_height * 7 // 8, rect_height - 100)  # 更靠近下边缘
                    elif edge_type == 'left':
                        rel_x = np.random.randint(50, full_rect_width // 8)  # 更靠近左边缘
                        rel_y = np.random.randint(100, rect_height - 200)
                    else:  # right
                        rel_x = np.random.randint(full_rect_width * 7 // 8, full_rect_width - 100)  # 更靠近右边缘
                        rel_y = np.random.randint(100, rect_height - 200)
                    
                    # 随机选择形状类型：0=矩形，1=椭圆，2=不规则多边形
                    shape_type = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])  # 更偏向不规则形状
                    block_w = np.random.randint(20, 61)  # 稍微减小尺寸
                    block_h = np.random.randint(15, 31)
                    color = int(np.random.randint(5, 15))  # 暗色
                    
                    _defect_positions.append((rel_x, rel_y, block_w, block_h, color, shape_type))
            
            print(f"等待结束，矩形重新从上方开始移动")
        else:
            # 继续等待，返回只有ROI的图像
            return test_image
    
    # 计算当前视口的水平位置
    # 超大矩形居中对称放置，计算起始X坐标
    global_rect_start_x = (viewport_count * viewport_width - full_rect_width) // 2
    
    # 当前视口的起始位置 (根据当前相机选择视口)
    viewport_start_x = _current_camera * viewport_width
    
    # 计算当前视口内矩形的相对位置
    rect_in_viewport_x = global_rect_start_x - viewport_start_x
    rect_y = _current_y_position
    
    # 检查当前视口中是否有矩形的可见部分
    if (rect_in_viewport_x < viewport_width and 
        rect_in_viewport_x + full_rect_width > 0 and 
        rect_y < height and rect_y + rect_height > 0):
        
        # 计算矩形在当前视口中的可见部分
        visible_x = max(0, rect_in_viewport_x)
        visible_width = min(viewport_width, rect_in_viewport_x + full_rect_width) - visible_x
        visible_y = max(0, rect_y)
        visible_height = min(height, rect_y + rect_height) - visible_y
        
        if visible_width > 0 and visible_height > 0:
            # 创建原始图像的副本，用于叠加半透明效果
            overlay = test_image.copy()
            
            # 绘制矩形的内部（半透明，灰度值较低）
            cv2.rectangle(overlay, 
                          (visible_x, visible_y), 
                          (visible_x + visible_width - 1, visible_y + visible_height - 1), 
                          120, -1)  # 内部灰度值为120

            # 将overlay图像与原图像按权重混合，实现半透明效果
            alpha = 0.6  # 透明度
            cv2.addWeighted(overlay, alpha, test_image, 1 - alpha, 0, test_image)
            
            # 绘制矩形边框（更深色，半透明度较低）
            border_overlay = test_image.copy()
            cv2.rectangle(border_overlay, 
                          (visible_x, visible_y), 
                          (visible_x + visible_width - 1, visible_y + visible_height - 1), 
                          20, 2)  # 边框灰度值为20，宽度为3

            # 将边框与当前图像混合
            border_alpha = 1  # 边框透明度较低
            cv2.addWeighted(border_overlay, border_alpha, test_image, 1 - border_alpha, 0, test_image)
            
            # 如果还没有缺陷位置信息，则创建
            if not _defect_positions:
                # 减少缺陷数量，增加靠近边缘的概率
                if np.random.random() < 0.7:  # 只有70%的机会生成缺陷
                    defect_count = np.random.randint(1, 5)  # 减少到1-4个缺陷
                    
                    # 生成偏向边缘的缺陷
                    for _ in range(defect_count):
                        # 决定缺陷位于哪个边缘区域，增加边缘概率
                        edge_type = np.random.choice(['top', 'bottom', 'left', 'right'], 
                                                   p=[0.3, 0.3, 0.2, 0.2])
                        
                        # 计算这个缺陷在全局矩形中的相对位置
                        if edge_type == 'top':
                            rel_x = np.random.randint(100, full_rect_width - 200)
                            rel_y = np.random.randint(50, rect_height // 8)  # 更靠近上边缘
                        elif edge_type == 'bottom':
                            rel_x = np.random.randint(100, full_rect_width - 200)
                            rel_y = np.random.randint(rect_height * 7 // 8, rect_height - 100)  # 更靠近下边缘
                        elif edge_type == 'left':
                            rel_x = np.random.randint(50, full_rect_width // 8)  # 更靠近左边缘
                            rel_y = np.random.randint(100, rect_height - 200)
                        else:  # right
                            rel_x = np.random.randint(full_rect_width * 7 // 8, full_rect_width - 100)  # 更靠近右边缘
                            rel_y = np.random.randint(100, rect_height - 200)
                        
                        # 随机选择形状类型：0=矩形，1=椭圆，2=不规则多边形
                        shape_type = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])  # 更偏向不规则形状
                        block_w = np.random.randint(20, 61)  # 稍微减小尺寸
                        block_h = np.random.randint(15, 31)
                        color = int(np.random.randint(5, 15))  # 暗色
                        
                        _defect_positions.append((rel_x, rel_y, block_w, block_h, color, shape_type))
            
            # 绘制当前视口中可见的缺陷
            for rel_x, rel_y, block_w, block_h, color, shape_type in _defect_positions:
                # 计算缺陷在全局坐标系中的位置
                global_defect_x = global_rect_start_x + rel_x
                global_defect_y = rect_y + rel_y
                
                # 计算缺陷在当前视口中的位置
                defect_in_viewport_x = global_defect_x - viewport_start_x
                defect_in_viewport_y = global_defect_y
                
                # 检查缺陷是否在当前视口和图像的可见区域内
                if (defect_in_viewport_x + block_w > 0 and defect_in_viewport_x < viewport_width and
                    defect_in_viewport_y + block_h > 0 and defect_in_viewport_y < height):
                    
                    # 确定缺陷在视口和图像内的可见部分
                    block_visible_x = max(0, defect_in_viewport_x)
                    block_visible_y = max(0, defect_in_viewport_y)
                    block_visible_w = min(viewport_width, defect_in_viewport_x + block_w) - block_visible_x
                    block_visible_h = min(height, defect_in_viewport_y + block_h) - block_visible_y
                    
                    if block_visible_w > 0 and block_visible_h > 0:
                        defect_overlay = test_image.copy()
                        
                        # 根据形状类型绘制不同形状的缺陷
                        if shape_type == 0:  # 矩形
                            cv2.rectangle(defect_overlay, 
                                         (block_visible_x, block_visible_y), 
                                         (block_visible_x + block_visible_w - 1, block_visible_y + block_visible_h - 1), 
                                         color, -1)
                        
                        elif shape_type == 1:  # 椭圆
                            center_x = block_visible_x + block_visible_w // 2
                            center_y = block_visible_y + block_visible_h // 2
                            axes_length = (block_visible_w // 2, block_visible_h // 2)
                            cv2.ellipse(defect_overlay, (center_x, center_y), axes_length, 
                                       0, 0, 360, color, -1)
                        
                        else:  # 不规则多边形
                            # 生成一个不规则多边形的点
                            points = []
                            center_x = block_visible_x + block_visible_w // 2
                            center_y = block_visible_y + block_visible_h // 2
                            num_points = np.random.randint(5, 9)  # 5-8个点
                            
                            for i in range(num_points):
                                angle = 2 * np.pi * i / num_points
                                # 添加一些随机性使多边形不规则
                                radius = min(block_visible_w, block_visible_h) // 2 * (0.7 + 0.3 * np.random.random())
                                x = int(center_x + radius * np.cos(angle))
                                y = int(center_y + radius * np.sin(angle))
                                points.append((x, y))
                            
                            points = np.array(points, np.int32)
                            points = points.reshape((-1, 1, 2))
                            cv2.fillPoly(defect_overlay, [points], color)
                        
                        # 将缺陷叠加到图像上，不透明度高
                        defect_alpha = 0.9
                        cv2.addWeighted(defect_overlay, defect_alpha, test_image, 1 - defect_alpha, 0, test_image)
    
    # 切换到下一个相机，如果所有相机都拍完了一轮，则移动矩形
    _current_camera = (_current_camera + 1) % _camera_count
    
    # 如果已经完成了一轮拍摄（回到第一个相机），则移动矩形
    if _current_camera == 0:
        # 移动矩形
        _current_y_position += _move_step
        
        # 检查矩形是否完全离开画面底部
        if _current_y_position > height:
            print("矩形已离开画面，开始等待5秒...")
            _is_waiting = True
            _waiting_until = current_time + 5.0  # 设置5秒后结束等待

    # print(f"当前相机: {_current_camera+1}/{_camera_count}, 矩形Y位置: {_current_y_position}")
    return test_image

def reset_test_image_state():
    """重置测试图像生成器的状态，可在需要时从外部调用"""
    global _current_y_position, _defect_positions, _current_camera, _waiting_until, _is_waiting
    _current_y_position = -4096  # 重置位置
    _defect_positions = []
    _current_camera = 0
    _waiting_until = 0
    _is_waiting = False
    print("测试图像生成器状态已重置")

if __name__ == "__main__":
    # 测试代码：生成一系列图像，模拟连续调用
    output_dir = "test_sequence"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(300):  # 增加到300帧以便观察多轮相机循环
        img = generate_test_image()
        cv2.imwrite(f"{output_dir}/frame_{i:03d}.jpg", img)
        print(f"生成第 {i+1}/300 帧")
        time.sleep(0.1)  # 添加短暂延迟模拟实际调用间隔
        
    print("测试序列生成完成，保存在", output_dir)