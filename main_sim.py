"""
模拟模式测试脚本 - 用于测试排除区域和系统运转
无需连接真实相机，使用合成图像进行测试
"""
import os
import sys
import cv2
import json
import numpy as np
import time
import copy

# 导入处理模块
import image_processor_hough as processor
import fused_image_processor

# ============================================================
# 配置
# ============================================================
FRAME_WIDTH = 2448
FRAME_HEIGHT = 2048

# 模拟的 ROI 区域（两条横穿画面的白色带状区域）
SIMULATED_ROIS = [
    {"x": 700, "y": 400, "width": 500, "height": 300, "group": 0},
    {"x": 1400, "y": 400, "width": 500, "height": 300, "group": 1},
    {"x": 700, "y": 950, "width": 500, "height": 300, "group": 2},
    {"x": 1400, "y": 950, "width": 500, "height": 300, "group": 3},
]

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    import json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 尝试解析为 JSON（config.yaml 实际是 JSON 格式）
            return json.loads(content)
    except Exception as e:
        print(f"[SIM] 加载配置失败: {e}")
        return {}


def generate_test_image(with_defects=True, use_real_image=None):
    """
    生成测试图像
    - 黑色背景
    - 在 ROI 区域内绘制白色矩形（模拟玻璃边缘）
    - 可选添加模拟缺陷
    """
    if use_real_image is not None:
        # 使用真实图像
        img = cv2.imread(use_real_image, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"[SIM] 使用真实图像: {use_real_image}")
            return img
        print(f"[SIM] 无法加载图像: {use_real_image}，使用合成图像")
    
    # 创建黑色背景
    img = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    
    # 在每个 ROI 区域内绘制白色矩形框（模拟玻璃边缘）
    for roi in SIMULATED_ROIS:
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        # 绘制白色矩形边框（模拟玻璃边缘）
        thickness = 15
        # 顶边
        cv2.rectangle(img, (x, y), (x + w, y + thickness), 200, -1)
        # 底边
        cv2.rectangle(img, (x, y + h - thickness), (x + w, y + h), 200, -1)
        # 左边
        cv2.rectangle(img, (x, y), (x + thickness, y + h), 200, -1)
        # 右边
        cv2.rectangle(img, (x + w - thickness, y), (x + w, y + h), 200, -1)
        
        # 可选：添加一些模拟缺陷（小的暗斑）
        if with_defects:
            # 在 ROI 中心附近添加一个小缺陷
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(img, (cx, cy), 10, 50, -1)
    
    return img


def visualize_exclusion_zones(img_gray, cam_index, line_name, output_path=None):
    """
    可视化排除区域在图像上的位置
    """
    # 转换为彩色图像以便标注
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # 获取排除区域
    zones = processor.get_exclusion_zones(line_name, cam_index)
    
    print(f"\n[SIM] 排除区域配置 (lineName={line_name}, cam_index={cam_index}):")
    if not zones:
        print("  (无排除区域)")
    else:
        print(f"  共 {len(zones)} 个区域:")
        for i, zone in enumerate(zones):
            x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
            print(f"    [{i}] x={x}, y={y}, w={w}, h={h}")
            # 绘制排除区域（红色半透明）
            overlay = img_color.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, img_color, 0.7, 0, img_color)
            # 绘制边框
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # 绘制 ROI 区域（绿色边框）
    for roi in SIMULATED_ROIS:
        x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 添加图例
    cv2.putText(img_color, f"Line: {line_name}, Cam Index: {cam_index}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_color, "Green: ROI regions", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_color, f"Red: Exclusion zones ({len(zones)} total)", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, img_color)
        print(f"[SIM] 可视化结果已保存: {output_path}")
    
    return img_color


def run_processing_test(config, cam_index, line_name, use_real_image=None):
    """
    运行处理测试
    """
    print(f"\n{'='*60}")
    print(f"[SIM] 开始处理测试")
    print(f"      lineName: {line_name}")
    print(f"      cam_index: {cam_index}")
    print(f"{'='*60}")
    
    # 生成或加载测试图像
    img_gray = generate_test_image(with_defects=True, use_real_image=use_real_image)
    print(f"[SIM] 图像尺寸: {img_gray.shape}")
    
    # 准备配置（注入运行时参数）
    test_config = copy.deepcopy(config)
    hip = test_config.setdefault('hough_inspector_params', {})
    hip['_RUNTIME_LINE_NAME'] = line_name
    hip['_RUNTIME_CAM_INDEX'] = cam_index
    
    # 调用处理函数
    try:
        t0 = time.perf_counter()
        report, annotated = fused_image_processor.process_image(
            img_gray, SIMULATED_ROIS, test_config, mode=1
        )
        elapsed = (time.perf_counter() - t0) * 1000
        
        print(f"\n[SIM] 处理完成，耗时: {elapsed:.1f} ms")
        print(f"[SIM] 结果:")
        print(f"      image_status: {report.get('image_status', 'N/A')}")
        print(f"      state_code: {report.get('state_code', 'N/A')}")
        print(f"      defects count: {len(report.get('defects', []))}")
        
        if report.get('defects'):
            print(f"\n[SIM] 检测到的缺陷:")
            for i, defect in enumerate(report.get('defects', [])):
                dtype = defect.get('type', '?')
                loc = defect.get('location', {})
                print(f"      [{i}] Type={dtype}, Location={loc}")
        
        return report, annotated
        
    except Exception as e:
        import traceback
        print(f"[SIM] 处理异常: {e}")
        traceback.print_exc()
        return None, None


def test_exclusion_zone_application():
    """
    测试排除区域是否正确应用到边缘检测
    """
    print("\n" + "="*60)
    print("[SIM] 测试排除区域边缘屏蔽效果")
    print("="*60)
    
    # 创建一个简单的边缘图像
    edge_img = np.zeros((300, 500), dtype=np.uint8)
    # 在整个区域内画一些边缘
    cv2.rectangle(edge_img, (10, 10), (490, 290), 255, 2)
    cv2.line(edge_img, (50, 50), (450, 250), 255, 2)
    
    print(f"[SIM] 原始边缘像素数: {np.count_nonzero(edge_img)}")
    
    # 模拟一个排除区域（在边缘图像坐标系中）
    test_zones = [
        {"x": 100, "y": 100, "width": 200, "height": 100}
    ]
    
    # 应用排除区域（ROI 偏移为 0）
    result = processor.apply_exclusion_zones_to_edges(edge_img, 0, 0, test_zones)
    
    print(f"[SIM] 屏蔽后边缘像素数: {np.count_nonzero(result)}")
    print(f"[SIM] 减少了: {np.count_nonzero(edge_img) - np.count_nonzero(result)} 像素")
    
    # 保存对比图
    compare = np.hstack([edge_img, result])
    cv2.imwrite("sim_output/exclusion_test_edges.png", compare)
    print("[SIM] 边缘对比图已保存: sim_output/exclusion_test_edges.png")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("GlassAlgo 模拟模式测试")
    print("="*60)
    
    # 创建输出目录
    os.makedirs("sim_output", exist_ok=True)
    
    # 加载配置
    config = load_config("config.yaml")
    if not config:
        print("[SIM] 错误: 无法加载配置文件")
        return
    
    # 从配置读取 lineName
    line_name = config.get('lineName', 'Line3')
    print(f"[SIM] 当前产线: {line_name}")
    
    # 检查是否有真实测试图像可用
    real_images = {
        "Line2": "line2cam3.BMP",
        "Line3": "line3cam3.BMP"
    }
    use_real = real_images.get(line_name)
    if use_real and os.path.exists(use_real):
        print(f"[SIM] 找到真实测试图像: {use_real}")
    else:
        use_real = None
        print("[SIM] 使用合成测试图像")
    
    # ========================================
    # 测试 1: 可视化排除区域
    # ========================================
    print("\n--- 测试 1: 可视化排除区域 ---")
    
    # 使用合成图像可视化
    test_img = generate_test_image(with_defects=False, use_real_image=use_real)
    
    # 测试 cam_index = 2 的排除区域
    vis_img = visualize_exclusion_zones(
        test_img, 
        cam_index=2, 
        line_name=line_name,
        output_path=f"sim_output/{line_name.lower()}_cam2_exclusion_zones.png"
    )
    
    # 对比：测试 cam_index = 0（不应该有排除区域）
    vis_img_no_zones = visualize_exclusion_zones(
        test_img, 
        cam_index=0, 
        line_name=line_name,
        output_path=f"sim_output/{line_name.lower()}_cam0_no_exclusion.png"
    )
    
    # ========================================
    # 测试 2: 排除区域边缘屏蔽
    # ========================================
    print("\n--- 测试 2: 排除区域边缘屏蔽 ---")
    test_exclusion_zone_application()
    
    # ========================================
    # 测试 3: 完整处理流程（cam_index=2）
    # ========================================
    print("\n--- 测试 3: 完整处理流程 (cam_index=2) ---")
    report, annotated = run_processing_test(
        config, 
        cam_index=2, 
        line_name=line_name,
        use_real_image=use_real
    )
    
    if annotated is not None:
        output_path = f"sim_output/{line_name.lower()}_cam2_annotated.png"
        cv2.imwrite(output_path, annotated)
        print(f"[SIM] 标注结果已保存: {output_path}")
    
    # ========================================
    # 测试 4: 对比不同 cam_index
    # ========================================
    print("\n--- 测试 4: 对比不同 cam_index ---")
    for test_cam_idx in [0, 1, 2, 3, 4]:
        zones = processor.get_exclusion_zones(line_name, test_cam_idx)
        status = f"有 {len(zones)} 个排除区域" if zones else "无排除区域"
        print(f"  cam_index={test_cam_idx}: {status}")
    
    # ========================================
    # 总结
    # ========================================
    print("\n" + "="*60)
    print("[SIM] 测试完成！")
    print("="*60)
    print("\n输出文件:")
    for f in os.listdir("sim_output"):
        print(f"  - sim_output/{f}")
    print("\n请检查 sim_output/ 目录中的图像以验证排除区域是否正确应用。")


if __name__ == "__main__":
    main()
