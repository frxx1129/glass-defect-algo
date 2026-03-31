"""
分路决策模拟器 (简化版)
模拟不同切片数和缺陷位置下的撤清触发逻辑

配置说明:
- 5相机 (Line2/Line3): cam0,cam1,cam2,cam3,cam4，边缘为cam0和cam4
- 4相机 (Line1): cam0,cam1,cam2,cam3，边缘为cam0和cam3

撤清标记约定:
- 0: 整片撤清
- 1: 最右边那片
- 2: 中间那片(3片时) 或 左边那片(2片时)
- 3: 最左边那片(仅3片时)
"""



def simulate_piece_count(cams_with_vertical: list[int], expected_cams: int, cam_vertical_coords: dict | None = None) -> int:
    """模拟切片数估计"""
    first_cam = 0
    last_cam = max(0, expected_cams - 1)
    
    valid_cuts = 0
    for ci in cams_with_vertical:
        is_valid = False
        if expected_cams == 5:
            # 简化逻辑：只要不是边缘相机就有效
            if ci != 0 and ci != 4:
                is_valid = True
        else:
            if ci != first_cam and ci != last_cam:
                is_valid = True
        
        if is_valid:
            valid_cuts += 1
            
    piece_count = valid_cuts + 1
    max_pieces = 4 if expected_cams >= 5 else 3
    return min(max_pieces, max(1, piece_count))


def simulate_marks(piece_count: int, defect_cam: int, defect_x_ratio: float,
                   expected_cams: int, cam_width: int = 2448) -> list[int]:
    """模拟撤清标记决策"""
    center = (expected_cams - 1) / 2.0
    x_px = defect_x_ratio * cam_width
    
    if piece_count <= 1:
        return [0]
    
    if piece_count == 2:
        if defect_cam < center:
            return [2]  # 左片
        elif defect_cam > center:
            return [1]  # 右片
        else:
            return [2] if x_px < cam_width / 2 else [1]
    
    elif piece_count == 3:
        if expected_cams == 5:
            if defect_cam <= 1:
                return [3]  # 左片
            elif defect_cam == 2:
                return [2]  # 中片
            else:
                return [1]  # 右片
        elif expected_cams == 4:
            if defect_cam == 0:
                return [3]  # 左片
            elif defect_cam == 3:
                return [1]  # 右片
            elif defect_cam == 1:
                return [3] if x_px < cam_width / 2 else [2]
            else:  # cam2
                return [1] if x_px > cam_width / 2 else [2]
    
    return [0]


def print_simulation():
    print("=" * 70)
    print("分路决策模拟 (简化版 - 依旧排除边缘两个相机)")
    print("=" * 70)
    
    # === 5相机配置 ===
    print("\n" + "=" * 70)
    print("【5相机配置】(Line2/Line3)")
    print("  边缘相机: cam0, cam4 | 中间相机: cam1, cam2, cam3")
    print("=" * 70)
    
    print("\n--- 切片数估计 ---")
    cases_5cam = [
        ([], "无中间相机看到竖直边"),
        ([1], "cam1看到竖直边"),
        ([2], "cam2看到竖直边"),
        ([3], "cam3看到竖直边"),
        ([1, 2], "cam1,cam2看到竖直边"),
        ([2, 3], "cam2,cam3看到竖直边"),
        ([1, 3], "cam1,cam3看到竖直边"),
        ([1, 2, 3], "cam1,2,3都看到竖直边"),
    ]
    for cams, desc in cases_5cam:
        pieces = simulate_piece_count(cams, 5)
        print(f"  {desc:35} -> {pieces}片")

    print("\n--- 5相机边缘切分测试 (边缘应当无效) ---")
    cases_5cam_edge = [
        ([0], {0: [1800]}, "Cam0(边缘)"),
        ([4], {4: [300]}, "Cam4(边缘)"),
        ([0, 2], {0: [1800], 2: [1200]}, "Cam0(边缘)+Cam2(有效)"),
    ]
    for cams, coords, desc in cases_5cam_edge:
        pieces = simulate_piece_count(cams, 5, coords)
        print(f"  {desc:35} -> {pieces}片")
    
    print("\n--- 切2片时的分路 ---")
    print("  布局: [左片(标记2) | 右片(标记1)]")
    positions_2p_5c = [
        (0, 0.5, "cam0"), (1, 0.5, "cam1"), (2, 0.3, "cam2左侧"),
        (2, 0.7, "cam2右侧"), (3, 0.5, "cam3"), (4, 0.5, "cam4"),
    ]
    for cam, x, desc in positions_2p_5c:
        m = simulate_marks(2, cam, x, 5)
        piece = "左片" if 2 in m else "右片"
        print(f"  缺陷在{desc:12} -> 标记{m} ({piece})")
    
    print("\n--- 切3片时的分路 ---")
    print("  布局: [左片(标记3) | 中片(标记2) | 右片(标记1)]")
    positions_3p_5c = [
        (0, 0.5, "cam0"), (1, 0.5, "cam1"), (2, 0.5, "cam2"),
        (3, 0.5, "cam3"), (4, 0.5, "cam4"),
    ]
    for cam, x, desc in positions_3p_5c:
        m = simulate_marks(3, cam, x, 5)
        piece = {3: "左片", 2: "中片", 1: "右片"}.get(m[0], "?")
        print(f"  缺陷在{desc:12} -> 标记{m} ({piece})")


if __name__ == "__main__":
    print_simulation()
