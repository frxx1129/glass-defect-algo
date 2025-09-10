import os
import json
import cv2
import time
import argparse
import image_processor_optimized

# 批处理 valid 文件夹下图片，复用现有算法/参数

def load_config(cfg_path: str = 'config.json'):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_template_rois(config, cam_idx: int = 0):
    camera_rois_cfg = config.get('camera_rois', {})
    roi_path = None
    if isinstance(camera_rois_cfg, dict):
        roi_path = camera_rois_cfg.get(str(cam_idx)) or camera_rois_cfg.get(cam_idx)
    elif isinstance(camera_rois_cfg, list) and cam_idx < len(camera_rois_cfg):
        roi_path = camera_rois_cfg[cam_idx]
    if not roi_path:
        roi_path = config.get('roi_template_file', 'roi_averaged_by_group_CORRECTED.json')
    try:
        with open(roi_path, 'r', encoding='utf-8') as f:
            averaged_data = json.load(f)
        # 取 source_image_count 最大的组
        best_group_key = max(averaged_data, key=lambda k: averaged_data[k].get('source_image_count', 0))
        rois = averaged_data[best_group_key]['averaged_rois']
        return rois
    except Exception as e:
        print(f"[批处理]: 加载ROI失败: {e}")
        return []

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def process_single_image(img_path, template_rois, config, out_dir, draw_contours=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[跳过]: 无法读取 {img_path}")
        return
    t0 = time.time()
    report, annotated = image_processor_optimized.process_image_from_memory_parallel(img, template_rois, config, draw_contours=draw_contours)
    dt = (time.time() - t0) * 1000.0
    base = os.path.splitext(os.path.basename(img_path))[0]
    json_out = os.path.join(out_dir, base + '.json')
    img_out = os.path.join(out_dir, base + '_annotated.jpg')
    try:
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        cv2.imwrite(img_out, annotated)
        print(f"[完成]: {base} 状态={report.get('image_status')} 缺陷数={len(report.get('defects', []))} 耗时={dt:.1f}ms")
    except Exception as e:
        print(f"[错误]: 保存结果失败 {e}")


def main():
    parser = argparse.ArgumentParser(description='批量处理 valid 目录下图片')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--input', default='valid', help='输入图片目录')
    parser.add_argument('--output', default='valid_results', help='输出结果目录')
    parser.add_argument('--ext', nargs='*', default=['.jpg', '.png', '.bmp'], help='允许的扩展名')
    parser.add_argument('--draw-contours', action='store_true', help='可选：在结果中绘制轮廓')
    args = parser.parse_args()

    cfg = load_config(args.config)
    template_rois = load_template_rois(cfg, cam_idx=0)
    if not template_rois:
        print('[警告]: 未加载到任何ROI，将继续处理但结果可能全部为OK。')

    ensure_dir(args.output)

    # 收集文件
    exts = set([e.lower() for e in args.ext])
    all_files = []
    for name in os.listdir(args.input):
        p = os.path.join(args.input, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            all_files.append(p)
    if not all_files:
        print('[提示]: 未找到匹配图片。')
        return

    print(f'[批处理]: 发现 {len(all_files)} 张图片，开始处理...')
    for fp in all_files:
        process_single_image(fp, template_rois, cfg, args.output, draw_contours=args.draw_contours)

if __name__ == '__main__':
    main()
