import os
import json
import argparse
import random
import csv
import time
from typing import Dict, Any

import image_processor_optimized
import cv2


def load_config(cfg_path: str = 'config.json') -> Dict[str, Any]:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_template_rois(config: Dict[str, Any], cam_idx: int = 0):
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
        best_group_key = max(averaged_data, key=lambda k: averaged_data[k].get('source_image_count', 0))
        return averaged_data[best_group_key]['averaged_rois']
    except Exception as e:
        print(f"[ROI加载失败]: {e}")
        return []


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clone_config_for_run(base_cfg: Dict[str, Any], gamma: float, sigma: float, clahe_clip: float, clahe_tile: int) -> Dict[str, Any]:
    # 深拷贝只针对我们要改的部分，避免不必要的巨大结构复制
    cfg = json.loads(json.dumps(base_cfg))  # 简单安全深拷贝
    # 兼容嵌套结构
    dd = cfg.get('defect_detection_params', {})
    pp = dd.get('preprocess_params', {})
    pp['GAMMA_VALUE'] = gamma
    pp['CANNY_SIGMA'] = sigma
    pp['CLAHE_CLIP_LIMIT'] = clahe_clip
    pp['CLAHE_TILE_TARGET_PX'] = clahe_tile
    dd['preprocess_params'] = pp
    cfg['defect_detection_params'] = dd
    return cfg


def summarize_defects(report: Dict[str, Any]):
    counts = {}
    for d in report.get('defects', []):
        t = d.get('type', 'UNK')
        counts[t] = counts.get(t, 0) + 1
    return counts


def run_sweep(image_path: str,
              base_config: Dict[str, Any],
              rois,
              runs: int,
              gamma_min: float,
              gamma_max: float,
              sigma_min: float,
              sigma_max: float,
              output_dir: str,
              seed: int | None,
              draw_contours: bool):
    """参数扫描（改版）
    目标：收集满足 1) 所有 ROI polygons_in_roi == 1 的样本（允许缺陷，记录统计）。
    gamma/sigma 在中心附近小范围随机；外部区间参数保持以兼容调用。
    """
    # 采样窗口（仍用固定中心范围）
    GAMMA_CENTER, GAMMA_HALF_RANGE = 0.749, 0.15   # 扩大到 ±0.15
    SIGMA_CENTER, SIGMA_HALF_RANGE = 0.140, 0.05   # 扩大到 ±0.05
    CLAHE_CLIP_MIN, CLAHE_CLIP_MAX = 1.0, 5.0
    CLAHE_TILE_MIN, CLAHE_TILE_MAX = 8, 128

    ensure_dir(output_dir)
    images_dir = os.path.join(output_dir, 'images')
    ensure_dir(images_dir)
    csv_path = os.path.join(output_dir, 'results.csv')

    if seed is not None:
        random.seed(seed)

    fieldnames = ['run','gamma','canny_sigma','clahe_clip','clahe_tile','image_status','state_code','defect_total','defect_types','output_image','output_report','elapsed_ms']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        valid_collected = 0
        attempt = 0
        while valid_collected < runs:
            attempt += 1
            gamma = random.uniform(GAMMA_CENTER - GAMMA_HALF_RANGE, GAMMA_CENTER + GAMMA_HALF_RANGE)
            sigma = random.uniform(SIGMA_CENTER - SIGMA_HALF_RANGE, SIGMA_CENTER + SIGMA_HALF_RANGE)
            clahe_clip = random.uniform(CLAHE_CLIP_MIN, CLAHE_CLIP_MAX)
            clahe_tile = random.randint(CLAHE_TILE_MIN, CLAHE_TILE_MAX)
            cfg = clone_config_for_run(base_config, gamma, sigma, clahe_clip, clahe_tile)

            t0 = time.time()
            report = image_processor_optimized.process_image(
                image_path,
                rois,
                images_dir,
                cfg,
                draw_contours=draw_contours,
                use_parallel=True
            )
            dt_ms = (time.time() - t0) * 1000.0
            if report is None:
                print(f"[尝试 {attempt}] 失败: 无法处理图像")
                continue

            rois_info = report.get('rois', [])
            if (not rois_info) or any(r.get('polygons_in_roi', 0) != 1 for r in rois_info):
                print(f"[尝试 {attempt}] 条件不满足(需要所有ROI=1轮廓)")
                continue

            # 允许缺陷，统计
            defect_counts = summarize_defects(report)
            total_defects = sum(defect_counts.values())

            valid_collected += 1
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_stub = f"{base_name}_g{gamma:.3f}_s{sigma:.3f}_run{valid_collected}".replace(' ','')
            img_name = f"{out_stub}.jpg"
            json_name = f"{out_stub}.json"
            generic_img = os.path.join(images_dir, f"{base_name}_processed.jpg")
            unique_img = os.path.join(images_dir, img_name)
            unique_json = os.path.join(images_dir, json_name)

            if os.path.isfile(generic_img):
                try:
                    os.replace(generic_img, unique_img)
                except Exception:
                    pass

            report['sweep_params'] = {
                'gamma_value': gamma,
                'canny_sigma': sigma,
                'clahe_clip_limit': clahe_clip,
                'clahe_tile_target_px': clahe_tile,
                'run_index': valid_collected,
                'attempt_index': attempt
            }
            try:
                with open(unique_json, 'w', encoding='utf-8') as jf:
                    json.dump(report, jf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[运行 {valid_collected}] JSON保存失败: {e}")

            writer.writerow({
                'run': valid_collected,
                'gamma': f"{gamma:.4f}",
                'canny_sigma': f"{sigma:.4f}",
                'clahe_clip': f"{clahe_clip:.3f}",
                'clahe_tile': clahe_tile,
                'image_status': report.get('image_status'),
                'state_code': report.get('state_code'),
                'defect_total': total_defects,
                'defect_types': json.dumps(defect_counts, ensure_ascii=False),
                'output_image': img_name,
                'output_report': json_name,
                'elapsed_ms': f"{dt_ms:.1f}"
            })
            print(f"[有效 {valid_collected}/{runs} | 尝试 {attempt}] g={gamma:.3f} s={sigma:.3f} clip={clahe_clip:.2f} tile={clahe_tile} 缺陷数={total_defects} 用时={dt_ms:.1f}ms -> {img_name}")

    print(f"完成，共 {runs} 次。结果 CSV: {csv_path}")


def run_multi_sweep(image_specs,
                    base_config: Dict[str, Any],
                    rois,
                    runs: int,
                    output_dir: str,
                    seed: int | None,
                    draw_contours: bool):
    """多图同步参数搜索：同一个 (gamma,sigma,CLAHE) 需同时满足各图片的约束。
    image_specs: 列表，每项: {
        'path': str,
        'required_roi_contours': int,
        'defect_types_only': list[str] | None
    }
    约束逻辑：
      - 每个 ROI 的 polygons_in_roi == required_roi_contours
      - 若 defect_types_only 不为空：所有缺陷类型必须属于该集合（允许 0 缺陷）
    成功后为每张图片各自保存图像/报告，公共参数写入 sweep_params。
    输出 CSV 一行对应一个成功参数集，列出所有图片是否满足及缺陷统计汇总。
    """
    ensure_dir(output_dir)
    images_root = os.path.join(output_dir, 'images')
    ensure_dir(images_root)
    csv_path = os.path.join(output_dir, 'multi_results.csv')

    # 参数范围与单图一致（中心 ± 扩大）
    GAMMA_CENTER, GAMMA_HALF_RANGE = 0.749, 0.15
    SIGMA_CENTER, SIGMA_HALF_RANGE = 0.140, 0.05
    CLAHE_CLIP_MIN, CLAHE_CLIP_MAX = 1.0, 5.0
    CLAHE_TILE_MIN, CLAHE_TILE_MAX = 8, 128

    if seed is not None:
        random.seed(seed)

    fieldnames = [
        'run','gamma','canny_sigma','clahe_clip','clahe_tile','attempts',
        'total_elapsed_ms','images','total_defects'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        valid_collected = 0
        attempt = 0
        while valid_collected < runs:
            attempt += 1
            gamma = random.uniform(GAMMA_CENTER - GAMMA_HALF_RANGE, GAMMA_CENTER + GAMMA_HALF_RANGE)
            sigma = random.uniform(SIGMA_CENTER - SIGMA_HALF_RANGE, SIGMA_CENTER + SIGMA_HALF_RANGE)
            clahe_clip = random.uniform(CLAHE_CLIP_MIN, CLAHE_CLIP_MAX)
            clahe_tile = random.randint(CLAHE_TILE_MIN, CLAHE_TILE_MAX)
            param_start = time.time()
            per_image_results = []  # 暂存，不立即落盘
            all_ok = True
            total_defects_sum = 0

            for spec in image_specs:
                img_path = spec['path']
                required = spec.get('required_roi_contours')
                allowed_types = spec.get('defect_types_only')
                # B 缺陷尺寸约束（针对 a800b_2_x 场景）：默认 <50mm * <50mm
                b_max_length = spec.get('b_defect_max_length_mm', 50.0)
                b_max_width = spec.get('b_defect_max_width_mm', 50.0)
                cfg = clone_config_for_run(base_config, gamma, sigma, clahe_clip, clahe_tile)
                report = image_processor_optimized.process_image(
                    img_path,
                    rois,
                    images_root,
                    cfg,
                    draw_contours=draw_contours,
                    use_parallel=True
                )
                if report is None:
                    print(f"[尝试 {attempt}] 图像 {img_path} 处理失败")
                    all_ok = False
                    break
                rois_info = report.get('rois', [])
                if (not rois_info) or any(r.get('polygons_in_roi', 0) != required for r in rois_info):
                    print(f"[尝试 {attempt}] {os.path.basename(img_path)} ROI轮廓不满足 {required}")
                    all_ok = False
                    break
                defects = report.get('defects', [])
                if allowed_types:
                    if any(d.get('type') not in allowed_types for d in defects):
                        print(f"[尝试 {attempt}] {os.path.basename(img_path)} 存在不允许的缺陷类型")
                        all_ok = False
                        break
                    # 对 a800b_2_x 的 B 缺陷尺寸判定（仅当允许列表为 B 且出现 B 缺陷时）
                    if set(allowed_types) == {'B'} and defects:
                        size_ok = True
                        for d in defects:
                            if d.get('type') == 'B':
                                loc = d.get('location', {})
                                length_mm = loc.get('length_mm')
                                width_mm = loc.get('width_mm')
                                if length_mm is None or width_mm is None:
                                    continue
                                if length_mm >= b_max_length or width_mm >= b_max_width:
                                    size_ok = False
                                    print(f"[尝试 {attempt}] {os.path.basename(img_path)} B缺陷尺寸超限 length={length_mm} width={width_mm} (阈值 {b_max_length}/{b_max_width})")
                                    break
                        if not size_ok:
                            all_ok = False
                            break
                defect_counts = summarize_defects(report)
                total_defects = sum(defect_counts.values())
                total_defects_sum += total_defects

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                img_dir = os.path.join(images_root, base_name)
                ensure_dir(img_dir)
                stub = f"{base_name}_g{gamma:.3f}_s{sigma:.3f}_clip{clahe_clip:.2f}_tile{clahe_tile}_runX".replace(' ','')
                generic_img = os.path.join(images_root, f"{base_name}_processed.jpg")
                target_img = os.path.join(img_dir, stub + '.jpg')
                target_json = os.path.join(img_dir, stub + '.json')
                # 暂存，稍后统一保存
                per_image_results.append({
                    'image': base_name,
                    'defects': defect_counts,
                    'roi_count': required,
                    'defect_total': total_defects,
                    'generic_img': generic_img,
                    'target_img': target_img,
                    'target_json': target_json,
                    'report': report,
                    'gamma': gamma,
                    'sigma': sigma,
                    'clahe_clip': clahe_clip,
                    'clahe_tile': clahe_tile
                })

            if not all_ok:
                continue

            valid_collected += 1
            elapsed_set_ms = (time.time() - param_start) * 1000.0
            # 统一保存：仅在全部图片通过后写入
            for r in per_image_results:
                rep = r['report']
                rep['sweep_params'] = {
                    'gamma_value': r['gamma'],
                    'canny_sigma': r['sigma'],
                    'clahe_clip_limit': r['clahe_clip'],
                    'clahe_tile_target_px': r['clahe_tile'],
                    'attempt_index': attempt,
                    'run_index': valid_collected
                }
                if os.path.isfile(r['generic_img']):
                    try:
                        os.replace(r['generic_img'], r['target_img'])
                    except Exception:
                        pass
                try:
                    with open(r['target_json'], 'w', encoding='utf-8') as jf:
                        json.dump(rep, jf, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[运行 {valid_collected}] 保存报告失败 {r['target_json']}: {e}")
            # 回填 run 号到已保存文件名（可选）——简单略过重命名以避免IO。
            writer.writerow({
                'run': valid_collected,
                'gamma': f"{gamma:.4f}",
                'canny_sigma': f"{sigma:.4f}",
                'clahe_clip': f"{clahe_clip:.3f}",
                'clahe_tile': clahe_tile,
                'attempts': attempt,
                'total_elapsed_ms': f"{elapsed_set_ms:.1f}",
                'images': ';'.join([r['image'] for r in per_image_results]),
                'total_defects': total_defects_sum
            })
            print(f"[组合有效 {valid_collected}/{runs} | 尝试 {attempt}] g={gamma:.3f} s={sigma:.3f} clip={clahe_clip:.2f} tile={clahe_tile} -> 所有图通过, 总缺陷={total_defects_sum}")

    print(f"多图扫描完成，共 {runs} 组。CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Gamma & Canny Sigma 参数随机扫描')
    parser.add_argument('--config', default='config.json', help='配置文件路径')
    parser.add_argument('--image', default='valid/b800b_x.BMP', help='输入图像路径 (默认 b800b_x.BMP)')
    parser.add_argument('--runs', type=int, default=100, help='运行次数')
    parser.add_argument('--gamma-min', type=float, default=0.5)
    parser.add_argument('--gamma-max', type=float, default=1.5)
    parser.add_argument('--sigma-min', type=float, default=0.1)
    parser.add_argument('--sigma-max', type=float, default=0.5)
    parser.add_argument('--output', default='param_sweep_results', help='输出目录')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (可复现)')
    parser.add_argument('--draw-contours', action='store_true', help='绘制轮廓')
    parser.add_argument('--multi', action='store_true', help='启用多图联合筛选')
    args = parser.parse_args()

    cfg = load_config(args.config)
    rois = load_template_rois(cfg, cam_idx=0)
    if not rois:
        print('[警告] 未找到 ROI，继续但可能无有效结果。')

    if args.multi:
        image_specs = [
            { 'path': 'valid/b800b_x.BMP', 'required_roi_contours': 1, 'defect_types_only': None },
            { 'path': 'valid/a800b_2_x.BMP', 'required_roi_contours': 1, 'defect_types_only': ['B'] },
            { 'path': 'valid/b800q.BMP', 'required_roi_contours': 1, 'defect_types_only': None },
            { 'path': 'valid/b800b_S.BMP', 'required_roi_contours': 1, 'defect_types_only': None }
        ]
        run_multi_sweep(
            image_specs=image_specs,
            base_config=cfg,
            rois=rois,
            runs=args.runs,
            output_dir=args.output,
            seed=args.seed,
            draw_contours=args.draw_contours
        )
    else:
        run_sweep(
            image_path=args.image,
            base_config=cfg,
            rois=rois,
            runs=args.runs,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            output_dir=args.output,
            seed=args.seed,
            draw_contours=args.draw_contours
        )


if __name__ == '__main__':
    main()
