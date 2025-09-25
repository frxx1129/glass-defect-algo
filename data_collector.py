"""Minimal data collector with extremely simplified logic.

Features:
- Connect to real cameras (uses CameraManager from existing project) sequentially in one process.
- Load averaged ROIs from a json file (same schema as existing averaged ROI files).
- For each grabbed frame, run simple line detection per ROI.
- If ANY line exists in an ROI, save ONLY that ROI cropped image into the current pane folder.
- Maintain a very lightweight pane state machine (time based):
    * A pane starts when first ROI with line appears while idle.
    * Continues accumulating until max_duration_s OR explicit idle timeout (gap_timeout_s since last saved ROI) then closes.
- When a pane closes, if there were any saved ROI images, rename the pane folder with suffix '_NG'.
- NO server, NO FastAPI, NO complex rejection logic.

Run:
    python data_collector.py

Config file: data_collector_config.json

"""
from __future__ import annotations
import os, sys, json, time, ctypes, traceback
from datetime import datetime
import threading
import cv2
import numpy as np

from camera_manager import CameraManager, MultiCameraSetup
from collector_hough_processor import process_frame

CONFIG_FILE = 'data_collector_config.json'

# ---------------- Utility -----------------

def load_config(path=CONFIG_FILE):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def pick_roi_group(averaged_json: dict, strategy: str):
    if not averaged_json:
        return []
    if strategy == 'largest_source_image_count':
        best_key = None; best_val = -1
        for k,v in averaged_json.items():
            cnt = v.get('source_image_count', 0)
            if cnt > best_val:
                best_val = cnt; best_key = k
        if best_key is None:
            best_key = next(iter(averaged_json))
        return averaged_json[best_key].get('averaged_rois', [])
    # fallback first
    first_key = next(iter(averaged_json))
    return averaged_json[first_key].get('averaged_rois', [])

def load_rois_from_file(path, strategy):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    group = pick_roi_group(data, strategy)
    out = []
    for r in group:
        try:
            x=int(r.get('x')); y=int(r.get('y'))
            w=int(r.get('w', r.get('width',0))); h=int(r.get('h', r.get('height',0)))
            if w>0 and h>0:
                out.append({'x':x,'y':y,'w':w,'h':h})
        except Exception:
            continue
    return out

def load_camera_rois(cfg):
    rs = cfg.get('roi_sources', {})
    strat = rs.get('default_group_strategy', 'largest_source_image_count')
    cam_entries = rs.get('cameras', [])
    # fallback: if roi_sources empty, build from camera_rois mapping
    if not cam_entries:
        cam_rois_map_cfg = cfg.get('camera_rois', {}) or {}
        for k,v in cam_rois_map_cfg.items():
            try:
                cam_entries.append({'index': int(k), 'file': v})
            except Exception:
                pass
    mapping = {}
    for entry in cam_entries:
        try:
            idx = int(entry.get('index'))
            file = entry.get('file')
            if not file or not os.path.exists(file):
                print(f"[ROI] 文件缺失: {file}")
                continue
            mapping[idx] = load_rois_from_file(file, strat)
            print(f"[ROI] 相机{idx} 加载ROI {len(mapping[idx])} 条 from {file}")
        except Exception as e:
            print(f"[ROI] 加载失败: {e}")
    return mapping

# ------------- Simple Camera Wrapper (single thread sequential) ---------

def init_cameras(config):
    cs = config.get('camera_setup', {})
    expected = int(cs.get('expected_cameras', 1) or 1)
    setup_tool = MultiCameraSetup({'camera_setup': cs})  # 传入完整结构以便其使用 network_interfaces 等
    setup_tool.bootstrap_mac_bindings()  # will not harm if already present
    # open sequentially by mapping from config after bootstrap
    bindings = cs.get('camera_bindings', [])
    cameras = []
    if not bindings:
        # fallback to physical index open attempts 0..expected-1
        for idx in range(expected):
            cam = CameraManager(idx)
            if cam.open():
                cameras.append(cam)
        return cameras
    unified_params = cs.get('unified_params', {}) or {}
    runtime_cfg = cs.get('runtime', {}) or {}
    for b in bindings:
        idx = b.get('index'); mac = b.get('mac')
        cam = CameraManager(idx, mac=mac)
    if cam.open():
            try:
                # 应用统一参数（与主程序保持一致调用）
                cam.set_params(unified_params)
        # NOTE: 若需要软件触发，可在此后调用 MVStartGrab + 周期 MVTriggerSoftware
        # runtime_cfg.get('use_software_trigger') 暂未强制实现触发循环, 简化为连续抓取模式
            except Exception as e:
                print(f"[Init] 应用相机参数失败 idx={idx}: {e}")
            cameras.append(cam)
    return cameras

# ------------- Pane State Machine (time based) -------------
class PaneSession:
    def __init__(self, root_dir, max_duration_s, gap_timeout_s):
        self.root_dir = root_dir
        self.max_duration_s = max_duration_s
        self.gap_timeout_s = gap_timeout_s
        self.reset()
    def reset(self):
        self.active = False
        self.start_ts = 0.0
        self.last_save_ts = 0.0
        self.folder = None
        self.saved_count = 0
        self.temp_folder = None
    def start(self):
        ts = time.time()
        self.active = True
        self.start_ts = ts
        self.last_save_ts = ts
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        self.folder = os.path.join(self.root_dir, ts_str)
        os.makedirs(self.folder, exist_ok=True)
        self.temp_folder = self.folder  # same during active
        print(f"[Pane] Started: {self.folder}")
    def save_roi(self, cam_idx, roi_idx, img_roi):
        fname = f"cam{cam_idx}_roi{roi_idx}_{self.saved_count+1:04d}.png"
        cv2.imwrite(os.path.join(self.folder, fname), img_roi)
        self.saved_count += 1
        self.last_save_ts = time.time()
    def should_close(self):
        if not self.active: return False
        now = time.time()
        if now - self.start_ts >= self.max_duration_s:
            return True
        if self.saved_count>0 and (now - self.last_save_ts) >= self.gap_timeout_s:
            return True
        return False
    def close(self):
        if not self.active:
            return None
        final_folder = self.folder
        had_ng = self.saved_count > 0
        self.active = False
        if had_ng:
            # rename with suffix _NG
            new_name = self.folder + '_NG'
            try:
                if not os.path.exists(new_name):
                    os.rename(self.folder, new_name)
                    final_folder = new_name
            except Exception as e:
                print(f"[Pane] Rename NG failed: {e}")
        print(f"[Pane] Closed: {final_folder} (saved={self.saved_count})")
        self.reset()
        return final_folder, had_ng

# ------------- Main Loop -------------

def run():
    cfg = load_config()
    out_root = cfg.get('output_root', 'simple_inspections')
    os.makedirs(out_root, exist_ok=True)
    cam_rois_map = load_camera_rois(cfg)
    pane_cfg = cfg.get('pane', {})
    max_dur = float(pane_cfg.get('max_duration_s', 8.0))
    gap_to = float(pane_cfg.get('gap_timeout_s', 2.0))
    min_roi_interval = float(pane_cfg.get('min_save_interval_s', 0.5))

    line_params_raw = cfg.get('line_detection', {})
    # convert theta if provided
    lp = dict(line_params_raw)
    if 'hough_theta_deg' in lp:
        lp['hough_theta'] = np.deg2rad(float(lp.pop('hough_theta_deg')))

    cameras = init_cameras(cfg)
    if not cameras:
        print('[Init] No cameras opened, exit.')
        return
    print(f"[Init] Opened {len(cameras)} camera(s).")

    pane = PaneSession(out_root, max_dur, gap_to)
    last_roi_save_time = {}

    try:
        while True:
            for cam_idx, cam in enumerate(cameras):
                # For simplicity: use SDK function MVGetOneFrameTimeout if available else skip. We assume it exists per typical SDK.
                try:
                    from MVGigE import MVGetOneFrameTimeout, MVST_SUCCESS
                    import numpy.ctypeslib as npct
                    # Query width/height only once? For simplicity each loop (cheap)
                    from MVGigE import MVGetWidth, MVGetHeight, MVGetPixelFormat
                    _, w = MVGetWidth(cam.handle); _, h = MVGetHeight(cam.handle)
                    _, pf = MVGetPixelFormat(cam.handle)
                    buf = (ctypes.c_ubyte * (w*h))()  # Mono8 assumption
                    res, out_w, out_h, out_fmt, frame_cnt = MVGetOneFrameTimeout(cam.handle, buf, len(buf), 50)  # 50ms timeout
                    if res != MVST_SUCCESS:
                        continue
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(out_h, out_w)
                except Exception:
                    continue

                # Ensure pane started when line appears
                rois_this_cam = cam_rois_map.get(cam_idx) or []
                if not rois_this_cam:
                    continue
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                try:
                    report, _, per_rois = process_frame(gray, rois_this_cam, cfg, thread_workers=int(cfg.get('system_params',{}).get('roi_threads',0) or 0) or None)
                except Exception as e:
                    print(f"[Proc] 相机{cam_idx} 处理异常: {e}")
                    continue
                # 缺陷判定: 只要任一 ROI defects 非空 -> 触发 pane
                has_defect = any(len(r[1].get('defects', []))>0 for r in per_rois)
                if has_defect and not pane.active:
                    pane.start()
                if pane.active and has_defect:
                    for (roi_idx, roi_report, roi_img) in per_rois:
                        if len(roi_report.get('defects', [])) <= 0:
                            continue
                        x = roi_report['x']; y=roi_report['y']; w=roi_report['w']; h=roi_report['h']
                        now = time.time()
                        key = (cam_idx, roi_idx)
                        if now - last_roi_save_time.get(key, 0.0) < min_roi_interval:
                            continue
                        raw_roi = gray[y:y+h, x:x+w]
                        # 保存 RAW
                        pane.save_roi(cam_idx, roi_idx, raw_roi)
                        try:
                            fname_ann = f"cam{cam_idx}_roi{roi_idx}_{pane.saved_count:04d}_ann.png"
                            cv2.imwrite(os.path.join(pane.folder, fname_ann), roi_img)
                        except Exception as e:
                            print(f"[Save] annotated失败: {e}")
                        last_roi_save_time[key] = now

                if pane.should_close():
                    pane.close()
            # small sleep to avoid 100% CPU (dependent on fps)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('\n[Main] Interrupted by user.')
    finally:
        if pane.active:
            pane.close()
        for cam in cameras:
            try: cam.close()
            except Exception: pass

if __name__ == '__main__':
    run()
