import cv2
import yaml
import json
import importlib
import sys

# 重定向 print 输出到列表
debug_lines = []
original_print = print

def custom_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    debug_lines.append(msg)

# 替换 print
import builtins
builtins.print = custom_print

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 加载处理器
image_processor = importlib.import_module('image_processor_hough')

# 加载 ROI
with open('line3cam3_roi_averaged_by_group.json', 'r', encoding='utf-8') as f:
    roi_data = json.load(f)
template_rois = roi_data['group_with_2_rois']['averaged_rois']
custom_print(f"ROI 配置: {template_rois}")

# 读取视频第一帧
cap = cv2.VideoCapture('inputs/line3/cam03_2026-01-09T11-00-21-289.webm')
ret, frame = cap.read()
cap.release()

if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 注入运行时参数
    hough_params = config.get('hough_inspector_params', {})
    hough_params['_RUNTIME_LINE_NAME'] = 'Line3'
    hough_params['_RUNTIME_CAM_INDEX'] = 2
    
    custom_print('处理第一帧...')
    pane_json, annotated = image_processor.process_image_from_memory_parallel(gray, template_rois, config)
    custom_print(f"结果: 状态={pane_json['image_status']}, state_code={pane_json['state_code']}, 缺陷={len(pane_json.get('defects',[]))}")
    
    # 保存带标注的图像
    cv2.imwrite('debug_frame1.png', annotated)
    custom_print('已保存 debug_frame1.png')
else:
    custom_print('无法读取视频')

# 恢复 print 并输出所有调试行
builtins.print = original_print
for line in debug_lines:
    original_print(line)

