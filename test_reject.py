import argparse
import json
import os
import sys
import time
import yaml

from rejection_controller import RejectionController


def load_config(path_candidates):
	for p in path_candidates:
		if os.path.exists(p):
			try:
				with open(p, 'r', encoding='utf-8') as f:
					if p.endswith(('.yml', '.yaml')):
						return yaml.safe_load(f)
					try:
						return yaml.safe_load(f)  # YAML 可解析 JSON
					except Exception:
						f.seek(0)
						return json.load(f)
			except Exception as e:
				print(f"[test] 读取配置失败 {p}: {e}")
	return {}


def resolve_marks_to_channels(marks, cfg, line_name):
	if marks is None:
		return []
	if not isinstance(marks, (list, tuple, set)):
		marks_iter = [marks]
	else:
		marks_iter = list(marks)
	mapping = {}
	try:
		mm_all = cfg.get('rejection_mark_to_channel', {}) or {}
		if isinstance(mm_all, dict):
			mapping = mm_all.get(line_name) or {}
	except Exception:
		mapping = {}
	channels = set()
	for m in marks_iter:
		try:
			key = str(int(m))
		except Exception:
			continue
		ch = mapping.get(key)
		if ch is None:
			try:
				ch = int(m) + 1  # 回退: mark+1
			except Exception:
				continue
		try:
			ch_i = int(ch)
			if 1 <= ch_i <= 8:
				channels.add(ch_i)
		except Exception:
			continue
	return sorted(channels)


def burst_pulse(controller: RejectionController, channels, burst_count, on_ms, cycle_ms):
	if not channels:
		print("[test] 无有效通道，跳过。")
		return
	on_ms = max(10, int(on_ms))
	cycle_ms = max(on_ms, int(cycle_ms))
	off_ms = max(0, cycle_ms - on_ms)
	on_s = on_ms / 1000.0
	off_s = off_ms / 1000.0
	print(f"[test] 开始脉冲串: channels={channels}, bursts={burst_count}, on={on_ms}ms, cycle={cycle_ms}ms")
	start = time.time()
	for i in range(int(burst_count)):
		# ON 同步
		for ch in channels:
			controller.send_command(ch, 0x01)
		time.sleep(on_s)
		# OFF 同步
		for ch in channels:
			controller.send_command(ch, 0x00)
		if i < burst_count - 1 and off_s > 0:
			time.sleep(off_s)
	elapsed = time.time() - start
	print(f"[test] 脉冲串完成，总耗时约 {elapsed:.2f}s")


def test_scan(controller: RejectionController, delay_ms: int = 300):
	print("[test] 顺序扫描通道 1..8")
	for ch in range(1, 9):
		controller.send_command(ch, 0x01)
		time.sleep(delay_ms / 1000.0)
		controller.send_command(ch, 0x00)
		time.sleep(0.1)
	print("[test] 扫描完成。")


def main():
	parser = argparse.ArgumentParser(description="剔废继电器测试程序 (脉冲串 / 通道扫描)")
	parser.add_argument('--port', default='COM5', help='串口号 (默认 COM5)')
	parser.add_argument('--baud', type=int, default=115200, help='波特率 (默认 115200)')
	parser.add_argument('--marks', type=int, nargs='*', help='要触发的 mark 集合 (0 基, 会映射为通道)')
	parser.add_argument('--line-name', default=None, help='产线名称(覆盖配置中的 lineName)')
	parser.add_argument('--burst-count', type=int, default=10, help='脉冲串次数 (默认10)')
	parser.add_argument('--on-ms', type=int, default=250, help='每次上电时长 ms (默认250)')
	parser.add_argument('--cycle-ms', type=int, default=500, help='总周期 ms (默认500, off=cycle-on)')
	parser.add_argument('--scan', action='store_true', help='执行 1..8 通道扫描测试')
	parser.add_argument('--dry', action='store_true', help='干运行(不真实开关, 仅打印包内容)')
	args = parser.parse_args()

	cfg = load_config(['config.yaml', 'config.json'])
	line_name = args.line_name or cfg.get('lineName') or 'Line1'

	# 初始化控制器
	controller = RejectionController(port=None if args.dry else args.port, baud=args.baud)

	if args.scan:
		test_scan(controller)

	if args.marks:
		channels = resolve_marks_to_channels(args.marks, cfg, line_name)
	else:
		channels = []

	if channels:
		burst_pulse(controller, channels, args.burst_count, args.on_ms, args.cycle_ms)
	else:
		if not args.scan:
			print("[test] 未提供 marks 或映射为空，使用 --marks 或 --scan 进行测试。")

	controller.close()


if __name__ == '__main__':
	main()
