"""统计管理器

恢复对产量(yield) 与 剔废(rejections) 的双计数持久化：
 - 每片玻璃进入+离开一次 => 产量 +1 (若之后未被/未曾剔废)
 - 自动 / 即时手动剔废：不计入该片产量 (本片不+1)
 - 滞后手动剔废：在先前已 +1 的情况下需回滚产量 (-1) 并增加剔废

兼容旧版本（仅剔废或旧签名）文件：若签名不匹配或缺字段则重置为 0,0。
返回 (date, yield, rejections)。
"""

import json
import hashlib
import base64
from datetime import datetime
import os

SECRET_KEY = "$$SWJTU$$GLASS&&"
STATS_FILE_PATH = "persistent_stats.dat"
REPORTS_DIR = "daily_reports"
FORMAT_VERSION = 3  # 1: 旧(含 yield) 2: 仅剔废 3: 恢复双计数

def save_stats(date_str: str, yield_count: int, rejection_count: int):
    """保存当日实时统计（产量 + 剔废）。"""
    try:
        data = {
            'date': date_str,
            'yield': int(yield_count),
            'rejections': int(rejection_count),
            'ver': FORMAT_VERSION
        }
        verify_str = f"{data['date']}-{data['yield']}-{data['rejections']}-{SECRET_KEY}"
        signature = hashlib.sha256(verify_str.encode('utf-8')).hexdigest()
        payload = {'data': data, 'signature': signature}
        encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode('utf-8'))
        with open(STATS_FILE_PATH, 'wb') as f:
            f.write(encoded)
    except Exception as e:
        print(f"❌ [统计管理器]: 保存实时统计失败: {e}")

def save_daily_report(date_str: str, yield_count: int, rejection_count: int):
    """保存每日归档报告 (含产量与剔废)。"""
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report = {
            'report_date': date_str,
            'total_yield': int(yield_count),
            'total_rejections': int(rejection_count),
            'ver': FORMAT_VERSION
        }
        verify_str = f"{report['report_date']}-{report['total_yield']}-{report['total_rejections']}-{SECRET_KEY}"
        signature = hashlib.sha256(verify_str.encode('utf-8')).hexdigest()
        wrapped = {'report': report, 'signature': signature}
        path = os.path.join(REPORTS_DIR, f"stats_report_{date_str}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(wrapped, f, ensure_ascii=False, indent=4)
        print(f"✅ [统计管理器]: 已保存每日报告 -> {path}")
    except Exception as e:
        print(f"❌ [统计管理器]: 保存每日报告失败: {e}")

def load_stats():
    """读取实时统计；失败/跨日返回 (today,0,0)。"""
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        with open(STATS_FILE_PATH, 'rb') as f:
            raw = f.read()
        payload = json.loads(base64.b64decode(raw).decode('utf-8'))
        data = payload.get('data', {})
        sig = payload.get('signature', '')
        date_str = data.get('date')
        rej = int(data.get('rejections', 0))
        yld = int(data.get('yield', 0))

        # 当前签名（格式3）
        verify_v3 = f"{date_str}-{yld}-{rej}-{SECRET_KEY}"
        sig_v3 = hashlib.sha256(verify_v3.encode('utf-8')).hexdigest()
        # 旧版本兼容（v2: yield 恒为0）
        verify_v2 = f"{date_str}-{0}-{rej}-{SECRET_KEY}"
        sig_v2 = hashlib.sha256(verify_v2.encode('utf-8')).hexdigest()
        # 最旧版本(可能无 rejections) 不再完全兼容，只要匹配 v3/v2 即接受
        if sig not in (sig_v3, sig_v2):
            print("⚠️ [统计管理器]: 校验失败，重置计数。")
            return today, 0, 0
        if date_str != today:
            print("ℹ️ [统计管理器]: 跨日，重置计数。")
            return today, 0, 0
        print(f"✅ [统计管理器]: 加载成功 (产量={yld}, 剔废={rej})")
        return date_str, yld, rej
    except FileNotFoundError:
        print("ℹ️ [统计管理器]: 无历史文件，初始化。")
        return today, 0, 0
    except Exception as e:
        print(f"❌ [统计管理器]: 读取失败: {e}")
        return today, 0, 0