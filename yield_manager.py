"""统计管理器 (精简版)

仅保留剔废(rejections) 计数的持久化/报告功能。
对历史版本兼容：旧文件含有 yield 字段与旧签名(无 rejections 或不同签名算法)时，读取后产量直接忽略并返回 0。
返回接口保持 (date, yield, rejections) 形式以避免外部调用崩溃，但 yield 恒为 0。
"""

import json
import hashlib
import base64
from datetime import datetime
import os

SECRET_KEY = "$$SWJTU$$GLASS&&"
STATS_FILE_PATH = "persistent_stats.dat"
REPORTS_DIR = "daily_reports"
FORMAT_VERSION = 2  # 1: 旧(含 yield) 2: 新(仅 rejections)

def save_stats(date_str: str, _unused_yield: int, rejection_count: int):
    """保存当日实时剔废统计 (兼容旧签名字段, yield 固定写 0)。"""
    try:
        data = {
            'date': date_str,
            'yield': 0,  # 占位保持字段，固定0
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

def save_daily_report(date_str: str, _unused_yield: int, rejection_count: int):
    """保存每日归档报告 (只含剔废)。"""
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        report = {
            'report_date': date_str,
            'total_rejections': int(rejection_count),
            'ver': FORMAT_VERSION
        }
        verify_str = f"{report['report_date']}-{report['total_rejections']}-{SECRET_KEY}"
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
        old_yield = int(data.get('yield', 0))  # 兼容读取

        # 新签名
        verify_new = f"{date_str}-{0}-{rej}-{SECRET_KEY}"
        sig_new = hashlib.sha256(verify_new.encode('utf-8')).hexdigest()
        # 旧签名(含 yield 但无 rejections 或不同结构)
        verify_old = f"{date_str}-{old_yield}-{rej}-{SECRET_KEY}"
        sig_old_variant = hashlib.sha256(verify_old.encode('utf-8')).hexdigest()
        verify_old_legacy = f"{date_str}-{old_yield}-{SECRET_KEY}"
        sig_old_legacy = hashlib.sha256(verify_old_legacy.encode('utf-8')).hexdigest()

        if sig not in (sig_new, sig_old_variant, sig_old_legacy):
            print("⚠️ [统计管理器]: 校验失败，重置计数。")
            return today, 0, 0
        if date_str != today:
            print("ℹ️ [统计管理器]: 跨日，重置计数。")
            return today, 0, 0
        print(f"✅ [统计管理器]: 加载成功 (剔废={rej})")
        return date_str, 0, rej
    except FileNotFoundError:
        print("ℹ️ [统计管理器]: 无历史文件，初始化。")
        return today, 0, 0
    except Exception as e:
        print(f"❌ [统计管理器]: 读取失败: {e}")
        return today, 0, 0