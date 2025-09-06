import json
import hashlib
import base64
from datetime import datetime
import os # <-- Import the os module

# Used for signing, can be any complex string
SECRET_KEY = "$$SWJTU$$GLASS&&" 
STATS_FILE_PATH = "persistent_stats.dat"
REPORTS_DIR = "daily_reports" # <-- New: Define a directory for reports

def save_stats(date_str, yield_count, rejection_count):
    """Saves the current operational stats to a secure binary file."""
    try:
        data = {'date': date_str, 'yield': yield_count, 'rejections': rejection_count}
        verify_str = f"{data['date']}-{data['yield']}-{data['rejections']}-{SECRET_KEY}"
        signature = hashlib.sha256(verify_str.encode('utf-8')).hexdigest()
        
        payload = {'data': data, 'signature': signature}
        encoded_payload = base64.b64encode(json.dumps(payload).encode('utf-8'))
        
        with open(STATS_FILE_PATH, 'wb') as f:
            f.write(encoded_payload)
        
    except Exception as e:
        print(f"❌ [统计管理器]: 保存实时统计文件时发生错误: {e}")

# --- NEW FUNCTION ---
def save_daily_report(date_str, yield_count, rejection_count):
    """
    Saves the final counts for a given day to a human-readable, signed JSON report.
    This file is for archival and external viewing.
    """
    try:
        # Ensure the reports directory exists
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        report_data = {
            "report_date": date_str,
            "total_yield": yield_count,
            "total_rejections": rejection_count
        }
        
        # Create a signature to verify the integrity of the report
        verify_str = f"{report_data['report_date']}-{report_data['total_yield']}-{report_data['total_rejections']}-{SECRET_KEY}"
        signature = hashlib.sha256(verify_str.encode('utf-8')).hexdigest()
        
        # Add the signature to the report
        report_data_with_signature = {
            "report": report_data,
            "signature": signature
        }
        
        # Define the file path
        report_path = os.path.join(REPORTS_DIR, f"stats_report_{date_str}.json")
        
        # Write the JSON file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data_with_signature, f, indent=4)
            
        print(f"✅ [统计管理器]: 已保存 {date_str} 的每日报告到 {report_path}")

    except Exception as e:
        print(f"❌ [统计管理器]: 保存每日报告时发生错误: {e}")


def load_stats():
    """
    Loads the operational stats from the secure binary file.
    Resets stats if the date has changed or the file is invalid.
    """
    # ... (This function remains unchanged) ...
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    try:
        with open(STATS_FILE_PATH, 'rb') as f:
            encoded_payload = f.read()
        
        payload = json.loads(base64.b64decode(encoded_payload).decode('utf-8'))
        data = payload['data']
        signature_from_file = payload['signature']
        
        rejections = data.get('rejections', 0)
        verify_str_new = f"{data['date']}-{data['yield']}-{rejections}-{SECRET_KEY}"
        expected_signature_new = hashlib.sha256(verify_str_new.encode('utf-8')).hexdigest()

        verify_str_old = f"{data['date']}-{data['yield']}-{SECRET_KEY}"
        expected_signature_old = hashlib.sha256(verify_str_old.encode('utf-8')).hexdigest()

        if signature_from_file != expected_signature_new and signature_from_file != expected_signature_old:
            print("⚠️ [统计管理器]: 统计文件校验失败！文件可能已被修改。将从0开始计数。")
            return today_str, 0, 0
            
        if data['date'] != today_str:
            print(f"ℹ️ [统计管理器]: 新的一天开始，统计将从0重新计数。")
            # The calling function in state_machine will handle saving the report
            return today_str, 0, 0
            
        loaded_yield = int(data['yield'])
        loaded_rejections = int(data.get('rejections', 0))
        print(f"✅ [统计管理器]: 成功加载本日统计: 产量 {loaded_yield}, 剔废 {loaded_rejections}")
        return data['date'], loaded_yield, loaded_rejections

    except FileNotFoundError:
        print("ℹ️ [统计管理器]: 未找到统计文件，将从0开始计数。")
        return today_str, 0, 0
    except Exception as e:
        print(f"❌ [统计管理器]: 读取统计文件时发生错误: {e}。将从0开始计数。")
        return today_str, 0, 0