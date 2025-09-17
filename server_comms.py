# --- START OF FILE server_comms.py (修改后) ---
import requests
import json
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit


def _mask_port(url: str) -> str:
    """返回不包含端口的 URL（保留协议、地址、路径与查询）。"""
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ''
        if ':' in hostname and not hostname.startswith('['):
            hostname = f"[{hostname}]"
        safe_netloc = hostname
        return urlunsplit((parts.scheme, safe_netloc, parts.path, "", parts.fragment))
    except Exception:
        return url.split(':')[0]

def _host_only(url: str) -> str:
    """仅返回主机地址（不含协议、端口、路径、查询）。"""
    try:
        parts = urlsplit(url)
        return parts.hostname or "<unknown-host>"
    except Exception:
        return "<unknown-host>"

# 注意：同步函数保持不变，因为它们需要返回值
def fetch_initial_state_from_server(settings):
    print(f"    [HTTP]: 准备获取初始状态...")
    try:
        server_ip = settings.server
        line_name = settings.lineName
        target_url = f"http://{server_ip}:8085/fastapi/glass/getAlgInitStatus?lineName={line_name}"
        response = requests.get(target_url, timeout=getattr(settings, 'http_get_timeout_s', 5))

        if response.status_code == 200:
            json_data = response.json()
            data = json_data.get("data")
            if data and isinstance(data, dict):
                print(f"    [HTTP]: 成功获取初始状态")
                return data
            else:
                print(f"    [HTTP]: 错误 - 响应中未找到有效的 'data' 字段。")
                return None
        else:
            host = _host_only(target_url)
            print(f"    [HTTP]: 请求失败 -> {host}")
            return None

    except Exception:
        try:
            host = _host_only(target_url)
        except Exception:
            host = "<unknown>"
        print(f"    [HTTP]: 网络/未知异常 -> {host}")
        return None


def fetch_collection_id_from_server(settings, collection_id_var):
    """获取 Collection ID (整数)，失败写入 -1。"""
    print(f"    [HTTP]: 准备获取 collection_id...")
    try:
        server_ip = settings.server
        line_name = settings.lineName
        target_url = f"http://{server_ip}:8085/fastapi/glass/getTodayCollectionByLine"
        request_url = f"{target_url}?lineName={line_name}"
        print(f"    [HTTP]: 正在向服务器请求 ({line_name})...")
        response = requests.get(request_url, timeout=getattr(settings, 'http_get_timeout_s', 5))
        if response.status_code == 200:
            json_data = response.json(); data = json_data.get("data")
            if isinstance(data, dict) and data.get("id") is not None:
                try:
                    collection_id_var.value = int(data.get("id"))
                except Exception:
                    collection_id_var.value = -1
                if collection_id_var.value >= 0:
                    print("    [HTTP]: 成功获取并更新 Collection ID")
                else:
                    print("    [HTTP]: ID 解析失败 -> -1")
            else:
                print("    [HTTP]: 响应缺少有效 data.id")
                collection_id_var.value = -1
        else:
            host = _host_only(request_url)
            print(f"    [HTTP]: 请求失败 -> {host}")
            collection_id_var.value = -1
    except Exception:
        try:
            host = _host_only(target_url)
        except Exception:
            host = "<unknown>"
        print(f"    [HTTP]: 网络/未知异常 -> {host}")
        collection_id_var.value = -1

# =================================================================
# --- 以下函数被修改为非阻塞 ---
# =================================================================

## 删除 periodic_stats_pusher：仅保留事件触发剔废广播。


def send_report_to_server(json_report, image_buffer, server_url, http_client, upload_timeout_s: float | None = None):
    """(修改) 使用非阻塞客户端上传缺陷报告。"""
    collection_id = json_report.get("collection_id", -1)
    try:
        collection_id = int(collection_id)
    except Exception:
        collection_id = -1
    rejection_time_str = json_report.get("rejection_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        rejection_dt = datetime.strptime(rejection_time_str, "%Y-%m-%d %H:%M:%S")
        time_str = rejection_dt.strftime('%H%M%S')
    except ValueError:
        time_str = "000000"
    cam_index = json_report.get("camera_index", "X")
    image_filename = f"{time_str}_{collection_id}_Cam{cam_index}.jpg"

    # 文件内容和名称
    file_data = (image_filename, image_buffer, 'image/jpeg')
    
    print(f"    [上传模块]: 正在提交报告上传任务")
    # 使用专门的文件上传方法
    # 允许从 shared_settings 传入 http_upload_timeout_s；此处无法直接访问 settings，按默认15s处理
    http_client.post_files(server_url, files=file_data, json_payload=json_report, timeout_s=upload_timeout_s)


def broadcast_rejections(settings, collection_id, rejection_count, http_client):
    """广播剔废数量（删除产量字段）。"""
    try:
        cid_raw = getattr(collection_id, 'value', collection_id)
        if isinstance(cid_raw, (bytes, bytearray)):
            cid_str = cid_raw.decode('utf-8', errors='replace')
        else:
            cid_str = str(cid_raw)
        cid_int = int(cid_str)
    except Exception:
        cid_int = -1

    payload = {
        "collectionId": cid_int,
        "rejection": int(rejection_count.value)
    }
    target_url = f"http://{settings.server}:8085/fastapi/glass/updateRejections"
    http_client.post(target_url, json=payload, timeout=getattr(settings, 'http_post_timeout_s', 5))