# --- START OF FILE http_client.py ---
import warnings
# 在 import requests 之前设置过滤，避免导入时弹 RequestsDependencyWarning
warnings.filterwarnings('ignore', message='Unable to find acceptable character detection dependency')

import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit


def _mask_port(url: str) -> str:
    """返回不包含端口的 URL（保留协议、地址、路径与查询）。"""
    try:
        parts = urlsplit(url)
        # 仅保留主机名，不携带端口/用户名/密码
        hostname = parts.hostname or ''
        # IPv6 主机名在 urlunsplit 时需要保留方括号
        if ':' in hostname and not hostname.startswith('['):
            hostname = f"[{hostname}]"
        safe_netloc = hostname
        return urlunsplit((parts.scheme, safe_netloc, parts.path, "", parts.fragment))
    except Exception:
        # 出现解析异常时，退化为移除冒号后的内容
        return url.split(':')[0]

def _host_only(url: str) -> str:
    """仅返回主机地址（不含协议、端口、路径、查询）。"""
    try:
        parts = urlsplit(url)
        return parts.hostname or "<unknown-host>"
    except Exception:
        return "<unknown-host>"

class NonBlockingHttpClient:
    """
    一个非阻塞的HTTP客户端，使用后台线程池来发送请求，
    避免阻塞主应用程序线程。
    """
    def __init__(self, max_workers=4):
        """
        初始化线程池。
        :param max_workers: 线程池中的最大并发工作线程数。
        """
        print(f"✅ [HTTP客户端]: 初始化非阻塞HTTP客户端，最大工作线程数: {max_workers}")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _send_request(self, method, url, **kwargs):
        """
        实际在后台线程中执行的请求函数。
        """
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code != 200:
                # 详细错误：方法、完整URL、状态码、原因、截断响应体
                body = None
                try:
                    body = response.text
                    if body is not None and len(body) > 500:
                        body = body[:500] + f"... (truncated {len(response.text) - 500} chars)"
                except Exception:
                    body = "<unable to read response body>"
                print(
                    f"    [HTTP后台]: 请求失败 -> {method.upper()} {url} | 状态码: {response.status_code} | 原因: {getattr(response, 'reason', '')} | 响应: {body}"
                )
        except requests.exceptions.RequestException as e:
            # 网络/协议异常，输出异常类型与详情
            print(
                f"    [HTTP后台]: 网络异常 -> {method.upper()} {url} | 异常: {e.__class__.__name__}: {e}"
            )
        except Exception as e:
            print(
                f"    [HTTP后台]: 未知错误 -> {method.upper()} {url} | 异常: {e.__class__.__name__}: {e}"
            )

    def post(self, url, **kwargs):
        """
        以非阻塞方式发送POST请求。
        """
        self.executor.submit(self._send_request, 'post', url, **kwargs)

    def get(self, url, **kwargs):
        """
        以非阻塞方式发送GET请求。
        """
        self.executor.submit(self._send_request, 'get', url, **kwargs)

    def post_files(self, url, files, json_payload, timeout_s: float | None = None):
        """
        专门用于文件上传的非阻塞POST方法。
        """
        # 为了传递files参数，我们需要一个专用的工作函数
        def _upload_task():
            try:
                # --- 新增: 递归清洗 JSON 数据中的 bytes，避免 json.dumps 抛出 TypeError ---
                def _sanitize(obj):
                    if isinstance(obj, (bytes, bytearray)):
                        # 尝试 utf-8 解码；失败则转 base64 文本，保证可 JSON 序列化
                        try:
                            return obj.decode('utf-8')
                        except Exception:
                            import base64
                            return base64.b64encode(obj).decode('ascii')
                    if isinstance(obj, dict):
                        return {k: _sanitize(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_sanitize(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_sanitize(v) for v in obj)
                    return obj
                safe_payload = _sanitize(json_payload)
                # 构建 multipart/form-data
                payload_tuple = ('payload', (None, json.dumps(safe_payload, ensure_ascii=False), 'application/json'))
                all_files = [('file', files), payload_tuple]

                to = 15 if timeout_s is None else float(timeout_s)
                response = requests.post(url, files=all_files, timeout=to)
                if response.status_code == 200:
                    print(f"    [上传模块-后台]: 报告发送成功。")
                else:
                    body = None
                    try:
                        body = response.text
                        if body is not None and len(body) > 500:
                            body = body[:500] + f"... (truncated {len(response.text) - 500} chars)"
                    except Exception:
                        body = "<unable to read response body>"
                    print(
                        f"    [上传模块-后台]: 上传失败 -> POST {url} | 状态码: {response.status_code} | 原因: {getattr(response, 'reason', '')} | 响应: {body}"
                    )
            except requests.exceptions.RequestException as e:
                print(
                    f"    [上传模块-后台]: 网络异常 -> POST {url} | 异常: {e.__class__.__name__}: {e}"
                )
            except Exception as e:
                print(
                    f"    [上传模块-后台]: 未知错误 -> POST {url} | 异常: {e.__class__.__name__}: {e}"
                )

        self.executor.submit(_upload_task)

    def post_files_batch(self, url, files_list, json_payload_list, timeout_s: float | None = None):
        """批量上传：files_list 为 [(filename, bytes, mime), ...]; json_payload_list 为 [report1, report2, ...]
        服务器要求: form-data 中多次出现字段名 fileList 作为图片数组；payload 字段为 JSON 数组字符串。
        不再附带额外字段。"""
        def _upload_task_batch():
            try:
                def _sanitize(obj):
                    if isinstance(obj, (bytes, bytearray)):
                        try:
                            return obj.decode('utf-8')
                        except Exception:
                            import base64
                            return base64.b64encode(obj).decode('ascii')
                    if isinstance(obj, dict):
                        return {k: _sanitize(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_sanitize(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_sanitize(v) for v in obj)
                    return obj
                safe_payloads = _sanitize(json_payload_list)

                multipart_parts = []
                # 仅添加图片
                for f in files_list:
                    if not f:
                        continue
                    # f: (filename, bytes, mime)
                    multipart_parts.append(('fileList', f))
                # 添加 payload JSON 数组
                multipart_parts.append(('payload', (None, json.dumps(safe_payloads, ensure_ascii=False), 'application/json')))

                to = 30 if timeout_s is None else float(timeout_s)
                response = requests.post(url, files=multipart_parts, timeout=to)
                if response.status_code == 200:
                    print(f"    [上传模块-后台]: 批量报告发送成功 (数量={len(files_list)}).")
                else:
                    body = None
                    try:
                        body = response.text
                        if body is not None and len(body) > 500:
                            body = body[:500] + f"... (truncated {len(response.text) - 500} chars)"
                    except Exception:
                        body = "<unable to read response body>"
                    print(f"    [上传模块-后台]: 批量上传失败 -> POST {url} | 状态码: {response.status_code} | 原因: {getattr(response, 'reason', '')} | 响应: {body}")
            except requests.exceptions.RequestException as e:
                print(f"    [上传模块-后台]: 批量网络异常 -> POST {url} | 异常: {e.__class__.__name__}: {e}")
            except Exception as e:
                print(f"    [上传模块-后台]: 批量未知错误 -> POST {url} | 异常: {e.__class__.__name__}: {e}")

        self.executor.submit(_upload_task_batch)


    def shutdown(self):
        """
        在应用程序关闭时，优雅地关闭线程池。
        """
        print("[HTTP客户端]: 正在关闭线程池...")
        self.executor.shutdown(wait=True)