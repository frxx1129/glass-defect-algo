# --- START OF FILE utils.py ---
import asyncio

class ConnectionManager:
    """Manages active WebSocket connections for camera streams."""
    def __init__(self, num_cameras):
        self.active_connections = [[] for _ in range(num_cameras)]
        self.yield_subscribers = []

    async def connect(self, ws, idx):
        await ws.accept()
        self.active_connections[idx].append(ws)

    def disconnect(self, ws, idx):
        if ws in self.active_connections[idx]:
            self.active_connections[idx].remove(ws)

    async def connect_yield(self, ws):
        await ws.accept()
        self.yield_subscribers.append(ws)

    def disconnect_yield(self, ws):
        if ws in self.yield_subscribers:
            self.yield_subscribers.remove(ws)

    async def broadcast(self, msg, idx):
        conns = list(self.active_connections[idx])
        if not conns:
            return
        results = await asyncio.gather(
            *[conn.send_text(msg) for conn in conns], return_exceptions=True
        )
        # 清理已失效的连接，确保后续不再尝试发送
        for conn, res in zip(conns, results):
            if isinstance(res, Exception):
                try:
                    if conn in self.active_connections[idx]:
                        self.active_connections[idx].remove(conn)
                except Exception:
                    pass

    async def broadcast_yield(self, msg):
        conns = list(self.yield_subscribers)
        if not conns:
            return
        results = await asyncio.gather(
            *[conn.send_text(msg) for conn in conns], return_exceptions=True
        )
        for conn, res in zip(conns, results):
            if isinstance(res, Exception):
                try:
                    if conn in self.yield_subscribers:
                        self.yield_subscribers.remove(conn)
                except Exception:
                    pass