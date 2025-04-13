from fastapi import FastAPI, Request,  WebSocket, WebSocketDisconnect
from loguru import logger
# 定义一个 WebSocket 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
                logger.info(f"已向客户端推送消息: {message}")
            except Exception as e:
                dead_connections.append(connection)
                logger.error(f"广播消息时出错: {str(e)}")
                continue
        
        # 清理已断开的连接
        for dead_connection in dead_connections:
            try:
                self.active_connections.remove(dead_connection)
            except ValueError:
                pass