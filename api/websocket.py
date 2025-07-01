from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from services import ConnectionManager

router = APIRouter()

# 全局变量，稍后在 main.py 中初始化
manager = None

def init_services(connection_manager: ConnectionManager):
    """初始化服务实例"""
    global manager
    manager = connection_manager

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接端点"""
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接（这里简单接收消息，可用于心跳检测）
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket) 