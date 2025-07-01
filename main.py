from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 导入配置
from core.config import logger
from core.database import create_tables

# 导入服务
from services import ConnectionManager, SSEService, DspyService

# 导入API路由
from api import (
    chat_router, 
    data_router, 
    version_router, 
    training_router, 
    websocket_router
)
from api.health import router as health_router

# 导入模型以确保表创建
from models import Interaction, Version

# 创建 FastAPI 应用
app = FastAPI(
    title="数据胶囊 API",
    description="基于 DsPy 的智能对话和模型训练系统",
    version="2.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
manager = ConnectionManager()
dspy_service = DspyService()
sse_service = SSEService(dspy_service.inference_processor, manager)

# 初始化各个模块的服务实例
from api.chat import init_services as init_chat_services
from api.training import init_services as init_training_services  
from api.websocket import init_services as init_websocket_services

init_chat_services(dspy_service, sse_service)
init_training_services(manager, dspy_service)
init_websocket_services(manager)

# 注册路由
app.include_router(chat_router, tags=["聊天"])
app.include_router(data_router, tags=["数据管理"])
app.include_router(version_router, tags=["版本管理"])
app.include_router(training_router, tags=["训练优化"])
app.include_router(websocket_router, tags=["WebSocket"])
app.include_router(health_router, tags=["健康检查"])

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("正在启动数据胶囊 API 服务...")
    
    # 创建数据库表
    create_tables()
    logger.info("数据库表创建完成")
    
    logger.info("数据胶囊 API 服务启动成功！")
    logger.info("API 文档地址: http://localhost:8080/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("数据胶囊 API 服务正在关闭...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 