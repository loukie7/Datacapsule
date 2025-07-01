# API 路由模块
from .chat import router as chat_router
from .data import router as data_router
from .version import router as version_router
from .training import router as training_router

__all__ = [
    "chat_router", 
    "data_router", 
    "version_router", 
    "training_router"
] 