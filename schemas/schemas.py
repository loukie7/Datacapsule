from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ResponseWrapper(BaseModel):
    """封装的响应模型"""
    status_code: int
    detail: str
    data: Any

class TrainingRequest(BaseModel):
    """训练请求模型"""
    ids: List[str]
    version: str 