import os
from loguru import logger
from dotenv import load_dotenv

from core.llm_manager import get_inference_manager, get_training_manager, get_entity_manager
from inference import InferenceProcessor, stream_predict  
from evaluation import EvaluationProcessor
from core.config import predictor_version

# 确保环境变量已加载
load_dotenv(override=True)

class InferenceService:
    """推理服务封装"""
    
    def __init__(self):
        # 使用统一的LLM管理器
        self.llm_manager = get_inference_manager()
        
        # 初始化处理器
        self.inference_processor = InferenceProcessor()
        self.eval_processor = EvaluationProcessor()
        
        # 初始化流式模型 - 使用模块级的stream_predict函数
        self.streaming_react = stream_predict
    
    @property
    def model(self):
        """获取推理模型"""
        return self.inference_processor.model
    
    @property
    def lm(self):
        """获取语言模型"""
        return self.llm_manager
    
    def get_version(self) -> str:
        """获取当前版本"""
        return self.inference_processor.get_version()
    
    def set_version(self, version: str):
        """设置版本"""
        self.inference_processor.set_version(version)
    
    def get_last_message(self):
        """获取最后一条消息"""
        return self.inference_processor.get_last_message()
    
    def load_model(self, file_path: str):
        """加载模型"""
        self.inference_processor.load_model(file_path)
        # 重新初始化流式响应 - 使用模块级的stream_predict函数
        self.streaming_react = stream_predict
    
    async def predict(self, question: str):
        """非流式预测"""
        result = self.model(question=question)
        return result
    
    async def stream_predict(self, question: str):
        """流式预测"""
        async for chunk in stream_predict(question, "default"):
            yield chunk 