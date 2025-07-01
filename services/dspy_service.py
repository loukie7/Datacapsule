import dspy
from loguru import logger
from dspy_inference import DspyInferenceProcessor
from dspy_evaluation import DspyEvaluationProcessor
from core.config import predictor_version

class DspyService:
    """DsPy 服务封装"""
    
    def __init__(self):
        # 初始化 DspyProcessor
        self.inference_processor = DspyInferenceProcessor()
        self.eval_processor = DspyEvaluationProcessor()
        
        # 初始化流式模型
        self.streaming_react = self.inference_processor.stream_predict
        
    @property
    def model(self):
        """获取推理模型"""
        return self.inference_processor.model
    
    @property
    def lm(self):
        """获取语言模型"""
        return self.inference_processor.lm
    
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
        # 重新初始化流式响应
        self.streaming_react = self.inference_processor.stream_predict
    
    async def predict(self, question: str):
        """非流式预测"""
        # 在新版本的 DSPy 中，直接使用模型进行预测
        pred = self.model
        result = pred(question=question)
        return result
    
    async def stream_predict(self, question: str):
        """流式预测"""
        async for chunk in self.inference_processor.stream_predict(question=question):
            yield chunk 