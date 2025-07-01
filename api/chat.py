from fastapi import APIRouter, Request, Body
import dspy
from loguru import logger
from schemas import ResponseWrapper
from services import SSEService, DspyService
from core.config import predictor_version
from core.database import SessionLocal
from models import Version

router = APIRouter()

# 全局服务实例（稍后在 main.py 中初始化）
dspy_service = None
sse_service = None

def init_services(dspy_svc: DspyService, sse_svc: SSEService):
    """初始化服务实例"""
    global dspy_service, sse_service
    dspy_service = dspy_svc
    sse_service = sse_svc

@router.post("/chat")
async def chat(
    request: Request, 
    prompt: str = Body(..., embed=True), 
    stream: int = Body(None, embed=True), 
    version: str = Body(None, embed=True)
):
    """聊天接口"""
    global predictor_version
    
    try:
        # 创建会话
        session = SessionLocal()
        
        # 更新预测器版本
        predictor_version = dspy_service.get_version()
        
        # 记录一个当前的版本号，如果版本号没有发生变化，则不需要进行操作
        if version and version != predictor_version:
            # 查询版本信息
            version_info = session.query(Version).filter(Version.version == version).first()
            if not version_info:
                return ResponseWrapper(
                    status_code=404, 
                    detail="error", 
                    data={"message": f"Version {version} not found"}
                )
            
            # 加载指定版本的模型文件
            logger.info(f"开始切换版本：{version}/{version_info.file_path}")
            file_path = version_info.file_path
            dspy_service.load_model(file_path)
            
            # 更新 predictor_version
            predictor_version = version
            dspy_service.set_version(version)
            logger.info(f"切换版本成功：{version},清除缓存")
        
        if stream == 1:
            # 流式返回：使用 sse-starlette 的 EventSourceResponse
            return sse_service.create_event_source_response(prompt)
        else:
            # 非流式返回：直接调用 ReAct 模块，获取最终答案
            dspyres = await dspy_service.predict(prompt)
            content = dspyres.answer
            reasoning = dspyres.reasoning
            return ResponseWrapper(
                status_code=200, 
                detail="success", 
                data={"content": content, "reasoning": reasoning}
            )
            
    except Exception as e:
        logger.error(f"聊天接口错误: {str(e)}")
        return ResponseWrapper(
            status_code=500, 
            detail="error", 
            data={"message": str(e)}
        )
    finally:
        if 'session' in locals():
            session.close() 