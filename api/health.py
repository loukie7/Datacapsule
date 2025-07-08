from fastapi import APIRouter
from schemas import ResponseWrapper

router = APIRouter()

@router.get("/health", response_model=ResponseWrapper)
async def health_check():
    """健康检查接口"""
    return ResponseWrapper(
        status_code=200, 
        detail="success", 
        data={"status": "healthy"}
    ) 