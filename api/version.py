from fastapi import APIRouter, Body
from loguru import logger
from schemas import ResponseWrapper
from core.database import SessionLocal
from models import Version

router = APIRouter()

@router.post("/create_version")
async def create_version(
    file_path: str = Body(..., embed=True), 
    old_version: str = Body(..., embed=True), 
    description: str = Body(..., embed=True)
):
    """创建新版本"""
    session = SessionLocal()
    try:
        # 解析旧版本号
        major, minor, patch = map(int, old_version.split('.'))
        
        # 递增版本号
        new_version = f"{major}.{minor}.{patch + 1}"
        
        # 检查新版本号是否已存在
        existing_version = session.query(Version).filter(Version.version == new_version).first()
        if existing_version:
            return ResponseWrapper(
                status_code=400, 
                detail="error", 
                data={"message": f"Version {new_version} already exists"}
            )
        
        # 创建新版本实例
        new_version_instance = Version(
            version=new_version,
            file_path=file_path,
            description=description
        )
        
        # 添加到会话并提交
        session.add(new_version_instance)
        session.commit()
        
        return ResponseWrapper(
            status_code=200, 
            detail="success", 
            data={"message": "Version created successfully", "new_version": new_version}
        )
    except Exception as e:
        session.rollback()
        return ResponseWrapper(
            status_code=500, 
            detail="error", 
            data={"message": str(e)}
        )
    finally:
        session.close()

@router.get("/versions", response_model=ResponseWrapper)
async def get_versions():
    """获取所有版本列表"""
    session = SessionLocal()
    try:
        # 查询所有版本并按创建时间排序
        versions = session.query(Version).order_by(Version.created_at.asc()).all()
        
        # 提取版本信息
        version_list = [
            {
                "version": version.version, 
                "file_path": version.file_path, 
                "description": version.description, 
                "created_at": version.created_at
            } 
            for version in versions
        ]
        
        return ResponseWrapper(
            status_code=200, 
            detail="success", 
            data={"versions": version_list}
        )
    except Exception as e:
        return ResponseWrapper(
            status_code=500, 
            detail="error", 
            data={"message": str(e)}
        )
    finally:
        session.close() 