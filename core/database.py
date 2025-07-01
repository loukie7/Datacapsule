from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import DATABASE_URL

# 创建数据库引擎
engine = create_engine(DATABASE_URL, echo=False)

# 创建基础模型类
Base = declarative_base()

# 创建会话工厂
SessionLocal = sessionmaker(bind=engine)

def create_tables():
    """创建所有数据表"""
    Base.metadata.create_all(engine)

def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 