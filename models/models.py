from sqlalchemy import Column, String, JSON, DateTime, Integer
from datetime import datetime
import uuid
from core.database import Base

class Interaction(Base):
    """交互记录模型"""
    __tablename__ = 'interactions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.now)
    question = Column(String)
    model = Column(String)
    version = Column(String)
    messages = Column(JSON)
    retrievmethod = Column(JSON)
    prompt = Column(String)
    modelResponse = Column(String)
    reasoning = Column(String)
    processingTime = Column(Integer)
    tokens = Column(JSON)

class Version(Base):
    """版本管理模型"""
    __tablename__ = 'versions'
    
    version = Column(String, primary_key=True)
    file_path = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.now) 