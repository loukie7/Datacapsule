from fastapi import APIRouter, Query
from typing import Dict
import json
import os
from datetime import datetime
from loguru import logger
from schemas import ResponseWrapper
from core.database import SessionLocal
from models import Interaction

router = APIRouter()

@router.post("/save_data")
async def save_data(data: Dict):
    """保存数据到 JSON 文件"""
    try:
        # 定义保存数据的文件路径
        file_path = "saved_data.json"
        
        # 检查文件是否存在，如果存在则读取现有数据
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        else:
            existing_data = []
        
        # 将新数据添加到现有数据中
        existing_data.append(data)
        
        # 将更新后的数据写回文件
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=2)
        
        return ResponseWrapper(
            status_code=200, 
            detail="success", 
            data={"message": "Data saved successfully"}
        )
    except Exception as e:
        return ResponseWrapper(
            status_code=500, 
            detail="error", 
            data={"message": str(e)}
        )

@router.post("/save_to_db")
async def save_to_db(data: Dict):
    """保存数据到 SQLite 数据库"""
    session = SessionLocal()
    try:
        # 检查是否已存在相同ID
        if data.get("id"):
            existing = session.query(Interaction).get(data["id"])
            if existing:
                return ResponseWrapper(
                    status_code=400,
                    detail="error",
                    data={"message": f"相同记录 {data['id']} 已存在"}
                )

        # 格式化 messages 和 retrievmethod 字段
        formatted_messages = json.dumps(data.get("messages"), ensure_ascii=False, indent=2)
        formatted_retrievmethod = json.dumps(data.get("retrievmethod"), ensure_ascii=False, indent=2)
        
        # 创建 Interaction 实例
        interaction = Interaction(
            id=data.get("id"),
            timestamp=datetime.fromisoformat(data.get("timestamp")),
            question=data.get("question"),
            model=data.get("model"),
            version=data.get("version"),
            messages=json.loads(formatted_messages),
            retrievmethod=json.loads(formatted_retrievmethod),
            prompt=data.get("prompt"),
            modelResponse=data.get("modelResponse"),
            reasoning=data.get("reasoning"),
            processingTime=data.get("processingTime"),
            tokens=data.get("tokens")
        )
        
        # 添加到会话并提交
        session.add(interaction)
        session.commit()
        
        return ResponseWrapper(
            status_code=200, 
            detail="success", 
            data={"message": "Data saved successfully to database"}
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

@router.delete("/interactions/{interaction_id}", response_model=ResponseWrapper)
async def delete_interaction(interaction_id: str):
    """删除交互记录"""
    session = SessionLocal()
    try:
        # 查询要删除的记录
        interaction = session.query(Interaction).filter(Interaction.id == interaction_id).first()
        
        if not interaction:
            return ResponseWrapper(
                status_code=404,
                detail="error",
                data={"message": f"ID为 {interaction_id} 的记录不存在"}
            )
        
        # 执行删除
        session.delete(interaction)
        session.commit()
        
        return ResponseWrapper(
            status_code=200,
            detail="success",
            data={"message": "记录删除成功", "deleted_id": interaction_id}
        )
    except Exception as e:
        session.rollback()
        return ResponseWrapper(
            status_code=500,
            detail="error",
            data={"message": f"删除失败: {str(e)}"}
        )
    finally:
        session.close()

@router.post("/editdata")
async def edit_data(data: Dict):
    """编辑数据记录"""
    session = SessionLocal()
    try:
        # 获取 messageId 和更新字段
        message_id = data.get("messageId")
        update_fields = data.get("updateFields", {})
        
        # 根据 messageId 查找记录
        interaction = session.query(Interaction).filter(Interaction.id == message_id).first()
        
        if not interaction:
            return ResponseWrapper(
                status_code=404, 
                detail="error", 
                data={"message": "Record not found"}
            )
        
        # 更新指定的字段
        for field, value in update_fields.items():
            if hasattr(interaction, field):
                if field in ["messages", "retrievmethod"]:
                    # 格式化 JSON 字段
                    setattr(interaction, field, json.loads(json.dumps(value, ensure_ascii=False, indent=2)))
                else:
                    setattr(interaction, field, value)
            else:
                return ResponseWrapper(
                    status_code=400, 
                    detail="error", 
                    data={"message": f"Field '{field}' does not exist"}
                )
         
        # 提交更改
        session.commit()
        
        return ResponseWrapper(
            status_code=200, 
            detail="success", 
            data={"message": "Data updated successfully"}
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

@router.get("/interactions/{interaction_id}", response_model=ResponseWrapper)
async def get_interaction_by_id(interaction_id: str):
    """根据ID获取交互记录"""
    session = SessionLocal()
    try:
        interaction = session.query(Interaction).filter(Interaction.id == interaction_id).first()
        
        if not interaction:
            return ResponseWrapper(
                status_code=404,
                detail="error",
                data={"message": f"ID为 {interaction_id} 的记录不存在"}
            )
        
        interaction_data = {
            "id": interaction.id,
            "timestamp": interaction.timestamp.isoformat(),
            "question": interaction.question,
            "model": interaction.model,
            "version": interaction.version,
            "messages": interaction.messages,
            "retrievmethod": interaction.retrievmethod,
            "prompt": interaction.prompt,
            "modelResponse": interaction.modelResponse,
            "reasoning": interaction.reasoning,
            "processingTime": interaction.processingTime,
            "tokens": interaction.tokens
        }
        
        return ResponseWrapper(
            status_code=200,
            detail="success",
            data=interaction_data
        )
    except Exception as e:
        return ResponseWrapper(
            status_code=500,
            detail="error",
            data={"message": f"查询失败: {str(e)}"}
        )
    finally:
        session.close()

@router.get("/interactions", response_model=ResponseWrapper)
async def get_interactions_by_version(
    version: str = Query(None),
    page: int = Query(1, ge=1, description="页码，从1开始"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量")
):
    """获取交互记录列表"""
    session = SessionLocal()
    try:
        # 构建查询
        if not version:
            # 移除获取最新版本逻辑，直接构建无版本过滤的查询
            base_query = session.query(
                Interaction.id,
                Interaction.question,
                Interaction.version,
                Interaction.model,
                Interaction.processingTime,
                Interaction.timestamp
            ).order_by(
                Interaction.timestamp.desc()
            )
        else:
            # 当指定版本时保持原有过滤逻辑
            base_query = session.query(
                Interaction.id,
                Interaction.question,
                Interaction.version,
                Interaction.model,
                Interaction.processingTime,
                Interaction.timestamp
            ).filter(
                Interaction.version == version
            ).order_by(
                Interaction.timestamp.desc()
            )

        # 分页处理
        total_count = base_query.count()
        total_pages = (total_count + page_size - 1) // page_size
        
        interactions = base_query.offset(
            (page - 1) * page_size
        ).limit(
            page_size
        ).all()

        # 构建响应数据
        interaction_list = [
            {
                "id": row.id,
                "question": row.question,
                "version": row.version,
                "model": row.model,
                "processingTime": row.processingTime,
                "timestamp": row.timestamp.isoformat()
            }
            for row in interactions
        ]
        
        return ResponseWrapper(
            status_code=200,
            detail="success",
            data={
                "version": version,
                "pagination": {
                    "total": total_count,
                    "total_pages": total_pages,
                    "current_page": page,
                    "page_size": page_size
                },
                "interactions": interaction_list
            }
        )
    except Exception as e:
        return ResponseWrapper(
            status_code=500, 
            detail="error", 
            data={"message": str(e)}
        )
    finally:
        session.close() 