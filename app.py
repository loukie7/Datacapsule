import os,io,sys
from dotenv import load_dotenv
load_dotenv(override=True)
from loguru import logger

# 设置 logger 日志级别
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.remove()  # 移除默认处理器
logger.add(sys.stderr, level=log_level)  # 添加新的处理器并设置日志级别
logger.info(f"日志级别设置为: {log_level}")

from fastapi import FastAPI, Request,  WebSocket, WebSocketDisconnect,Query, Body,File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import dspy
from pydantic import BaseModel, Field
from typing import List,Dict, Any
import tempfile
import json

from sqlalchemy import create_engine, Column, String, JSON, DateTime,Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid
import asyncio
from broadcast import ConnectionManager
from dspy_inference import DspyInferenceProcessor
from dspy_evaluation import DspyEvaluationProcessor



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


manager = ConnectionManager()
# 初始化 DspyProcessor
dspy_processor = DspyInferenceProcessor()
# 初始化流式模型
streaming_react = dspy_processor.stream_predict

eval_processor = DspyEvaluationProcessor()


predictor_version = "1.0.0"

# 定义数据库模型
Base = declarative_base()

# 创建数据库引擎
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///interactions.db"), echo=False)

Base.metadata.create_all(engine)

# 创建会话
SessionLocal = sessionmaker(bind=engine)

# 定义封装的响应模型
class ResponseWrapper(BaseModel):
    status_code: int
    detail: str
    data: Any

class Interaction(Base):
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

# 新增版本管理模型
class Version(Base):
    __tablename__ = 'versions'
    
    version = Column(String, primary_key=True)
    file_path = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.now)

# 新增请求体模型
class TrainingRequest(BaseModel):
    ids: List[str]
    version: str


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接（这里简单接收消息，可用于心跳检
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 异步生成器：流式返回 ReAct 模块的回答，并在结束后通过 websocket 推送 prompt 历史
async def stream_react_response(prompt: str):
    global streaming_react
    try:
        # 跟踪上一次的内容，用于增量更新
        last_reasoning = ""
        last_answer = ""
        
        # 修改这里：直接调用 streaming_react 函数
        async for chunk in streaming_react(question=prompt):
            # 假设每个 chunk 为 Prediction 对象，包含 answer 与 reasoning 字段
            if chunk:
                # 获取当前的 reasoning 和 answer
                current_reasoning = getattr(chunk, "reasoning", "") or ""
                current_answer = getattr(chunk, "answer", "") or ""
                
                # 计算增量内容
                reasoning_delta = current_reasoning[len(last_reasoning):] if current_reasoning else ""
                answer_delta = current_answer[len(last_answer):] if current_answer else ""
                
                # 只有当有新内容时才发送
                if reasoning_delta or answer_delta:
                    data = {
                        "reasoning_delta": reasoning_delta,
                        "answer_delta": answer_delta,
                        "reasoning": current_reasoning,  # 也可以选择只发送增量
                        "answer": current_answer        # 也可以选择只发送增量
                    }
                    logger.info(f"增量数据: {json.dumps(data)}")
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # 更新上一次的内容
                    last_reasoning = current_reasoning
                    last_answer = current_answer
        
        # 流式结束后的处理...
        last_message = dspy_processor.get_last_message()
        
        # 检查 last_message 是否为 None 或不包含必要字段
        if not last_message:
            error_data = {"error": "无法获取消息历史", "message": "处理请求时发生错误"}
            logger.error(f"last_message 为空或无效")
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # 构造一个只包含所需字段的新字典
        data_to_send = {
            "question": prompt,
            "prompt": last_message.get("prompt"),
            "messages": last_message.get("messages"),
            "timestamp": last_message.get("timestamp"),
            "uuid": last_message.get("uuid"),
            "model": last_message.get("model"),
            "version": predictor_version
        }

        # 从 response 中提取 choices 第一个元素的 message 的 content 字段
        try:
            # 检查 response 是否存在且包含必要字段
            if "response" in last_message and last_message["response"] and "choices" in last_message["response"]:
                data_to_send["content"] = last_message["response"].choices[0].message.content
                tokens = {}
                if "usage" in last_message:
                    tokens["completion_tokens"] = last_message["usage"].get("completion_tokens", 0)
                    tokens["prompt_tokens"] = last_message["usage"].get("prompt_tokens", 0)
                    tokens["total_tokens"] = last_message["usage"].get("total_tokens", 0)
                    data_to_send["tokens"] = tokens
            else:
                data_to_send["content"] = None
                data_to_send["tokens"] = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
                logger.warning("response 字段不存在或格式不正确")
        except (KeyError, IndexError, AttributeError) as e:
            # 如果不存在该字段则设为 None 或者按需处理
            data_to_send["content"] = None
            data_to_send["tokens"] = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
            logger.error(f"提取 content 时出错：{e}")
        
        # 将数据转换为 JSON 字符串
        json_message = json.dumps(data_to_send, ensure_ascii=False, indent=2)
        logger.info(json_message)
        
        # 修改：不再通过 websocket 广播，而是通过流式返回完整消息
        yield f"data: {json.dumps({'prompt_history': json_message})}\n\n"
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        # 捕获所有异常，返回错误信息
        error_message = str(e)
        logger.error(f"stream_react_response 发生错误: {error_message}")
        error_data = {"error": "处理请求失败", "message": error_message}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: Request, prompt: str = Body(..., embed=True), stream: int = Body(None, embed=True), version: str = Body(None, embed=True)):
    
    global predictor_version
    global streaming_react  # 添加全局声明
    try:
        # 创建会话
        session = SessionLocal()
        pred = dspy_processor.model
        
        predictor_version =dspy_processor.get_version()
        # 记录一个当前的版本号，如果版本号没有发生变化，则不需要进行操作
        if version and version != predictor_version:
            # 查询版本信息
            version_info = session.query(Version).filter(Version.version == version).first()
            if not version_info:
                return ResponseWrapper(status_code=404, detail="error", data={"message": f"Version {version} not found"})
            
            # 加载指定版本的模型文件todo
            logger.info(f"开始切换版本：{version}/{version_info.file_path}")
            file_path = version_info.file_path
            dspy_processor.load_model(file_path)
            # 更新 predictor_version
            predictor_version = version
            dspy_processor.set_version(version)
            logger.info(f"切换版本成功：{version},清除缓存")
            # 重新初始化 streaming_react
            streaming_react = dspy_processor.stream_predict  # 修改这里：直接赋值函数引用，不要调用
        
        if stream == 1:
            # 流式返回：包装生成器，media_type 为 "text/event-stream"
            return StreamingResponse(stream_react_response(prompt), media_type="text/event-stream")
        else:
            # 非流式返回：直接调用 ReAct 模块，获取最终答案
            # 为pred设置独立的llm配置
            with dspy.llm_config(model=dspy_processor.lm):
                pred = dspy_processor.model
                dspyres = pred(question=prompt)
            content = dspyres.answer
            reasoning = dspyres.reasoning
            return ResponseWrapper(status_code=200, detail="success", data={"content": content, "reasoning": reasoning})
    except Exception as e:
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})
    finally:
        session.close()
# 新增的 API 方法：接收数据并保存到 JSON 文件
@app.post("/save_data")
async def save_data(data: Dict):
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
        
        return ResponseWrapper(status_code=200, detail="success", data={"message": "Data saved successfully"})
    except Exception as e:
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})



# 新增的 API 方法：接收数据并保存到 SQLite 数据库
@app.post("/save_to_db")
async def save_to_db(data: Dict):
    try:
        # 创建会话
        session = SessionLocal()

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
        
        return ResponseWrapper(status_code=200, detail="success", data={"message": "Data saved successfully to database"})
    except Exception as e:
        session.rollback()
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})
    finally:
        session.close()

@app.delete("/interactions/{interaction_id}", response_model=ResponseWrapper)
async def delete_interaction(interaction_id: str):
    try:
        session = SessionLocal()
        
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

# 新增的 API 方法：接收数据并更新 SQLite 数据库中的记录
@app.post("/editdata")
async def edit_data(data: Dict):
    try:
        # 创建会话
        session = SessionLocal()
        
        # 获取 messageId 和更新字段
        message_id = data.get("messageId")
        update_fields = data.get("updateFields", {})
        
        # 根据 messageId 查找记录
        interaction = session.query(Interaction).filter(Interaction.id == message_id).first()
        
        if not interaction:
            return ResponseWrapper(status_code=404, detail="error", data={"message": "Record not found"})
        
        # 更新指定的字段
        for field, value in update_fields.items():
            if hasattr(interaction, field):
                if field in ["messages", "retrievmethod"]:
                    # 格式化 JSON 字段
                    setattr(interaction, field, json.loads(json.dumps(value, ensure_ascii=False, indent=2)))
                else:
                    setattr(interaction, field, value)
            else:
                return ResponseWrapper(status_code=400, detail="error", data={"message": f"Field '{field}' does not exist"})
         
        # 提交更改
        session.commit()
        
        return ResponseWrapper(status_code=200, detail="success", data={"message": "Data updated successfully"})
    except Exception as e:
        session.rollback()
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})
    finally:
        session.close()

@app.get("/interactions/{interaction_id}", response_model=ResponseWrapper)
async def get_interaction_by_id(interaction_id: str):
    try:
        session = SessionLocal()
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

@app.get("/interactions", response_model=ResponseWrapper)
async def get_interactions_by_version(
    version: str = Query(None),
    page: int = Query(1, ge=1, description="页码，从1开始"),
    page_size: int = Query(10, ge=1, le=100, description="每页数量")
):
    try:
        session = SessionLocal()
        
        # 获取最新版本（如果未指定）
        # latest_version = session.query(Version.version)\
        #                           .order_by(Version.created_at.desc())\
        #                           .first()
        #     if not latest_version:
        #         return ResponseWrapper(status_code=404, detail="error", data={"message": "无可用版本"})
        #     version = latest_version[0]
        # 修改后的代码片段
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
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})
    finally:
        session.close() 

# 全局优化任务跟踪
optimization_tasks = {}

# 异步优化任务
async def run_dspy_optimization(training_data: List[Dict], version: str, ids: List[str]):
    task_id = f"optimization_task_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        from dspy.teleprompt import BootstrapFewShot
        from dspy.evaluate import Evaluate
        from dspy.evaluate.metrics import answer_exact_match

        # 更新状态并发送开始消息
        logger.info(f"开始优化任务 {task_id}，数据量: {len(training_data)}，版本: {version}")
        optimization_tasks[task_id] = "loading_data"
        await manager.broadcast(json.dumps({
            "type": "optimization_status",
            "data": {
                "task_id": task_id,
                "status": "loading_data",
                "progress": 5,
                "message": "正在准备训练数据..."
            }
        }))

        # 创建训练集
        trainset = [dspy.Example(question=x["question"],reasoning=x["reasoning"], answer=x["modelResponse"]).with_inputs("question") for x in training_data]
        logger.info(f"任务 {task_id}: 已创建训练集，共 {len(trainset)} 条数据")
        
        # 更新状态
        optimization_tasks[task_id] = "preparing_model"
        await manager.broadcast(json.dumps({
            "type": "optimization_status",
            "data": {
                "task_id": task_id,
                "status": "preparing_model",
                "progress": 10,
                "message": "正在准备模型..."
            }
        }))
        
        # 从最新版本加载预测模型
        session = SessionLocal()
        
        # 修改这里：使用 dspy_processor 的 model 而不是 eval_processor 的 model
        # 因为 DspyEvaluationProcessor 没有 model 属性
        predict = dspy_processor.model
        logger.info(f"任务 {task_id}: 已加载模型")
        
        # 设置优化器
        teleprompter = BootstrapFewShot(metric=eval_processor.llm_biological_metric, max_labeled_demos=15)
        
        # 更新状态
        optimization_tasks[task_id] = "optimizing"
        await manager.broadcast(json.dumps({
            "type": "optimization_status",
            "data": {
                "task_id": task_id,
                "status": "optimizing",
                "progress": 15,
                "message": "正在进行模型优化..."
            }
        }))
        
        # 编译优化
        logger.info(f"任务 {task_id}: 开始编译优化")
        compiled_predictor = teleprompter.compile(predict, trainset=trainset)
        logger.info(f"任务 {task_id}: 编译优化完成")
        
        # 更新状态
        optimization_tasks[task_id] = "saving_model"
        await manager.broadcast(json.dumps({
            "type": "optimization_status",
            "data": {
                "task_id": task_id,
                "status": "saving_model",
                "progress": 50,
                "message": "正在保存优化后的模型..."
            }
        }))
        
        # 确保目录存在
        os.makedirs("dspy_program", exist_ok=True)
        last_version = session.query(Version.version).order_by(Version.created_at.desc()).first().version
        
        
        # 保存优化后的模型
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"dspy_program/program_v{last_version}_{timestamp}.pkl"
        compiled_predictor.save(output_path, save_program=False)
        logger.info(f"任务 {task_id}: 已保存模型到 {output_path}")
        
        # 解析当前版本号，生成新版本号
        # 从数据库获取最新版本号，原生新增
        major, minor, patch = map(int, last_version.split('.'))
        new_version = f"{major}.{minor}.{patch + 1}"
        
        # 描述信息
        description = f"基于 {version} 版本，使用 {len(ids)} 条数据优化生成的新版本"
        
        # 创建新版本
        new_version_instance = Version(
            version=new_version,
            file_path=output_path,
            description=description
        )
        
        session.add(new_version_instance)
        session.commit()
        logger.info(f"任务 {task_id}: 已创建新版本 {new_version}")
        
        # 更新状态为完成
        optimization_tasks[task_id] = "completed"
        
        # 通过 WebSocket 广播版本更新消息
        await manager.broadcast(json.dumps({
            "type": "version_update",
            "data": {
                "old_version": version,
                "new_version": new_version,
                "description": description,
                "model_path": output_path,
                "training_ids": ids,
                "progress": 100,
                "message": f"优化完成，已创建新版本{new_version}"
            }
        }))
        logger.info(f"任务 {task_id}: 优化任务完成")
        
    except Exception as e:
        # 记录错误并通过 WebSocket 发送失败消息
        error_message = str(e)
        logger.error(f"任务 {task_id} 失败: {error_message}")
        optimization_tasks[task_id] = f"failed: {error_message}"
        
        await manager.broadcast(json.dumps({
            "type": "optimization_failed",
            "data": {
                "version": version,
                "error": error_message,
                "task_id": task_id,
                "progress": 0,
                "message": f"优化失败: {error_message}"
            }
        }))
    finally:
        if 'session' in locals():
            session.close()

@app.post("/addtraining", response_model=ResponseWrapper)
async def add_training(request: TrainingRequest, background_tasks: BackgroundTasks):  # 使用新模型
    session = None
    try:
        # 获取ID列表
        ids = request.ids
        version = request.version
        
        # 参数校验
        if not ids:
            return ResponseWrapper(
                status_code=400,
                detail="error",
                data={"message": "未提供有效ID列表"}
            )
        if not version:
            return ResponseWrapper(
                status_code=400,
                detail="error",
                data={"message": "必须提供版本号参数"}
            )

        session = SessionLocal()
        
        # 查询数据库并收集数据
        training_data = []
        for interaction_id in ids:
            interaction = session.query(Interaction).get(interaction_id)
            if interaction:
                training_data.append({
                    "id": interaction.id,
                    "question": interaction.question,
                    "reasoning": interaction.reasoning,
                    "modelResponse": interaction.modelResponse,
                    "timestamp": interaction.timestamp.isoformat()
                })

        if not training_data:
            return ResponseWrapper(
                status_code=404,
                detail="error",
                data={"message": "未找到匹配的记录"}
            )

        # 生成任务ID
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        task_id = f"optimization_task_{version}_{timestamp}"
        
        # 在后台启动优化任务前先设置状态
        optimization_tasks[task_id] = "pending"
        
        # 将训练数据和任务信息保存为全局变量，以便后台任务使用
        # 这样可以避免在后台任务中重新查询数据库
        task_info = {
            "training_data": training_data,
            "version": version,
            "ids": [item["id"] for item in training_data],
            "task_id": task_id
        }
        
        # 添加后台任务 - 使用普通函数而不是异步函数
        background_tasks.add_task(
            start_optimization_task,
            task_info
        )
        
        # 立即返回响应，不等待优化任务完成
        logger.info(f"已创建优化任务 {task_id}，将在后台处理 {len(training_data)} 条数据")
        return ResponseWrapper(
            status_code=200,
            detail="success",
            data={
                "message": f"成功收集 {len(training_data)} 条训练数据，已创建后台优化任务",
                "task_id": task_id,
                "exported_ids": [item["id"] for item in training_data],
                "version": version,  # 返回版本号用于验证
                "optimization_status": "pending"  # 返回初始优化状态
            }
        )

    except Exception as e:
        logger.error(f"创建优化任务失败: {str(e)}")
        return ResponseWrapper(
            status_code=500,
            detail="error",
            data={"message": f"处理失败: {str(e)}"}
        )
    finally:
        if session:
            session.close()

# 新增函数：启动优化任务的普通函数
def start_optimization_task(task_info):
    """启动优化任务的普通函数，用于后台任务"""
    # 创建一个新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 在新的事件循环中运行异步任务
    try:
        # 发送初始通知
        loop.run_until_complete(manager.broadcast(json.dumps({
            "type": "optimization_created",
            "data": {
                "task_id": task_info["task_id"],
                "status": "pending",
                "progress": 0,
                "message": f"已创建优化任务，准备处理 {len(task_info['training_data'])} 条数据",
                "version": task_info["version"],
                "ids": task_info["ids"]
            }
        })))
        
        # 设置状态为 running
        optimization_tasks[task_info["task_id"]] = "running"
        
        # 执行实际的优化任务
        loop.run_until_complete(run_dspy_optimization(
            task_info["training_data"], 
            task_info["version"], 
            task_info["ids"]
        ))
    except Exception as e:
        logger.error(f"优化任务执行失败: {str(e)}")
        # 设置任务状态为失败
        optimization_tasks[task_info["task_id"]] = f"failed: {str(e)}"
        # 发送失败通知
        loop.run_until_complete(manager.broadcast(json.dumps({
            "type": "optimization_failed",
            "data": {
                "version": task_info["version"],
                "error": str(e),
                "task_id": task_info["task_id"],
                "progress": 0,
                "message": f"优化失败: {str(e)}"
            }
        })))
    finally:
        # 关闭事件循环
        loop.close()

# 新增的 API 方法：创建新版本
@app.post("/create_version")
async def create_version(file_path: str = Body(..., embed=True), old_version: str = Body(..., embed=True), description: str = Body(..., embed=True)):
    try:
        # 创建会话
        session = SessionLocal()
        
        # 解析旧版本号
        major, minor, patch = map(int, old_version.split('.'))
        
        # 递增版本号
        new_version = f"{major}.{minor}.{patch + 1}"
        
        # 检查新版本号是否已存在
        existing_version = session.query(Version).filter(Version.version == new_version).first()
        if existing_version:
            return ResponseWrapper(status_code=400, detail="error", data={"message": f"Version {new_version} already exists"})
        
        # 创建新版本实例
        new_version_instance = Version(
            version=new_version,
            file_path=file_path,
            description=description
        )
        
        # 添加到会话并提交
        session.add(new_version_instance)
        session.commit()
        
        return ResponseWrapper(status_code=200, detail="success", data={"message": "Version created successfully", "new_version": new_version})
    except Exception as e:
        session.rollback()
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})
    finally:
        session.close()

@app.get("/versions", response_model=ResponseWrapper)
async def get_versions():
    try:
        # 创建会话
        session = SessionLocal()
        
        # 查询所有版本并按创建时间排序
        versions = session.query(Version).order_by(Version.created_at.asc()).all()
        
        # 提取版本号
        version_list = [{"version": version.version, "file_path": version.file_path, "description": version.description, "created_at": version.created_at} for version in versions]
        
        return ResponseWrapper(status_code=200, detail="success", data={"versions": version_list})
    except Exception as e:
        return ResponseWrapper(status_code=500, detail="error", data={"message": str(e)})
    finally:
        session.close()

@app.get("/health",response_model=ResponseWrapper)
async def health_check():
    return ResponseWrapper(status_code=200, detail="success", data={"status": "healthy"})

# 添加一个 API 端点查询优化任务状态
@app.get("/optimization_status/{task_id:path}", response_model=ResponseWrapper)
async def get_optimization_status(task_id: str):
    try:
        if task_id in optimization_tasks:
            status = optimization_tasks[task_id]
            return ResponseWrapper(
                status_code=200,
                detail="success",
                data={
                    "task_id": task_id,
                    "status": status
                }
            )
        else:
            return ResponseWrapper(
                status_code=404,
                detail="error",
                data={"message": f"未找到对应的优化任务: {task_id}"}
            )
    except Exception as e:
        return ResponseWrapper(
            status_code=500,
            detail="error",
            data={"message": f"查询失败: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
