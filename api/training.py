from fastapi import APIRouter, BackgroundTasks
import json
import os
import asyncio
from datetime import datetime
from typing import List, Dict
from loguru import logger
from schemas import ResponseWrapper, TrainingRequest
from core.database import SessionLocal
from core.config import optimization_tasks
from models import Interaction, Version

router = APIRouter()

# 全局变量，稍后在 main.py 中初始化
dspy_service = None
sse_service = None

def init_services(dspy_svc, sse_svc):
    """初始化服务实例"""
    global dspy_service, sse_service
    dspy_service = dspy_svc
    sse_service = sse_svc

# 异步优化任务
async def run_dspy_optimization(training_data: List[Dict], version: str, ids: List[str]):
    """运行 DsPy 优化任务"""
    task_id = f"optimization_task_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    session = None
    
    try:
        from dspy.teleprompt import BootstrapFewShot
        from dspy.evaluate import Evaluate
        from dspy.evaluate.metrics import answer_exact_match

        # 更新状态并发送开始消息
        logger.info(f"开始优化任务 {task_id}，数据量: {len(training_data)}，版本: {version}")
        optimization_tasks[task_id] = "loading_data"
        await sse_service.broadcast_event("optimization_status", {
            "task_id": task_id,
            "status": "loading_data",
            "progress": 5,
            "message": "正在准备训练数据..."
        })

        # 创建训练集
        import dspy
        trainset = [
            dspy.Example(
                question=x["question"],
                reasoning=x["reasoning"], 
                answer=x["modelResponse"]
            ).with_inputs("question") 
            for x in training_data
        ]
        logger.info(f"任务 {task_id}: 已创建训练集，共 {len(trainset)} 条数据")
        
        # 更新状态
        optimization_tasks[task_id] = "preparing_model"
        await sse_service.broadcast_event("optimization_status", {
            "task_id": task_id,
            "status": "preparing_model",
            "progress": 10,
            "message": "正在准备模型..."
        })
        
        # 从最新版本加载预测模型
        session = SessionLocal()
        predict = dspy_service.model
        logger.info(f"任务 {task_id}: 已加载模型")
        
        # 设置优化器
        teleprompter = BootstrapFewShot(
            metric=dspy_service.eval_processor.llm_biological_metric, 
            max_labeled_demos=15
        )
        
        # 更新状态
        optimization_tasks[task_id] = "optimizing"
        await sse_service.broadcast_event("optimization_status", {
            "task_id": task_id,
            "status": "optimizing",
            "progress": 15,
            "message": "正在进行模型优化..."
        })
        
        # 编译优化
        logger.info(f"任务 {task_id}: 开始编译优化")
        compiled_predictor = teleprompter.compile(predict, trainset=trainset)
        logger.info(f"任务 {task_id}: 编译优化完成")
        
        # 更新状态
        optimization_tasks[task_id] = "saving_model"
        await sse_service.broadcast_event("optimization_status", {
            "task_id": task_id,
            "status": "saving_model",
            "progress": 50,
            "message": "正在保存优化后的模型..."
        })
        
        # 确保目录存在
        os.makedirs("dspy_program", exist_ok=True)
        last_version = session.query(Version.version).order_by(Version.created_at.desc()).first().version
        
        # 保存优化后的模型
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"dspy_program/program_v{last_version}_{timestamp}.pkl"
        compiled_predictor.save(output_path, save_program=False)
        logger.info(f"任务 {task_id}: 已保存模型到 {output_path}")
        
        # 解析当前版本号，生成新版本号
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
        
        # 通过 SSE 广播版本更新消息
        await sse_service.broadcast_event("version_update", {
            "old_version": version,
            "new_version": new_version,
            "description": description,
            "model_path": output_path,
            "training_ids": ids,
            "progress": 100,
            "message": f"优化完成，已创建新版本{new_version}"
        })
        logger.info(f"任务 {task_id}: 优化任务完成")
        
    except Exception as e:
        # 记录错误并通过 SSE 发送失败消息
        error_message = str(e)
        logger.error(f"任务 {task_id} 失败: {error_message}")
        optimization_tasks[task_id] = f"failed: {error_message}"
        
        await sse_service.broadcast_event("optimization_failed", {
            "version": version,
            "error": error_message,
            "task_id": task_id,
            "progress": 0,
            "message": f"优化失败: {error_message}"
        })
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
        loop.run_until_complete(sse_service.broadcast_event("optimization_created", {
            "task_id": task_info["task_id"],
            "status": "pending",
            "progress": 0,
            "message": f"已创建优化任务，准备处理 {len(task_info['training_data'])} 条数据",
            "version": task_info["version"],
            "ids": task_info["ids"]
        }))
        
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
        loop.run_until_complete(sse_service.broadcast_event("optimization_failed", {
            "version": task_info["version"],
            "error": str(e),
            "task_id": task_info["task_id"],
            "progress": 0,
            "message": f"优化失败: {str(e)}"
        }))
    finally:
        # 关闭事件循环
        loop.close()

@router.post("/addtraining", response_model=ResponseWrapper)
async def add_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """添加训练数据并启动优化任务"""
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
        task_info = {
            "training_data": training_data,
            "version": version,
            "ids": [item["id"] for item in training_data],
            "task_id": task_id
        }
        
        # 添加后台任务
        background_tasks.add_task(start_optimization_task, task_info)
        
        # 立即返回响应，不等待优化任务完成
        logger.info(f"已创建优化任务 {task_id}，将在后台处理 {len(training_data)} 条数据")
        return ResponseWrapper(
            status_code=200,
            detail="success",
            data={
                "message": f"成功收集 {len(training_data)} 条训练数据，已创建后台优化任务",
                "task_id": task_id,
                "exported_ids": [item["id"] for item in training_data],
                "version": version,
                "optimization_status": "pending"
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

@router.get("/optimization_status/{task_id:path}", response_model=ResponseWrapper)
async def get_optimization_status(task_id: str):
    """查询优化任务状态"""
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