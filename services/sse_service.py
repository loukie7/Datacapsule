import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Set, List
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from loguru import logger
from core.config import predictor_version

class SSEService:
    """Server-Sent Events 服务"""
    
    def __init__(self, dspy_processor):
        self.dspy_processor = dspy_processor
        # 添加 SSE 连接管理 - 使用字典存储连接和对应的队列
        self.connections: Dict[str, asyncio.Queue] = {}

    async def add_connection(self, connection_id: str, queue: asyncio.Queue):
        """添加 SSE 连接"""
        self.connections[connection_id] = queue
        logger.info(f"新增 SSE 连接 {connection_id}，当前连接数: {len(self.connections)}")

    async def remove_connection(self, connection_id: str):
        """移除 SSE 连接"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            logger.info(f"移除 SSE 连接 {connection_id}，当前连接数: {len(self.connections)}")

    async def broadcast_event(self, event: str, data: Dict[str, Any]):
        """向所有 SSE 连接广播事件"""
        if not self.connections:
            logger.info("没有活跃的 SSE 连接，跳过广播")
            return

        message = ServerSentEvent(
            data=json.dumps(data, ensure_ascii=False),
            event=event
        )
        
        # 向所有连接的队列中推送消息
        disconnected = []
        for connection_id, queue in self.connections.items():
            try:
                # 使用 nowait 避免阻塞，如果队列满了就跳过
                queue.put_nowait(message)
            except asyncio.QueueFull:
                logger.warning(f"连接 {connection_id} 的队列已满，跳过这次广播")
            except Exception as e:
                logger.warning(f"向 SSE 连接 {connection_id} 发送消息失败: {e}")
                disconnected.append(connection_id)
        
        # 清理断开的连接
        for connection_id in disconnected:
            await self.remove_connection(connection_id)
        
        logger.info(f"广播事件 '{event}' 到 {len(self.connections) - len(disconnected)} 个连接")

    async def events_stream(self) -> AsyncGenerator[ServerSentEvent, None]:
        """SSE 事件流生成器，用于状态推送"""
        # 为每个连接创建唯一 ID
        connection_id = f"sse_{asyncio.current_task().get_name()}_{id(asyncio.current_task())}"
        
        # 创建广播队列
        broadcast_queue = asyncio.Queue(maxsize=100)  # 限制队列大小防止内存泄漏
        
        await self.add_connection(connection_id, broadcast_queue)
        
        try:
            # 发送连接确认
            yield ServerSentEvent(
                data=json.dumps({"message": "SSE 连接已建立", "connection_id": connection_id}, ensure_ascii=False),
                event="connected"
            )
            
            # 持续监听广播队列
            while True:
                try:
                    # 等待广播消息，设置超时以便定期发送心跳
                    message = await asyncio.wait_for(broadcast_queue.get(), timeout=30.0)
                    yield message
                except asyncio.TimeoutError:
                    # 发送心跳消息
                    yield ServerSentEvent(
                        data=json.dumps({"timestamp": asyncio.get_event_loop().time()}, ensure_ascii=False),
                        event="heartbeat"
                    )
                    
        except Exception as e:
            logger.error(f"SSE 连接 {connection_id} 异常: {e}")
        finally:
            await self.remove_connection(connection_id)

    async def stream_react_response(self, prompt: str) -> AsyncGenerator[ServerSentEvent, None]:
        """
        使用 sse-starlette 的流式 ReAct 响应生成器
        """
        try:
            # 跟踪上一次的内容，用于增量更新
            last_reasoning = ""
            last_answer = ""
            
            # 直接调用 streaming_react 函数
            async for chunk in self.dspy_processor.stream_predict(question=prompt):
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
                            "reasoning": current_reasoning,
                            "answer": current_answer
                        }
                        logger.info(f"SSE增量数据: {json.dumps(data, ensure_ascii=False)}")
                        
                        # 使用 ServerSentEvent 创建事件
                        yield ServerSentEvent(
                            data=json.dumps(data, ensure_ascii=False),
                            event="chat_stream"
                        )
                        
                        # 强制刷新输出，确保实时传输
                        await asyncio.sleep(0)  # 让出控制权，确保数据被发送
                        
                        # 更新上一次的内容
                        last_reasoning = current_reasoning
                        last_answer = current_answer
            
            # 流式结束后的处理
            async for completion_event in self._send_completion_event(prompt):
                yield completion_event
            
        except Exception as e:
            # 捕获所有异常，返回错误信息
            error_message = str(e)
            logger.error(f"stream_react_response 发生错误: {error_message}")
            error_data = {"error": "处理请求失败", "message": error_message}
            
            yield ServerSentEvent(
                data=json.dumps(error_data, ensure_ascii=False),
                event="error"
            )
            
    async def _send_completion_event(self, prompt: str):
        """发送完成事件"""
        last_message = self.dspy_processor.get_last_message()
        
        # 检查 last_message 是否为 None 或不包含必要字段
        if not last_message:
            error_data = {"error": "无法获取消息历史", "message": "处理请求时发生错误"}
            logger.error(f"last_message 为空或无效")
            yield ServerSentEvent(
                data=json.dumps(error_data, ensure_ascii=False),
                event="error"
            )
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
                # 统一 tokens 字段结构
                if "usage" in last_message and last_message["usage"]:
                    tokens = {
                        "completion_tokens": last_message["usage"].get("completion_tokens", 0),
                        "prompt_tokens": last_message["usage"].get("prompt_tokens", 0),
                        "total_tokens": last_message["usage"].get("total_tokens", 0)
                    }
                    data_to_send["tokens"] = tokens
                else:
                    data_to_send["tokens"] = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
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
        
        # 通过 SSE 返回完整消息
        yield ServerSentEvent(
            data=json.dumps({'prompt_history': json_message}, ensure_ascii=False),
            event="completion"
        )
    
    def create_event_source_response(self, prompt: str) -> EventSourceResponse:
        """创建聊天的 EventSourceResponse"""
        return EventSourceResponse(
            self.stream_react_response(prompt),
            ping=5,  # 每5秒发送一次ping，提高响应性
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*"
            }
        )

    def create_events_response(self) -> EventSourceResponse:
        """创建状态推送的 EventSourceResponse"""
        return EventSourceResponse(
            self.events_stream(),
            ping=30,  # 每30秒发送一次ping
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache", 
                "Expires": "0",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*"
            }
        ) 