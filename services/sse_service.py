import json
import asyncio
import time
import uuid
import os
from typing import AsyncGenerator, Dict, Any, Set, List
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from loguru import logger
from core.config import predictor_version
from inference import stream_predict
from core.llm_manager import get_inference_manager

def make_json_safe(obj):
    """将对象转换为JSON安全的格式"""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # 处理自定义对象
        return make_json_safe(obj.__dict__)
    else:
        return obj

class SSEService:
    """Server-Sent Events 服务"""
    
    def __init__(self, inference_processor):
        self.inference_processor = inference_processor
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

    async def stream_react_response_char_by_char(self, prompt: str) -> AsyncGenerator[ServerSentEvent, None]:
        """
        字符级别的流式 ReAct 响应生成器 - 逐字显示效果
        """
        try:
            # 跟踪推理过程
            final_answer = ""
            final_reasoning = ""
            tool_calls = []
            trajectory = {}
            
            # 当前显示的完整内容
            current_full_reasoning = ""
            current_full_answer = ""
            
            # 直接调用 streaming_react 函数
            async for step in stream_predict(query=prompt):
                if step:
                    # 获取步骤信息
                    step_type = step.get('step_type', 'unknown')
                    reasoning = step.get('reasoning', '')
                    answer = step.get('answer', '')
                    step_trajectory = step.get('trajectory', {})
                    step_tool_calls = step.get('tool_calls', [])
                    
                    # 更新轨迹和工具调用
                    trajectory.update(step_trajectory)
                    tool_calls = step_tool_calls
                    
                    # 逐字输出reasoning（思考过程 - 在浅色部分流式显示）
                    # 注意：当step_type为final_answer时，不再更新reasoning，专注于answer的流式输出
                    if reasoning and reasoning != current_full_reasoning and step_type != 'final_answer':
                        async for char_data in self._stream_text_char_by_char(
                            current_full_reasoning, reasoning, "reasoning", step_type, step_tool_calls, 
                            ""  # 推理过程中answer始终保持为空
                        ):
                            yield char_data
                        current_full_reasoning = reasoning
                    
                    # 只有在最终答案阶段才逐字输出answer（在文本部分流式显示）
                    if step_type == 'final_answer' and answer and answer != current_full_answer:
                        async for char_data in self._stream_text_char_by_char(
                            current_full_answer, answer, "answer", step_type, step_tool_calls, current_full_reasoning
                        ):
                            yield char_data
                        current_full_answer = answer
                    # 推理过程中不设置answer内容，保持为空
                    
                    # 记录最终内容
                    if step_type == 'final_answer':
                        final_answer = answer
                        final_reasoning = reasoning
                    elif step_type == 'error':
                        final_answer = answer
                        final_reasoning = reasoning
                    
                    # 如果是最终答案或错误，结束流式传输
                    if step_type == 'final_answer' or step_type == 'error':
                        break
            
            # 记录最终内容
            logger.info(f"字符级流式传输完成，最终内容 - Answer: {final_answer[:100]}... | Reasoning: {final_reasoning[:100]}...")
            
            # 流式结束后发送完成事件
            async for completion_event in self._send_completion_event(prompt, final_reasoning, final_answer, tool_calls, trajectory):
                yield completion_event
            
        except Exception as e:
            logger.error(f"字符级流式传输异常: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            
            # 发送错误事件
            error_data = make_json_safe({
                "reasoning": f"处理请求时发生错误: {str(e)}",
                "answer": "很抱歉，处理您的请求时出现了错误。请稍后再试。",
                "step_type": "error",
                "tool_calls": []
            })
            
            yield ServerSentEvent(
                data=json.dumps(error_data, ensure_ascii=False),
                event="chat_stream"
            )

    async def _stream_text_char_by_char(self, previous_text: str, new_text: str, 
                                       content_type: str, step_type: str, tool_calls: list, other_content: str = "", 
                                       char_delay: float = None) -> AsyncGenerator[ServerSentEvent, None]:
        """
        将文本按字符逐个发送的辅助方法
        
        Args:
            previous_text: 之前已显示的文本
            new_text: 新的完整文本
            content_type: 内容类型 ("reasoning" 或 "answer")
            step_type: 步骤类型
            tool_calls: 工具调用信息
            other_content: 另一种类型的完整内容（用于保持另一字段的完整性）
            char_delay: 每个字符的延迟时间（秒），如果为None则使用默认值
        """
        # 计算需要新增的文本部分
        if new_text.startswith(previous_text):
            # 新文本是之前文本的扩展
            new_part = new_text[len(previous_text):]
        else:
            # 完全新的文本
            new_part = new_text
            previous_text = ""
        
        # 如果没有新内容，直接返回
        if not new_part:
            return
        
        # 逐字符发送新增内容
        current_text = previous_text
        for char in new_part:
            current_text += char
            
            # 构建数据
            if content_type == "reasoning":
                data = {
                    "reasoning_delta": char,
                    "answer_delta": "",
                    "reasoning": current_text,
                    "answer": "",  # 推理阶段不传递answer内容
                    "step_type": step_type,
                    "tool_calls": tool_calls
                }
            else:  # answer
                data = {
                    "reasoning_delta": "",
                    "answer_delta": char,
                    "reasoning": other_content,  # 保持reasoning内容完整
                    "answer": current_text,
                    "step_type": step_type,
                    "tool_calls": tool_calls
                }
            
            # 确保数据是JSON安全的
            safe_data = make_json_safe(data)
            
            # 发送字符
            yield ServerSentEvent(
                data=json.dumps(safe_data, ensure_ascii=False),
                event="chat_stream"
            )
            
            # 控制字符显示速度 - 推理过程稍快，最终答案稍慢营造更好的阅读体验
            if char_delay is not None:
                await asyncio.sleep(char_delay)
            elif content_type == "reasoning":
                await asyncio.sleep(0.03)  # 30ms per character for reasoning (faster)
            else:  # answer
                await asyncio.sleep(0.06)  # 60ms per character for final answer (slower for better reading)

    async def stream_react_response(self, prompt: str) -> AsyncGenerator[ServerSentEvent, None]:
        """
        使用 sse-starlette 的流式 ReAct 响应生成器 (段落级别显示)
        """
        try:
            # 跟踪推理过程
            final_answer = ""
            final_reasoning = ""
            tool_calls = []
            trajectory = {}
            
            # 直接调用 streaming_react 函数
            async for step in stream_predict(query=prompt):
                if step:
                    # 获取步骤信息
                    step_type = step.get('step_type', 'unknown')
                    content = step.get('content', '')
                    reasoning = step.get('reasoning', '')
                    answer = step.get('answer', '')
                    step_trajectory = step.get('trajectory', {})
                    step_tool_calls = step.get('tool_calls', [])
                    
                    # 更新轨迹和工具调用
                    trajectory.update(step_trajectory)
                    tool_calls = step_tool_calls
                    
                    # 直接使用stream_predict返回的reasoning，不需要再次累积
                    current_reasoning = reasoning
                    current_answer = answer
                    
                    # 记录最终内容
                    if step_type == 'final_answer':
                        final_answer = answer
                        final_reasoning = reasoning
                    elif step_type == 'error':
                        final_answer = answer
                        final_reasoning = reasoning
                    
                    # 发送流式数据
                    data = {
                        "reasoning_delta": "",  # 不发送增量，发送完整内容
                        "answer_delta": "",     # 不发送增量，发送完整内容
                        "reasoning": current_reasoning,
                        "answer": current_answer,
                        "step_type": step_type,
                        "tool_calls": step_tool_calls
                    }
                    
                    # 确保数据是JSON安全的
                    safe_data = make_json_safe(data)
                    
                    logger.info(f"SSE流式数据[{step_type}]: reasoning={current_reasoning[:100]}... | answer={current_answer[:100]}...")
                    
                    # 使用 ServerSentEvent 创建事件
                    yield ServerSentEvent(
                        data=json.dumps(safe_data, ensure_ascii=False),
                        event="chat_stream"
                    )
                    
                    # 强制刷新输出，确保实时传输
                    await asyncio.sleep(0.1)
                    
                    # 如果是最终答案，结束流式传输
                    if step_type == 'final_answer' or step_type == 'error':
                        break
            
            # 记录最终内容
            logger.info(f"流式传输完成，最终内容 - Answer: {final_answer[:100]}... | Reasoning: {final_reasoning[:100]}...")
            
            # 流式结束后发送完成事件
            async for completion_event in self._send_completion_event(prompt, final_reasoning, final_answer, tool_calls, trajectory):
                yield completion_event
            
            # 推理数据已经在stream_predict中处理，不需要额外保存
            
        except Exception as e:
            # 捕获所有异常，返回错误信息
            error_message = str(e)
            logger.error(f"stream_react_response 发生错误: {error_message}")
            error_data = {"error": "处理请求失败", "message": error_message}
            
            yield ServerSentEvent(
                data=json.dumps(error_data, ensure_ascii=False),
                event="error"
            )
            
    async def _send_completion_event(self, prompt: str, final_reasoning: str = "", final_answer: str = "", tool_calls: list = None, trajectory: dict = None):
        """发送完成事件，包含详细的推理过程和工具调用信息"""
        
        # 构建消息历史，将工具调用信息添加到用户消息中（前端从这里提取召回方法）
        user_content = prompt
        if tool_calls:
            # 将工具调用信息添加到用户消息的内容中，供前端提取
            tool_calls_section = self._format_tool_calls_for_frontend(tool_calls)
            user_content += f"\n\n{tool_calls_section}"
        
        messages = [{"role": "user", "content": user_content}]
        
        # 构建content，只包含最终答案，不包含推理过程（推理过程已在思考框中显示）
        if final_answer:
            # 只包含最终答案，不包含推理摘要
            formatted_content = f"[[ ## answer ## ]]\n{final_answer}\n[[ ## completed ## ]]"
            logger.info(f"使用流式传输的最终答案构造completion事件，包含{len(tool_calls) if tool_calls else 0}个工具调用")
        else:
            # 如果没有答案，使用默认格式
            formatted_content = f"[[ ## answer ## ]]\n我可以帮助您处理各种海洋生物信息相关的任务。\n[[ ## completed ## ]]"
            logger.warning(f"最终答案为空，使用默认格式，包含{len(tool_calls) if tool_calls else 0}个工具调用")
        
        # 获取当前时间戳（转换为毫秒）
        current_timestamp = int(time.time() * 1000)
        
        # 构造完成事件数据
        completion_data = {
            "question": prompt,
            "prompt": prompt,
            "messages": messages,
            "timestamp": current_timestamp,
            "uuid": str(uuid.uuid4()),
            "model": f"{os.getenv('LLM_TYPE', 'openai')}/{os.getenv('LLM_MODEL', 'gpt-3.5-turbo')}",
            "version": predictor_version,
            "content": formatted_content,
            "tokens": {
                "completion_tokens": len(final_answer.split()) if final_answer else 50,
                "prompt_tokens": len(prompt.split()) * 2,
                "total_tokens": (len(final_answer.split()) if final_answer else 50) + len(prompt.split()) * 2
            }
        }
        
        # 确保completion数据是JSON安全的
        safe_completion_data = make_json_safe(completion_data)
        
        logger.info(f"发送completion事件，时间戳: {current_timestamp}")
        
        # 通过 SSE 返回完整消息
        yield ServerSentEvent(
            data=json.dumps({'prompt_history': json.dumps(safe_completion_data, ensure_ascii=False)}, ensure_ascii=False),
            event="completion"
        )
    
    def _format_tool_calls_for_frontend(self, tool_calls):
        """将工具调用格式化为前端期望的格式，只包含工具名称和参数"""
        if not tool_calls:
            return ""
            
        formatted_calls = []
        
        for i, tool_call in enumerate(tool_calls):
            method = tool_call.get('method', tool_call.get('tool_name', ''))
            args = tool_call.get('args', tool_call.get('arguments', {}))
            
            # 前端期望的格式：只包含工具名称和参数，不包含结果
            call_text = f"[[ ## tool_name_{i+1} ## ]] {method} [[ ## tool_args_{i+1} ## ]] {json.dumps(make_json_safe(args), ensure_ascii=False)}"
            formatted_calls.append(call_text)
        
        return "\n".join(formatted_calls)
    
    def _format_tool_calls_for_message(self, tool_calls):
        """将工具调用格式化为消息内容，包含完整信息（用于调试等）"""
        if not tool_calls:
            return ""
            
        formatted_calls = []
        
        for i, tool_call in enumerate(tool_calls):
            method = tool_call.get('method', tool_call.get('tool_name', ''))
            args = tool_call.get('args', tool_call.get('arguments', {}))
            result = tool_call.get('result', tool_call.get('observation', ''))
            
            # 完整格式，包含工具名称、参数和结果
            call_text = f"[[ ## tool_name_{i+1} ## ]] {method} [[ ## tool_args_{i+1} ## ]] {json.dumps(make_json_safe(args), ensure_ascii=False)}"
            if result:
                # 如果result是字符串，直接使用；否则转换为JSON字符串
                if isinstance(result, str):
                    call_text += f" [[ ## tool_result_{i+1} ## ]] {result}"
                else:
                    # 确保result是JSON安全的
                    safe_result = make_json_safe(result)
                    call_text += f" [[ ## tool_result_{i+1} ## ]] {json.dumps(safe_result, ensure_ascii=False)}"
            
            formatted_calls.append(call_text)
        
        return "\n\n".join(formatted_calls)
    
    def _format_trajectory_for_message(self, trajectory):
        """将轨迹格式化为消息内容，专注于推理过程"""
        if not trajectory:
            return ""
        
        formatted_sections = []
        formatted_sections.append("[[ ## 推理过程 ## ]]")
        
        # 按顺序格式化思考过程
        idx = 0
        while f"thought_{idx}" in trajectory:
            thought = trajectory.get(f"thought_{idx}", "")
            tool_name = trajectory.get(f"tool_name_{idx}", "")
            observation = trajectory.get(f"observation_{idx}", "")
            #步骤 {idx+1}
            if thought:
                formatted_sections.append(f"{thought}")
            
            # 如果有工具调用，简单说明（不包含详细结果）
            if tool_name and tool_name != "finish":
                formatted_sections.append(f"→ 使用工具: {tool_name}")
            
            # 如果有观察结果，提供简要总结
            if observation:
                # 简化观察结果显示
                if isinstance(observation, tuple) and len(observation) == 2:
                    count, _ = observation
                    formatted_sections.append(f"→ 找到 {count} 个相关结果")
                elif isinstance(observation, list):
                    formatted_sections.append(f"→ 找到 {len(observation)} 个相关结果")
                else:
                    # 截断过长的观察结果
                    obs_str = str(observation)
                    if len(obs_str) > 100:
                        formatted_sections.append(f"→ 获得结果: {obs_str[:100]}...")
                    else:
                        formatted_sections.append(f"→ 获得结果: {obs_str}")
            
            idx += 1
        
        return "\n\n".join(formatted_sections)

    def create_event_source_response(self, prompt: str) -> EventSourceResponse:
        """创建聊天的 EventSourceResponse - 使用字符级流式输出"""
        return EventSourceResponse(
            self.stream_react_response_char_by_char(prompt),
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