"""
Mini-React 推理处理器模块
将原dspy_inference.py中的主要逻辑迁移过来
"""
import os
import time
import uuid
import asyncio
import json
from loguru import logger
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Any

from miniReact import ReAct, Module, streamify, LM
from signatures import MarineBiologyKnowledgeQueryAnswer
from tools import InferenceTools
from agents import get_agent

# 确保环境变量已加载
load_dotenv(override=True)

MAX_ITERS = int(os.getenv("MAX_ITERS", "10"))

class InferenceProcessor:
    """推理处理器"""
    
    def __init__(self):
        # 初始化工具
        self.tools = InferenceTools()
        
        # 使用miniReact原生的LM实例，与agents.py保持一致
        self.llm_manager = LM(
            model_name=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            api_base=os.getenv('BASE_URL', 'https://api.openai.com/v1'),
            api_key=os.getenv('API_KEY')
        )
        
        # 初始化版本号
        self.predictor_version = "1.0.0"
        
        # 初始化 ReAct 模型
        self.model = self._create_react_model()
        
        # 使用 streamify 包装，获得支持流式返回的模块
        self.streaming_model = streamify(self.model)
    
    def _create_react_model(self):
        """创建 ReAct 模型"""
        # 使用签名对象
        signature = MarineBiologyKnowledgeQueryAnswer
        
        # 创建工具函数列表
        tools = [
            self.tools.marine_species_query,
            self.tools.find_nodes_by_node_type,
            self.tools.batch_find_nodes_by_node_type,
            self.tools.get_unique_vector_query_results,
            self.tools.get_node_attribute,
            self.tools.get_adjacent_node_descriptions,
            self.tools.nodes_count
            
        ]
        
        # 创建 ReAct 模型
        return ReAct(
            signature=signature,
            tools=tools,
            max_iters=MAX_ITERS,
            lm=self.llm_manager
        )
    
    def get_last_message(self):
        """获取最后一条消息历史，包含推理过程和工具调用信息"""
        try:
            # 尝试从语言模型获取历史记录
            if hasattr(self.llm_manager, 'history') and self.llm_manager.history:
                last_msg = self.llm_manager.history[-1]
                # 如果获取到的消息格式正确，返回它
                if isinstance(last_msg, dict):
                    return last_msg
            
            # 如果没有历史记录，创建一个包含推理过程的默认消息格式
            current_time = time.time()
            
            # 构建包含推理轨迹的消息历史
            messages = []
            
            # 添加系统消息
            messages.append({
                "role": "system",
                "content": "你是一个专业的海洋生物知识问答助手，能够使用多种工具来检索和分析海洋生物信息。"
            })
            
            # 如果有最近的推理过程，添加到消息中
            if hasattr(self, '_last_trajectory') and self._last_trajectory:
                trajectory_content = self._format_trajectory_for_message(self._last_trajectory)
                if trajectory_content:
                    messages.append({
                        "role": "system", 
                        "content": trajectory_content
                    })
            
            # 如果有最近的工具调用，添加到消息中
            if hasattr(self, '_last_tool_calls') and self._last_tool_calls:
                tool_calls_content = self._format_tool_calls_for_message(self._last_tool_calls)
                if tool_calls_content:
                    messages.append({
                        "role": "system",
                        "content": tool_calls_content
                    })
            
            default_message = {
                "prompt": "",
                "messages": messages,
                "timestamp": current_time,
                "uuid": str(uuid.uuid4()),
                "model": f"{os.getenv('LLM_TYPE', 'openai')}-{os.getenv('LLM_MODEL', 'gpt-3.5-turbo')}",
                "response": None,
                "usage": None
            }
            
            logger.info(f"创建包含推理过程的消息，时间戳: {current_time} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))})")
            
            return default_message
            
        except Exception as e:
            logger.error(f"获取最后消息时出错: {str(e)}")
            # 返回None以便SSE服务能够优雅处理
            return None
    
    def _format_trajectory_for_message(self, trajectory):
        """将轨迹格式化为消息内容"""
        if not trajectory:
            return ""
        
        formatted_parts = []
        
        # 按顺序格式化思考、工具调用和观察结果
        idx = 0
        while f"thought_{idx}" in trajectory:
            thought = trajectory.get(f"thought_{idx}", "")
            tool_name = trajectory.get(f"tool_name_{idx}", "")
            tool_args = trajectory.get(f"tool_args_{idx}", {})
            observation = trajectory.get(f"observation_{idx}", "")
            
            if thought:
                formatted_parts.append(f"思考 {idx+1}: {thought}")
            if tool_name and tool_name != "finish":
                formatted_parts.append(f"调用工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
            if observation:
                formatted_parts.append(f"观察结果: {observation}")
            
            idx += 1
        
        return "\n\n".join(formatted_parts)
    
    def _format_tool_calls_for_message(self, tool_calls):
        """将工具调用格式化为消息内容，符合前端期望的格式"""
        formatted_calls = []
        
        for i, tool_call in enumerate(tool_calls):
            method = tool_call.get('method', '')
            args = tool_call.get('args', {})
            result = tool_call.get('result', '')
            
            # 使用前端期望的格式
            call_text = f"[[ ## tool_name_{i+1} ## ]]\n{method}\n[[ ## tool_args_{i+1} ## ]]\n{json.dumps(args, ensure_ascii=False)}"
            if result:
                call_text += f"\n[[ ## tool_result_{i+1} ## ]]\n{result}"
            
            formatted_calls.append(call_text)
        
        return "\n\n".join(formatted_calls)
    
    def set_last_inference_data(self, trajectory, tool_calls):
        """设置最后一次推理的数据，用于构建消息历史"""
        self._last_trajectory = trajectory
        self._last_tool_calls = tool_calls
    
    def load_model(self, file_path):
        """加载指定版本的模型"""
        try:
            # 这里可以实现模型加载逻辑
            logger.info(f"加载模型文件: {file_path}")
            # 重新创建模型
            self.model = self._create_react_model()
            self.streaming_model = streamify(self.model)
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return False
    
    def set_version(self, version):
        """设置当前预测器版本"""
        self.predictor_version = version
    
    def get_version(self):
        """获取当前预测器版本"""
        return self.predictor_version
    
    def predict(self, question):
        """非流式预测"""
        try:
            result = self.model(question=question)
            # 确保返回的对象有必要的属性
            if not hasattr(result, 'reasoning'):
                result.reasoning = getattr(result, 'answer', '') or ''
            if not hasattr(result, 'answer'):
                result.answer = getattr(result, 'reasoning', '') or ''
            return result
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            # 返回一个简单的预测结果
            return type('Prediction', (), {
                'answer': f'处理您的请求时出现错误: {str(e)}',
                'reasoning': f'发生错误: {str(e)}'
            })()
    
    # 删除这个方法，移动到类外部


class StreamingReAct:
    """支持流式输出的React推理器"""
    
    def __init__(self, signature, tools, max_iters=5, lm=None):
        self.signature = signature
        # 正确处理工具列表，使用函数名作为键
        self.tools = {tool.__name__: tool for tool in tools}
        self.max_iters = max_iters
        self.lm = lm
        
        # 创建基础的React模型
        from miniReact import ReAct
        self.react_model = ReAct(
            signature=signature,
            tools=tools,
            max_iters=max_iters,
            lm=lm
        )
    
    def stream_forward(self, query: str):
        """流式前向推理，捕获每个步骤的详细信息"""
        
        # 重置状态
        self.trajectory = {}
        self.step_count = 0
        self.tool_calls = []
        
        # 构建输入
        input_data = {
            "query": query,
            "tools": self.tools,
            "examples": self.examples if hasattr(self, 'examples') else []
        }
        
        # 执行推理
        try:
            # 调用mini-react的forward方法
            result = self.react_model._call_with_potential_trajectory_truncation(
                self.react_model.react, self.trajectory, lm=self.lm, **input_data
            )
            
            # 从结果中提取trajectory
            if hasattr(result, 'trajectory'):
                self.trajectory = result.trajectory
            elif hasattr(result, 'demo') and hasattr(result.demo[0], 'trajectory'):
                self.trajectory = result.demo[0].trajectory
            else:
                # 如果没有trajectory，创建一个基本的
                self.trajectory = {
                    "final_answer": str(result) if result else "无法生成答案",
                    "reasoning": "推理过程未记录"
                }
            
            logger.info(f"推理完成，trajectory包含 {len(self.trajectory)} 个键")
            
            # 生成流式步骤
            yield from self._generate_stream_steps()
            
        except Exception as e:
            logger.error(f"推理过程中发生错误: {str(e)}")
            # 生成错误步骤
            yield {
                "step_type": "error",
                "reasoning": f"推理过程中发生错误: {str(e)}",
                "answer": "很抱歉，处理您的问题时出现了错误。请稍后再试。",
                "tool_calls": []
            }
    
    def _generate_stream_steps(self):
        """从trajectory生成流式步骤"""
        
        # 初始化变量
        current_reasoning_parts = []
        current_tool_calls = []
        step_count = 0
        
        # 遍历trajectory，按顺序处理
        while True:
            thought_key = f"thought_{step_count}"
            tool_name_key = f"tool_name_{step_count}"
            tool_args_key = f"tool_args_{step_count}"
            observation_key = f"observation_{step_count}"
            
            # 检查是否还有更多步骤
            if thought_key not in self.trajectory:
                break
            
            thought = self.trajectory.get(thought_key, "")
            tool_name = self.trajectory.get(tool_name_key, "")
            tool_args = self.trajectory.get(tool_args_key, {})
            observation = self.trajectory.get(observation_key, "")
            
            # 1. 发送思考步骤
            if thought:
                reasoning_text = f"思考步骤 {step_count + 1}: {thought}"
                current_reasoning_parts.append(reasoning_text)
                
                yield {
                    "step_type": "thinking",
                    "reasoning": "\n\n".join(current_reasoning_parts),
                    "answer": "",
                    "tool_calls": current_tool_calls.copy()
                }
                
                # 模拟思考时间
                time.sleep(0.3)
            
            # 2. 发送工具调用步骤
            if tool_name and tool_name != "finish":
                tool_call_info = {
                    "method": tool_name,
                    "args": tool_args,
                    "result": None  # 结果稍后填充
                }
                
                # 添加工具调用的推理描述
                tool_reasoning = f"调用工具: {tool_name}"
                if tool_args:
                    args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
                    tool_reasoning += f"({args_str})"
                
                current_reasoning_parts.append(tool_reasoning)
                
                yield {
                    "step_type": "tool_call",
                    "reasoning": "\n\n".join(current_reasoning_parts),
                    "answer": "",
                    "tool_calls": current_tool_calls + [tool_call_info]
                }
                
                # 模拟工具调用时间
                time.sleep(0.5)
                
                # 3. 发送观察结果步骤
                if observation:
                    # 填充工具调用结果
                    tool_call_info["result"] = observation
                    current_tool_calls.append(tool_call_info)
                    
                    # 添加观察结果的推理描述
                    observation_reasoning = f"观察结果: {observation}"
                    current_reasoning_parts.append(observation_reasoning)
                    
                    yield {
                        "step_type": "observation",
                        "reasoning": "\n\n".join(current_reasoning_parts),
                        "answer": "",
                        "tool_calls": current_tool_calls.copy()
                    }
                    
                    # 模拟处理时间
                    time.sleep(0.3)
            
            # 处理finish工具调用
            elif tool_name == "finish":
                final_answer = tool_args.get("answer", "")
                if final_answer:
                    yield {
                        "step_type": "final_answer",
                        "reasoning": "\n\n".join(current_reasoning_parts),
                        "answer": final_answer,
                        "tool_calls": current_tool_calls.copy()
                    }
                    break
            
            step_count += 1
            
            # 安全检查，防止无限循环
            if step_count > 20:
                logger.warning("步骤数超过20，可能出现无限循环，强制退出")
                break
        
        # 如果没有找到finish，检查是否有直接的答案
        if step_count == 0 or not any(step.get("step_type") == "final_answer" for step in []):
            # 尝试从trajectory中获取最终答案
            final_answer = self.trajectory.get("final_answer", "")
            if not final_answer:
                # 如果没有final_answer，尝试其他可能的键
                for key in ["answer", "result", "output"]:
                    if key in self.trajectory:
                        final_answer = str(self.trajectory[key])
                        break
            
            if not final_answer:
                final_answer = "无法生成答案，请检查输入或重试。"
            
            yield {
                "step_type": "final_answer",
                "reasoning": "\n\n".join(current_reasoning_parts) if current_reasoning_parts else "推理过程已完成",
                "answer": final_answer,
                "tool_calls": current_tool_calls.copy()
            }
        
        # 保存工具调用信息
        self.tool_calls = current_tool_calls 


# 独立的流式预测函数
async def stream_predict(query: str, agent_name: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
    """
    真正的流式预测函数，返回推理步骤和最终答案
    解决streamify批处理问题，实现真正的实时流式推理
    """
    logger.info(f"开始流式预测: {query}")
    
    try:
        # 获取agent实例
        agent = get_agent(agent_name)
        if not agent:
            logger.error(f"Agent '{agent_name}' 创建失败")
            yield {
                "step_type": "error",
                "reasoning": f"Agent '{agent_name}' 不存在",
                "answer": "很抱歉，智能体初始化失败。请检查系统配置。",
                "tool_calls": []
            }
            return
        
        logger.info(f"Agent创建成功，类型: {type(agent)}, ReAct模型类型: {type(agent.react_model)}")
        
        # 测试基本工具功能
        try:
            logger.info("测试工具功能...")
            from tools import InferenceTools
            test_tools = InferenceTools()
            
            # 测试向量搜索工具
            vector_result = test_tools.get_unique_vector_query_results("鱼", top_k=1)
            logger.info(f"向量搜索测试结果: {vector_result}")
            
            # 测试计数工具
            count_result = test_tools.nodes_count(["test1", "test2"])
            logger.info(f"计数工具测试结果: {count_result}")
            
        except Exception as e:
            logger.error(f"工具功能测试失败: {e}")
        
        # 准备追踪变量
        reasoning_parts = []
        tool_calls = []
        trajectory = {}
        
        logger.info("开始真正的流式推理...")
        
        # 先发送一个初始步骤，让用户知道推理已开始
        yield {
            "step_type": "thinking",
            "reasoning": f"收到您的问题：{query}",
            "answer": "",
            "tool_calls": []
        }
        await asyncio.sleep(0.1)
        
        # 发送第二个步骤，表示开始分析
        yield {
            "step_type": "thinking", 
            "reasoning": f"收到您的问题：{query}\n\n正在启动AI推理引擎...",
            "answer": "",
            "tool_calls": []
        }
        await asyncio.sleep(0.2)
        
        # 发送第三个步骤，表示开始调用模型
        yield {
            "step_type": "thinking",
            "reasoning": f"收到您的问题：{query}\n\n正在启动AI推理引擎...\n\n开始深度分析您的问题...",
            "answer": "",
            "tool_calls": []
        }
        await asyncio.sleep(0.2)
        
        # 调用ReAct模型并处理轨迹
        try:
            logger.info("开始调用ReAct模型...")
            
            # 调用ReAct模型
            result = agent.react_model(question=query)
            
            logger.info(f"ReAct模型调用完成，result类型: {type(result)}")
            
            # 详细检查result对象的属性
            logger.info(f"Result属性: {[attr for attr in dir(result) if not attr.startswith('_')][:10]}")
            if hasattr(result, '__dict__'):
                logger.info(f"Result.__dict__: {list(result.__dict__.keys()) if result.__dict__ else 'Empty'}")
            
            # 提取轨迹数据 - 检查多种可能的轨迹位置
            trajectory = {}
            if hasattr(result, 'trajectory') and result.trajectory:
                trajectory = result.trajectory
                logger.info(f"从result.trajectory获取到轨迹数据，包含{len(trajectory)}个元素: {list(trajectory.keys())[:10]}")
            elif hasattr(result, 'demo') and result.demo and len(result.demo) > 0:
                demo = result.demo[0]
                if hasattr(demo, 'trajectory') and demo.trajectory:
                    trajectory = demo.trajectory
                    logger.info(f"从result.demo[0].trajectory获取到轨迹数据，包含{len(trajectory)}个元素: {list(trajectory.keys())[:10]}")
                else:
                    logger.warning("demo对象中没有轨迹数据")
            else:
                logger.warning("未找到轨迹数据，检查result的属性:")
                logger.warning(f"result类型: {type(result)}")
                logger.warning(f"result属性: {dir(result)[:10]}")
                # 检查是否可以直接从result获取推理信息
                if hasattr(result, 'answer'):
                    answer_str = str(result.answer)[:100] if result.answer else ""
                    logger.info(f"找到直接答案: {answer_str}...")
                
            # 如果trajectory为空，尝试从result构造基本的推理步骤
            if not trajectory:
                logger.warning("轨迹数据为空，尝试直接从result获取答案")
                
                # 发送分析步骤
                yield {
                    "step_type": "thinking",
                    "reasoning": "正在分析您的问题，请稍候...",
                    "answer": "",
                    "tool_calls": []
                }
                await asyncio.sleep(0.5)
                
                # 尝试从多个属性获取答案
                final_answer = ""
                if hasattr(result, 'answer') and result.answer:
                    final_answer = str(result.answer)
                elif hasattr(result, 'content') and result.content:
                    final_answer = str(result.content)
                elif hasattr(result, 'response') and result.response:
                    final_answer = str(result.response)
                else:
                    # 尝试从result的字符串表示中获取
                    result_str = str(result)
                    if result_str and result_str not in ["None", "<object>", "object"]:
                        final_answer = result_str
                    else:
                        final_answer = "抱歉，暂时无法处理您的问题。请检查系统配置或联系管理员。"
                
                logger.info(f"从空轨迹中提取的答案: {final_answer[:100]}...")
                
                # 发送最终答案
                yield {
                    "step_type": "final_answer", 
                    "reasoning": "分析完成",
                    "answer": final_answer,
                    "tool_calls": []
                }
                return
            
            # 解析轨迹并生成流式步骤
            formatted_steps = []
            tool_calls = []
            step_idx = 0
            max_steps = 20  # 防止无限循环
            
            logger.info(f"开始解析轨迹，trajectory包含键: {list(trajectory.keys()) if trajectory else 'None'}")
            
            # 检查轨迹数据的实际格式
            if trajectory:
                # 输出前几个键来调试格式
                sample_keys = list(trajectory.keys())[:5]
                logger.info(f"轨迹样本键: {sample_keys}")
                
                # 检查是否有thought_0, tool_name_0等格式的键
                has_thought_keys = any(key.startswith('thought_') for key in trajectory.keys())
                has_tool_keys = any(key.startswith('tool_name_') for key in trajectory.keys())
                logger.info(f"是否包含thought_键: {has_thought_keys}, 是否包含tool_name_键: {has_tool_keys}")
            
            while step_idx < max_steps and f"thought_{step_idx}" in trajectory:
                thought = trajectory.get(f"thought_{step_idx}", "")
                tool_name = trajectory.get(f"tool_name_{step_idx}", "")
                tool_args = trajectory.get(f"tool_args_{step_idx}", {})
                observation = trajectory.get(f"observation_{step_idx}", "")
                
                # 安全地处理切片操作，确保变量是字符串类型
                thought_str = str(thought)[:50] if thought else ""
                observation_str = str(observation)[:50] if observation else ""
                logger.info(f"步骤 {step_idx}: thought='{thought_str}...', tool='{tool_name}', args={tool_args}, obs='{observation_str}...'")
                
                # 发送思考步骤
                if thought:
                    step_text = f"步骤 {step_idx + 1}: {thought}"
                    formatted_steps.append(step_text)
                    
                    yield {
                        "step_type": "thinking",
                        "reasoning": "\n\n".join(formatted_steps),
                        "answer": "",
                        "tool_calls": tool_calls.copy()
                    }
                    
                    await asyncio.sleep(0.3)
                
                # 处理工具调用
                if tool_name and tool_name != "finish":
                    tool_call_info = {
                        "method": tool_name,
                        "args": tool_args,
                        "result": observation
                    }
                    tool_calls.append(tool_call_info)
                    
                    tool_text = f"步骤 {step_idx + 1}.1: 调用工具 {tool_name}"
                    formatted_steps.append(tool_text)
                    
                    yield {
                        "step_type": "tool_call",
                        "reasoning": "\n\n".join(formatted_steps),
                        "answer": "",
                        "tool_calls": tool_calls.copy()
                    }
                    
                    await asyncio.sleep(0.3)
                    
                    # 发送观察结果
                    if observation:
                        obs_text = f"步骤 {step_idx + 1}.2: 观察结果 - {observation}"
                        formatted_steps.append(obs_text)
                        
                        yield {
                            "step_type": "observation",
                            "reasoning": "\n\n".join(formatted_steps),
                            "answer": "",
                            "tool_calls": tool_calls.copy()
                        }
                        
                        await asyncio.sleep(0.3)
                
                # 处理finish工具
                elif tool_name == "finish":
                    final_answer = tool_args.get("answer", "") if tool_args else ""
                    
                    completion_text = f"步骤 {step_idx + 1}.3: 生成最终答案"
                    formatted_steps.append(completion_text)
                    
                    # 先发送thinking步骤，表明开始生成答案
                    yield {
                        "step_type": "thinking",
                        "reasoning": "\n\n".join(formatted_steps),
                        "answer": "",
                        "tool_calls": tool_calls.copy()
                    }
                    
                    await asyncio.sleep(0.3)
                    
                    # 然后发送final_answer，让SSE服务逐字符处理
                    yield {
                        "step_type": "final_answer",
                        "reasoning": "\n\n".join(formatted_steps),
                        "answer": final_answer,
                        "tool_calls": tool_calls.copy()
                    }
                    
                    logger.info(f"推理完成，最终答案: {final_answer[:100]}...")
                    return
                
                step_idx += 1
            
            # 如果没有执行任何步骤，说明轨迹格式不符合预期
            if step_idx == 0:
                logger.warning("没有找到符合预期格式的轨迹步骤，尝试其他方式解析")
                
                # 尝试直接从轨迹中找到答案相关的键
                possible_answer_keys = ['final_answer', 'answer', 'result', 'output']
                final_answer = ""
                
                for key in possible_answer_keys:
                    if key in trajectory:
                        final_answer = str(trajectory[key])
                        logger.info(f"从轨迹键'{key}'中找到答案: {final_answer[:100]}...")
                        break
                
                # 如果还是没找到，尝试从result对象获取
                if not final_answer and hasattr(result, 'answer'):
                    final_answer = str(result.answer)
                    logger.info(f"从result.answer中获取答案: {final_answer[:100]}...")
                
                # 尝试其他可能的属性
                if not final_answer:
                    for attr in ['response', 'text', 'output', 'content']:
                        if hasattr(result, attr):
                            final_answer = str(getattr(result, attr))
                            logger.info(f"从result.{attr}中获取答案: {final_answer[:100]}...")
                            break
                
                if not final_answer:
                    # 尝试从result的字符串表示中获取信息
                    result_str = str(result)
                    if result_str and result_str not in ["None", "<object>", "object"]: 
                        final_answer = result_str
                        logger.info(f"从result字符串表示中获取答案: {final_answer[:100]}...")
                    else:
                        # 最后的备用方案：提供一个友好的错误提示
                        query_str = str(query)[:50] if query else "您的问题"
                        final_answer = f"很抱歉，我在处理您的问题{query_str}...时遇到了一些技术问题。请稍后再试，或者联系管理员检查系统配置。"
                        logger.warning(f"无法从任何地方获取到答案，result类型: {type(result)}, 内容: {result_str[:100]}")
                    
            else:
                # 如果没有finish工具，尝试从result中提取答案
                if hasattr(result, 'answer') and result.answer:
                    final_answer = str(result.answer)
                elif hasattr(result, 'content') and result.content:
                    final_answer = str(result.content)
                else:
                    query_str = str(query)[:50] if query else "您的问题"
                    final_answer = f"基于已有信息，我已尽力分析了您的问题{query_str}...，但无法提供详细答案。请尝试重新描述问题或提供更多细节。"
            
            # 先发送thinking步骤表明开始生成答案（如果有formatted_steps）
            if formatted_steps:
                yield {
                    "step_type": "thinking",
                    "reasoning": "\n\n".join(formatted_steps) + "\n\n正在生成最终答案...",
                    "answer": "",
                    "tool_calls": tool_calls
                }
                await asyncio.sleep(0.3)
            
            # 然后发送final_answer让SSE服务逐字符处理
            yield {
                "step_type": "final_answer",
                "reasoning": "\n\n".join(formatted_steps) if formatted_steps else "推理过程完成",
                "answer": final_answer,
                "tool_calls": tool_calls
            }
            
            return
            
        except Exception as e:
            logger.error(f"ReAct模型调用失败: {e}")
            import traceback
            logger.error(f"完整错误信息: {traceback.format_exc()}")
            
            # 提供更友好的错误信息
            error_msg = str(e)
            if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                friendly_msg = "网络连接问题，请检查网络连接或稍后再试。"
            elif "api" in error_msg.lower() or "key" in error_msg.lower():
                friendly_msg = "API配置问题，请检查API密钥和服务地址配置。"
            else:
                friendly_msg = f"系统错误：{error_msg}。请稍后再试或联系管理员。"
            
            yield {
                "step_type": "error",
                "reasoning": f"处理问题“{query}”时发生错误",
                "answer": friendly_msg,
                "tool_calls": []
            }
            return
    
    except Exception as e:
        logger.error(f"流式预测过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"完整错误信息: {traceback.format_exc()}")
        
        # 尝试使用简单的直接LLM调用作为备用方案
        try:
            logger.info("尝试使用备用方案: 直接LLM调用")
            from core.llm_manager import get_inference_manager
            
            llm_manager = get_inference_manager()
            
            # 发送备用处理步骤
            yield {
                "step_type": "thinking",
                "reasoning": "正在使用备用方案处理您的问题...",
                "answer": "",
                "tool_calls": []
            }
            await asyncio.sleep(0.5)
            
            # 直接调用LLM
            simple_prompt = f"""你是一个海洋生物知识专家。请回答以下问题：

{query}

请用中文回答，并尽可能提供详细和准确的信息。"""
            
            backup_answer = llm_manager.complete(simple_prompt)
            
            # 发送备用答案
            yield {
                "step_type": "final_answer",
                "reasoning": "使用备用方案完成分析",
                "answer": backup_answer,
                "tool_calls": []
            }
            return
            
        except Exception as backup_error:
            logger.error(f"备用方案也失败: {backup_error}")
        
        # 最终的错误处理
        yield {
            "step_type": "error",
            "reasoning": f"处理问题“{query}”时发生错误",
            "answer": f"很抱歉，我无法处理您的问题“{query}”。请稍后再试，或者检查系统配置。错误信息: {str(e)}",
            "tool_calls": []
        }