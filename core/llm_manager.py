"""
统一的LLM管理器模块
基于第一性原理设计，支持标准OpenAI协议和Azure OpenAI协议
参考mini-react的设计思路，实现协议统一和配置管理
"""
import os
import json
import httpx
from typing import Dict, List, Optional, Any, Union, AsyncIterable
from loguru import logger
from dotenv import load_dotenv
from urllib.parse import urljoin
from enum import Enum


class LLMProtocol(Enum):
    """LLM协议类型枚举"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    QWEN = "qwen"


class LLMConfig:
    """
    统一的LLM配置管理器
    基于单例模式，确保全局配置的一致性
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._debug = False
            cls._instance._load_from_env()
        return cls._instance
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        load_dotenv(override=True)
        
        # 基础配置
        self._config.update({
            "llm_type": os.getenv("LLM_TYPE", "openai"),
            "model_name": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            "api_key": os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("BASE_URL") or os.getenv("OPENAI_API_BASE"),
            "timeout": float(os.getenv("LLM_TIMEOUT", "60.0")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")) if os.getenv("LLM_MAX_TOKENS") else None,
        })
        
        # Azure OpenAI 特定配置
        if self._config["llm_type"] == "azure_openai":
            self._config.update({
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
                "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            })
        
        # 优化模型配置（用于训练和评估），默认使用主模型配置
        self._config.update({
            "train_llm_type": os.getenv("Train_LLM_TYPE", self._config["llm_type"]),
            "train_model_name": os.getenv("Train_LLM_MODEL", self._config["model_name"]),
            "train_api_key": os.getenv("Train_OPENAI_API_KEY", self._config["api_key"]),
            "train_base_url": os.getenv("Train_OPENAI_BASE_URL", self._config["base_url"]),
        })
        
        # 实体提取模型配置，默认使用主模型配置
        self._config.update({
            "entity_llm_type": os.getenv("ALI_LLM_TYPE", self._config["llm_type"]),
            "entity_model_name": os.getenv("ALI_LLM_MODEL", self._config["model_name"]),
            "entity_api_key": os.getenv("ALI_OPENAI_API_KEY", self._config["api_key"]),
            "entity_base_url": os.getenv("ALI_OPENAI_BASE_URL", self._config["base_url"]),
        })
        
        # 调试模式
        if os.getenv("LLM_DEBUG", "").lower() in ("1", "true", "yes"):
            self._debug = True
        
        logger.info(f"LLM配置已加载: {self._config['llm_type']} - {self._config['model_name']}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """设置配置项"""
        self._config[key] = value
        logger.info(f"配置已更新: {key} = {value}")
    
    def enable_debug(self):
        """启用调试模式"""
        self._debug = True
        logger.info("LLM调试模式已启用")
    
    def disable_debug(self):
        """禁用调试模式"""
        self._debug = False
        logger.info("LLM调试模式已禁用")
    
    def is_debug_enabled(self) -> bool:
        """检查是否启用调试模式"""
        return self._debug
    
    def get_protocol(self, config_prefix: str = "") -> LLMProtocol:
        """
        获取协议类型
        
        Args:
            config_prefix: 配置前缀，用于获取特定用途的配置(如train_, entity_)
        """
        llm_type_key = f"{config_prefix}llm_type" if config_prefix else "llm_type"
        llm_type = self._config.get(llm_type_key, "openai")
        
        try:
            return LLMProtocol(llm_type)
        except ValueError:
            logger.warning(f"未知的LLM类型: {llm_type}，使用默认OpenAI协议")
            return LLMProtocol.OPENAI


class BaseAPIClient:
    """
    基础API客户端
    提供统一的HTTP客户端抽象
    """
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        
        # 构建HTTP客户端
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self.client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout
        )
    
    def post(self, url: str, data: dict, **kwargs) -> httpx.Response:
        """发送POST请求"""
        return self.client.post(url, json=data, **kwargs)
    
    def close(self):
        """关闭客户端"""
        if hasattr(self, 'client'):
            self.client.close()
    
    def __del__(self):
        self.close()


class OpenAIAPIClient(BaseAPIClient):
    """标准OpenAI协议客户端"""
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        # 处理不同的API基地址格式
        if "qianfan.baidubce.com" in base_url:
            # 百度千帆API，使用v2路径，不需要v1
            if not base_url.endswith('/'):
                base_url += '/'
        else:
            # 标准OpenAI API
            if not base_url.endswith('/'):
                base_url += '/'
            if not base_url.endswith('v1/'):
                base_url += 'v1/'
        
        super().__init__(base_url, api_key, timeout)
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """调用聊天完成API"""
        
        # 统一使用OpenAI兼容格式（千帆ModelBuilder完全兼容）
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        url = "chat/completions"
        
        response = self.post(url, data)
        
        if response.status_code == 200:
            return response.json()
        else:
            # 改进错误处理
            try:
                if "application/json" in response.headers.get("content-type", ""):
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        if "error" in error_data:
                            error_info = error_data["error"]
                            if isinstance(error_info, dict):
                                error_message = error_info.get("message", str(error_info))
                            else:
                                error_message = str(error_info)
                        else:
                            error_message = error_data.get("message", str(error_data))
                    else:
                        error_message = str(error_data)
                else:
                    error_message = response.text
            except Exception:
                error_message = response.text
            
            raise Exception(f"API调用失败 (状态码: {response.status_code}): {error_message}")


class AzureOpenAIAPIClient(BaseAPIClient):
    """Azure OpenAI协议客户端"""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment_name: str, timeout: float = 60.0):
        # Azure OpenAI的URL格式
        base_url = f"{endpoint}/openai/deployments/{deployment_name}/"
        
        super().__init__(base_url, api_key, timeout)
        self.api_version = api_version
        self.deployment_name = deployment_name
        
        # Azure OpenAI使用api-key头而不是Bearer
        self.client.headers.update({
            "api-key": api_key,
            "Content-Type": "application/json"
        })
        # 移除Bearer认证
        if "Authorization" in self.client.headers:
            del self.client.headers["Authorization"]
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """调用Azure OpenAI聊天完成API"""
        data = {
            "messages": messages,
            **kwargs
        }
        
        # Azure OpenAI不需要在请求体中包含model参数
        if "model" in data:
            del data["model"]
        
        url = f"chat/completions?api-version={self.api_version}"
        response = self.post(url, data)
        
        if response.status_code == 200:
            return response.json()
        else:
            # 改进错误处理 - Azure版本
            try:
                if "application/json" in response.headers.get("content-type", ""):
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        if "error" in error_data:
                            error_info = error_data["error"]
                            if isinstance(error_info, dict):
                                error_message = error_info.get("message", str(error_info))
                            else:
                                error_message = str(error_info)
                        else:
                            error_message = error_data.get("message", str(error_data))
                    else:
                        error_message = str(error_data)
                else:
                    error_message = response.text
            except Exception:
                error_message = response.text
            
            raise Exception(f"Azure OpenAI API调用失败 (状态码: {response.status_code}): {error_message}")


class UnifiedLLMManager:
    """
    统一的LLM管理器
    基于第一性原理设计，支持多种协议的统一调用
    兼容mini-react框架的LM接口
    """
    
    def __init__(self, config_prefix: str = ""):
        """
        初始化LLM管理器
        
        Args:
            config_prefix: 配置前缀，用于获取特定用途的配置(如train_, entity_)
        """
        self.config = LLMConfig()
        self.config_prefix = config_prefix
        self.client = None
        
        # 为了兼容mini-react框架，添加期望的属性
        self._update_compatibility_attributes()
        
        self._setup_client()
        
        # 更新兼容性属性
        self._update_compatibility_attributes()
    
    def _setup_client(self):
        """设置API客户端"""
        protocol = self.config.get_protocol(self.config_prefix)
        
        if protocol == LLMProtocol.AZURE_OPENAI:
            self._setup_azure_client()
        else:
            self._setup_openai_client()
        
        # 设置完客户端后再次更新兼容性属性，确保所有属性都正确
        self._update_compatibility_attributes()
    
    def _setup_openai_client(self):
        """设置标准OpenAI协议客户端"""
        base_url_key = f"{self.config_prefix}base_url" if self.config_prefix else "base_url"
        api_key_key = f"{self.config_prefix}api_key" if self.config_prefix else "api_key"
        
        base_url = self.config.get_config(base_url_key)
        api_key = self.config.get_config(api_key_key)
        timeout = self.config.get_config("timeout")
        
        if not base_url:
            raise ValueError(f"缺少API base URL配置: {base_url_key}")
        if not api_key:
            raise ValueError(f"缺少API密钥配置: {api_key_key}")
        
        self.client = OpenAIAPIClient(base_url, api_key, timeout)
        logger.info(f"已初始化OpenAI客户端: {base_url}")
    
    def _setup_azure_client(self):
        """设置Azure OpenAI协议客户端"""
        endpoint = self.config.get_config("azure_endpoint")
        api_key = self.config.get_config("api_key")
        api_version = self.config.get_config("api_version")
        deployment_name = self.config.get_config("deployment_name")
        timeout = self.config.get_config("timeout")
        
        if not endpoint:
            raise ValueError("缺少Azure OpenAI endpoint配置")
        if not api_key:
            raise ValueError("缺少Azure OpenAI API密钥配置")
        if not deployment_name:
            raise ValueError("缺少Azure OpenAI deployment名称配置")
        
        self.client = AzureOpenAIAPIClient(endpoint, api_key, api_version, deployment_name, timeout)
        logger.info(f"已初始化Azure OpenAI客户端: {endpoint}")
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        model_key = f"{self.config_prefix}model_name" if self.config_prefix else "model_name"
        return self.config.get_config(model_key, "gpt-3.5-turbo")
    
    def _get_api_base(self) -> str:
        """获取API基础URL"""
        base_url_key = f"{self.config_prefix}base_url" if self.config_prefix else "base_url"
        return self.config.get_config(base_url_key, "")
    
    def _update_compatibility_attributes(self):
        """更新兼容性属性，使其兼容mini-react框架"""
        # 设置模型名称
        model_key = f"{self.config_prefix}model_name" if self.config_prefix else "model_name"
        self.model_name = self.config.get_config(model_key, "gpt-3.5-turbo")
        
        # 设置API基础URL
        base_url_key = f"{self.config_prefix}base_url" if self.config_prefix else "base_url"
        self.api_base = self.config.get_config(base_url_key, "")
        
        # 设置其他兼容性属性
        api_key_key = f"{self.config_prefix}api_key" if self.config_prefix else "api_key"
        self.api_key = self.config.get_config(api_key_key, "")
        
        # 添加history属性以兼容某些功能
        if not hasattr(self, 'history'):
            self.history = []
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        统一的聊天接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            聊天响应
        """
        model_name = self.get_model_name()
        
        # 合并默认参数
        params = {
            "temperature": self.config.get_config("temperature", 0.7),
            "max_tokens": self.config.get_config("max_tokens"),
            **kwargs
        }
        
        # 移除None值
        params = {k: v for k, v in params.items() if v is not None}
        
        if self.config.is_debug_enabled():
            logger.info(f"LLM调用: {model_name}")
            logger.info(f"参数: {params}")
            logger.info(f"消息: {messages}")
        
        try:
            response = self.client.chat_completion(
                model=model_name,
                messages=messages,
                **params
            )
            
            if self.config.is_debug_enabled():
                logger.info(f"API响应类型: {type(response)}")
                logger.info(f"API响应内容: {response}")
            
            # 处理不同类型的响应
            content = ""
            model_info = model_name
            usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            if isinstance(response, str):
                # 如果响应是字符串，直接使用
                content = response
            elif isinstance(response, dict):
                # 标准OpenAI格式
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if isinstance(choice, dict) and "message" in choice:
                        content = choice["message"].get("content", "")
                    else:
                        content = str(choice)
                elif "result" in response:
                    # 百度千帆可能的格式
                    content = response["result"]
                elif "output" in response:
                    # 其他可能的格式
                    content = response["output"]
                else:
                    # 如果都没有，尝试直接转字符串
                    content = str(response)
                
                # 只有当response是字典时才调用get方法
                model_info = response.get("model", model_name)
                usage_info = response.get("usage", usage_info)
            else:
                # 其他类型，尝试转字符串
                content = str(response)
            
            result = {
                "content": content,
                "model": model_info,
                "usage": usage_info
            }
            
            if self.config.is_debug_enabled():
                logger.info(f"最终结果: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            import traceback
            logger.error(f"完整错误信息: {traceback.format_exc()}")
            return {"content": f"调用语言模型时出错: {str(e)}", "error": str(e)}
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        文本完成接口
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            完成的文本
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, **kwargs)
        return response["content"]
    
    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """直接调用接口，返回兼容miniReact的字典格式"""
        try:
            result = self.complete(prompt, **kwargs)
            # 返回miniReact期望的格式
            return {
                "content": result,
                "choices": [{"message": {"content": result}}]
            }
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            error_msg = f"调用语言模型时出错: {str(e)}"
            return {
                "content": error_msg,
                "choices": [{"message": {"content": error_msg}}],
                "error": str(e)
            }
    
    def close(self):
        """关闭客户端"""
        if self.client:
            self.client.close()
    
    def __del__(self):
        self.close()


# 全局实例
_inference_manager = None
_training_manager = None
_entity_manager = None


def get_inference_manager() -> UnifiedLLMManager:
    """获取推理LLM管理器"""
    global _inference_manager
    if _inference_manager is None:
        _inference_manager = UnifiedLLMManager()
    return _inference_manager


def get_training_manager() -> UnifiedLLMManager:
    """获取训练LLM管理器"""
    global _training_manager
    if _training_manager is None:
        _training_manager = UnifiedLLMManager("train_")
    return _training_manager


def get_entity_manager() -> UnifiedLLMManager:
    """获取实体提取LLM管理器"""
    global _entity_manager
    if _entity_manager is None:
        _entity_manager = UnifiedLLMManager("entity_")
    return _entity_manager


def setup_openai(api_key: str, model: str = "gpt-3.5-turbo", base_url: str = "https://api.openai.com/v1/"):
    """
    快速设置OpenAI配置
    
    Args:
        api_key: OpenAI API密钥
        model: 模型名称
        base_url: API基础URL
    """
    config = LLMConfig()
    config.set_config("api_key", api_key)
    config.set_config("model_name", model)
    config.set_config("base_url", base_url)
    config.set_config("llm_type", "openai")
    
    # 重新初始化全局管理器
    global _inference_manager
    _inference_manager = None
    
    logger.info(f"已设置OpenAI配置: {model} @ {base_url}")


def setup_azure_openai(endpoint: str, api_key: str, deployment_name: str, api_version: str = "2023-12-01-preview"):
    """
    快速设置Azure OpenAI配置
    
    Args:
        endpoint: Azure OpenAI端点
        api_key: API密钥
        deployment_name: 部署名称
        api_version: API版本
    """
    config = LLMConfig()
    config.set_config("azure_endpoint", endpoint)
    config.set_config("api_key", api_key)
    config.set_config("deployment_name", deployment_name)
    config.set_config("api_version", api_version)
    config.set_config("llm_type", "azure_openai")
    
    # 重新初始化全局管理器
    global _inference_manager
    _inference_manager = None
    
    logger.info(f"已设置Azure OpenAI配置: {deployment_name} @ {endpoint}")


def setup_deepseek(api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com/v1/"):
    """
    快速设置DeepSeek配置
    
    Args:
        api_key: DeepSeek API密钥
        model: 模型名称
        base_url: API基础URL
    """
    config = LLMConfig()
    config.set_config("api_key", api_key)
    config.set_config("model_name", model)
    config.set_config("base_url", base_url)
    config.set_config("llm_type", "deepseek")
    
    # 重新初始化全局管理器
    global _inference_manager
    _inference_manager = None
    
    logger.info(f"已设置DeepSeek配置: {model} @ {base_url}")


# 兼容性函数
def chat(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """全局聊天接口"""
    manager = get_inference_manager()
    return manager.chat(messages, **kwargs)


def complete(prompt: str, **kwargs) -> str:
    """全局完成接口"""
    manager = get_inference_manager()
    return manager.complete(prompt, **kwargs)


def enable_debug():
    """启用调试模式"""
    LLMConfig().enable_debug()


def disable_debug():
    """禁用调试模式"""
    LLMConfig().disable_debug() 