import os
import sys
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv(override=True)

# 设置日志配置
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.remove()  # 移除默认处理器
logger.add(sys.stderr, level=log_level)  # 添加新的处理器并设置日志级别
logger.info(f"日志级别设置为: {log_level}")

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///interactions.db")

# 全局变量
predictor_version = "1.0.0"

# 全局优化任务跟踪
optimization_tasks = {} 