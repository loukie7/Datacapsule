import os
import dspy
from dotenv import load_dotenv
from loguru import logger

# 确保环境变量已加载
load_dotenv(override=True)

class DspyEvaluationProcessor:
    def __init__(self):
        # 初始化评估用的语言模型
        self.eval_lm = dspy.LM(
            f'{os.getenv("Train_LLM_TYPE")}/{os.getenv("Train_LLM_MODEL")}',
            base_url=os.getenv("Train_OPENAI_BASE_URL"),
            api_key=os.getenv("Train_OPENAI_API_KEY"),
            stream=True  # 直接在创建模型时启用流式模式
        )
        # 移除全局配置，避免影响其他模块
        # dspy.configure(lm=self.eval_lm)
    
    # 评估相关功能
    class BiologicalRetrievalEvaluation(dspy.Signature):
        """评估生物检索任务的推理步骤质量"""
        question = dspy.InputField(desc="用户的查询问题")
        standard_reasoning = dspy.InputField(desc="标准的推理步骤")
        predicted_reasoning = dspy.InputField(desc="模型产生的推理步骤")
        evaluation_score = dspy.OutputField(desc="评分(0-100)")
        evaluation_feedback = dspy.OutputField(desc="详细的评分解释，包括各个方面的得分")

    class LLMBiologicalEvaluator(dspy.Module):
        def __init__(self, eval_lm):
            super().__init__()
            # 使用传入的评估模型
            self.eval_lm = eval_lm
            
            # 使用思维链方式增强评估能力，直接提供指令，并使用专用评估模型
            self.eval_chain = dspy.ChainOfThought(
                DspyEvaluationProcessor.BiologicalRetrievalEvaluation,
                instructions="""
                您是一位专业的生物检索质量评估专家。您的任务是评估模型产生的生物检索推理步骤质量。
            
            请根据以下标准进行评分(总分100分):
            
            1. 检索条件识别准确性 (20分)
               - 是否正确识别了所有检索条件
               - 是否正确区分了精确条件和模糊条件
            
            2. 需求识别准确性 (10分)
               - 是否正确识别了查询中的所有需求
            
            3. 检索策略合理性 (40分)
               - 是否先执行精确检索，后执行模糊检索 (10分)
               - 后续检索步骤是否基于前面步骤的结果 (10分)
               - 筛选顺序是否从限制性强的条件开始 (10分)
               - 前面步骤检索到的内容是否把全部信息传递给后面检索所使用的工具 (10分)
            
            4. 结果整合正确性和完整性 (30分)
               - 答案准确性，与标准答案相比，核心事实是否一致，即使表达方式不同，只要核心信息正确也应得高分 (25分)
               - 提取所有需要汇总的信息  (5分)
            
            评估须知：
                1. 在评估答案准确性时，请比对预测结果与标准答案的内容，理解语义等价性而非只做字符匹配
                2. 即使表达方式不同，只要内容实质相同，也应给予高分
                3. 同一事实的不同表述方式应被视为正确，如"共有3种"和"总共有三种"表达的是相同含义
                4. 对每个评分维度提供详细分析和具体理由
                """
            )
            # 显式设置评估链使用评估模型
            self.eval_chain.lm = self.eval_lm
        
        def forward(self, example, prediction):
            """评估预测与标准答案的匹配程度
            
            Args:
                example: 包含标准答案的示例
                prediction: 模型的预测结果
                
            Returns:
                float: 0-1之间的分数
            """
            # 如果没有推理步骤，使用简单的答案匹配
            if not hasattr(example, 'reasoning') or not hasattr(prediction, 'reasoning'):
                return 1.0 if dspy.evaluate.answer_exact_match(example, prediction) else 0.0
            
            # 准备标准推理步骤
            standard_reasoning = "\n".join(example.reasoning) if isinstance(example.reasoning, list) else example.reasoning
            
            # 获取预测的推理步骤
            predicted_reasoning = prediction.reasoning if hasattr(prediction, 'reasoning') else ""
            
            try:
                # 直接使用评估链，不再使用 context 管理器
                # 因为我们已经在创建模型时启用了流式模式，并显式设置了评估链使用评估模型
                evaluation = self.eval_chain(
                    question=example.question,
                    standard_reasoning=standard_reasoning,
                    predicted_reasoning=predicted_reasoning
                )
                
                # 将分数从0-100转换为0-1
                try:
                    score = float(evaluation.evaluation_score) / 100.0
                    # 边界处理
                    score = max(0.0, min(1.0, score))
                    logger.info(f"评估结果: {score} (问题: {example.question[:30]}...)")
                    return score
                except:
                    # 如果分数转换失败，默认返回0.5
                    logger.warning(f"评估分数转换失败，使用默认分数0.5")
                    return 0.5
            except Exception as e:
                logger.error(f"评估过程出错: {str(e)}")
                # 出错时返回默认分数
                return 0.5
    
    def llm_biological_metric(self, example, pred, trace=None, frac=1.0):
        """使用大模型评估函数"""
        try:
            # 创建评估器实例，传入评估模型
            evaluator = self.LLMBiologicalEvaluator(self.eval_lm)
            
            # 确保在 litellm 客户端级别启用流式模式
            if hasattr(evaluator.eval_lm, 'client'):
                evaluator.eval_lm.client.stream = True
                logger.info("已在评估模型客户端级别启用流式模式")
            
            # 执行评估
            result = evaluator(example, pred)
            return result
        except Exception as e:
            logger.error(f"评估指标计算出错: {str(e)}")
            # 出错时返回默认分数
            return 0.5