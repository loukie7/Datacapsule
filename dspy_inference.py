import os
import dspy
from react_tools import ReActTools, GraphVectorizer
from dotenv import load_dotenv
from dspy_query_db import MarineSpeciesQuery
import json
from loguru import logger


# 确保环境变量已加载
load_dotenv(override=True)

MAX_ITERS = int(os.getenv("MAX_ITERS","10"))

class DspyInferenceProcessor:
    def __init__(self):
        self.ragtool = ReActTools()
        self.graphvectorizer = GraphVectorizer()
        self.query_processor = MarineSpeciesQuery(os.getenv("SPECIES_DB_URL","./.dbs/marine_species.db"))
        # 初始化语言模型
        self.lm = dspy.LM(
            f'{os.getenv("LLM_TYPE")}/{os.getenv("LLM_MODEL")}',
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY")
        )
        
        # 配置 dspy 使用该语言模型
        dspy.configure(lm=self.lm)

        # 初始化版本号
        self.predictor_version = "1.0.0"
        # 初始化 RactModel
        self.model = self.RactModel(self)
        # 使用 streamify 包装，获得支持流式返回的模块
        self.streaming_model = dspy.streamify(self.model)
    
    def find_nodes_by_node_type(self, start_node, trget_node_type):
        '''
        此方法会根据传入的节点名称，在图数据中以该节点为起点查找包含指定节点类型的节点列表，并返回节点数量与节点列表。
        start_node 为开始查找的树节点名称，只允许单个节点、
        trget_node_type 目标节点类型,只允许单个类型名称
        返回值为从该节点开始，包含指定属性名的节点数量与节点列表
        已知图数据中存在一系列的海洋生物相关信息：
        1. ⽣物分类学图数据：包括"拉丁学名", "命名年份", "作者", "中文学名",
        2. ⽣物科属于数据："界", "门", "纲", "目", "科", "属", "种"(种即是中文学名),它们的从属关系是: 界 -> 门 -> 纲 -> 目 -> 科 ->属 ->种 。
        3. ⽣物特征图数据：包括"自然分布地", "生物特征","生活习性"等。
        本方法可以根据给定的节点名称，在图数据中以此节点为起点查找包含指定该属性的节点或节点列表，例如1："盲鳗科" "种" 则会返回 盲鳗科所有的种，例如2："盲鳗科" "界" 则会返回 盲鳗科对应的界， 。
        4. 因为本方法需要的参数是精准的节点属性名称(或节点类型名)，建议查询的节点类型属于"自然分布地", "生物特征", "生活习性"等时,或查询返回为空时、查询失败时，先通过get_unique_vector_query_results方法获取准确的节点名称，再通过本方法获取对应的节点信息。

        Args:
            start_node: 开始查找的节点名称
            trget_node_type: 目标节点类型
        Returns:
            count: 节点数量
            nodes: 节点列表
        '''
        nodes = self.ragtool.find_nodes_by_node_type(start_node, trget_node_type)
        # 如果nodes为空，则返回0,不为为空时，则返回节点数量与节点列表
        if not nodes:
            return 0,[]
        count = len(nodes)
        return count,nodes

    def batch_find_nodes_by_node_type(self, start_nodes, trget_node_type):
        '''
        此方法会根据传入包含多个开始节点的列表，批量查询指定目标节点类型的节点列表，返回多条查询的结果集。
        Args:
            start_nodes: 开始查找的节点名称列表
            trget_node_type: 目标节点类型
        Returns:
            traget_nodes_list: 多条查询结果的列表
        '''
        # 字典格式为，key为节点名称，value为包含指定属性名的节点数量与节点列表
        traget_nodes_list = {}
        for node in start_nodes:
            count, nodes = self.find_nodes_by_node_type(start_nodes, trget_node_type)
            traget_nodes_list[node] = {"count": count, "nodes": nodes}
        return traget_nodes_list

    def get_unique_vector_query_results(self, query, node_type=None, search_type="all", top_k=1, better_than_threshold=0.65):
        """通过向量搜索，获取与查询最相关的实体或关系
        Args:
            query: 搜索查询文本
            node_type: 实体类型筛选条件，如果为None则不筛选。可选值包括：
                - species (种、中文名)
                - 界
                - 门
                - 纲
                - 目
                - 科
                - 属
                - 位置
                - 繁殖特征
                - 行为特征
                - 体型
                - 体色
                - 体长
                - 特殊特征
            search_type: 搜索类型，'all'/'entity'/'relation'
            top_k: 返回结果的数量
            better_than_threshold: 相似度阈值，只返回相似度高于此值的结果
        Returns:
            list: 搜索结果，精准的实体名列表
        """
        try:
            # 添加超时控制
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            # 使用线程池执行可能耗时的操作
            with ThreadPoolExecutor() as executor:
                # 设置超时时间（例如10秒）
                future = executor.submit(self.graphvectorizer.search, query, node_type, search_type, top_k, better_than_threshold)
                try:
                    result = future.result(timeout=10)  # 10秒超时
                    return result
                except TimeoutError:
                    logger.error(f"向量搜索超时: query={query}, node_type={node_type}")
                    return []  # 超时返回空列表
        except Exception as e:
            # 捕获所有异常，确保不会导致整个流程崩溃
            logger.error(f"向量搜索出错: {str(e)}, query={query}, node_type={node_type}")
            return []  # 出错返回空列表

    def get_node_attribute(self,node_id):
        '''
        根据节点id获取所有属性，包括中文学名、拉丁学名、命名年份、作者、node_type
        Args:
            node_id: 节点id
        Returns:
            list: 属性列表
        '''
        return self.ragtool.get_node_attribute(node_id)
    def get_adjacent_node_descriptions(self, nodenames):
        '''
        此方法会根据传入的节点列表，获取每个节点相邻所有节点描述，合并到一个列表中返回，非精准检索，谨慎使用
        Args:
            nodenames: 节点名称列表
        Returns:
            list: 相邻节点描述列表
        '''
        return self.ragtool.get_adjacent_node_descriptions(nodenames)

    def nodes_count(self, nodes):
        '''
        此方法会根据传入的节点列表，统计数量，返回数量
        Args:
            nodes: 节点列表
        Returns:
            int: 节点数量
        '''
        if not nodes:
            return 0
        return len(nodes)
    
    def MarineSpeciesQuery(self,query):
        """根据自然语言查询数据库
        Args:
            natural_language_query: 用户的自然语言查询
            
        Returns:
            查询结果和解释
        """
        result = self.query_processor.query_database(query)
        return  self.query_processor.format_query_results(result)
    
    # 定义签名类
    class MarineBiologyKnowledgeQueryAnswer(dspy.Signature):
        """
        针对复杂检索问题的增强签名。
        此签名能够：
        1. 分析用户问题，提取精确检索条件和模糊检索条件
        2. 确定检索顺序和优先级策略
        3. 对多实体结果进行遍历查询
        4. 按照检索需求有序组织答案
        """
        # 输入字段
        question = dspy.InputField(desc="用户的原始问题")
        # 输出字段
        answer = dspy.OutputField(desc="根据检索结果综合形成的完整答案，确保涵盖所有检索需求，使用中文回复")
    
    # 建议添加的问题分类签名
    class QuestionClassifier(dspy.Signature):
        """对用户问题进行分类"""
        question = dspy.InputField(desc="用户的原始问题")
        question_type = dspy.OutputField(desc="问题类型，可能的值包括：实体查询/关系查询/属性查询/统计查询等")
        search_strategy = dspy.OutputField(desc="建议的检索策略：向量检索/图检索/混合检索")
        key_entities = dspy.OutputField(desc="问题中的关键实体列表")
    
    # 定义 RactModel 类
    class RactModel(dspy.Module):
        def __init__(self, processor):
            super().__init__()
            # 保存外部类的引用
            self.processor = processor
            # 利用 ReAct 将工具函数集成进来
            self.react = dspy.ReAct(
                DspyInferenceProcessor.MarineBiologyKnowledgeQueryAnswer,
                max_iters = MAX_ITERS,
                tools=[
                    processor.find_nodes_by_node_type, 
                    processor.batch_find_nodes_by_node_type,
                    processor.get_unique_vector_query_results,
                    processor.get_node_attribute,
                    processor.get_adjacent_node_descriptions,
                    processor.nodes_count
                ]
            )
        
        def forward(self, question):
            return self.react(question=question)
    
    def get_last_message(self):
        """获取最后一条消息历史"""
        return self.lm.history[-1] if self.lm.history else None
    
    def load_model(self, file_path):
        """加载指定版本的模型"""
        result = self.model.load(file_path)
        # 加载模型后清除缓存
        dspy.settings.configure(cache=None)
        return result
    
    def set_version(self, version):
        """设置当前预测器版本"""
        self.predictor_version = version
    
    def get_version(self):
        """获取当前预测器版本"""
        return self.predictor_version
    
    def predict(self, question):
        """非流式预测"""
        return self.model(question=question)
    
    def stream_predict(self, question):
        """流式预测，实现真正的增量输出"""
        try:
            # 创建一个跟踪状态的对象
            class StreamState:
                def __init__(self):
                    self.last_answer = ""
                    self.last_reasoning = ""
                    self.is_first_chunk = True
                    
            state = StreamState()
            
            # 使用 dspy 的流式模型获取结果
            async def real_stream():
                # 首先发送一个空的初始状态
                if state.is_first_chunk:
                    initial_prediction = type('Prediction', (), {
                        'answer': '',
                        'reasoning': '思考中...'
                    })
                    state.is_first_chunk = False
                    yield initial_prediction
                
                # 启动非流式预测（在后台运行）
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                # 创建一个执行器来运行阻塞的预测
                with ThreadPoolExecutor() as executor:
                    # 提交预测任务到线程池
                    future = executor.submit(self.predict, question)
                    
                    # 每隔一小段时间检查一次结果，模拟流式输出
                    while not future.done():
                        await asyncio.sleep(0.2)  # 等待200毫秒
                        # 发送思考中的状态
                        thinking_prediction = type('Prediction', (), {
                            'answer': state.last_answer,
                            'reasoning': state.last_reasoning + "."  # 添加一个点表示思考
                        })
                        state.last_reasoning += "."
                        yield thinking_prediction
                    
                    # 获取最终结果
                    try:
                        final_result = future.result()
                        # 如果最终结果可用，分段返回
                        if hasattr(final_result, 'answer') and hasattr(final_result, 'reasoning'):
                            # 将答案和推理过程分成多个部分
                            answer_parts = self._split_text(final_result.answer, 10)  # 分成约10个部分
                            reasoning_parts = self._split_text(final_result.reasoning, 5)  # 分成约5个部分
                            
                            # 先返回完整的推理过程
                            for i, reasoning_part in enumerate(reasoning_parts):
                                current_reasoning = "".join(reasoning_parts[:i+1])
                                prediction = type('Prediction', (), {
                                    'answer': state.last_answer,
                                    'reasoning': current_reasoning
                                })
                                state.last_reasoning = current_reasoning
                                yield prediction
                                await asyncio.sleep(0.1)  # 短暂停顿
                            
                            # 然后逐步返回答案
                            for i, answer_part in enumerate(answer_parts):
                                current_answer = "".join(answer_parts[:i+1])
                                prediction = type('Prediction', (), {
                                    'answer': current_answer,
                                    'reasoning': final_result.reasoning
                                })
                                state.last_answer = current_answer
                                yield prediction
                                await asyncio.sleep(0.1)  # 短暂停顿
                    except Exception as e:
                        logger.error(f"获取预测结果时出错: {str(e)}")
                        error_prediction = type('Prediction', (), {
                            'answer': '处理您的请求时出现错误',
                            'reasoning': f'发生错误: {str(e)}'
                        })
                        yield error_prediction
            
            return real_stream()
        except Exception as e:
            logger.error(f"流式预测出错: {str(e)}")
            # 如果流式预测失败，尝试使用非流式预测
            try:
                logger.info("尝试使用非流式预测作为备选方案")
                result = self.predict(question)
                # 将非流式结果转换为可迭代对象以模拟流式返回
                async def mock_stream():
                    yield result
                return mock_stream()
            except Exception as e2:
                logger.error(f"备选预测也失败: {str(e2)}")
                raise e  # 重新抛出原始异常
    
    def _split_text(self, text, num_parts):
        """将文本分成大约 num_parts 个部分"""
        if not text:
            return [""]
        
        # 计算每部分的大致长度
        part_length = max(1, len(text) // num_parts)
        parts = []
        
        for i in range(0, len(text), part_length):
            parts.append(text[i:i + part_length])
        
        return parts