"""
Mini-React 工具函数模块
将原dspy_inference.py中的工具方法迁移到这里
"""
import os
from agents import ReActTools, GraphVectorizer
from dotenv import load_dotenv
from query_db import MarineSpeciesQuery
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

# 确保环境变量已加载
load_dotenv(override=True)

class InferenceTools:
    """推理工具函数集合"""
    
    def __init__(self):
        self.ragtool = ReActTools()
        self.graphvectorizer = GraphVectorizer()
        self.query_processor = MarineSpeciesQuery(os.getenv("SPECIES_DB_URL", "./.dbs/marine_species.db"))
    
    def find_nodes_by_node_type(self, start_node, target_node_type):
        """
        此方法会根据传入的节点名称，在图数据中以该节点为起点查找包含指定节点类型的节点列表，并返回节点数量与节点列表。
        start_node 为开始查找的树节点名称，只允许单个节点、
        target_node_type 目标节点类型,只允许单个类型名称
        返回值为从该节点开始，包含指定属性名的节点数量与节点列表
        已知图数据中存在一系列的海洋生物相关信息：
        1. ⽣物分类学图数据：包括"拉丁学名", "命名年份", "作者", "中文学名",
        2. ⽣物科属于数据："界", "门", "纲", "目", "科", "属", "种"(种即是中文学名),它们的从属关系是: 界 -> 门 -> 纲 -> 目 -> 科 ->属 ->种 。
        3. ⽣物特征图数据：包括"自然分布地", "生物特征","生活习性"等。
        本方法可以根据给定的节点名称，在图数据中以此节点为起点查找包含指定该属性的节点或节点列表，例如1："盲鳗科" "种" 则会返回 盲鳗科所有的种，例如2："盲鳗科" "界" 则会返回 盲鳗科对应的界， 。
        4. 因为本方法需要的参数是精准的节点属性名称(或节点类型名)，建议查询的节点类型属于"自然分布地", "生物特征", "生活习性"等时,或查询返回为空时、查询失败时，先通过get_unique_vector_query_results方法获取准确的节点名称，再通过本方法获取对应的节点信息。

        Args:
            start_node: 开始查找的节点名称
            target_node_type: 目标节点类型
        Returns:
            count: 节点数量
            nodes: 节点列表
        """
        try:
            logger.info(f"工具调用: find_nodes_by_node_type(start_node='{start_node}', target_node_type='{target_node_type}')")
            nodes = self.ragtool.find_nodes_by_node_type(start_node, target_node_type)
            # 如果nodes为空，则返回0,不为为空时，则返回节点数量与节点列表
            if not nodes:
                logger.info(f"工具返回: 未找到节点")
                return {"count": 0, "nodes": []}
            count = len(nodes)
            logger.info(f"工具返回: 找到{count}个节点")
            return {"count": count, "nodes": list(nodes)}
        except Exception as e:
            logger.error(f"工具调用失败: find_nodes_by_node_type - {str(e)}")
            return {"count": 0, "nodes": [], "error": str(e)}

    def batch_find_nodes_by_node_type(self, start_nodes, target_node_type):
        """
        此方法会根据传入包含多个开始节点的列表，批量查询指定目标节点类型的节点列表，返回多条查询的结果集。
        Args:
            start_nodes: 开始查找的节点名称列表
            target_node_type: 目标节点类型
        Returns:
            target_nodes_list: 多条查询结果的列表
        """
        # 字典格式为，key为节点名称，value为包含指定属性名的节点数量与节点列表
        target_nodes_list = {}
        for node in start_nodes:
            result = self.find_nodes_by_node_type(node, target_node_type)
            target_nodes_list[node] = result
        return target_nodes_list

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
            logger.info(f"工具调用: get_unique_vector_query_results(query='{query}', node_type='{node_type}')")
            # 使用线程池执行可能耗时的操作
            with ThreadPoolExecutor() as executor:
                # 设置超时时间（例如10秒）
                future = executor.submit(self.graphvectorizer.search, query, node_type, search_type, top_k, better_than_threshold)
                try:
                    result = future.result(timeout=10)  # 10秒超时
                    logger.info(f"工具返回: {len(result) if result else 0}个向量搜索结果")
                    return result
                except TimeoutError:
                    logger.error(f"向量搜索超时: query={query}, node_type={node_type}")
                    return []  # 超时返回空列表
        except Exception as e:
            # 捕获所有异常，确保不会导致整个流程崩溃
            logger.error(f"向量搜索出错: {str(e)}, query={query}, node_type={node_type}")
            return []  # 出错返回空列表

    def get_node_attribute(self, node_id):
        """
        根据节点id获取所有属性，包括中文学名、拉丁学名、命名年份、作者、node_type
        Args:
            node_id: 节点id
        Returns:
            list: 属性列表
        """
        return self.ragtool.get_node_attribute(node_id)
    
    def get_adjacent_node_descriptions(self, nodenames):
        """
        此方法会根据传入的节点列表，获取每个节点相邻所有节点描述，合并到一个列表中返回，非精准检索，谨慎使用
        Args:
            nodenames: 节点名称列表
        Returns:
            list: 相邻节点描述列表
        """
        return self.ragtool.get_adjacent_node_descriptions(nodenames)

    def nodes_count(self, nodes):
        """
        此方法会根据传入的节点列表，统计数量，返回数量
        Args:
            nodes: 节点列表
        Returns:
            int: 节点数量
        """
        if not nodes:
            return 0
        return len(nodes)
    
    def marine_species_query(self, query):
        """根据自然语言查询数据库，用于数据类问题的查询
        Args:
            query: 用户的自然语言查询
            
        Returns:
            查询结果和解释
        """
        try:
            logger.info(f"工具调用: marine_species_query(query='{query}')")
            result = self.query_processor.query_database(query)
            formatted_result = self.query_processor.format_query_results(result)
            logger.info(f"工具返回: 数据库查询成功，返回结果长度={len(str(formatted_result))}")
            return formatted_result
        except Exception as e:
            logger.error(f"工具调用失败: marine_species_query - {str(e)}")
            return f"数据库查询失败: {str(e)}。建议尝试使用其他工具进行查询。" 