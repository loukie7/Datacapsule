import os
import numpy as np
from openai import OpenAI
from loguru import logger
import networkx as nx
import dspy
from typing import List
from nanovector_db import NanoVectorDB


MAX_BATCH_SIZE = os.getenv("MAX_BATCH_SIZE")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K","3"))
BETTER_THAN_THRESHOLD =  float(os.getenv("BETTER_THAN_THRESHOLD","0.7"))
WORKING_DIR =os.getenv("RAG_DIR","graph_data")

client = OpenAI(base_url=os.getenv("EMBEDDING_MODEL_BASE_URL"),api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),)

# 定义节点类型的层级顺序
NODE_HIERARCHY = {
        "界": 1,
        "门": 2,
        "纲": 3,
        "目": 4,
        "科": 5,
        "属": 6,
        "种": 7,
        "中文学名": 7,
        "自然分布地": 8,
        "生活习性": 8,
        "生物特征": 8,
        "经济性": 8,
        "保护信息": 8,
        "食性":8,
        "繁殖特征":8,
        "行为特征":8,
        "体型":8,
        "体色":8,
        "体长":8,
        "特殊特征":8
}

class ReActTools:
    def __init__(self):
        logger.info("ReActTools initialized")
        GRAPHML_DIR = os.getenv("GRAPHML_DIR","graph_chunk_entity_relation_clean.graphml")
        logger.info("init-ReActTools")
        logger.info(f"{WORKING_DIR}/{GRAPHML_DIR}")
        if os.path.exists(f"{WORKING_DIR}/{GRAPHML_DIR}"):
            self.nx = nx.read_graphml(f"{WORKING_DIR}/{GRAPHML_DIR}")
        
        # 判断是否正确加载到网络图
        if self.nx and self.nx.number_of_nodes() >0:
            logger.info(f"NetworkX graph loaded successfully! have nodes: {self.nx.number_of_nodes()}")
            self.nx_nodes=self.nx.nodes(data=True)
            self.entity_type_map = {}
            for node in self.nx_nodes:
                item = node[1]
                id = node[0]
                entity_type = item.get('node_type') 
                if entity_type:  # 只处理包含entity_type的节点
                    if entity_type not in self.entity_type_map:
                        self.entity_type_map[entity_type] = {}
                    self.entity_type_map[entity_type][id] = item
                else:
                    logger.warning(f"Warning: Node {id} missing node_type attribute")
        else:
            logger.error("NetworkX graph is empty!")

        self.dim = int(os.getenv("EMBEDDING_DIM",1536))
        self.vectorizer = GraphVectorizer(WORKING_DIR)
       
    def openai_embedding_function(self,texts: List[str]):
        
        response = client.embeddings.create(
            input=texts,
            model=os.getenv("EMBEDDING_MODEL")
        )
        return [x.embedding for x in response.data]
    
    def find_nodes_by_node_type(self,start_node,attr_name):
        '''
        根据开始节点名查找具有指定属性节点，返回节点信息,节点不存时返回None
        '''
        logger.info(f"开始查找 - 起始节点: '{start_node}', 目标属性: '{attr_name}'")
        checked_nodes = []
        nodes = set()
        self.find_neighbors_recursive(start_node, attr_name, nodes, checked_nodes, depth=0)
        logger.info(f"查找完成 - 找到 {len(nodes)} 个节点: {nodes}")
        return nodes
                

    def find_neighbors_recursive(self,node, target, nodes, checked_nodes, depth=0):
        """
        递归查询某一节点的邻居，并根据目标进行逐层判断，确保递进朝一个方向。
        :param node: 当前节点
        :param target: 目标节点的类型
        :param nodes: 已找到的目标节点列表
        :param checked_nodes: 已检查的节点列表
        :param depth: 当前递归深度（用于日志缩进）
        """
        indent = "  " * depth
        logger.debug(f"{indent}检查节点: '{node}' (递归深度: {depth}, 已检查节点数: {len(checked_nodes)})")
        checked_nodes.append(node)  # 标记当前节点已检查
        
        # 添加异常处理，检查节点是否存在
        try:
            if node not in self.nx.nodes:
                logger.warning(f"{indent}节点 '{node}' 不存在于图中")
                return
            
            source_node_type = self.nx.nodes[node].get("node_type")
            if not source_node_type:
                logger.warning(f"{indent}节点 '{node}' 没有node_type属性")
                return
            
            logger.debug(f"{indent}当前节点类型: '{source_node_type}'")
        except Exception as e:
            logger.error(f"{indent}处理节点 '{node}' 时出错: {str(e)}")
            return
            
        # 获取当前节点和目标节点的层级
        source_level = NODE_HIERARCHY.get(source_node_type, float('inf'))
        target_level = NODE_HIERARCHY.get(target, float('inf'))
        logger.debug(f"{indent}层级比较 - 当前节点: {source_level}, 目标节点: {target_level}")
        
        if source_level == target_level:
            logger.info(f"{indent}找到目标节点! '{node}' (类型: {source_node_type})")
            nodes.add(node)
            return
            
        # 获取邻居节点
        try:
            # 获取所有相邻节点（包括入边和出边）
            neighbors = list(self.nx.neighbors(node))  # 获取出边邻居
            predecessors = list(self.nx.predecessors(node))  # 获取入边邻居
            all_neighbors = list(set(neighbors + predecessors))  # 合并并去重
            logger.debug(f"{indent}找到 {len(all_neighbors)} 个邻居节点（包括入边和出边）")
        except Exception as e:
            logger.error(f"{indent}获取节点 '{node}' 的邻居时出错: {str(e)}")
            return
            
        for neighbor in all_neighbors:
            # 跳过已检查的节点
            if neighbor in checked_nodes:
                logger.debug(f"{indent}跳过已检查的节点: '{neighbor}'")
                continue
                
            try:
                neighbor_type = self.nx.nodes[neighbor].get("node_type")
                if not neighbor_type:
                    logger.debug(f"{indent}邻居节点 '{neighbor}' 没有node_type属性，跳过")
                    continue
                    
                neighbor_level = NODE_HIERARCHY.get(neighbor_type, float('inf'))
                logger.debug(f"{indent}检查邻居: '{neighbor}' (类型: {neighbor_type}, 层级: {neighbor_level})")
                
                # 如果是目标节点，则添加到结果列表
                if neighbor_type == target or (neighbor_level == 7 and neighbor_level == target_level):
                    logger.info(f"{indent}找到目标节点! '{neighbor}' (类型: {neighbor_type})")
                    nodes.add(neighbor)
                    # 如果目标比当前节点层级高，停止递归并返回目标节点
                    if target_level <= source_level:
                        logger.debug(f"{indent}目标层级({target_level})小于等于当前层级({source_level})，停止递归")
                        return
                else:
                    if NODE_HIERARCHY.get(neighbor_type, float('inf')) <= 7:
                        if target_level < source_level and neighbor_level < source_level:
                            logger.debug(f"{indent}向上递归: '{neighbor}' (当前层级: {source_level}, 邻居层级: {neighbor_level}, 目标层级: {target_level})")
                            self.find_neighbors_recursive(neighbor, target, nodes, checked_nodes, depth+1)
                        elif target_level > source_level and neighbor_level > source_level:
                            logger.debug(f"{indent}向下递归: '{neighbor}' (当前层级: {source_level}, 邻居层级: {neighbor_level}, 目标层级: {target_level})")
                            self.find_neighbors_recursive(neighbor, target, nodes, checked_nodes, depth+1)
                        else:
                            logger.debug(f"{indent}不符合递归条件，跳过邻居: '{neighbor}'")
                    else:
                        logger.debug(f"{indent}邻居层级 > 7，跳过: '{neighbor}' (层级: {neighbor_level})")
            except Exception as e:
                logger.warning(f"{indent}处理邻居节点 '{neighbor}' 时出错: {str(e)}")
                continue
        
        logger.debug(f"{indent}完成节点 '{node}' 的所有邻居检查")

    # 查询指定节点所有属性
    def get_node_attribute(self,node_id):
        '''
        根据节点id获取所有属性，包括中文学名、拉丁学名、命名年份、作者、node_type
        :param node_id: 节点id
        :return: 属性值
        '''
        return self.nx.nodes[node_id]
    
    def get_adjacent_node_descriptions(self,nodenames):
        '''
        根据列表中节点名获取所有相邻节点的description
        :param node_id: 节点id
        :return: 所有相依节点信息集合
        '''
        result = set()
        for nodename in nodenames:
            # 获取出边邻居
            for neighbor in self.nx.neighbors(nodename):
                description = self.nx.nodes[neighbor].get("description")
                if description:
                    result.add(description)
            # 获取入边邻居
            for predecessor in self.nx.predecessors(nodename):
                description = self.nx.nodes[predecessor].get("description")
                if description:
                    result.add(description)
        return list(result)

class GraphVectorizer:
    def __init__(self, db_path: str=None, openai_api_key: str = None):
        """初始化向量化器
        
        Args:
            db_path: 向量数据库存储路径
            openai_api_key: OpenAI API密钥，如果不提供则从环境变量获取
        """
        if db_path is None:
            db_path = WORKING_DIR
        self.db = NanoVectorDB(db_path)

    
    def _get_embedding(self, text: str) -> list[float]:
        """获取文本的向量表示"""
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL"),
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    def vectorize_graph(self, graph_file: str):
        """将知识图谱中的实体和关系向量化并存储
        
        Args:
            graph_file: GraphML文件路径
        """
        # 读取图谱
        G = nx.read_graphml(graph_file)
        
        # 向量化并存储实体
        for node, attrs in G.nodes(data=True):
            # 构建实体描述文本
            entity_desc = f"实体ID: {node}"
            if 'node_type' in attrs:
                entity_desc += f", 类型: {attrs['node_type']}"
            if 'name' in attrs:
                entity_desc += f", 名称: {attrs['name']}"
            
            # 获取实体向量
            embedding = self._get_embedding(entity_desc)
            
            # 存储实体向量
            self.db.add_entity(
                entity_id=node,
                entity_type=attrs.get('node_type'),
                entity_name=attrs.get('name'),
                embedding=embedding
            )
        
        # 向量化并存储关系
        for source, target, attrs in G.edges(data=True):
            # 构建关系描述文本
            relation_desc = f"关系: 从 {source} 到 {target}"
            if 'relation' in attrs:
                relation_desc += f", 类型: {attrs['relation']}"
            
            # 获取关系向量
            embedding = self._get_embedding(relation_desc)
            
            # 存储关系向量
            self.db.add_relation(
                source_id=source,
                target_id=target,
                relation_type=attrs.get('relation'),
                embedding=embedding
            )
    
    def search(self, query: str, node_type: str = None, search_type: str = 'all', top_k: int = 5, better_than_threshold: float = BETTER_THAN_THRESHOLD):
        """搜索与查询最相关的实体或关系
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
                - 自然分布地
                - 食性
                - 繁殖特征
                - 生活习性
                - 体型
                - 体色
                - 体长
                - 特殊特征
            k: 返回的结果数量
            search_type: 搜索类型，'all'/'entity'/'relation'
            better_than_threshold: 相似度阈值，只返回相似度高于此值的结果
        
        Returns:
            list: 搜索结果，精准的实体名列表
        """
        # 获取查询向量
        query_embedding = self._get_embedding(query)
        results = []
        
        if search_type in ['all', 'entity']:
            entities = self.db.search_entities(query_embedding, k=100)  # 获取更多结果用于筛选
            # 按node_type筛选
            if node_type:
                entities = [e for e in entities if e['entity_type'] == node_type]
            results.extend(entities)
        
        if search_type in ['all', 'relation']:
            results.extend(self.db.search_relations(query_embedding, k=100))  # 获取更多结果用于筛选
        
        # 按相似度阈值筛选
        results = [r for r in results if r['similarity'] >= better_than_threshold]
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
