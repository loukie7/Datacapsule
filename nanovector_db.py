import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from pathlib import Path
import networkx as nx
from loguru import logger

class NanoVectorDB:
    def __init__(self, db_path: str):
        """初始化向量数据库
        
        Args:
            db_path: 数据库文件存储路径
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.entity_vectors_file = self.db_path / 'entity_vectors.json'
        self.relation_vectors_file = self.db_path / 'relation_vectors.json'
        logger.info(f"初始化向量数据库: {self.db_path}/entity_vectors.json, relation_vectors.json" )
        # 初始化存储文件
        if not self.entity_vectors_file.exists():
            logger.info(f"文件不存在，开始创建向量数据库: {self.db_path}/entity_vectors.json, relation_vectors.json" )
            self._save_vectors(self.entity_vectors_file, [])
        if not self.relation_vectors_file.exists():
            logger.info(f"文件不存在，开始创建向量数据库: {self.db_path}/relation_vectors.json" )
            self._save_vectors(self.relation_vectors_file, [])
        logger.info(f"开始缓存向量数据: {self.db_path}/entity_vectors.json, relation_vectors.json" )
        # 缓存向量数据
        self.entity_vectors_cache = self._load_vectors(self.entity_vectors_file)
        self.relation_vectors_cache = self._load_vectors(self.relation_vectors_file)
        logger.info(f"已缓存实体向量 {len(self.entity_vectors_cache)} 条，关系向量 {len(self.relation_vectors_cache)} 条")

    def _save_vectors(self, file_path: Path, vectors: list):
        """保存向量数据到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vectors, f, ensure_ascii=False)
    
    def _load_vectors(self, file_path: Path) -> list:
        """从文件加载向量数据"""
        logger.info(f"开始加载向量数据库: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载向量数据库: {file_path}")
        return data
    
    def add_entity(self, entity_id: str, entity_type: str, entity_name: str, embedding: list):
        """添加实体向量"""
        self.entity_vectors_cache.append({
            'entity_id': entity_id,
            'entity_type': entity_type,
            'entity_name': entity_name,
            'embedding': embedding
        })
        self._save_vectors(self.entity_vectors_file, self.entity_vectors_cache)
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, embedding: list):
        """添加关系向量"""
        self.relation_vectors_cache.append({
            'source_id': source_id,
            'target_id': target_id,
            'relation_type': relation_type,
            'embedding': embedding
        })
        self._save_vectors(self.relation_vectors_file, self.relation_vectors_cache)
    
    def search_entities(self, query_embedding: list, k: int = 5) -> list:
        """搜索最相似的实体"""
        results = []
        
        for entity in self.entity_vectors_cache:
            similarity = 1 - self._cosine_distance(query_embedding, entity['embedding'])
            results.append({
                'type': 'entity',
                'id': entity['entity_id'],
                'entity_type': entity['entity_type'],
                'name': entity['entity_name'],
                'similarity': similarity
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def search_relations(self, query_embedding: list, k: int = 5) -> list:
        """搜索最相似的关系"""
        results = []
        
        for relation in self.relation_vectors_cache:
            similarity = 1 - self._cosine_distance(query_embedding, relation['embedding'])
            results.append({
                'type': 'relation',
                'source': relation['source_id'],
                'target': relation['target_id'],
                'relation_type': relation['relation_type'],
                'similarity': similarity
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]
    
    def _cosine_distance(self, v1: list, v2: list) -> float:
        """计算余弦距离"""
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        dot_product = np.dot(v1_array, v2_array)
        norm_v1 = np.linalg.norm(v1_array)
        norm_v2 = np.linalg.norm(v2_array)
        return 1 - dot_product / (norm_v1 * norm_v2)