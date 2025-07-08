"""
Mini-React 实体提取模块
将原entity_extraction.py中的dspy功能迁移到mini-react
"""
import os
import io
import sys
import json
import networkx as nx
from dotenv import load_dotenv
from loguru import logger

from miniReact import Signature, InputField, OutputField, Predict, Module, LM, set_model
from core.llm_manager import get_entity_manager

# 设置环境变量
load_dotenv(override=True)

# 定义用于分类的签名
class ClassifyDistributionSignature(Signature):
    """将生物的自然分布地文本拆分为多个具体的地理位置实体"""
    def __init__(self):
        super().__init__(
            input_fields={
                "text": InputField()
            },
            output_fields={
                "locations": OutputField(desc="从文本中提取的地理位置列表")
            }
        )

class ClassifyHabitsSignature(Signature):
    """将生物的生活习性文本拆分为多个具体的习性特征实体"""
    def __init__(self):
        super().__init__(
            input_fields={
                "text": InputField()
            },
            output_fields={
                        "feeding": OutputField(desc="食性信息"),
        "reproduction": OutputField(desc="繁殖信息"),
        "behavior": OutputField(desc="行为特征")
            }
        )

class ClassifyFeaturesSignature(Signature):
    """将生物的生物特征文本拆分为多个具体的特征实体"""
    def __init__(self):
        super().__init__(
            input_fields={
                "text": InputField()
            },
            output_fields={
                        "body_shape": OutputField(desc="体型特征"),
        "body_color": OutputField(desc="体色特征"),
        "body_size": OutputField(desc="体长信息"),
        "special_features": OutputField(desc="特殊特征")
            }
        )

# 创建提取器模块
class DistributionExtractor(Module):
    def __init__(self, lm=None):
        super().__init__()
        self.predictor = Predict(ClassifyDistributionSignature())
        if lm:
            self.predictor.lm = lm
    
    def forward(self, text):
        return self.predictor(text=text)

class HabitsExtractor(Module):
    def __init__(self, lm=None):
        super().__init__()
        self.predictor = Predict(ClassifyHabitsSignature())
        if lm:
            self.predictor.lm = lm
    
    def forward(self, text):
        return self.predictor(text=text)

class FeaturesExtractor(Module):
    def __init__(self, lm=None):
        super().__init__()
        self.predictor = Predict(ClassifyFeaturesSignature())
        if lm:
            self.predictor.lm = lm
    
    def forward(self, text):
        return self.predictor(text=text)

# 设置Mini-React的语言模型
def setup_react():
    """设置语言模型"""
    try:
        llm_type = os.getenv("ALI_LLM_TYPE", os.getenv("LLM_TYPE", "openai"))
        model_name = os.getenv("ALI_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
        api_key = os.getenv("ALI_OPENAI_API_KEY")
        base_url = os.getenv("ALI_OPENAI_BASE_URL")
        
        if api_key and base_url:
            lm = LM(
                model_name,  # 直接使用模型名称，不添加前缀
                api_base=base_url,
                api_key=api_key
            )
            set_model(model_name)  # 也是直接使用模型名称
            return lm
        else:
            logger.warning("语言模型配置不完整，使用默认配置")
            set_model("gpt-3.5-turbo")
            return None
    except Exception as e:
        logger.error(f"设置语言模型失败: {str(e)}")
        return None

# 主函数
def process_entities():
    """处理实体提取"""
    # 设置Mini-React
    lm = setup_react()
    
    # 初始化提取器
    distribution_extractor = DistributionExtractor(lm)
    habits_extractor = HabitsExtractor(lm)
    features_extractor = FeaturesExtractor(lm)
    
    # 读取JSON数据
    with open('/Users/idw/rags/modellens_dspyv3.0/docs/demo130.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 遍历每个生物实体
    print("开始处理生物实体数据...")
    print(f"共加载 {len(data)} 个生物实体数据")
    
    for entity_index, entity in enumerate(data):
        # 使用中文学名作为唯一标识符
        entity_id = entity['中文学名']
        print(f"\n[{entity_index+1}/{len(data)}] 正在处理生物: {entity_id}（拉丁学名: {entity['拉丁学名']}）")
        print(f"  分类信息: 界={entity['界']}, 门={entity['门']}, 纲={entity['纲']}, 目={entity['目']}, 科={entity['科']}, 属={entity['属']}, 种={entity['种']}")
        
        # 安全获取命名信息
        naming_year = entity.get('命名年份', '未知')
        if not isinstance(naming_year, str):
            naming_year = str(naming_year)
        author = entity.get('作者', '未知')
        print(f"  命名信息: 命名年份={naming_year}, 作者={author}")
        
        # 添加实体节点
        G.add_node(entity_id,
                   中文学名=entity['中文学名'],
                   拉丁学名=entity['拉丁学名'],
                   命名年份=naming_year,
                   作者=entity.get('作者', ''),
                   node_type='种')
        
        # 添加命名年份、作者、拉丁学名节点
        year_node_id = f"年份_{naming_year}"
        G.add_node(year_node_id, name=naming_year, node_type="命名年份")
        G.add_edge(entity_id, year_node_id, relation="命名于")
        
        author_node_id = f"作者_{author}"
        G.add_node(author_node_id, name=author, node_type="作者")
        G.add_edge(entity_id, author_node_id, relation="作者为")
        
        latin_name_node_id = f"拉丁学名_{entity['拉丁学名']}"
        G.add_node(latin_name_node_id, name=entity['拉丁学名'], node_type="拉丁学名")
        G.add_edge(entity_id, latin_name_node_id, relation="拉丁学名")
        
        # 添加分类层级关系
        print(f"  构建分类层级关系...")
        taxonomy_levels = ['界', '门', '纲', '目', '科', '属']
        for i in range(len(taxonomy_levels)):
            current_level = taxonomy_levels[i]
            current_value = entity[current_level]
            
            if not G.has_node(current_value):
                G.add_node(current_value, node_type=current_level)
                print(f"    - 添加{current_level}节点: {current_value}")
            
            if i > 0:
                previous_level = taxonomy_levels[i-1]
                previous_value = entity[previous_level]
                G.add_edge(previous_value, current_value, relation='包含')
                print(f"    - 添加关系: {previous_value} 包含 {current_value}")
        
        # 连接属到物种实体
        G.add_edge(entity['属'], entity_id, relation='包含')
        print(f"    - 添加关系: {entity['属']} 包含 {entity_id}")
        
        # 处理自然分布地
        print(f"  处理 {entity_id} 的自然分布地信息...")
        try:
            print(f"  原始自然分布地文本: {entity['自然分布地']}")
            distribution_result = distribution_extractor(entity['自然分布地'])
            print(f"  提取到的地理位置: {distribution_result.locations}")
            
            # 处理位置数据
            locations = []
            if isinstance(distribution_result.locations, str):
                if ',' in distribution_result.locations:
                    locations = distribution_result.locations.split(',')
                    for location in locations:
                        if '，' in location:
                            locations.extend(location.split('，'))
                else:
                    locations = [distribution_result.locations]
                locations = [location.strip() for location in locations if location.strip()]
            else:
                locations = distribution_result.locations
           
            for location in locations:
                if location and location.strip() and location != "无信息" and location != "不明确":
                    location_id = f"{location}"
                    G.add_node(location_id, name=location, node_type='自然分布地')
                    G.add_edge(entity_id, location_id, relation='分布于')
                    print(f"    - 添加地理位置: {location}")
        except Exception as e:
            print(f"  处理自然分布地时出错: {e}")
            location_id = f"{entity['自然分布地']}"
            G.add_node(location_id, name=entity['自然分布地'], node_type='自然分布地')
            G.add_edge(entity_id, location_id, relation='分布于')

        # 处理生活习性
        print(f"  处理 {entity_id} 的生活习性信息...")
        try:
            print(f"  原始生活习性文本: {entity['生活习性']}")
            habits_result = habits_extractor(entity['生活习性'])
            print(f"  食性={habits_result.feeding}, 繁殖={habits_result.reproduction}, 行为={habits_result.behavior}")
            
            # 添加各种习性信息
            for habit_type, habit_value in [
                ('食性为', habits_result.feeding),
                ('繁殖特征', habits_result.reproduction),
                ('行为特征', habits_result.behavior)
            ]:
                if habit_value and "无具体" not in habit_value and "不明确" not in habit_value:
                    habit_id = f"{habit_value}"
                    G.add_node(habit_id, name=habit_value, node_type='生活习性')
                    G.add_edge(entity_id, habit_id, relation=habit_type)
                    print(f"    - 添加{habit_type}: {habit_value}")
        except Exception as e:
            print(f"  处理生活习性时出错: {e}")
            habits_id = f"{entity['生活习性']}"
            G.add_node(habits_id, name=entity['生活习性'], node_type='生活习性')
            G.add_edge(entity_id, habits_id, relation='生活习性')

        # 处理生物特征
        print(f"  处理 {entity_id} 的生物特征信息...")
        try:
            print(f"  原始生物特征文本: {entity['生物特征']}")
            features_result = features_extractor(entity['生物特征'])
            print(f"  提取结果: 体型={features_result.body_shape}, 体色={features_result.body_color}, 体长={features_result.body_size}, 特殊特征={features_result.special_features}")
            
            # 添加各种特征信息
            for feature_type, feature_value in [
                ('体型为', features_result.body_shape),
                ('体色为', features_result.body_color),
                ('体长为', features_result.body_size),
                ('特殊特征', features_result.special_features)
            ]:
                if feature_value and "无具体" not in feature_value and "不明确" not in feature_value:
                    feature_id = f"{feature_value}"
                    G.add_node(feature_id, name=feature_value, node_type='生物特征')
                    G.add_edge(entity_id, feature_id, relation=feature_type)
                    print(f"    - 添加{feature_type}: {feature_value}")
        except Exception as e:
            print(f"  处理生物特征时出错: {e}")
            features_id = f"{entity['生物特征']}"
            G.add_node(features_id, name=entity['生物特征'], node_type='生物特征')
            G.add_edge(entity_id, features_id, relation='生物特征')

    # 保存为GraphML格式
    output_file = '/Users/idw/rags/modellens_dspyv3.0/graph_data_new/graph_entity_relation_detailed.graphml'
    print(f"\n保存知识图谱到文件: {output_file}")
    nx.write_graphml(G, output_file, encoding='utf-8')
    print(f"已成功生成详细的实体关系图: {output_file}")
    print(f"图谱统计信息:")
    print(f"  - 总节点数: {G.number_of_nodes()}")
    print(f"  - 总边数: {G.number_of_edges()}")
    
    # 统计各类型节点数量
    node_types = {}
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"  - 节点类型统计:")
    for node_type, count in node_types.items():
        print(f"    * {node_type}: {count}个节点")
    
    print(f"处理完成!")

class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        """初始化实体提取器"""
        self.entity_manager = get_entity_manager()
        
        # 初始化提取器
        self.distribution_extractor = DistributionExtractor(self.entity_manager)
        self.habits_extractor = HabitsExtractor(self.entity_manager)
        self.features_extractor = FeaturesExtractor(self.entity_manager)
    
    def extract_distribution(self, text):
        """提取分布地信息"""
        return self.distribution_extractor(text)
    
    def extract_habits(self, text):
        """提取生活习性"""
        return self.habits_extractor(text)
    
    def extract_features(self, text):
        """提取生物特征"""
        return self.features_extractor(text)

if __name__ == "__main__":
    process_entities() 