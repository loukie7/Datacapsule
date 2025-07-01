import os,io,sys
import json
import networkx as nx
import dspy

# 定义用于分类的签名
class ClassifyDistribution(dspy.Signature):
    """将生物的自然分布地文本拆分为多个具体的地理位置实体。"""
    text = dspy.InputField()
    locations = dspy.OutputField(description="从文本中提取的地理位置列表")

class ClassifyHabits(dspy.Signature):
    """将生物的生活习性文本拆分为多个具体的习性特征实体。"""
    text = dspy.InputField()
    feeding = dspy.OutputField(description="食性信息")
    reproduction = dspy.OutputField(description="繁殖信息")
    behavior = dspy.OutputField(description="行为特征")

class ClassifyFeatures(dspy.Signature):
    """将生物的生物特征文本拆分为多个具体的特征实体。"""
    text = dspy.InputField()
    body_shape = dspy.OutputField(description="体型特征")
    body_color = dspy.OutputField(description="体色特征")
    body_size = dspy.OutputField(description="体长信息")
    special_features = dspy.OutputField(description="特殊特征")

# 创建分类器
class DistributionExtractor(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict(ClassifyDistribution)
    
    def forward(self, text):
        return self.classifier(text=text)

class HabitsExtractor(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict(ClassifyHabits)
    
    def forward(self, text):
        return self.classifier(text=text)

class FeaturesExtractor(dspy.Module):
    def __init__(self):
        self.classifier = dspy.Predict(ClassifyFeatures)
    
    def forward(self, text):
        return self.classifier(text=text)

# 设置DSPy的语言模型
def setup_dspy():
    ali= dspy.LM(
            f'deepseek/{os.getenv("ALI_LLM_MODEL")}',
            base_url=os.getenv("ALI_OPENAI_BASE_URL"),
            api_key=os.getenv("ALI_OPENAI_API_KEY")
    )
    dspy.settings.configure(lm=ali)
    
# 主函数
def process_entities():
    # 设置DSPy
    setup_dspy()
    
    # 初始化提取器
    distribution_extractor = DistributionExtractor()
    habits_extractor = HabitsExtractor()
    features_extractor = FeaturesExtractor()
    
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
        
        # 安全获取命名信息，处理可能缺失的字段
        naming_year = entity.get('命名年份', '未知')
        # 如果命令年份不是字符串则转换为字符串
        if not isinstance(naming_year, str):
            naming_year = str(naming_year)
        author = entity.get('作者', '未知')
        print(f"  命名信息: 命名年份={naming_year}, 作者={author}")
        
        # 添加实体节点，包含基本属性，使用get方法安全获取可能缺失的字段
        G.add_node(entity_id,
                   中文学名=entity['中文学名'],
                   拉丁学名=entity['拉丁学名'],  # 添加拉丁学名属性
                   命名年份=naming_year,
                   作者=entity.get('作者', ''),
                   node_type='种')  # 将'species'改为'种'
        
        # 添加命名年份节点
        year_node_id = f"年份_{naming_year}"
        G.add_node(year_node_id, name=naming_year, node_type="命名年份")
        G.add_edge(entity_id, year_node_id, relation="命名于")
        # 添加作者节点
        author_node_id = f"作者_{author}"
        G.add_node(author_node_id, name=author, node_type="作者")
        G.add_edge(entity_id, author_node_id, relation="作者为")
        # 添加拉丁学名节点
        latin_name_node_id = f"拉丁学名_{entity['拉丁学名']}"
        G.add_node(latin_name_node_id, name=entity['拉丁学名'], node_type="拉丁学名")
        G.add_edge(entity_id, latin_name_node_id, relation="拉丁学名")
        # 添加分类层级关系
        print(f"  构建分类层级关系...")
        taxonomy_levels = ['界', '门', '纲', '目', '科', '属']  # 移除'种'级别
        for i in range(len(taxonomy_levels)):
            current_level = taxonomy_levels[i]
            current_value = entity[current_level]
            
            # 添加分类节点
            if not G.has_node(current_value):
                G.add_node(current_value, node_type=current_level)
                print(f"    - 添加{current_level}节点: {current_value}")
            
            # 添加分类关系边
            if i > 0:
                previous_level = taxonomy_levels[i-1]
                previous_value = entity[previous_level]
                G.add_edge(previous_value, current_value, relation='包含')
                print(f"    - 添加关系: {previous_value} 包含 {current_value}")
        
        # 直接连接属到物种实体，跳过种级别
        G.add_edge(entity['属'], entity_id, relation='包含')
        print(f"    - 添加关系: {entity['属']} 包含 {entity_id}")
        
        # 处理自然分布地
        print(f"  处理 {entity_id} 的自然分布地信息...")
        try:
            print(f"  原始自然分布地文本: {entity['自然分布地']}")
            distribution_result = distribution_extractor(entity['自然分布地'])
            print(f"  提取到的地理位置: {distribution_result.locations}")
            # 如果是字符串则根据','分割成字符串，还需要'，'的分割
            locations = []
            if isinstance(distribution_result.locations, str):
                # 如果是'，'的还需要分割
                if ',' in distribution_result.locations:
                    locations = distribution_result.locations.split(',')
                # 再循环判断一下是否有'，'的
                for location in locations:
                    if '，' in location:
                        locations.append(location)
                    # 去除空格
                locations = [location.strip() for location in locations] 
            else:
                locations = distribution_result.locations
           
            for idx, location in enumerate(locations):
                # 过滤掉无效的地理位置信息
                if location and location.strip() and location != "无信息" and location != "不明确":
                    location_id = f"{location}"
                    G.add_node(location_id, name=location, node_type='自然分布地')
                    G.add_edge(entity_id, location_id, relation='分布于')
                    print(f"    - 添加地理位置: {location}")
        except Exception as e:
            print(f"  处理自然分布地时出错: {e}")
            # 如果分类失败，添加原始文本
            location_id = f"{entity['自然分布地']}"
            G.add_node(location_id, name=entity['自然分布地'], node_type='自然分布地')
            G.add_edge(entity_id, location_id, relation='分布于')
            print(f"  使用原始文本作为地理位置: {entity['自然分布地']}")

        
        # 处理生活习性
        print(f"  处理 {entity_id} 的生活习性信息...")
        try:
            print(f"  原始生活习性文本: {entity['生活习性']}")
            habits_result = habits_extractor(entity['生活习性'])
            print(f"  食性={habits_result.feeding}, 繁殖={habits_result.reproduction}, 行为={habits_result.behavior}")
            
            # 添加食性信息
            if habits_result.feeding and "无具体" not in habits_result.feeding and "不明确" not in habits_result.feeding:
                feeding_id = f"{habits_result.feeding}"
                G.add_node(feeding_id, name=habits_result.feeding, node_type='生活习性')
                G.add_edge(entity_id, feeding_id, relation='食性为') 
                print(f"    - 添加食性: {habits_result.feeding}")
            
            # 添加繁殖信息
            if habits_result.reproduction and "无具体" not in habits_result.reproduction and "不明确" not in habits_result.reproduction:
                reproduction_id = f"{habits_result.reproduction}"
                G.add_node(reproduction_id, name=habits_result.reproduction, node_type='生活习性')
                G.add_edge(entity_id, reproduction_id, relation='繁殖特征')  # 修改关系方向
                print(f"    - 添加繁殖信息: {habits_result.reproduction}")
            
            # 添加行为特征
            if habits_result.behavior and "无具体" not in habits_result.behavior and "不明确" not in habits_result.behavior:
                behavior_id = f"{habits_result.behavior}"
                G.add_node(behavior_id, name=habits_result.behavior, node_type='生活习性')
                G.add_edge(entity_id, behavior_id, relation='行为特征')  # 修改关系方向
                print(f"    - 添加行为特征: {habits_result.behavior}")
        except Exception as e:
            print(f"  处理生活习性时出错: {e}")
            # 如果分类失败，添加原始文本
            habits_id = f"{entity['生活习性']}"
            G.add_node(habits_id, name=entity['生活习性'], node_type='生活习性')
            G.add_edge(entity_id, habits_id, relation='生活习性')
            print(f"  使用原始文本作为生活习性: {entity['生活习性']}")

        
        # 处理生物特征
        print(f"  处理 {entity_id} 的生物特征信息...")
        try:
            print(f"  原始生物特征文本: {entity['生物特征']}")
            features_result = features_extractor(entity['生物特征'])
            print(f"  提取结果: 体型={features_result.body_shape}, 体色={features_result.body_color}, 体长={features_result.body_size}, 特殊特征={features_result.special_features}")
            
            
            # 添加体型特征
            if features_result.body_shape and "无具体" not in features_result.body_shape and "不明确" not in features_result.body_shape:
                shape_id = f"{features_result.body_shape}"
                G.add_node(shape_id, name=features_result.body_shape, node_type='生物特征')
                G.add_edge(entity_id, shape_id, relation='体型为')  # 修改关系方向
                print(f"    - 添加体型特征: {features_result.body_shape}")
            
            # 添加体色特征
            if features_result.body_color and "无具体" not in features_result.body_color and "不明确" not in features_result.body_color:
                color_id = f"{features_result.body_color}"
                G.add_node(color_id, name=features_result.body_color, node_type='生物特征')
                G.add_edge(entity_id, color_id, relation='体色为')  # 修改关系方向
                print(f"    - 添加体色特征: {features_result.body_color}")
            
            # 添加体长信息
            if features_result.body_size and "无具体" not in features_result.body_size and "不明确" not in features_result.body_size:
                size_id = f"{features_result.body_size}"
                G.add_node(size_id, name=features_result.body_size, node_type='生物特征')
                G.add_edge(entity_id, size_id, relation='体长为')  # 修改关系方向
                print(f"    - 添加体长信息: {features_result.body_size}")
            
            # 添加特殊特征
            if features_result.special_features and "无具体" not in features_result.special_features and "不明确" not in features_result.special_features:
                special_id = f"{features_result.special_features}"
                G.add_node(special_id, name=features_result.special_features, node_type='生物特征')
                G.add_edge(entity_id, special_id, relation='特殊特征')  # 修改关系方向
                print(f"    - 添加特殊特征: {features_result.special_features}")
        except Exception as e:
            print(f"  处理生物特征时出错: {e}")
            # 如果分类失败，添加原始文本
            features_id = f"{entity['生物特征']}"
            G.add_node(features_id, name=entity['生物特征'], node_type='生物特征')
            G.add_edge(entity_id, features_id, relation='生物特征')
            print(f"  使用原始文本作为生物特征: {entity['生物特征']}")

    
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


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    process_entities()

    