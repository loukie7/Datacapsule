"""
Mini-React 签名定义模块
将原dspy_inference.py中的签名类迁移到这里
"""
from miniReact import Signature, InputField, OutputField

# 创建海洋生物知识查询签名
MarineBiologyKnowledgeQueryAnswer = Signature(
    input_fields={
        "question": InputField(desc="用户的原始问题")
    },
    output_fields={
        "answer": OutputField(desc="根据检索结果综合形成的完整答案，确保涵盖所有检索需求。重要：必须使用中文回复，绝不能使用英文。答案应该详细、准确，并且完全使用中文表达。")
    },
    instructions="""你是一个海洋生物知识专家，能够回答关于海洋生物的各种问题。你有以下工具可以使用：

1. get_unique_vector_query_results - 通过向量搜索获取相关实体
2. find_nodes_by_node_type - 查找指定类型的节点
3. batch_find_nodes_by_node_type - 批量查找节点
4. get_node_attribute - 获取节点属性
5. get_adjacent_node_descriptions - 获取相邻节点描述
6. nodes_count - 统计节点数量
7. marine_species_query - 自然语言数据库查询

当用户提出问题时，你需要：
1. 分析问题的类型和需求
2. 选择合适的工具进行信息检索
3. 根据检索结果提供准确、详细的中文答案

请务必使用中文回复，不要使用英文。"""
)

class QuestionClassifier(Signature):
    """对用户问题进行分类"""
    def __init__(self):
        super().__init__(
            input_fields={
                "question": InputField(desc="用户的原始问题")
            },
            output_fields={
                "question_type": OutputField(desc="问题类型，可能的值包括：实体查询/关系查询/属性查询/统计查询等"),
                "search_strategy": OutputField(desc="建议的检索策略：向量检索/图检索/混合检索"),
                "key_entities": OutputField(desc="问题中的关键实体列表")
            }
        ) 