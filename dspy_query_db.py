import os
import json
import re
import dspy
import sqlite3
from dotenv import load_dotenv
# 设置DSPy的语言模型
def setup_dspy():
    load_dotenv(override=True)
    
    if os.getenv("Train_LLM_MODEL"):
        Train = dspy.LM(
            f'deepseek/{os.getenv("Train_LLM_MODEL")}',
            base_url=os.getenv("Train_OPENAI_BASE_URL"),
            api_key=os.getenv("Train_OPENAI_API_KEY")
        )
        dspy.settings.configure(lm=Train)
    else:
        # 默认使用OpenAI
        dspy.settings.configure(lm="openai")


# 在已有的签名定义之后添加
class NaturalLanguageToSQL(dspy.Signature):
    """将自然语言查询转换为SQL语句。注意：返回纯SQL文本，不要包含```sql或```等代码块标记。
    重要：保持原始查询中的中文词汇不变，不要自动转换为拉丁文或英文。
    当查询涉及到地理位置（distributions表中的location字段）时，必须使用LIKE语句而不是精确匹配，
    例如：WHERE location LIKE '%东海%' 而不是 WHERE location = '东海'"""
    query = dspy.InputField(description="用户的自然语言查询")
    db_schema = dspy.InputField(description="数据库的表结构信息")
    sql = dspy.OutputField(description="生成的SQL查询语句，必须是纯SQL文本，对地理位置使用LIKE操作符")
    explanation = dspy.OutputField(description="SQL查询的解释")

# 在已有的提取器类之后添加
class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(NaturalLanguageToSQL)
    
    def forward(self, query, db_schema):
        return self.generator(query=query, db_schema=db_schema)
    
# 查询相关类
class MarineSpeciesQuery:
    def __init__(self, db_path):
        """初始化查询器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        setup_dspy()
    
    def query_database(self, natural_language_query):
        """根据自然语言查询数据库
        
        Args:
            natural_language_query: 用户的自然语言查询
            
        Returns:
            查询结果和解释
        """
        # 先获取表中实际的值
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT family FROM species")
            families = [row[0] for row in cursor.fetchall()]
        
        # 获取数据库表结构
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            db_schema = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                column_info = []
                for col in columns:
                    column_info.append({
                        "name": col[1],
                        "type": col[2]
                    })
                
                db_schema.append({
                    "table": table_name,
                    "columns": column_info
                })
            
            db_schema_str = json.dumps(db_schema, ensure_ascii=False, indent=2)
            
            # 当拼接db_schema_enriched时，添加关于location的使用说明
            db_schema_enriched = json.dumps(db_schema, ensure_ascii=False, indent=2)
            
            # 添加额外使用提示
            location_usage_hint = """
            重要提示：当查询涉及地理位置时，请使用LIKE操作符而不是等号(=)。
            例如：
            正确: WHERE d.location LIKE '%东海%'
            错误: WHERE d.location = '东海'
            
            这是因为地理位置通常需要模糊匹配，一个物种可能分布在多个地区，
            或者地理位置描述可能包含其他词汇。
            """
            
            # 初始化SQL生成器
            sql_generator = SQLGenerator()
            
            # 生成SQL
            result = sql_generator(natural_language_query, db_schema_enriched + "\n" + location_usage_hint)
            
            # 清理SQL，移除Markdown代码块标记
            sql = result.sql
            sql = re.sub(r'```sql\s*', '', sql)  # 移除开始的```sql
            sql = re.sub(r'\s*```\s*$', '', sql)  # 移除结束的```
            
            try:
                # 执行SQL查询
                print(f"执行SQL查询: {sql}")
                cursor.execute(sql)
                
                # 获取列名
                column_names = [description[0] for description in cursor.description]
                
                # 获取查询结果
                rows = cursor.fetchall()
                
                # 转换为字典列表
                results = []
                for row in rows:
                    result_dict = {}
                    for i, col_name in enumerate(column_names):
                        result_dict[col_name] = row[i]
                    results.append(result_dict)
                
                return {
                    "success": True,
                    "query": natural_language_query,
                    "sql": sql,  # 使用清理后的SQL
                    "explanation": result.explanation,
                    "results": results,
                    "column_names": column_names,
                    "row_count": len(rows)
                }
            except Exception as e:
                print(f"SQL执行错误: {e}")
                return {
                    "success": False,
                    "query": natural_language_query,
                    "sql": sql,  # 使用清理后的SQL
                    "explanation": result.explanation,
                    "error": str(e)
                }

    def format_query_results(self, query_result):
        """格式化查询结果
        
        Args:
            query_result: 查询结果字典
            
        Returns:
            格式化的结果字符串
        """
        if not query_result["success"]:
            return f"查询失败: {query_result['error']}\n原始SQL: {query_result['sql']}"
        
        output = []
        output.append(f"查询: {query_result['query']}")
        output.append(f"SQL: {query_result['sql']}")
        output.append(f"解释: {query_result['explanation']}")
        output.append(f"找到 {query_result['row_count']} 条结果:")
        
        if query_result['row_count'] > 0:
            # 计算每列的最大宽度
            widths = {}
            for col in query_result['column_names']:
                widths[col] = len(col)
            
            for row in query_result['results']:
                for col in query_result['column_names']:
                    val = str(row[col]) if row[col] is not None else 'NULL'
                    widths[col] = max(widths[col], len(val))
            
            # 生成表头
            header = " | ".join(col.ljust(widths[col]) for col in query_result['column_names'])
            separator = "-+-".join("-" * widths[col] for col in query_result['column_names'])
            
            output.append(header)
            output.append(separator)
            
            # 生成数据行
            for row in query_result['results']:
                row_str = " | ".join(
                    str(row[col]).ljust(widths[col]) if row[col] is not None else 'NULL'.ljust(widths[col])
                    for col in query_result['column_names']
                )
                output.append(row_str)
        
        return "\n".join(output)
    

if __name__ == "__main__":
    # 直接使用查询处理器示例
    query_processor = MarineSpeciesQuery("marine_species.db")
    result = query_processor.query_database("分布在东海的盲鳗科哪些生物?有多少？")
    formatted_result = query_processor.format_query_results(result)
    print(formatted_result)
