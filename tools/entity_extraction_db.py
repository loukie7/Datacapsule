import os
import json
import re
import dspy
import sqlite3
from dotenv import load_dotenv

# 定义用于分类的签名
class ClassifyDistribution(dspy.Signature):
    """将生物的自然分布地文本拆分为多个具体的地理位置实体。"""
    text = dspy.InputField()
    locations = dspy.OutputField(description="从文本中提取的地理位置列表")

class ClassifyHabits(dspy.Signature):
    """从生物的生活习性文本中提取数值特征。"""
    text = dspy.InputField()
    depth = dspy.OutputField(description="栖息水深（米）")
    temperature = dspy.OutputField(description="温度范围（摄氏度）")
    egg_count = dspy.OutputField(description="产卵量（粒）")
    other_numerical = dspy.OutputField(description="其他数值特征，格式为[{名称, 数值, 单位}]")

class ClassifyFeatures(dspy.Signature):
    """从生物的生物特征文本中提取数值特征。"""
    text = dspy.InputField()
    body_length = dspy.OutputField(description="体长或全长（厘米或米）")
    body_weight = dspy.OutputField(description="体重（克或千克）")
    other_numerical = dspy.OutputField(description="其他数值特征，格式为[{名称, 数值, 单位}]")



# 创建提取器
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
    load_dotenv(override=True)
    
    if os.getenv("ALI_LLM_MODEL"):
        ali = dspy.LM(
            f'deepseek/{os.getenv("ALI_LLM_MODEL")}',
            base_url=os.getenv("ALI_OPENAI_BASE_URL"),
            api_key=os.getenv("ALI_OPENAI_API_KEY")
        )
        dspy.settings.configure(lm=ali)
    else:
        # 默认使用OpenAI
        dspy.settings.configure(lm="openai")

# 海洋生物数据处理类
class MarineSpeciesProcessor:
    def __init__(self, db_path):
        """初始化处理器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        setup_dspy()
        self.distribution_extractor = DistributionExtractor()
        self.habits_extractor = HabitsExtractor()
        self.features_extractor = FeaturesExtractor()
        self._setup_database()
    
    def _setup_database(self):
        """创建数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            # 物种基本信息表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS species (
                id INTEGER PRIMARY KEY,
                latin_name TEXT NOT NULL,
                naming_year INTEGER,
                author TEXT,
                chinese_name TEXT,
                kingdom TEXT,
                phylum TEXT,
                class TEXT,
                order_name TEXT,
                family TEXT,
                genus TEXT,
                species_name TEXT,
                body_length TEXT
            )
            ''')
            
            # 地理分布表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS distributions (
                id INTEGER PRIMARY KEY,
                species_id INTEGER,
                location TEXT,
                FOREIGN KEY (species_id) REFERENCES species(id)
            )
            ''')
            
            # 数值特性表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS numerical_traits (
                id INTEGER PRIMARY KEY,
                species_id INTEGER,
                trait_type TEXT,
                trait_name TEXT,
                value REAL,
                unit TEXT,
                FOREIGN KEY (species_id) REFERENCES species(id)
            )
            ''')
            
            # 原始文本描述表
            conn.execute('''
            CREATE TABLE IF NOT EXISTS descriptions (
                id INTEGER PRIMARY KEY,
                species_id INTEGER,
                description_type TEXT,
                content TEXT,
                FOREIGN KEY (species_id) REFERENCES species(id)
            )
            ''')
    
    def _extract_body_length(self, text):
        """从生物特征文本中提取体长信息
        
        Args:
            text: 生物特征文本
            
        Returns:
            提取的体长信息或None
        """
        # 匹配常见的体长表述格式
        patterns = [
            r'体长(?:为)?(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:厘米|cm|CM)',
            r'体长(?:为)?约(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:厘米|cm|CM)',
            r'全长(?:为)?(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:厘米|cm|CM)',
            r'全长(?:为)?约(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:厘米|cm|CM)',
            r'全长可达(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:米|m|M)',
            r'体长约(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:厘米|cm|CM)',
            r'全长约(\d+(?:[.．]\d+)?(?:\s*[-－~～至]\s*\d+(?:[.．]\d+)?)?)\s*(?:厘米|cm|CM)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_locations(self, distribution_text):
        """使用DSPy从分布地文本中提取具体地理位置
        
        Args:
            distribution_text: 分布地文本
            
        Returns:
            地理位置列表
        """
        try:
            result = self.distribution_extractor(distribution_text)
            locations = []
            
            # 处理返回的地理位置，可能是字符串或列表
            if isinstance(result.locations, str):
                # 如果是字符串，按逗号分割
                if ',' in result.locations:
                    locations.extend([loc.strip() for loc in result.locations.split(',')])
                elif '，' in result.locations:
                    locations.extend([loc.strip() for loc in result.locations.split('，')])
                else:
                    locations.append(result.locations.strip())
            else:
                # 如果是列表，直接使用
                locations = result.locations
            
            # 过滤无效位置
            filtered_locations = []
            for loc in locations:
                if loc and loc.strip() and loc != "无信息" and loc != "不明确":
                    filtered_locations.append(loc.strip())
            
            return filtered_locations
        except Exception as e:
            print(f"提取地理位置时出错: {e}")
            return []
    
    def _extract_numerical_traits_from_habits(self, text):
        """从生活习性文本中提取数值特征
        
        Args:
            text: 生活习性文本
            
        Returns:
            数值特征列表，每项包含trait_name, value, unit
        """
        try:
            result = self.habits_extractor(text)
            traits = []
            
            # 处理栖息水深
            if result.depth and result.depth not in ["无", "未知", "不明确"]:
                # 提取数字和单位
                match = re.search(r'(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?)\s*(米|m)', result.depth)
                if match:
                    value_str = match.group(1)
                    unit = match.group(2)
                    
                    # 如果是范围，取平均值
                    if '-' in value_str or '~' in value_str:
                        parts = re.split(r'[-~]', value_str)
                        try:
                            value = (float(parts[0].strip()) + float(parts[1].strip())) / 2
                        except:
                            value = float(parts[0].strip())
                    else:
                        value = float(value_str)
                    
                    traits.append({
                        "name": "栖息水深",
                        "value": value,
                        "unit": unit
                    })
            
            # 处理温度范围
            if result.temperature and result.temperature not in ["无", "未知", "不明确"]:
                match = re.search(r'(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?)\s*(°C|℃)', result.temperature)
                if match:
                    value_str = match.group(1)
                    unit = match.group(2)
                    
                    if '-' in value_str or '~' in value_str:
                        parts = re.split(r'[-~]', value_str)
                        try:
                            value = (float(parts[0].strip()) + float(parts[1].strip())) / 2
                        except:
                            value = float(parts[0].strip())
                    else:
                        value = float(value_str)
                    
                    traits.append({
                        "name": "适宜温度",
                        "value": value,
                        "unit": unit
                    })
            
            # 处理产卵量
            if result.egg_count and result.egg_count not in ["无", "未知", "不明确"]:
                match = re.search(r'(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?万?\s*)(粒|个)', result.egg_count)
                if match:
                    value_str = match.group(1)
                    unit = match.group(2)
                    
                    # 处理"万"单位
                    multiplier = 10000 if "万" in value_str else 1
                    value_str = value_str.replace("万", "").strip()
                    
                    if '-' in value_str or '~' in value_str:
                        parts = re.split(r'[-~]', value_str)
                        try:
                            value = (float(parts[0].strip()) + float(parts[1].strip())) / 2 * multiplier
                        except:
                            value = float(parts[0].strip()) * multiplier
                    else:
                        value = float(value_str) * multiplier
                    
                    traits.append({
                        "name": "产卵量",
                        "value": value,
                        "unit": unit
                    })
            
            # 处理其他数值特征
            if result.other_numerical and isinstance(result.other_numerical, list):
                for trait in result.other_numerical:
                    if isinstance(trait, dict) and 'name' in trait and 'value' in trait and 'unit' in trait:
                        traits.append(trait)
            
            return traits
        except Exception as e:
            print(f"从生活习性提取数值特征时出错: {e}")
            return []
    
    def _extract_numerical_traits_from_features(self, text):
        """从生物特征文本中提取数值特征
        
        Args:
            text: 生物特征文本
            
        Returns:
            数值特征列表，每项包含trait_name, value, unit
        """
        try:
            result = self.features_extractor(text)
            traits = []
            
            # 处理体长
            if result.body_length and result.body_length not in ["无", "未知", "不明确"]:
                match = re.search(r'(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?)\s*(厘米|cm|CM|米|m)', result.body_length)
                if match:
                    value_str = match.group(1)
                    unit = match.group(2)
                    
                    # 单位标准化
                    if unit.lower() in ['cm', 'cm', '厘米']:
                        unit = '厘米'
                    elif unit.lower() in ['m', 'm', '米']:
                        unit = '米'
                    
                    # 如果是范围，取平均值
                    if '-' in value_str or '~' in value_str:
                        parts = re.split(r'[-~]', value_str)
                        try:
                            value = (float(parts[0].strip()) + float(parts[1].strip())) / 2
                        except:
                            value = float(parts[0].strip())
                    else:
                        value = float(value_str)
                    
                    traits.append({
                        "name": "体长",
                        "value": value,
                        "unit": unit
                    })
            
            # 处理体重
            if result.body_weight and result.body_weight not in ["无", "未知", "不明确"]:
                match = re.search(r'(\d+(?:\.\d+)?(?:\s*[-~]\s*\d+(?:\.\d+)?)?)\s*(克|g|千克|kg)', result.body_weight)
                if match:
                    value_str = match.group(1)
                    unit = match.group(2)
                    
                    # 单位标准化
                    if unit.lower() in ['g', 'g', '克']:
                        unit = '克'
                    elif unit.lower() in ['kg', 'kg', '千克']:
                        unit = '千克'
                    
                    if '-' in value_str or '~' in value_str:
                        parts = re.split(r'[-~]', value_str)
                        try:
                            value = (float(parts[0].strip()) + float(parts[1].strip())) / 2
                        except:
                            value = float(parts[0].strip())
                    else:
                        value = float(value_str)
                    
                    traits.append({
                        "name": "体重",
                        "value": value,
                        "unit": unit
                    })
            
            # 处理其他数值特征
            if result.other_numerical and isinstance(result.other_numerical, list):
                for trait in result.other_numerical:
                    if isinstance(trait, dict) and 'name' in trait and 'value' in trait and 'unit' in trait:
                        traits.append(trait)
            
            return traits
        except Exception as e:
            print(f"从生物特征提取数值特征时出错: {e}")
            return []
    
    def process_json_file(self, json_file_path):
        """处理JSON文件，提取数据并存入SQLite数据库
        
        Args:
            json_file_path: JSON文件路径
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            species_data = json.load(f)
        
        print(f"开始处理生物实体数据...")
        print(f"共加载 {len(species_data)} 个生物实体数据")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for entity_index, entity in enumerate(species_data):
                # 使用中文学名作为标识
                entity_id = entity['中文学名']
                print(f"\n[{entity_index+1}/{len(species_data)}] 正在处理生物: {entity_id}（拉丁学名: {entity['拉丁学名']}）")
                
                # 从生物特征中提取体长
                body_length = None
                if '生物特征' in entity:
                    body_length = self._extract_body_length(entity['生物特征'])
                
                # 安全获取命名信息，处理可能缺失的字段
                naming_year = entity.get('命名年份', None)
                # 如果命令年份不是字符串则转换为字符串
                if naming_year and not isinstance(naming_year, int):
                    try:
                        naming_year = int(naming_year)
                    except:
                        naming_year = None
                
                # 插入物种基本信息
                cursor.execute('''
                INSERT INTO species (
                    latin_name, naming_year, author, chinese_name, 
                    kingdom, phylum, class, order_name, 
                    family, genus, species_name, body_length
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity.get('拉丁学名', ''),
                    naming_year,
                    entity.get('作者', ''),
                    entity.get('中文学名', ''),
                    entity.get('界', ''),
                    entity.get('门', ''),
                    entity.get('纲', ''),
                    entity.get('目', ''),
                    entity.get('科', ''),
                    entity.get('属', ''),
                    entity.get('种', ''),
                    body_length
                ))
                
                species_id = cursor.lastrowid
                print(f"  已添加物种基本信息，ID: {species_id}")
                
                # 处理原始描述文本
                for desc_type in ['生活习性', '生物特征']:
                    if desc_type in entity:
                        cursor.execute('''
                        INSERT INTO descriptions (species_id, description_type, content)
                        VALUES (?, ?, ?)
                        ''', (species_id, desc_type, entity[desc_type]))
                
                # 处理地理分布
                if '自然分布地' in entity:
                    print(f"  处理 {entity_id} 的自然分布地信息...")
                    print(f"  原始自然分布地文本: {entity['自然分布地']}")
                    locations = self._extract_locations(entity['自然分布地'])
                    
                    for location in locations:
                        cursor.execute('''
                        INSERT INTO distributions (species_id, location)
                        VALUES (?, ?)
                        ''', (species_id, location))
                        print(f"    - 添加地理位置: {location}")
                
                # 处理生活习性中的数值特征
                if '生活习性' in entity:
                    print(f"  处理 {entity_id} 的生活习性信息...")
                    print(f"  原始生活习性文本: {entity['生活习性']}")
                    traits = self._extract_numerical_traits_from_habits(entity['生活习性'])
                    
                    for trait in traits:
                        cursor.execute('''
                        INSERT INTO numerical_traits (
                            species_id, trait_type, trait_name, value, unit
                        ) VALUES (?, ?, ?, ?, ?)
                        ''', (
                            species_id, 
                            '生活习性',
                            trait['name'], 
                            trait['value'],
                            trait['unit']
                        ))
                        print(f"    - 添加数值特征: {trait['name']} = {trait['value']} {trait['unit']}")
                
                # 处理生物特征中的数值特征
                if '生物特征' in entity:
                    print(f"  处理 {entity_id} 的生物特征信息...")
                    print(f"  原始生物特征文本: {entity['生物特征']}")
                    traits = self._extract_numerical_traits_from_features(entity['生物特征'])
                    
                    for trait in traits:
                        cursor.execute('''
                        INSERT INTO numerical_traits (
                            species_id, trait_type, trait_name, value, unit
                        ) VALUES (?, ?, ?, ?, ?)
                        ''', (
                            species_id, 
                            '生物特征',
                            trait['name'], 
                            trait['value'],
                            trait['unit']
                        ))
                        print(f"    - 添加数值特征: {trait['name']} = {trait['value']} {trait['unit']}")
            
            conn.commit()
            print(f"\n数据已成功导入到数据库: {self.db_path}")
            
            # 输出统计信息
            cursor.execute("SELECT COUNT(*) FROM species")
            species_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM distributions")
            distributions_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM numerical_traits")
            traits_count = cursor.fetchone()[0]
            
            print(f"数据库统计信息:")
            print(f"  - 总物种数: {species_count}")
            print(f"  - 总地理分布记录: {distributions_count}")
            print(f"  - 总数值特征记录: {traits_count}")
            print(f"处理完成!")


# 使用示例
if __name__ == "__main__":
    processor = MarineSpeciesProcessor("./dbs/marine_species.db")
    # 加工单个JSON文件
    # processor.process_json_file("docs/demo_18.json")
    