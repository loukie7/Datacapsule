<h1 align="center">🌟 Datacapsule 🌟</h1>

Datacapsule是一个基于知识图谱的多路召回解决方案，旨在通过多路召回技术，实现精准的知识检索。该解决方案涵盖了检索系统、实体关系抽取、实体属性抽取、实体链接、结构化数据库构建以及问答系统等多个功能模块，为信息检索和应用提供了强大的支持。

<br>

<div align="center">
  <img src="https://img.shields.io/badge/解决方案-red" />
  <img src="https://img.shields.io/badge/免费-blue" />
  <img src="https://img.shields.io/badge/Python-yellow" />
  <img src="https://img.shields.io/badge/JavaScript-orange" />
  <img src="https://img.shields.io/badge/TypeScript-blue" />
  <br />
  <img src="https://img.shields.io/badge/English-lightgrey" />
  <img src="https://img.shields.io/badge/简体中文-lightgrey" />
</div>

# 背景
知识图谱的多路召回技术是一种在信息检索领域中广泛使用的技术，它通过构建一个包含丰富实体和关系的图形数据库，使得用户可以通过多种方式（如关键字、实体链接等）来查询相关信息。这种方法不仅可以提高信息检索的效率，还可以帮助用户更好地理解和利用数据中的复杂关系。

但是传统知识图谱构建的速度和效率有限，因此需要一种更高效的构建图谱方法，同时基于图谱的检索的效率也存在问题，因此，我们提出了一种基于知识图谱的多路召回技术解决方案，旨在提高构建效率并优化检索效果。
对用户的问题进行深入理解，首先判断用户问题的实体中是否在图谱中，如果不在直接通过向量检索得到答案。

如果在实体中，在判断用户问题的种类，如：实体查询：如"什么是台湾盲鳗？"；关系查询：如"台湾盲鳗和蒲氏黏盲鳗有什么关系？"；属性查询：如"蒲氏黏盲鳗的生活习性是什么？"；统计查询：如"盲鳗科有多少种？"。实体查询、关系查询、属性查询通过图结构检索召回；统计查询通过结构化检索召回


# 主要功能介绍

1. **功能设计图**：
     ![功能设计图](./images/function-diagram.png)

2. **项目文件结构概要**：

- backend/（后端服务目录）
  - dspy_program/（DSPy模型及程序目录）
    - retrieval_demo_18.json（小型示例数据集）
    - retrieval_demo_130.json（完整规模数据集）
    - optimized_program.pkl（优化后的DSPy程序）
    - signature.py（DSPy签名定义文件）
    - examples/（训练示例数据）
  - graph_data_new/（知识图谱数据目录）
    - knowledge_graph-1.html（知识图谱可视化文件）
    - knowledge_graph-1.graphml（知识图谱数据文件）
    - vectors/（向量数据存储目录）
      - bio_vectors.json（生物实体向量数据）
      - relation_vectors.json（关系向量数据）
  - tools/（工具类模块目录）
    - entity_extraction.py（实体抽取工具）
    - entity_extraction_db.py（结构化数据库构建工具）
  - .dspy_cache/（DSPy缓存目录）
  - app.py（主应用入口）
  - dspy_evaluation.py（评估模块）
  - dspy_inference.py（推理模块）
  - dspy_query_db.py（数据库查询模块）
  - nanovector_db.py（向量数据库实现）
  - react_tools.py（图谱查询与向量检索工具）
  - requirements.txt（依赖包列表）
  - .env（环境配置文件）

- frontend/（前端服务目录）
  - src/（源代码目录）
    - components/（组件目录）
      - Chat/（聊天相关组件）
      - Graph/（知识图谱展示组件）
      - UI/（界面元素组件）
    - hooks/（React钩子函数）
    - services/（服务调用模块）
    - App.tsx（应用主组件）
    - main.tsx（入口文件）
  - public/（静态资源目录）
    - images/（图片资源）
  - package.json（项目配置和依赖）
  - vite.config.ts（Vite配置文件）
  - tailwind.config.js（TailwindCSS配置）
  - .env.example（环境变量示例）

3. **知识图谱与结构化数据库构建**：基于dspy作为意图识别方法去处理实体抽取，构建图谱信息，对应`entity_extraction.py`模块，将构建的图谱信息抽取为结构化信息存储进数据库中，对应`entity_extraction_db.py`模块。

4. **知识图谱存储与管理**：基于 NetworkX 实现的知识图谱存储和管理功能，支持实体关系的动态构建和查询，对应 `react_tools.py` 中的 `ReActTools` 模块。

5. **向量数据库检索**：基于 NanoVector 实现的轻量级向量数据库，支持高效的语义相似度检索，对应 `nanovector_db.py` 中的 `NanoVectorDB` 模块。

6. **基于图谱的多路召回方法**：

   - 基于 Chain of Thought 的推理系统
   - 支持多轮对话的上下文理解
   - 形成了一个完整的推理和查询系统
      `dspy_inference.py`  整合各种检索方式；提供统一的查询接口
      `dspy_query_db.py`  处理结构化数据查询
      `react_tools.py`  整合向量检索和图检索，`ReActTools`类负责图结构检索，`GraphVectorizer`类负责向量检索，调用 `NanoVectordb.py` 的功能
      `nanovector_db.py`  封装了NanoVector库，提供了向量数据库的查询、存储和向量相似度计算功能
      `dspy_evaluation.py`  确保推理质量和模型优化

   系统协同工作流程：
   1. 用户发起查询 → `dspy_inference.py`
   - 接收用户问题
   - 负责整体推理流程控制
   - 判断问题中的实体是否在知识图谱中：
     * 不在图谱中：直接使用向量检索获取答案
     * 在图谱中：进一步判断问题类型
   - 问题类型判断和对应的检索策略：
     * 实体查询（使用图结构检索） 
       例如："什么是台湾盲鳗？"
     * 关系查询（使用图结构检索）
       例如："台湾盲鳗和蒲氏黏盲鳗有什么关系？"
     * 属性查询（使用图结构检索）
       例如："蒲氏黏盲鳗的生活习性是什么？"
     * 统计查询（使用结构化检索）
       例如："虎鲨目的生物有多少种？"

   2. 多路检索阶段：
      a) 向量检索路径：
         `dspy_inference.py → react_tools.py (GraphVectorizer类) → nanovector_db.py`
         - 将问题转换为向量
         - 计算向量相似度
         - 返回相关实体
            b) 图结构检索路径：
            `dspy_inference.py → react_tools.py (ReActTools类)`
         - 基于实体进行图遍历
         - 查找相关节点和关系
         - 返回结构化知识
            c) 结构化检索路径：
            `dspy_inference.py → dspy_query_db.py`
         - 将自然语言转换为SQL
         - 查询结构化数据库
         - 返回精确匹配结果
   3. 结果整合与推理：
      - `dspy_inference.py` 整合多路检索结果
      - 使用 DSPy 进行推理和答案生成
      - 生成结构化的回答
   4. 评估与优化：
      `dspy_evaluation.py`
      - 评估答案质量
      - 收集用户反馈
      - 用于模型优化
      - 更新优化器数据
   5. 返回结果给用户：
      - 流式返回答案
      - 保存交互记录
      - 更新系统状态

   对应 `dspy_inference.py` 、 `dspy_evaluation.py` 和 `dspy_query_db.py` 模块。

7. **实时通信与状态同步**：
   - WebSocket 实现的实时消息推送
   - 支持流式输出的对话响应
   - 优化器状态的实时反馈
   对应 `broadcast.py` 和 `app.py` 中的 WebSocket 实现。

8. **模型优化器**：
   - 支持基于用户反馈的模型优化
   - 版本管理和回滚功能
   - 优化器过程可视化
   对应 `dspy_evaluation.py` 中的评估优化模块。

9. **数据库管理系统**：
   - SQLite 存储用户交互记录
   - 支持向量数据的批量处理
   - 数据版本控制
   对应 `dspy_query_db.py` 中的数据库管理功能。

10. **前端交互界面**：

   - 基于 React 18 + Vite 的现代化界面
   - 实时对话窗口
   - 用户问答对收集
   - 推理过程展示
   - 优化进度展示
      对应前端 `frontend` 目录的实现。

11. **系统监控与日志**：
   - 基于 loguru 的分级日志系统
   - 性能监控和错误追踪
   - API 调用统计
      对应各模块中的日志记录实现。

12. **环境配置管理**：
    - 支持多种 LLM 模型配置
    - 灵活的环境变量管理
    - 多环境部署支持
    对应 `.env` 和 `.env.example` 的配置管理。


# 技术框架
## **前端技术栈**
- 开发语言：JavaScript+TypeScript
- 前端框架：React 18 + Vite
- UI 框架：TailwindCSS
- 开发工具：
  * 构建工具：Vite
- 实时通信：WebSocket 客户端

## **后端技术栈**
- 开发语言：Python (推荐版本：3.8+)
- Web 框架：FastAPI
- 数据库：
  * 结构化数据：SQLite
  * 向量数据库：NanoVector (轻量级向量数据库)
  * 图结构信息存储：NetworkX (用于知识图谱存储)
- 知识抽取：
  * 实体&关系抽取：DSPy + CoT (Chain of Thought)
- AI 模型：
  * Embedding 模型：支持 配置见 backend/.env.example
  * 大语言模型：支持 OpenAI/DeepSeek等，配置见 backend/.env.example
- 开发工具：
  * 依赖管理：pip
  * 环境管理：python-dotenv
  * 日志系统：loguru

## **系统架构**
- 前后端分离架构
- WebSocket 实时通信
- 向量检索 + 图检索 + text2sql混合召回
- DSPy 意图理解和推理

**本项目主要关注解决方案的实现，部分代码由cursor生成提效**


# 项目依赖
详情参考requirements.txt 


# 快速开始

## 1. 安装依赖
```bash
pip install -r backend/requirements.txt
```
注意：如果安装时报错，可能是requirements.txt文件格式问题，建议：
- 复制requirements.txt内容到新文件
- 检查并删除可能的特殊字符
- 使用新创建的依赖文件进行安装

## 2. 配置环境变量
在backend目录下创建.env文件，并按照.env.example模板进行配置。主要配置项如下：

a) 大语言模型配置：
    本项目使用DSPy进行意图识别，需要配置两个独立的模型：
    Dspy官方中文文档信息：https://www.aidoczh.com/dspy/
    1. 问答/推理模型：用于处理用户查询和推理
    2. 优化模型：用于模型优化
    两个模型可以使用相同或不同的配置，支持OpenAI-SDK格式的模型：
    - OpenAI API系列：GPT-3.5/4/4o
    - DeepSeek系列：deepseek-chat/coder
    - 阿里云系列：Qwen/通义千问
    - 百度文心系列：ERNIE-Bot
    - Ollama本地部署
    - HuggingFace部署
    - VLLM高性能部署


    # 问答/推理模型配置（用于处理用户查询和推理）
    LLM_TYPE="deepseek"                # 模型类型(可替换为其他模型)
    API_KEY="sk-xxxxxxxxxxxxxxxx"             # API密钥
    BASE_URL="xxxxxxxxxxxxxxxxxxxxx"  # API基础地址
    LLM_MODEL="deepseek-chat"          # 具体的模型名称
    
    # Ollama配置（本地部署方案，适合离线环境）
    # LLM_TYPE="ollama_chat"           # 设置为使用Ollama本地模型
    # API_KEY=""                       # Ollama本地部署不需要API密钥
    # BASE_URL="http://localhost:11434" # Ollama服务的本地地址
    # LLM_MODEL="xxxxxxxxxxxxx"           # 使用的具体模型
    
    # 优化模型配置（用于模型后优化,如果不优化可以忽略）
    Train_LLM_TYPE="deepseek"            # 优化模型类型(可替换为其他模型)
    Train_LLM_MODEL="deepseek-chat" # 优化使用的具体模型
    Train_OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxx"  # 优化模型的API密钥
    Train_OPENAI_BASE_URL="xxxxxxxxxxxxxxxxxxxxx"  # 优化模型的API地址

b) 系统环境配置（核心路径和参数设置）：
   ```
   RAG_DIR="graph_data_new"              # 知识图谱数据存储目录
   LOG_LEVEL="DEBUG"                     # 日志级别（可选：DEBUG, INFO, WARNING, ERROR）
   DATABASE_URL="sqlite:///.dbs/interactions.db"  # 交互数据库路径
   SPECIES_DB_URL="./.dbs/marine_species.db"     # 物种数据库路径
   ```

c) 向量检索配置（影响检索效果的关键参数）：
   ```
   VECTOR_SEARCH_TOP_K=3                 # 向量检索返回的最大结果数
   BETTER_THAN_THRESHOLD=0.7             # 相似度筛选阈值（0-1之间）
   GRAPHML_DIR="graph_entity_relation_detailed.graphml"  # 知识图谱存储文件
   ```

d) Embedding模型配置（文本向量化参数）：
   ```
   MAX_BATCH_SIZE=100                    # 批处理大小，影响处理速度
   EMBEDDING_MAX_TOKEN_SIZE=8192         # 单次处理的最大token数
   EMBEDDING_DIM=1024                    # 向量维度
   EMBEDDING_MODEL="xxxxxxxxxxxxxxx"   # 使用的embedding模型
   EMBEDDING_MODEL_BASE_URL="xxxxxxxxxxxxxxxxxxxxx"
   EMBEDDING_MODEL_API_KEY="your-embedding-api-key"  # embedding服务的API密钥
   ```

重要注意事项：
- 所有标注为 "your-xxx-api-key" 的配置项必须替换为您申请的实际API密钥
- API密钥可以从相应的服务提供商平台获取：
- 请确保在运行程序前完成所有必要的配置
- 配置文件中的路径可以根据实际部署环境进行调整
- 建议在正式部署前在测试环境中验证配置的正确性

## 3. 运行服务
## 环境配置
本项目使用环境变量来配置API和WebSocket地址。

### 配置步骤
1. 复制`.env.example`文件并重命名为`.env`（或`.env.development`、`.env.production`等）
2. 根据你的环境修改文件中的变量值

### 可用环境变量
- `VITE_API_URL`: 后端API地址
- `VITE_WS_URL`: WebSocket服务地址

### 启动后端服务
```bash
cd backend
python app.py
```
### 启动前端服务
```bash
cd frontend 
npm install
- 开发环境: `npm run dev` (使用`.env.development`或`.env`中的配置)
- 生产构建: `npm run build` (使用`.env.production`中的配置)
```
## 4. 数据处理说明
本项目提供了两种数据处理方式：
1. 使用内置示例数据（默认方式）
2. 使用自定义数据：
   - 使用 tools/entity_extraction.py 进行图数据抽取
   - 使用 entity_extraction_db.py 进行结构化数据抽取与存储
   - 处理后的数据将自动存储在配置文件指定的位置：
     * 图数据：保存在 RAG_DIR 指定目录
     * 结构化数据：保存在 SPECIES_DB_URL 指定的数据库文件


## 5. 运行步骤
**启动成功后的界面如下**：
![启动成功界面](./images/startup-success.jpg)


**实体不在图谱中的问题：**

![非实体信息截图](./images/非实体信息截图.jpg)

**补充说明**：当用户查询的实体不存在于知识图谱中时，系统会自动切换至向量检索策略。当前配置使用 `top_k=1` 参数，仅返回相似度最高的单个结果。这种设计在处理专业领域内的模糊查询时表现良好，但面对领域外查询时存在局限性：

1. 对于需要综合多个信息源的复杂问题，单一结果可能不够全面
2. 对于统计类问题（如"有多少种..."），系统只能基于有限上下文回答
3. 对于非专业领域的问题，缺乏足够的背景知识进行准确响应

此限制是系统当前设计的权衡结果，可通过以下方式改进：
- 在 `dspy_inference.py` 中调整 `top_k` 参数以获取更多结果
- 对非领域问题实现智能转发至通用模型
- 扩展知识图谱覆盖更广泛的实体信息

**实体在图谱中的问题：**

- **实体查询问题：**

  ![实体信息查询](./images/实体信息查询.jpg)

- **关系查询问题：**

  ![关系信息查询](./images/关系信息查询.jpg)

- **属性查询问题：**

  ![属性信息查询](./images/属性信息查询.jpg)

- **统计查询问题：**

  ![统计信息查询](./images/统计信息查询.jpg)

  问题的正确性可以去～backend/docs/demo130.json自行验证

- **知识图谱展现：**

  - 点击首页link链接即可即可获取知识图谱信息

- **构建优化样本**：

  - 人工去修改前端页面中“推理过程”和“模型返回”中的内容
  - 目前架构下小样本的优化数据（30-50条）能取得一定的效果
  - ![优化样本](./images/优化样本.jpg)

- **优化样本：**

  - ![训练所有样本](./images/训练所有样本.jpg)

  



### DSPy 意图理解机制

1. **零样本理解能力**：

   - DSPy 框架使用 ReAct（Reasoning+Acting）模式，允许大模型在无需预训练的情况下理解用户意图
   - 系统通过 `dspy_inference.py` 中的 `ReActModel` 类集成了多种工具函数
   - 大模型根据问题语义自动选择最合适的工具，例如：
     * 实体问题："什么是台湾盲鳗？" → 调用 `find_nodes_by_node_type`
     * 统计问题："盲鳗科有多少种？" → 调用适当的计数和查询方法

2. **零样本理解的实现原理**：

   - 在 `dspy_inference.py` 中，ReAct 模块会自动解析每个工具函数的签名和文档字符串：
     ```python
     # dspy_inference.py 中的核心代码
     self.react = dspy.ReAct(
         DspyInferenceProcessor.MarineBiologyKnowledgeQueryAnswer,
         max_iters = MAX_ITERS,
         tools=[
             processor.find_nodes_by_node_type, 
             processor.get_unique_vector_query_results,
             # ...其他工具
         ]
     )
     ```

   - 工具函数的详细文档提供了关键上下文，如 `find_nodes_by_node_type` 中的描述：
     ```python
     def find_nodes_by_node_type(self, start_node, trget_node_type):
         '''
         此方法会根据传入的节点名称，在图数据中以该节点为起点查找包含指定节点类型的节点列表。
         start_node 为开始查找的树节点名称，只允许单个节点。
         trget_node_type 目标节点类型,只允许单个类型名称。
         返回值为从该节点开始，包含指定属性名的节点数量与节点列表。
         已知图数据中存在一系列的海洋生物相关信息：
         1. ⽣物分类学图数据：包括"拉丁学名", "命名年份", "作者", "中文学名"
         2. ⽣物科属于数据："界", "门", "纲", "目", "科", "属", "种"...
         '''
     ```

   - DSPy 内部生成隐式提示，引导模型如何为不同问题选择工具：
     * 当问题包含"台湾盲鳗是什么"时，模型理解这是查询特定实体的描述
     * 当问题包含"盲鳗科有多少种"时，模型理解这需要计数操作

   - 大模型的思维链能力（在 `react_tools.py` 中体现）让系统能够：
     * 分析问题中的关键实体和关系
     * 规划多步检索策略
     * 根据中间结果调整后续操作

   这种零样本理解能力不依赖于预先定义的硬编码规则，而是依托于:
   1. 函数的清晰命名和文档
   2. DSPy的提示工程自动化
   3. 大模型的上下文理解能力
   4. ReAct框架的推理-行动循环机制

3. **工具选择机制**：
   ```python
   self.react = dspy.ReAct(
       DspyInferenceProcessor.MarineBiologyKnowledgeQueryAnswer,
       max_iters = MAX_ITERS,
       tools=[processor.find_nodes_by_node_type, ...]
   )
   ```
   - 模型通过思考链（Chain-of-Thought）分析问题特征
   - 基于问题类型动态选择工具组合
   - 无需硬编码规则即可处理多种问题类型

###  DSPy 优化原理与效果

1. **优化技术本质**：
   - DSPy 优化不是传统的参数微调，而是**提示工程自动化**
   - 系统通过 `dspy_evaluation.py` 中的评估器收集用户反馈数据
   - 优化过程存储在 `dspy_program` 目录中的程序文件（.pkl 和 .json）

2. **优化流程**：
   ```python
   # app.py 中的优化逻辑
   async def run_dspy_optimization(training_data: List[Dict], version: str, ids: List[str]):
       # 收集优化数据
       # 构建评估指标
       # 优化推理程序
       # 保存优化后的模型
   ```
   - 收集用户提问和反馈数据作为优化样本
   - 使用 BiologicalRetrievalEvaluation 评估推理质量
   - 应用多次迭代优化，生成更精确的思考链模板

3. **优化效果**：
   - **意图理解增强**：系统能更准确区分实体查询、关系查询、属性查询和统计查询
   - **工具选择优化**：模型学会更高效地组合检索工具，减少不必要的检索步骤
   - **推理模式改进**：通过分析成功案例，系统生成更结构化的推理路径
   - **领域适应性**：优化后的系统表现出更强的领域特定理解能力，尤其在海洋生物学术语处理上

4. **版本比较**：
   - 通过比较 `program_v1.0.1_20250302192606.json` 和 `program_v1.0.3_20250315154834.pkl` 可见优化效果
   
   
   

## 6. 交流与问题讨论
## About data

### 1. 数据源替换

#### 内置数据源替换

本项目包含两个内置示例数据集（`demo18.json`和`demo130.json`），它们结构相同但数据量不同。替换步骤：

```bash
# 替换小型测试数据集
cp your_small_dataset.json backend/docs/demo18.json

# 替换完整数据集
cp your_full_dataset.json backend/docs/demo130.json
```

两个数据集共享相同的结构和字段，仅在数据量上有区别，方便您进行快速测试和完整训练。

#### 自定义数据引入

引入您自己的领域数据需要以下全面调整：

1. **准备JSON格式数据**
   - 系统优先支持JSON格式，包含实体、关系和属性字段

2. **实体抽取与图谱构建**
   - 使用`tools/entity_extraction.py`从JSON中提取实体并构建图谱
   - 需修改抽取逻辑以适配您的数据结构
   - 自定义实体类型和关系类型映射

3. **建立结构化数据库**
   - 使用`tools/entity_extraction_db.py`创建关系型数据库
   - 调整数据库表结构设计
   - 修改字段映射和索引策略

4. **DSPy组件全面调整**

   a. `dspy_inference.py`：
      - 重新定义问题类型和意图分类
      - 修改`MarineBiologyKnowledgeQueryAnswer`签名类及描述
      - 调整ReAct工具选择逻辑和参数
      - 自定义推理流程和决策路径

   b. `dspy_evaluation.py`：
      - 重新设计评估指标和权重
      - 修改`BiologicalRetrievalEvaluation`签名以匹配新领域
      - 调整评分标准和反馈机制

   c. `dspy_query_db.py`：
      - 重构SQL生成逻辑
      - 调整`NaturalLanguageToSQL`提示
      - 修改数据库查询和结果格式化

   d. `react_tools.py`：
      - 重新定义`NODE_HIERARCHY`以匹配新领域的层级关系
      - 调整图检索算法和路径选择逻辑
      - 修改向量检索参数和阈值

5. **配置文件调整**
   - 修改`.env.example`和`.env`中的模型参数
   - 调整向量检索参数和阈值
   - 更新数据路径和文件名

6. **优化数据准备**
   
   - 创建领域特定的示例问答对
   - 编写标准推理路径作为优化基准
   - 设计适合新领域的评估样本

### 2. 数据场景适配性

### 最佳适用场景
- **有明确标准答案的领域**：如百科知识、产品目录、技术规范等
- **结构化程度高的数据**：实体关系明确、属性定义清晰的知识库
- **专业垂直领域**：如本项目示例中的海洋生物学分类系统

### 需要额外工作的场景
- **非量化评估内容**：如论文概要、观点分析等主观内容
- **需要推理的场景**：需要复杂逻辑推导的问题
- **多源异构数据**：来自不同格式、不同结构的混合数据

在这些场景中，您需要设计自定义评估指标才能有效衡量系统表现。

### 3. 数据处理全流程（未来规划）

数据清洗和切分是我们下一阶段的重点开发方向，将实现以下流程：

### 数据预处理流程

1. **版面识别转换**
   - PDF等文档通过版面识别模型转换为结构化Markdown
   - 关键步骤：自动识别→结构化转换→人工校验

2. **智能内容切分**
   - 多种切分策略：固定长度、语义分割、页面分割、递归分块
   - 自适应切分：根据内容特点自动选择最佳切分方式
   - 切分后进行人工复核确保质量

3. **多模态向量化**
   - 文本：使用大规模语言模型生成向量表示
   - 图像：通过多模态模型处理，提取视觉与文本语义
   - 表格：专用模型转换为结构化文本后向量化
   - 所有非文本内容经过人工确认后再进行向量化

4. **结构化处理**（可选）
   - 通过大模型将非结构化内容转换为JSON格式
   - 字段粒度和深度可根据业务需求定制
   - 支持复杂嵌套结构和多级关系

5. **多级索引构建**
   - 向量索引：所有内容的语义向量进入向量数据库
   - 实体索引：抽取的实体及关系进入专用索引
   - 结构化索引：JSON数据导入关系型数据库
   - 混合索引：支持多路召回和交叉验证



## 系统局限性与改进方向

### 当前意图识别模块的局限

1. **流式输出支持有限**
   - 当前框架不支持真正的增量式流式输出
   - 大型响应可能导致前端等待时间延长
   - 用户体验在复杂查询时可能受到影响

2. *优化效果量化挑战**
   - 优化效果不易在量化指标上直观体现
   - 领域适应性提升难以精确衡量
   - 对比测试基准尚不完善

3. **架构灵活性不足**
   - 现有框架与业务逻辑耦合度较高
   - 难以快速适应新领域和新需求
   - 未来目标：发展为可配置的中间件形态，支持插件式开发

### 复杂查询处理能力

1. **多条件筛选查询支持情况**
   - 系统原则上支持多条件筛选的统计查询
   - 例如："体长3m以上，生活在东海的虎鲨目鲨鱼有多少种？"

2. **查询精度依赖因素**
   - 查询精度高度依赖于结构化数据的字段粒度
   - 关键条件：
     * 用户筛选条件必须与`entity_extraction_db.py`处理的结构化数据字段匹配
     * 查询字段需作为独立属性存储（如"体长"、"自然分布地"）
     * 若属性被合并（如多种特征合并为"生物特征"），查询精度将显著降低

3. **改进方向**
   - 优化实体抽取逻辑，支持更细粒度的属性识别
   - 增强结构化数据处理，改进属性分离与标准化
   - 提升模糊匹配能力，处理非精确条件表述
   - 引入自动字段映射，实现用户查询与数据字段的智能对应

### 响应效率提升策略

1. **本地部署优化**
   - 本地模型部署可显著提升整体响应速度
   - 推荐搭配高性能推理框架：
     * [VLLM](https://github.com/vllm-project/vllm)：支持高效批处理和KV缓存
     * [Xinference](https://github.com/xorbitsai/xinference)：分布式推理支持和资源优化
   - 模型选择建议：
     * 不推荐本地部署小参数模型（7B/14B），推理质量难以满足复杂推理需求
   
2. **API服务选择**
   - 不同服务提供商性能差异显著
   - 服务对比分析：
     * DeepSeek官方API：功能完整但响应较慢，适合非实时场景
   - 选择建议：
     * 对成本敏感的场景，可在保证基本性能的前提下选择性价比更高的服务商
     * 建议在正式部署前进行多服务商的性能和成本对比测试
   
   
   
## 图谱管理与展示说明

### 图数据库与可视化优化

1. **当前图谱管理架构**
   - 采用轻量级图数据库实现（基于NetworkX）
   - 特点与局限：
     * 高效灵活，便于集成和部署
     * 缺少专业的图数据库管理界面
     * 不支持复杂的可视化配置和交互操作
   - 未来规划：
     * 集成专业图数据库（如Neo4j或TigerGraph）
     * 开发管理员控制台，支持图谱结构调整
     * 优化存储结构，提升大规模图谱处理能力

2. **知识图谱展现优化**
   - 当前实现：
     * 基础HTML展示（`knowledge_graph-1.html`）
     * 简单网络图布局，缺乏交互功能
     * 节点和边的样式未经专业设计
   - 改进计划：
     * 引入专业图可视化库（如ECharts、Graphin等）
     * 实现自适应布局和缩放功能
     * 支持节点分组、过滤和高亮等交互特性

3. **推理过程展示说明**
   - 当前设计：
     * 系统故意保留并展示详细的推理过程
     * 目的：方便开发者和用户深入理解系统决策路径
     * 有助于调试和验证推理质量
   - 可配置选项：
     * 生产环境可通过配置隐藏详细推理过程
     * 研发环境可保留完整思考链用于开发和优化
     * 后续版本将提供更精细的展示控制选项



## 7. 下一步计划
### **从解决方案到端到端产品**：

1. **当前定位与局限**
   - 目前开源内容本质上是一套技术解决方案
   - 主要挑战：
     * 用户需更换数据集时，需修改大量代码
     * 定制化程度高，可复用性有限
     * 技术门槛较高，不适合非技术团队直接使用

2. **产品化发展路线**
   - 核心转变：从代码修改到配置驱动
   - 规划功能：
     * 可视化配置界面：意图识别框架签名、评估方案等
     * 模块化设计：支持即插即用的组件替换
     * 低代码/无代码接口：降低使用门槛
     * 自动化工作流：简化数据预处理和模型优化过程
   - 目标：大幅降低企业知识库构建与维护成本

3. **"数据胶囊"产品愿景**
   - 产品名称由来：Datacapsule（数据胶囊）—— 小小胶囊蕴含庞大能量
   - 核心价值主张：
     * 降低企业知识构建难度
     * 形成企业闭环的知识壁垒
     * 释放大模型在垂直领域的潜力
   - 适用场景：
     * 企业专有知识管理
     * 专业领域智能问答
     * 行业知识图谱构建与应用

### 开放协作邀请

我们诚挚邀请对知识图谱、大模型应用、数据处理等领域感兴趣的开发者加入项目。如有兴趣，请扫描README文件末尾的二维码与我们联系，一起探索知识增强的未来！

   

## 8.鸣谢

**项目鸣谢**：十分感谢百度飞桨AI技术生态部：梦姐、楠哥和张翔、新飞同学对本项目的大力支持与帮助！

**项目核心贡献者**：Loukie7、Alex—鹏哥

对项目感兴趣的同学可以扫码添加好友，后续会成立产品交流社群

![二维码](./images/二维码.jpg)
