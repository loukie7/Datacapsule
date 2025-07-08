<div align="center">
  <img src="./images/datacapsule.jpg" alt="Datacapsule Logo" width="400" />
  <h1>✨ Datacapsule</h1>
  <p><strong>Everything for precision</strong></p>
  <p>Datacapsule是一个基于知识图谱的多路召回解决方案，旨在通过多路召回技术，实现精准的知识检索。该解决方案涵盖了检索系统、实体关系抽取、实体属性抽取、实体链接、结构化数据库构建以及问答系统等多个功能模块，为信息检索和应用提供了强大的支持。</p>
</div>

<div align="center">

  <img src="https://img.shields.io/github/stars/loukie7/Datacapsule?style=for-the-badge&logo=github&color=yellow" alt="Stars" />
  <img src="https://img.shields.io/github/forks/loukie7/Datacapsule?style=for-the-badge&logo=github&color=blue" alt="Forks" />
  <img src="https://img.shields.io/github/issues/loukie7/Datacapsule?style=for-the-badge&logo=github&color=red" alt="Issues" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License" />

<br>

  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-green?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18+-blue?style=for-the-badge&logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-blue?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />

  <br>

  <a href="./readme.md"><img src="https://img.shields.io/badge/English-lightgrey?style=for-the-badge" alt="English" /></a>
  <a href="./readme_en.md"><img src="https://img.shields.io/badge/中文文档-lightgrey?style=for-the-badge" alt="中文文档" /></a>

  <br>

  <a href="https://github.com/loukie7/Datacapsule-webui">📱 Frontend Repository</a> •
  <a href="https://github.com/loukie7/Datacapsule/wiki">📚 Documentation</a> •
  <a href="https://github.com/loukie7/Datacapsule/discussions">💬 Discussions</a>

</div>

---

## 🚀 项目概述

Datacapsule是一个先进的基于知识图谱的多路召回解决方案，结合了图数据库、向量检索和智能推理的强大功能，提供精准的信息检索和问答能力。系统智能地通过多个检索路径（向量检索、图遍历和结构化数据库查询）路由查询，以提供全面准确的响应。

### 🌟 核心特性

- **🔍 多路径检索**：在向量检索、图遍历和SQL查询之间进行智能路由
- **🧠 智能问题理解**：自动将查询分类为实体、关系、属性和统计问题
- **📊 知识图谱管理**：使用NetworkX进行动态图构建和可视化
- **⚡ 轻量级向量数据库**：内置NanoVector进行高效语义检索
- **🔄 实时通信**：使用SSE（服务器发送事件）进行流式响应
- **🎯 Mini-React框架**：轻量级智能推理调度器
- **🌐 现代化前端**：React 18 + Vite + TailwindCSS界面
- **📈 性能优化**：结构化数据缓存和高效查询处理

---

## 🏗️ 系统架构

![系统架构图](./images/function-diagram.png)

### 🔧 技术栈

#### 后端
- **框架**：FastAPI
- **数据库**：SQLite + NanoVector + NetworkX
- **AI集成**：Mini-React + 标准OpenAI协议
- **通信**：SSE（服务器发送事件）
- **语言**：Python 3.8+

#### 前端
- **框架**：React 18 + Vite
- **样式**：TailwindCSS
- **状态管理**：React Hooks
- **通信**：SSE客户端
- **语言**：TypeScript + JavaScript

---

## 🎯 查询类型与检索策略

| 查询类型 | 示例 | 检索方法 |
|----------|------|----------|
| **实体查询** | "什么是台湾盲鳗？" | 图结构检索 |
| **关系查询** | "物种A和物种B有什么关系？" | 图遍历 |
| **属性查询** | "物种X的生活习性是什么？" | 图属性搜索 |
| **统计查询** | "科Y有多少种？" | 结构化数据库查询 |
| **一般查询** | 不包含图谱实体的问题 | 向量相似度搜索 |

---

## 🚀 快速开始

### 前置要求
- Python 3.8+
- Node.js 16+
- Git

### 1. 克隆仓库
```bash
git clone https://github.com/loukie7/Datacapsule.git
cd Datacapsule
```

### 2. 后端设置
```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 使用您的API密钥和配置编辑.env文件
```

### 3. 配置
编辑`.env`文件进行设置：

```env
# LLM配置
LLM_TYPE="openai"
API_KEY="your-api-key"
BASE_URL="https://api.openai.com/v1"
LLM_MODEL="gpt-3.5-turbo"

# Embedding配置
EMBEDDING_MODEL="text-embedding-ada-002"
EMBEDDING_MODEL_API_KEY="your-embedding-api-key"

# 系统配置
LOG_LEVEL="INFO"
DATABASE_URL="sqlite:///.dbs/interactions.db"
VECTOR_SEARCH_TOP_K=3
```

### 4. 启动后端服务
```bash
python main.py
```

### 5. 前端设置
前端设置请访问[Datacapsule WebUI仓库](https://github.com/loukie7/Datacapsule-webui)。

---

## 📊 演示截图

### 启动成功界面
![启动成功](./images/startup-success.jpg)

### 查询示例

#### 实体信息查询
![实体查询](./images/实体信息查询.jpg)

#### 关系查询
![关系查询](./images/关系信息查询.jpg)

#### 属性查询
![属性查询](./images/属性信息查询.jpg)

#### 统计查询
![统计查询](./images/统计信息查询.jpg)

---

## 🗓️ 版本路线图

### 📅 版本历史

#### v1.0 (2025-04-11)
- 🎉 Datacapsule 1.0首次发布
- 基于WebSocket的实时通信
- DSPy智能推理框架
- Litellm LLM调用集成
- 基础知识图谱构建

#### v1.1 (2025-07-08) - 当前版本
- 🔄 **通信升级**：从WebSocket迁移到SSE（服务器发送事件）
- 🧠 **框架优化**：用轻量级Mini-React调度器替换DSPy
- 🔗 **API简化**：移除Litellm依赖，使用标准OpenAI协议
- 🏗️ **架构重构**：改进代码结构和可维护性

#### v1.2 (即将推出)
- 📄 **文档处理**：增强文档解析能力
- ✂️ **文本分割**：高级文本分割策略
- 🤖 **智能体优化**：改进智能体检索策略
- 🔍 **搜索增强**：更好的语义搜索和排序

---

## 🛠️ 数据处理

### 内置数据
系统包含海洋生物学示例数据集：
- `docs/demo_18.json` - 小型测试数据集
- `docs/demo_130.json` - 完整数据集

### 自定义数据集成
1. **准备JSON数据**：用实体、关系和属性构建数据结构
2. **图构建**：使用`utils/entity_extraction.py`进行图构建
3. **数据库设置**：使用`utils/entity_extraction_db.py`进行结构化存储
4. **配置**：在`.env`中更新路径和参数

---

## 🔧 高级配置

### 向量搜索参数
```env
VECTOR_SEARCH_TOP_K=3           # 返回结果数量
BETTER_THAN_THRESHOLD=0.7       # 相似度阈值
EMBEDDING_DIM=1024              # 向量维度
MAX_BATCH_SIZE=100              # 处理批次大小
```

### 数据库配置
```env
DATABASE_URL="sqlite:///.dbs/interactions.db"
SPECIES_DB_URL="./.dbs/marine_species.db"
RAG_DIR="graph_data_new"
```

---

## 🤝 贡献

我们欢迎贡献！详情请联系项目负责人

### 开发设置
1. Fork仓库
2. 创建功能分支
3. 进行更改
4. 如适用，添加测试
5. 提交拉取请求

---

## 📈 性能与优化

### 本地部署
- **VLLM**：批处理的高性能推理
- **Xinference**：分布式推理支持
- **Ollama**：本地模型部署

### API服务选项
- **OpenAI**：可靠性能的标准API
- **DeepSeek**：成本效益的替代方案
- **自定义端点**：自托管解决方案

---

## 🎯 使用场景

### 理想应用
- **知识管理**：企业知识库
- **专业问答**：特定领域问答
- **研究工具**：学术和科学信息检索
- **文档**：技术文档搜索

### 领域适配性
- **结构化数据**：清晰的实体关系层次
- **专业领域**：专业术语和概念
- **事实信息**：可验证和精确的数据

---

## 🔮 未来计划

### 产品演进
- **配置驱动**：可视化配置界面
- **模块化设计**：基于插件的架构
- **无代码界面**：降低技术门槛
- **企业功能**：多租户支持、高级分析

### 技术路线图
- **图数据库**：Neo4j/TigerGraph集成
- **可视化**：高级图可视化工具
- **可扩展性**：分布式处理能力
- **多模态**：支持图像、文档和多媒体

---

## 📄 许可证

本项目在MIT许可证下授权 - 详情请查看[LICENSE](LICENSE)文件。

---

## 🙏 鸣谢

**项目鸣谢**：十分感谢百度飞桨AI技术生态部：梦姐、楠哥和张翔、新飞同学对本项目的大力支持与帮助！

**项目核心贡献者**：Loukie7、Alex—鹏哥

对项目感兴趣的同学可以扫码添加好友，后续会成立产品交流社群

<div align="center">
  <img src="./images/二维码.jpg" alt="WeChat QR Code" width="200" />
</div>

---

<div align="center">
  <p>⭐ 给我们的GitHub点个星 — 这很有帮助！</p>
  <p>由Datacapsule团队倾情打造 ❤️</p>
</div>