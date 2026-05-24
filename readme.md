# RAG 论文问答系统

基于检索增强生成（RAG）的论文问答系统。将 PDF 论文切分、向量化后存入 ChromaDB，通过 DeepSeek 大模型实现基于论文内容的问答，支持引用溯源。

## 技术栈

- Python 3.12+
- pdfplumber — PDF 文本提取
- sentence-transformers — 文本向量化 (multilingual-e5-small)
- ChromaDB — 向量数据库
- DeepSeek API — 大模型生成回答
- OpenAI SDK

## 快速开始

```bash
# 1. 安装依赖
pip install pdfplumber sentence-transformers chromadb openai

# 2. 设置 API Key
# Windows: 系统环境变量 DEEPSEEK_API_KEY=sk-你的key
# Linux/Mac: export DEEPSEEK_API_KEY=sk-你的key

# 3. 放入论文
# 将 PDF 文件放入 papers/ 目录

# 4. 构建向量索引
python build_index.py           # 增量（仅新增）
python build_index.py --full    # 全量重建

# 5. 开始问答
python query.py
```

## 项目结构

```
rag_papers/
  README.md              项目说明
  build_index.py          构建向量索引（增量/全量）
  query.py                交互式问答
  requirements.txt        Python 依赖
  papers/                 待处理的 PDF 论文
  chroma_db/              向量数据库文件
  src/
    chunker.py            PDF 提取、文本清洗、切分
    embedder.py           向量嵌入模型
    vectordb.py           ChromaDB 增删查
    rag.py                RAG 管线（检索 + 生成）
```

## RAG 流程

```
PDF 论文 -> 提取文字 -> 清洗 -> 切分段落
                                    |
Embedding 模型 -> 向量化 -> 存入 ChromaDB
                                    |
用户中文提问 -> 向量化 -> 检索Top5段落 -> 拼入Prompt -> DeepSeek回答
```

核心参数：每块 500 词、相邻重叠 100 词、检索 Top 5。

## 索引管理

- `python build_index.py` — 增量模式，只处理新增 PDF
- `python build_index.py --full` — 全量重建
- 向量库按文件名判重，已入库文件不会重复索引
- 删除论文：从 papers/ 移除后 `--full` 重建

## 注意事项

- Embedding 模型首次运行需下载约 118MB，之后走本地缓存
- DeepSeek API Key 通过系统环境变量传入，不硬编码
- 向量数据库存储在 `chroma_db/` 目录，可直接备份迁移
- 支持中文提问 + 英文论文的跨语言检索