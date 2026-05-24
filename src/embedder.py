"""
向量嵌入模型
依赖: pip install sentence-transformers
"""
from sentence_transformers import SentenceTransformer

# 首次运行自动下载模型（约 118MB）
MODEL_NAME = "intfloat/multilingual-e5-small"


def load_embedder():
    """加载多语言 Embedding 模型（优先本地缓存）"""
    print(f"  加载模型: {MODEL_NAME}")

    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        print("  (从本地缓存加载)")
    except Exception:
        print("  (首次运行需下载约 118MB)")
        model = SentenceTransformer(MODEL_NAME)

    return model


def embed_chunks(model, chunks):
    """
    将 chunks 转为向量。

    e5 模型要求文本加前缀：
    - 文档段落加 "passage: "
    - 查询加 "query: "（在 rag.py 中处理）
    """
    texts = [f"passage: {chunk['content']}" for chunk in chunks]
    print(f"  正在编码 {len(texts)} 个文本块...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def embed_query(model, query):
    """将查询转为向量"""
    return model.encode([f"query: {query}"], show_progress_bar=False)[0]
