"""
ChromaDB 向量数据库操作
依赖: pip install chromadb
"""
import os
import chromadb
from chromadb.config import Settings


def get_collection(db_path="./chroma_db", collection_name="papers"):
    """获取或创建 ChromaDB 集合"""
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(name=collection_name)


def add_chunks(collection, chunks, embeddings):
    """将 chunks 及其向量存入 ChromaDB"""
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
                 for chunk in chunks]

    # 分批插入，避免一次太多
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end]
        )
    print(f"  已存入 {len(chunks)} 条")


def search(collection, query_embedding, top_k=5):
    """检索最相关的 K 个段落"""
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    # 整理成易用格式
    retrieved = []
    for i in range(len(results['ids'][0])):
        retrieved.append({
            "content": results['documents'][0][i],
            "source": results['metadatas'][0][i]['source'],
            "chunk_id": results['metadatas'][0][i]['chunk_id'],
        })
    return retrieved
