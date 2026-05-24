"""
Step 1: 构建论文向量索引
==========================
读取 papers/ 目录下所有 PDF，切分 -> Embedding -> 存入 ChromaDB

用法:
  python build_index.py          增量模式（只处理新增 PDF）
  python build_index.py --full   全量模式（重新处理所有 PDF）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chunker  import process_papers, process_pdf_list
from src.embedder import load_embedder, embed_chunks
from src.vectordb import get_collection, add_chunks

BASE = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE, "papers")
DB_PATH    = os.path.join(BASE, "chroma_db")


def get_existing_sources(collection):
    """获取已入库的论文文件名列表"""
    try:
        results = collection.get()
        if results and results['metadatas']:
            return set(m['source'] for m in results['metadatas'] if m)
    except Exception:
        pass
    return set()


def main():
    full_rebuild = "--full" in sys.argv

    print("=" * 50)
    print("Step 1: 构建论文向量索引" + (" (全量)" if full_rebuild else " (增量)"))
    print("=" * 50)

    # 获取所有 PDF 文件
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("错误: papers/ 目录没有 PDF 文件。")
        return

    # 增量模式：过滤已入库的文件
    if not full_rebuild:
        collection = get_collection(db_path=DB_PATH)
        existing = get_existing_sources(collection)
        pdf_files = [f for f in pdf_files if f not in existing]
        if not pdf_files:
            print("没有新增 PDF，索引已是最新。")
            print(f"  已入库论文数: {len(existing)}")
            return
        print(f"  已入库: {len(existing)} 篇, 新增: {len(pdf_files)} 篇")

    # ① 提取 PDF 文本并切分
    print("\n[1/3] 提取并切分 PDF...")
    chunks = process_pdf_list([os.path.join(PAPERS_DIR, f) for f in pdf_files])

    if not chunks:
        print("错误: 没有提取到任何文本。")
        return

    # ② 加载 Embedding 模型并编码
    print("\n[2/3] 加载 Embedding 模型...")
    model = load_embedder()
    embeddings = embed_chunks(model, chunks)

    # ③ 存入 ChromaDB（追加模式）
    print("\n[3/3] 存入向量数据库...")
    collection = get_collection(db_path=DB_PATH)
    add_chunks(collection, chunks, embeddings)

    # 统计
    all_sources = get_existing_sources(collection) if not full_rebuild else set()
    all_sources.update(set(c['source'] for c in chunks))

    print(f"\n索引构建完成。")
    print(f"  本次处理: {len(pdf_files)} 篇论文, {len(chunks)} 个文本块")
    print(f"  向量库总计: {len(all_sources)} 篇论文")
    print(f"  存储路径: {DB_PATH}")
    print(f"\n下一步: python query.py")


if __name__ == "__main__":
    main()
