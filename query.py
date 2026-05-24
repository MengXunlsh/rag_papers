"""
Step 2: 交互式问答
====================
加载已构建的向量索引，输入问题，检索 + LLM 生成回答

用法: python query.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedder import load_embedder, embed_query
from src.vectordb import get_collection, search
from src.rag      import build_prompt, generate_answer

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "chroma_db")


def main():
    # 检查 API Key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("请先设置环境变量: set DEEPSEEK_API_KEY=sk-xxx")
        return

    # 加载模型和数据库
    print("加载 Embedding 模型...")
    model = load_embedder()

    print("加载向量数据库...")
    collection = get_collection(db_path=DB_PATH)

    # 交互循环
    print("\n" + "=" * 50)
    print("RAG 论文问答系统已就绪")
    print("输入问题开始，输入 exit 退出")
    print("=" * 50 + "\n")

    while True:
        query = input("提问: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("再见!")
            break

        # 检索
        print("  检索中...")
        q_emb = embed_query(model, query)
        results = search(collection, q_emb, top_k=5)

        # 生成回答
        messages = build_prompt(query, results)
        print("\n回答:", end=" ")
        answer = generate_answer(messages, stream=True)

        # 展示来源
        print("\n引用来源:")
        seen = set()
        for r in results:
            key = r['source']
            if key not in seen:
                print(f"  - {key} (块{r['chunk_id']})")
                seen.add(key)
        print()


if __name__ == "__main__":
    main()
