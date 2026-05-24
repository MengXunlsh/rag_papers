"""
RAG 检索 -> 生成 管线
依赖: pip install openai
"""
import os
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# RAG 专用 System Prompt
SYSTEM_PROMPT = (
    "你是一个学术论文助手。请基于下面提供的论文片段回答用户的问题。"
    "每个片段标注了 [来源: 文件名 块号]，回答时引用来源。"
    "如果提供的片段无法回答问题，请如实说明，不要编造。"
    "用中文回答，保持专业、准确。"
)


def build_prompt(query, retrieved_chunks):
    """构建 RAG prompt：拼接检索到的段落 + 用户问题"""
    # 拼接参考材料
    contexts = []
    for chunk in retrieved_chunks:
        source_label = f"[来源: {chunk['source']}, 块{chunk['chunk_id']}]"
        contexts.append(f"{source_label}\n{chunk['content']}")

    context_text = "\n\n---\n\n".join(contexts)

    # 构建消息
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"参考论文片段：\n\n{context_text}\n\n"
            f"用户问题：{query}\n\n"
            f"请基于以上片段回答，并引用具体来源。"
        )}
    ]


def generate_answer(messages, stream=True):
    """调用 DeepSeek 生成回答"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.3,
        stream=stream,
    )

    if stream:
        full = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full += text
        print()
        return full
    else:
        return response.choices[0].message.content
