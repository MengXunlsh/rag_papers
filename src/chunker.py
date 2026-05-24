"""
PDF 提取与文本切分
依赖: pip install pdfplumber
"""
import os
import re
import pdfplumber


def extract_text(pdf_path):
    """从 PDF 提取纯文本，去页眉页脚"""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    """
    清洗学术论文文本：
    - 去引用标记 [1], [2,3], (Author, 2020)
    - 去图表编号 (Fig. 1, Table 2)
    - 合并没有句号结尾的断行
    - 去多余空白
    """
    # 去引用标记
    text = re.sub(r'\[[\d,\s-]+\]', '', text)
    text = re.sub(r'\(\w+\s*et\s*al\.,?\s*\d{4}\)', '', text)
    # 去图表编号
    text = re.sub(r'(Fig\.?\s*\d+|Figure\s*\d+|Table\s*\d+)', '', text, flags=re.IGNORECASE)
    # 合并断行（行末非标点结尾则与下一行合并）
    text = re.sub(r'(?<![.!?:])\n(?!\n)', ' ', text)
    # 压缩连续空白和空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # 去纯数字行（页码）
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def chunk_text(text, source_name, chunk_size=500, overlap=100):
    """
    将文本切分为带来源标记的 chunk 列表。

    策略：先按段落分，超过 chunk_size 的段落再按固定长度 + 重叠切。

    返回: [{"content": "...", "source": "file.pdf", "chunk_id": 0}, ...]
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    chunk_id = 0

    for para in paragraphs:
        words = para.split()
        if len(words) <= chunk_size:
            if len(' '.join(words)) > 20:  # 过滤太短的碎片
                chunks.append({
                    "content": ' '.join(words),
                    "source": source_name,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
        else:
            # 超长段落：滑动窗口切分
            start = 0
            while start < len(words):
                segment = words[start:start + chunk_size]
                if len(segment) < 10:
                    break
                chunks.append({
                    "content": ' '.join(segment),
                    "source": source_name,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
                start += chunk_size - overlap

    return chunks


def process_papers(papers_dir):
    """处理 papers/ 目录下所有 PDF，返回 chunks 列表"""
    pdf_files = [f for f in os.listdir(papers_dir) if f.lower().endswith('.pdf')]
    pdf_paths = [os.path.join(papers_dir, f) for f in pdf_files]
    return process_pdf_list(pdf_paths)


def process_pdf_list(pdf_paths):
    """处理指定的 PDF 文件列表，返回 chunks 列表"""
    all_chunks = []

    if not pdf_paths:
        return all_chunks

    for path in pdf_paths:
        fname = os.path.basename(path)
        print(f"  处理: {fname}")
        try:
            raw = extract_text(path)
            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned, fname)
            all_chunks.extend(chunks)
            print(f"    提取 {len(cleaned)} 字符, 切出 {len(chunks)} 个块")
        except Exception as e:
            print(f"    失败: {e}")

    print(f"\n  共 {len(all_chunks)} 个文本块\n")
    return all_chunks
