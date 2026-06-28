"""文件加载：PDF / TXT / MD / CSV / 图片。"""
import os
from pathlib import Path
from typing import List, Tuple

import fitz
from langchain_core.documents import Document


def load_file(file_path: str) -> Tuple[List[Document], List[str]]:
    """按扩展名分发到对应加载器。"""
    ext = Path(file_path).suffix.lower()
    if ext in (".txt", ".md", ".csv"):
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png"):
        return load_image(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def load_txt(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        return [], []
    return [Document(page_content=text, metadata={"source": source, "page": 0, "type": "text"})], []


def load_pdf(file_path: str) -> Tuple[List[Document], List[str]]:
    """按页提取文字 + 内嵌图片。"""
    source = Path(file_path).name
    documents = []
    image_paths = []

    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": source, "page": page_num + 1, "type": "text"},
            ))

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_filename = f"{Path(file_path).stem}_p{page_num+1}_img{img_index+1}.{base_image['ext']}"
            image_save_path = os.path.join("images", image_filename)
            with open(image_save_path, "wb") as f:
                f.write(base_image["image"])
            image_paths.append(image_save_path)

            documents.append(Document(
                page_content="",
                metadata={"source": source, "page": page_num + 1, "type": "image", "image_path": image_save_path},
            ))

    doc.close()
    return documents, image_paths


def load_image(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    return [
        Document(page_content="", metadata={"source": source, "page": 0, "type": "image", "image_path": file_path}),
    ], [file_path]
