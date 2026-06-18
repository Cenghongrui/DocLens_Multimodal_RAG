import os
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from langchain_core.documents import Document

def load_file(file_path: str) -> Tuple[List[Document], List[str]]:
    ext = Path(file_path).suffix.lower() 

    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png"):
        return load_image(file_path)
    else:
        raise ValueError(f"不支持该文件格式: {ext}")


def load_text(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return [], []

    doc = Document(
        page_content=text,
        metadata={"source": source, "page": 0, "type": "text"},
    )
    return [doc], []



def load_pdf(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    documents = []
    image_paths = []

    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        # 提取文字
        text = page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_num + 1,
                    "type": "text",
                }
            ))

        # 提取图片
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"{Path(file_path).stem}_p{page_num+1}_img{img_index+1}.{image_ext}"
            image_save_path = os.path.join("images", image_filename)

            with open(image_save_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(image_save_path)

            # 图片先占位，page_content 等 vision.py 填充
            documents.append(Document(
                page_content="",
                metadata={
                    "source": source,
                    "page": page_num + 1,
                    "type": "image",
                    "image_path": image_save_path,
                }
            ))

    doc.close()
    return documents, image_paths



def load_image(file_path: str) -> Tuple[List[Document], List[str]]:
    source = Path(file_path).name
    doc = Document(
        page_content="",
        metadata={
            "source": source,
            "page": 0,
            "type": "image",
            "image_path": file_path,
        },
    )
    return [doc], [file_path]