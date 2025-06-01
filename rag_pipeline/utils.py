from __future__ import annotations
from typing import List
from langchain.schema import Document
from rag_pipeline import config
import os
import base64
import cv2
from pathlib import Path
from pdf2image import convert_from_path
import requests
from rag_pipeline.text_splitter import MarkdownHeaderTextSplitter


def encode_image(image_path, image_size=(837, 1012)):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=image_size, interpolation=cv2.INTER_CUBIC)

    except Exception as _:
        print(f"Error when encoding image: {image_path}")
        return ""

    _, ext = os.path.splitext(image_path)
    _, encoded_image = cv2.imencode(ext, img)
    encoded_string = base64.b64encode(encoded_image).decode("utf-8")
    return encoded_string


def get_page_number(filename):
    if filename.startswith("page_") and filename.endswith(".png"):
        try:
            return int(filename[5:-4])  # "page_X.png" -> X 추출
        except ValueError:
            return float("inf")
    return float("inf")


def pdf_to_docs(file_path: Path) -> List[Document]:
    temp_img_dir = Path("./data/temp_img")
    os.makedirs(temp_img_dir, exist_ok=True)

    pdf_name = file_path.stem  # 확장자 없이 파일명 추출
    images = convert_from_path(str(file_path))

    # 각 페이지를 png로 저장
    for idx, image in enumerate(images):
        output_path = temp_img_dir / f"page_{idx+1}.png"
        image.save(output_path, "PNG")

    print(f"PDF successfully converted: {pdf_name} -> {len(images)} pages")

    # Text Extraction
    all_texts = []

    for filename in sorted(os.listdir(temp_img_dir), key=get_page_number):
        img_path = os.path.join(temp_img_dir, filename)
        if not os.path.isfile(img_path):
            continue

        # Encode image
        image_url = encode_image(img_path, image_size=(837, 1012))
        _, image_ext = os.path.splitext(filename)
        image_ext = image_ext.lstrip(".")  # e.g. png, jpg
        image_url = f"data:image/{image_ext};base64,{image_url}"

        payload = {
            "model": config.REMOTE_LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract all the text from the image: {image_url}.",
                }
            ],
        }
        response = requests.post(config.REMOTE_LLM_URL, json=payload).json()
        text = response["choices"][0]["message"]["content"]

        print(f"Successfully extracted text from: {filename}\n")
        all_texts.append(f"{text.strip()}\n")

    combined_texts = "\n".join(all_texts)

    # Split extracted text
    headers_to_split_on = [("#", "Header1"), ...]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split=headers_to_split_on, strip_headers=False
    )

    split_contents = md_splitter.split_text(combined_texts)
    print("\nSuccesfully split text!")

    return split_contents


def img_to_docs(file_path: Path) -> List[Document]:
    # Text Extraction
    all_texts = []

    for filename in sorted(os.listdir(file_path), key=get_page_number):
        img_path = os.path.join(file_path, filename)
        if not os.path.isfile(img_path):
            continue

        # Encode image
        image_url = encode_image(img_path, image_size=(837, 1012))
        _, image_ext = os.path.splitext(filename)
        image_ext = image_ext.lstrip(".")  # e.g. png, jpg
        image_url = f"data:image/{image_ext};base64,{image_url}"

        payload = {
            "model": config.REMOTE_LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Extract all the text from the image: {image_url}.",
                }
            ],
        }
        response = requests.post(config.REMOTE_LLM_URL, json=payload).json()
        text = response["choices"][0]["message"]["content"]

        print(f"Successfully extracted text from: {filename}\n")
        all_texts.append(f"{text.strip()}\n")

    combined_texts = "\n".join(all_texts)

    # Split extracted text
    headers_to_split_on = [("#", "Header1"), ...]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split=headers_to_split_on, strip_headers=False
    )

    split_contents = md_splitter.split_text(combined_texts)
    print("\nSuccesfully split text!")

    return split_contents


def build_payload_for_summary_generation(query_text: str) -> dict:
    payload = {
        "model": config.REMOTE_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful assistant that generates summaries based on the provided question.""",
            },
            {"role": "user", "content": f"[Question]:{query_text}"},
        ],
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False,
        "n": 1,
    }
    return payload


def build_payload_for_hyde(query_text: str) -> dict:
    payload = {
        "model": config.REMOTE_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful assistant that generates answers based on the provided question and context.""",
            },
            {"role": "user", "content": f"[Question]:{query_text}"},
        ],
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False,
        "n": 1,
    }
    return payload


def build_payload_for_llm_answer(query_text: str, context: str) -> dict:
    payload = {
        "model": config.REMOTE_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful assistant that generates answers based on the provided question and context.""",
            },
            {
                "role": "user",
                "content": f"[Question]:{query_text}, [Context]:{context}",
            },
        ],
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False,
        "n": 1,
    }
    return payload


def build_payload_for_complexity_check(query_text: str) -> dict:
    """Determine if the question requires simple retrieval or complex multi-hop reasoning."""
    payload = {
        "model": config.REMOTE_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful assistant that generates answers based on the provided question and context.""",
            },
            {
                "role": "user",
                "content": f"[Question]:{query_text}, [Context]:{context}",
            },
        ],
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False,
        "n": 1,
    }
    return payload


def build_payload_for_complexity_check(query_text: str) -> dict:
    """Determine if the question requires simple retrieval or complex multi-hop reasoning."""
    payload = {
        "model": config.REMOTE_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """Determine if the user's question requires simple retrieval or complex multi-hop reasoning.
                
                Simple questions can be answered with a single retrieval operation and direct generation of an answer.
                Complex questions require multiple reasoning steps and retrievals to reach a final answer.
                
                Reply with ONLY one of two options:
                - "simple": If the question can be answered directly with a single retrieval.
                - "complex": If the question requires multi-hop reasoning and multiple retrievals.
                
                Respond with only the word "simple" or "complex".
                """,
            },
            {"role": "user", "content": query_text},
        ],
        "max_tokens": 3000,
        "temperature": 0.1,
        "top_p": 0.95,
        "stream": False,
        "n": 1,
    }
    return payload
