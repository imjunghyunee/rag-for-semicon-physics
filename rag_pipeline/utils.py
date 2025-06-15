from __future__ import annotations
from typing import List
from langchain.schema import Document
from rag_pipeline import config
import os
import base64
import cv2
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI
from rag_pipeline.text_splitter import MarkdownHeaderTextSplitter

# Initialize OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)


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

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all the text from the image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content

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

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all the text from the image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content

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


def generate_summary(query_text: str) -> str:
    """Generate summary for the given query using OpenAI API."""
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates summaries based on the provided question.",
            },
            {
                "role": "user",
                "content": f"[Question]:{query_text}",
            },
        ],
        max_tokens=5000,
        temperature=0.5,
        top_p=0.95,
    )
    return response.choices[0].message.content


def generate_hyde_document(query_text: str) -> str:
    """Generate hypothetical document for HyDE using OpenAI API."""
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates answers based on the provided question and context.",
            },
            {
                "role": "user",
                "content": f"[Question]:{query_text}",
            },
        ],
        max_tokens=5000,
        temperature=0.5,
        top_p=0.95,
    )
    return response.choices[0].message.content


def generate_llm_answer(query_text: str, context: str) -> str:
    """Generate final answer using OpenAI API with improved context handling."""
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant specializing in semiconductor physics that generates accurate answers based on the provided question and context.

When answering:
1. Use the provided context as your primary information source
2. If the context includes previous questions and answers, build upon that knowledge
3. Provide technically accurate information for semiconductor physics domain
4. Be clear and concise while being comprehensive
5. If information is insufficient, acknowledge the limitations

The context may include:
- Previous questions and answers from earlier steps
- Current retrieved documents
- Technical specifications and data""",
            },
            {
                "role": "user",
                "content": f"Question: {query_text}\n\nContext:\n{context}",
            },
        ],
        max_tokens=5000,
        temperature=0.3,
        top_p=0.95,
    )
    return response.choices[0].message.content


def check_query_complexity(query_text: str) -> str:
    """Determine if the question requires simple retrieval or complex multi-hop reasoning."""

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in semiconductor physics who specializes in categorizing question complexity.

Analyze questions to determine if they require simple retrieval or complex multi-hop reasoning.

SIMPLE questions typically:
- Ask for a single definition, formula, or concept
- Can be answered with direct information retrieval
- Have clear, straightforward answers in textbooks
- Examples: "What is the bandgap of silicon?", "Define hole mobility", "What is the formula for electron drift velocity?"

COMPLEX questions typically:
- Require multiple steps of reasoning or calculation
- Need integration of multiple concepts
- Ask for analysis, comparison, or multi-step problem solving
- Involve design decisions or trade-off analysis
- Examples: "How does temperature affect both carrier concentration and mobility in silicon devices?", "Analyze the trade-offs between different doping strategies for optimizing transistor performance", "Compare the effects of different gate materials on MOSFET threshold voltage and explain the underlying physics"

Additional complexity indicators:
- Words like: "compare", "analyze", "trade-off", "optimize", "design", "both", "relationship between", "how does", "calculate and", "determine the effect", "multi-step"
- Questions requiring synthesis of multiple physics concepts
- Problems involving device design or performance optimization

Respond with ONLY one word: "simple" or "complex" """,
                },
                {
                    "role": "user",
                    "content": f"Categorize this semiconductor physics question: {query_text}",
                },
            ],
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
        )

        decision = response.choices[0].message.content.strip().lower()

        # 추가 검증 로직
        if decision not in ["simple", "complex"]:
            print(
                f"Warning: Invalid complexity decision '{decision}', applying fallback logic"
            )
            # 폴백 로직: 특정 키워드 기반 판별
            complex_indicators = [
                "compare",
                "analyze",
                "trade-off",
                "optimize",
                "design",
                "multiple",
                "both",
                "relationship between",
                "how does",
                "calculate and",
                "determine the effect",
                "multi-step",
                "synthesis",
                "integration",
                "performance",
                "efficiency",
            ]

            question_lower = query_text.lower()

            # 복잡성 점수 계산
            complexity_score = sum(
                1 for indicator in complex_indicators if indicator in question_lower
            )

            # 추가적인 복잡성 지표들
            if len(query_text.split()) > 15:  # 긴 질문은 복잡할 가능성이 높음
                complexity_score += 1
            if "?" in query_text and query_text.count("?") > 1:  # 여러 질문
                complexity_score += 1
            if any(word in question_lower for word in ["step", "process", "procedure"]):
                complexity_score += 1

            decision = "complex" if complexity_score >= 2 else "simple"

        print(f"Question complexity determined as: {decision}")
        return decision

    except Exception as e:
        print(f"Error in complexity check: {e}")
        # 에러 발생 시 안전한 기본값
        return "simple"


def extract_variables(query_text: str) -> str:
    """Extract meaningful variables and their values from the user query."""
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are an expert in semiconductor physics who specializes in extracting variables and their values from technical questions.

Extract all meaningful variables, parameters, values, and units mentioned in the question. Present them in a structured dictionary format.

Examples:
- Question: "What is the electron mobility in silicon at 300K with doping concentration of 1e16 cm^-3?"
- Variables: {"temperature": "300K", "doping_concentration": "1e16 cm^-3", "material": "silicon", "carrier_type": "electron", "property": "mobility"}

- Question: "Calculate the threshold voltage for a MOSFET with gate oxide thickness of 5nm and substrate doping of 1e15 cm^-3"
- Variables: {"device_type": "MOSFET", "gate_oxide_thickness": "5nm", "substrate_doping": "1e15 cm^-3", "property": "threshold voltage"}

Include units when specified, material properties, device types, physical conditions, and any numerical values.
If no clear variables are found, return an empty dictionary.

Respond ONLY with the dictionary format as a string.""",
            },
            {
                "role": "user",
                "content": f"Extract variables from this semiconductor physics question: {query_text}",
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )

    try:
        variables_str = response.choices[0].message.content.strip()
        # Validate that it's a reasonable dictionary-like string
        if variables_str.startswith("{") and variables_str.endswith("}"):
            return variables_str
        else:
            return "{}"
    except Exception as e:
        print(f"Error extracting variables: {e}")
        return "{}"
