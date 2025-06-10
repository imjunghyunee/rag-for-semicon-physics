import os
import glob
from llama_parse import LlamaParse
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 입력 및 출력 디렉토리 설정
input_dir = "/home/jhwang/rag-for-semicon-physics/text"
output_dir = "/home/jhwang/rag-for-semicon-physics/data/text_extracted_md"

# 출력 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(output_dir, exist_ok=True)

# parsing instruction 을 지정합니다.
parsing_instruction = """You are parsing a semiconductor physics textbook. Please accurately extract all content including: 
    mathematical equations, tables, figures descriptions and technical terminology.
    Preserve the structure and formatting, especially for tables in markdown format.
    All the equations should be in LaTeX format, and tables should be formatted as markdown tables."""

# LlamaParse 설정 (멀티모달 모델 사용)
parser = LlamaParse(
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    vendor_multimodal_api_key=os.environ["OPENAI_API_KEY"],
    result_type="markdown",  # markdown 형식으로 결과 저장
    language="en",  # 반도체 물리학 교재는 영어로 가정
    system_prompt_append=parsing_instruction,
    verbose=True,
)

# PDF 파일 목록 가져오기
pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
pdf_files.sort()  # 파일명 순으로 정렬

print(f"찾은 PDF 파일 수: {len(pdf_files)}")

# 각 PDF 파일 처리
for pdf_file in pdf_files:
    try:
        print(f"\n처리 중: {os.path.basename(pdf_file)}")

        # PDF 파싱
        documents = parser.load_data(file_path=pdf_file)

        # 파일명에서 확장자 제거
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]

        # 각 페이지별로 별도 파일로 저장
        for i, doc in enumerate(documents):
            # 페이지 번호를 포함한 파일명 생성 (0부터 시작하므로 +1)
            output_file = os.path.join(output_dir, f"{base_name}_page_{i+1:03d}.md")

            # 마크다운 파일로 저장
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(doc.text)

            print(f"저장 완료: {output_file}")

        print(f"총 페이지 수: {len(documents)}")

    except Exception as e:
        print(f"오류 발생 - {os.path.basename(pdf_file)}: {str(e)}")
        continue

print("\n모든 파일 처리 완료!")
