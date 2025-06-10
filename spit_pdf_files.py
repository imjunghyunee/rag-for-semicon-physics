import os
from PyPDF2 import PdfReader, PdfWriter

# 1. Input and Output paths
input_pdf_path = (
    "/home/jhwang/rag-for-semicon-physics/data/neamen_split_by_chapter/chapter15.pdf"
)
output_dir = "/home/jhwang/rag-for-semicon-physics/data/neamen_split_by_chapter"

# 2. Define chapter page ranges (start inclusive, end exclusive)
chapter_ranges = {
    # "chapter1": [26, 47],
    # "chapter2": [50, 78],
    # "chapter3": [83, 126],
    # "chapter4": [131, 175],
    # "chapter5": [181, 210],
    # "chapter6": [217, 259],
    # "chapter7": [266, 295],
    # "chapter8": [301, 349],
    # "chapter9": [356, 391],
    # "chapter10": [396, 459],
    # "chapter11": [468, 509],
    # "chapter12": [516, 586],
    "chapter15-1": [1, 21],
    "chapter15-2": [21, 35],
    # "chapter13": [596, 637],
    # "chapter14": [643, 690],
    # "chapter15": [695, 729],
}

# 3. Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# 4. Load the original PDF
reader = PdfReader(input_pdf_path)
total_pages = len(reader.pages)

# 5. Split and save each chapter
for chapter_name, (start_page, end_page) in chapter_ranges.items():
    writer = PdfWriter()

    # Adjust for 0-based indexing
    start_idx = start_page - 1
    end_idx = end_page - 1  # end_page is exclusive, so we exclude this page

    # Validate bounds
    if start_idx >= total_pages or start_idx < 0:
        print(f"⚠️ Skipping {chapter_name}: start_page {start_page} is out of bounds.")
        continue
    if end_idx > total_pages:
        end_idx = total_pages

    for i in range(start_idx, end_idx):
        writer.add_page(reader.pages[i])

    output_path = os.path.join(output_dir, f"{chapter_name}.pdf")
    with open(output_path, "wb") as out_file:
        writer.write(out_file)

    print(f"✅ Saved {chapter_name}: pages {start_page}-{end_page - 1} → {output_path}")

print("🎉 All chapters have been processed.")
