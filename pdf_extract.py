"""
This script processes PDF documents and extracts text, images, and tables using 
modernized PyMuPDF features, including layout analysis and LLM-ready markdown.

Usage:
  python3 pdf_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/

Requirements:
  pip install pymupdf pymupdf4llm pymupdf-layout rapidocr-onnxruntime Pillow
"""

import os
import csv
import json
import argparse
import re
import pymupdf.layout
import pymupdf
import pymupdf4llm
from pathlib import Path

from rapidocr import RapidOCR

# Initialize RapidOCR engine once
engine = RapidOCR()


def extract_ocr_from_image_file(image_path):
    """
    Extracts text from an image file using RapidOCR.
    """
    try:
        # RapidOCR returns a RapidOCROutput object in newer versions
        output = engine(str(image_path))
        if output and output.txts:
            # output.txts is a tuple of detected text segments
            return "\n".join(output.txts)
    except Exception:
        return None
    return None


def process_ocr_content(md_text, output_path):
    """
    Finds image links in markdown, performs OCR, and returns:
    1. MD text with OCR in invisible comments after image links.
    2. TXT text with OCR in place of image links.
    """
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    md_content = md_text
    txt_content = md_text
    
    matches = re.findall(img_regex, md_text)
    for alt_text, img_rel_path in matches:
        full_path = output_path / img_rel_path
        if full_path.exists():
            print(f"Performing OCR on {img_rel_path}...")
            ocr_text = extract_ocr_from_image_file(full_path)
            
            target = f"![{alt_text}]({img_rel_path})"
            
            # Update MD content with invisible comment
            if ocr_text:
                # Escape --> to avoid breaking the HTML comment
                safe_ocr = ocr_text.replace("-->", "-- >")
                md_comment = f" <!-- OCR_TEXT: {safe_ocr} -->"
            else:
                md_comment = " <!-- OCR failed or no text detected -->"
            
            md_content = md_content.replace(target, target + md_comment, 1)
            
            # Update TXT content with visible OCR block
            replacement = f"\n\n[START_OCR_TEXT_FROM_IMAGE: {img_rel_path}]\n"
            if ocr_text:
                replacement += ocr_text
            else:
                replacement += "[No text detected or OCR failed]"
            replacement += "\n[END_OCR_TEXT_FROM_IMAGE]\n\n"
            
            txt_content = txt_content.replace(target, replacement, 1)
            
    return md_content, txt_content


def extract_from_pdf(pdf_path, output_path):
    """
    Extracts text, images, and tables from a single PDF document using 
    advanced PyMuPDF features (pymupdf4llm and layout analysis).

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path where parsed PDF output will be saved.
    """
    pdf_path = Path(pdf_path).absolute()
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = pdf_path.stem
    out_img_dir_name = f"{base_name}_images"
    out_tab_path = output_path / f"{base_name}_tables"
    out_tab_path.mkdir(exist_ok=True)

    # 1. Advanced Layout-Aware Text/Markdown Extraction with Images
    print(f"Extracting layout-aware markdown and images from: {pdf_path}")
    
    # We change directory to output_path so that pymupdf4llm saves images 
    # with relative paths in the markdown content.
    original_cwd = os.getcwd()
    try:
        os.chdir(output_path)
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path), 
            write_images=True, 
            image_path=out_img_dir_name,
            image_format="png"
        )
    finally:
        os.chdir(original_cwd)
    
    # 2. Process images for OCR (enriched MD and TXT versions)
    print("Processing images for OCR...")
    enriched_md, txt_text = process_ocr_content(md_text, output_path)

    # Save the Markdown version (contains relative image links + OCR comments)
    with open(output_path / f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(enriched_md)

    # Save the .txt version with OCR'd image text in place
    with open(output_path / f"{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(txt_text)

    # 3. Extract and save JSON structure
    doc = pymupdf.open(str(pdf_path))
    doc_json = []
    for page in doc:
        # sort=True ensures a logical reading order.
        page_json = json.loads(page.get_text("json", sort=True))
        doc_json.append(page_json)

    with open(output_path / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(doc_json, f, indent=2)

    # 4. Extract and save tables as CSV
    print("Extracting tables...")
    for i, page in enumerate(doc):
        tabs = page.find_tables()
        for tab_index, tab in enumerate(tabs):
            table_data = tab.extract()
            with open(out_tab_path / f"page_{i}_table_{tab_index}.csv", "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)

    print(f"Extraction complete. Results saved in: {output_path}")


def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Modernized PDF parser using advanced PyMuPDF features."
    )
    parser.add_argument(
        "-f", "--file_path", type=str, help="Path to the PDF document.", required=True
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Directory where to store the parsed results.",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.file_path):
        print(f"Error: PDF file not found at '{args.file_path}'")
        exit(1)

    extract_from_pdf(args.file_path, args.out_path)
