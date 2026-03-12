"""
This script processes PDF documents and extracts text, images, and tables using 
modernized PyMuPDF features, including layout analysis and LLM-ready markdown.

Usage:
  python3 pdf_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/

Requirements:
  pip install pymupdf pymupdf4llm pymupdf-layout rapidocr-onnxruntime pytesseract Pillow ollama
"""

import os
import csv
import json
import argparse
import re
import time
import pymupdf.layout
import pymupdf
import pymupdf4llm
from pathlib import Path

import pytesseract
from PIL import Image
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec
from ollama import Client

class HybridOCR:
    def __init__(self):
        self.engines = {}
        # Mapping from Tesseract Script to RapidOCR rec_lang enum
        self.script_to_lang = {
            "Latin": LangRec.EN,
            "Cyrillic": LangRec.CYRILLIC,
            "Han": LangRec.CH,
            "Japanese": LangRec.JAPAN,
            "Korean": LangRec.KOREAN,
            "Arabic": LangRec.ARABIC,
            "Devanagari": LangRec.DEVANAGARI,
        }

    def get_engine(self, lang=LangRec.CH):
        if lang not in self.engines:
            # rec_lang is passed via nested key "Rec.lang_type"
            # use_cls is under "Global.use_cls"
            self.engines[lang] = RapidOCR(params={
                "Rec.lang_type": lang, 
                "Global.use_cls": True
            })
        return self.engines[lang]

    def extract_text(self, image_path):
        try:
            # Phase 1: Routing & Orientation Detection
            osd = pytesseract.image_to_osd(str(image_path))
            
            # Parse OSD output
            osd_data = {}
            for line in osd.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    osd_data[key.strip()] = value.strip()
            
            script = osd_data.get("Script", "Latin")
            rotate = int(osd_data.get("Rotate", 0))
            
            # Phase 2: Logic - Map Script to RapidOCR model
            rec_lang = self.script_to_lang.get(script, LangRec.EN)
            
            # Phase 3: Extraction
            engine = self.get_engine(rec_lang)
            
            img = Image.open(image_path)
            if rotate != 0:
                img = img.rotate(-rotate, expand=True)
            
            output = engine(img)
            if output and output.txts:
                return "\n".join(output.txts)
        except Exception:
            # Fallback to default engine if OSD fails
            try:
                engine = self.get_engine(LangRec.CH)
                output = engine(str(image_path))
                if output and output.txts:
                    return "\n".join(output.txts)
            except:
                return None
        return None

# Initialize HybridOCR engine once
hybrid_ocr = HybridOCR()


def extract_ocr_from_image_file(image_path):
    """
    Extracts text from an image file using Hybrid OCR (Tesseract + RapidOCR).
    """
    return hybrid_ocr.extract_text(image_path)


def extract_ollama_from_image_file(image_path, model, host, timeout, retries=2):
    """
    Extracts description and text from an image file using Ollama with retries.
    """
    client = Client(host=host, timeout=timeout)

    prompt_simple = """
        **Role**: You are a Senior Cyber Threat Intelligence (CTI) Analyst.

        Generate a detailed description of the content in the provided image extracted from a CTI technical report.

        Explain the image context and content in detail.

        If there is any text embedded in the image, extract the text and translate it if it is not in English.

        If code is present, such as in a code snippet or code analysis window, extract all of the code text in detail, verbatim.

        If the image contains a system overview graph or flowchart, describe the directional flow of data/attacks.

        **Format Constraints**: Start the response with a line containing `### Image Category: <label>`, where <label> must be chosen among the following: Company Logo, Application Screeshot, Web Page, Code Snippet, Code Analysis, Traffic Analysis, C2 Infrastructure, Phishing Message, Text Document, Flow Chart, System Diagram, Data Table, or Unknown.
        """

    for attempt in range(retries + 1):
        try:
            response = client.generate(
                model=model,
                prompt=prompt_simple,
                images=[image_path],
                stream=False,
                options={
                    "temperature": 0.1,
                },
            )
            return response["response"]
        except Exception as e:
            if attempt < retries:
                print(
                    f"\nWarning: Ollama query failed for {image_path.name} (attempt {attempt + 1}/{retries + 1}): {e}. Retrying in 2s..."
                )
                time.sleep(2)
            else:
                print(
                    f"\nError: Ollama query failed after {retries + 1} attempts for {image_path.name}: {e}"
                )
                return None


def process_image_text_for_md(md_text, output_path, image_text_map, mode="ocr"):
    """
    Finds image links in markdown and appends an invisible comment with extracted text.
    """
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    md_content = md_text
    label = "OCR" if mode == "ocr" else "Ollama"

    matches = re.findall(img_regex, md_text)
    for alt_text, img_rel_path in matches:
        text = image_text_map.get(img_rel_path)
        if text:
            # Escape -- to avoid breaking the comment
            safe_text = text.replace("--", "- -")
            comment = f"\n<!-- {label}: {safe_text} -->\n"
            target = f"![{alt_text}]({img_rel_path})"
            md_content = md_content.replace(target, target + comment, 1)

    return md_content


def process_image_text_for_txt(md_text, output_path, image_text_map, mode="ocr"):
    """
    Finds image links in markdown and replaces them with extracted text for the .txt version.
    """
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    txt_content = md_text
    label = "OCR" if mode == "ocr" else "OLLAMA"

    matches = re.findall(img_regex, md_text)
    for alt_text, img_rel_path in matches:
        text = image_text_map.get(img_rel_path)

        replacement = f"\n\n[START_{label}_TEXT_FROM_IMAGE: {img_rel_path}]\n"
        if text:
            replacement += text
        else:
            replacement += f"[No text detected or {label} failed]"
        replacement += f"\n[END_{label}_TEXT_FROM_IMAGE]\n\n"

        target = f"![{alt_text}]({img_rel_path})"
        txt_content = txt_content.replace(target, replacement, 1)

    return txt_content


def extract_from_pdf(
    pdf_path,
    output_path,
    mode="ocr",
    model=None,
    ollama_host="http://localhost:11434",
    ollama_timeout=300,
    ollama_retries=2,
):
    """
    Extracts text, images, and tables from a single PDF document using 
    advanced PyMuPDF features (pymupdf4llm and layout analysis).

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path where parsed PDF output will be saved.
        mode: Image text extraction mode ('ocr' or 'ollama').
        model: Ollama model name (if mode is 'ollama').
        ollama_host: Ollama host URL.
        ollama_timeout: Ollama API timeout in seconds.
        ollama_retries: Number of Ollama API retries on failure.
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
    
    # Pre-calculate OCR or Ollama analysis for all images to avoid redundant processing
    image_text_map = {}
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    matches = re.findall(img_regex, md_text)
    for _, img_rel_path in matches:
        if img_rel_path not in image_text_map:
            full_path = output_path / img_rel_path
            if full_path.exists():
                if mode == "ollama":
                    print(
                        f"Querying Ollama ({model}) for {img_rel_path}...",
                        end=" ",
                        flush=True,
                    )
                    start_time = time.time()
                    image_text_map[img_rel_path] = extract_ollama_from_image_file(
                        full_path, model, ollama_host, ollama_timeout, ollama_retries
                    )
                    elapsed_time = time.time() - start_time
                    print(f"done in {elapsed_time:.2f}s")
                else:
                    print(f"Performing OCR on {img_rel_path}...")
                    image_text_map[img_rel_path] = extract_ocr_from_image_file(
                        full_path
                    )

    # Save the Markdown version with invisible comments
    md_with_text = process_image_text_for_md(
        md_text, output_path, image_text_map, mode
    )
    with open(output_path / f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(md_with_text)

    # Save the .txt version with extracted image text in place
    print("Generating the text version...")
    txt_text = process_image_text_for_txt(
        md_text, output_path, image_text_map, mode
    )
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
    parser.add_argument(
        "--mode",
        choices=["ocr", "ollama"],
        default="ocr",
        help="Image text extraction mode: 'ocr' (default) or 'ollama'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl:32b",
        help="Ollama model name (default: qwen3-vl:32b).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=300,
        help="Ollama API timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--ollama-retries",
        type=int,
        default=2,
        help="Number of Ollama API retries on failure (default: 2).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.file_path):
        print(f"Error: PDF file not found at '{args.file_path}'")
        exit(1)

    extract_from_pdf(
        args.file_path,
        args.out_path,
        args.mode,
        args.model,
        args.ollama_host,
        args.ollama_timeout,
        args.ollama_retries,
    )
