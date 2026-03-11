"""
This script processes PDF documents and extracts text, images, and tables using Docling.

Usage:
  python3 docling_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/

Requirements:
  pip install docling rapidocr-onnxruntime pytesseract Pillow pandas
"""

import os
import argparse
import json
import re
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

import pytesseract
from PIL import Image
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec

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
            
            # Load and rotate image if Tesseract detects orientation
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


def process_ocr_for_md(md_text, output_path, ocr_map):
    """
    Finds image links in markdown and appends an invisible comment with OCR text.
    """
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    md_content = md_text
    
    matches = re.findall(img_regex, md_text)
    for alt_text, img_rel_path in matches:
        ocr_text = ocr_map.get(img_rel_path)
        if ocr_text:
            # Escape --> to avoid breaking the comment
            safe_ocr_text = ocr_text.replace("-->", "-- >")
            comment = f" <!-- OCR: {safe_ocr_text} -->"
            target = f"![{alt_text}]({img_rel_path})"
            md_content = md_content.replace(target, target + comment, 1)
                
    return md_content


def process_ocr_for_txt(md_text, output_path, ocr_map):
    """
    Finds image links in markdown and replaces them with OCR text for the .txt version.
    """
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    txt_content = md_text
    
    matches = re.findall(img_regex, md_text)
    for alt_text, img_rel_path in matches:
        ocr_text = ocr_map.get(img_rel_path)
        
        replacement = f"\n\n[START_OCR_TEXT_FROM_IMAGE: {img_rel_path}]\n"
        if ocr_text:
            replacement += ocr_text
        else:
            replacement += "[No text detected or OCR failed]"
        replacement += "\n[END_OCR_TEXT_FROM_IMAGE]\n\n"
        
        target = f"![{alt_text}]({img_rel_path})"
        txt_content = txt_content.replace(target, replacement, 1)
            
    return txt_content


def extract_from_pdf(pdf_path, output_path):
    """
    Extracts text, images, and tables from a single PDF document using Docling.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path where parsed PDF output will be saved.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure pipeline options for Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR for images
    pipeline_options.do_table_structure = True  # Enable table structure extraction
    pipeline_options.images_scale = 2.0  # Scale images for better quality
    pipeline_options.generate_picture_images = True # Required for image extraction in v2

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    result = converter.convert(pdf_path)
    doc = result.document

    base_name = pdf_path.stem
    out_img_path = output_path / f"{base_name}_images"
    out_img_path.mkdir(exist_ok=True)

    # 1. Manually save images and update URIs so the exporter can link them
    # Note: doc.pictures is a convenient way to access all PictureItems
    for i, picture in enumerate(doc.pictures):
        pil_img = picture.get_image(doc)
        img_filename = f"image_{i}.png"
        pil_img.save(out_img_path / img_filename)
        
        # Set the URI for the referenced image mode
        if picture.image:
            picture.image.uri = f"{base_name}_images/{img_filename}"

    # 2. Save as JSON
    with open(output_path / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, indent=2)

    # 3. Save as Markdown (Docling's primary output) with image references
    # Since we set the URIs manually, REFERENCED mode will use them.
    markdown_content = doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)

    # Pre-calculate OCR for all images to avoid redundant processing
    ocr_map = {}
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    matches = re.findall(img_regex, markdown_content)
    for _, img_rel_path in matches:
        if img_rel_path not in ocr_map:
            full_path = output_path / img_rel_path
            if full_path.exists():
                print(f"Performing OCR on {img_rel_path}...")
                ocr_map[img_rel_path] = extract_ocr_from_image_file(full_path)

    # Save the Markdown version with invisible OCR comments
    md_with_ocr = process_ocr_for_md(markdown_content, output_path, ocr_map)
    with open(output_path / f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(md_with_ocr)

    # 4. Save as Plain Text with OCR'd image text in place
    print("Generating the text version...")
    txt_text = process_ocr_for_txt(markdown_content, output_path, ocr_map)
    with open(output_path / f"{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(txt_text)

    # 5. Extract and save tables as CSV
    out_tab_path = output_path / f"{base_name}_tables"
    out_tab_path.mkdir(exist_ok=True)
    
    for i, table_element in enumerate(doc.tables):
        # Docling can export tables to pandas dataframes
        df = table_element.export_to_dataframe()
        csv_filename = f"table_{i}.csv"
        df.to_csv(out_tab_path / csv_filename, index=False)

    print(f"Extraction complete. Results saved in: {output_path}")


def parse_arguments():
    """Parses command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Parse PDF document and extract text, images, and tables using Docling."
    )
    parser.add_argument(
        "-f", "--file_path", type=str, help="Path to the PDF document.", required=True
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Directory where to store the parsed document and extracted images.",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.file_path):
        print(f"Error: PDF file not found at '{args.file_path}'")
        exit(1)

    extract_from_pdf(args.file_path, args.out_path)
