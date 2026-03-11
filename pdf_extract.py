"""
This script processes PDF documents and extracts images.

Usage:
  python3 pdf_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/

Requirements:

  Install Tesseract and multi-language support:
    
    sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-fra tesseract-ocr-ita tesseract-ocr-por tesseract-ocr-kor tesseract-ocr-chi-sim tesseract-ocr-rus tesseract-ocr-ukr tesseract-ocr-jpn
"""

import os
import csv
import json
import argparse

import pymupdf

import io
import base64
import pytesseract
from PIL import Image
from PIL import ImageEnhance


def extract_text_from_image(img_base64):
    """
    Extracts text from an image (provided as base64) using OCR.

    Args:
        img_base64: The image data as a base64 encoded string.

    Returns:
        The extracted text as a string, or None if an error occurs or no text is found.
    """

    try:
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert(
            "RGB"
        )  # Ensure RGB for pytesseract

        # Optional: Preprocessing can significantly improve OCR accuracy
        # Example: Convert to grayscale
        image = image.convert("L")  # Convert to grayscale

        # Example: Enhance contrast (more examples below)
        from PIL import ImageEnhance

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)  # Increase contrast

        text = pytesseract.image_to_string(
            image, lang="eng+spa+por+rus+ita+fra+ukr+chi_sim+kor+jpn"
        )
        if text:
            return text.strip()  # Remove leading/trailing whitespace

    except Exception as e:
        # print(f"Error during OCR: {e}")
        return None

    return None


def doc_json_to_txt(doc_json):
    """
    Converts the JSON data from a PDF document to a plain text string. This includes converting images to text via Tesseract OCR.
    """
    doc_text = ""
    for page in doc_json:
        if not "blocks" in page:
            continue
        for block in page["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if not "spans" in line:
                        continue
                    for span in line["spans"]:
                        if "text" in span and len(span["text"]) > 0:
                            doc_text += span["text"] + " "
                    doc_text += "\n"
                doc_text += "\n"
            elif "image" in block and "ext" in block:
                img_base64 = block["image"]
                img_type = block["ext"]
                img_text = extract_text_from_image(img_base64)
                if img_text:
                    doc_text += "\n\n[START_TEXT_EXTRACTED_FROM_IMAGE]\n\n"
                    doc_text += img_text
                    doc_text += "\n\n[END_TEXT_EXTRACTED_FROM_IMAGE]\n\n"
        doc_text += "\n"
    return doc_text


def extract_from_pdf(pdf_path, output_path):
    """
    Extracts text, images, and tables from a single PDF document using PyMuPDF.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path where parsed PDF output will be saved.

    Saves:
        - Text (plus in-place image OCR)
        - Images
        - Tables
    """

    doc = pymupdf.open(pdf_path)

    # Extract and save text
    doc_json = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_json = json.loads(page.get_text("json"))  # Get json text and images
        doc_json.append(page_json)

    base_name = os.path.basename(pdf_path)
    file_name, ext = os.path.splitext(base_name)
    with open(
        os.path.join(output_path, file_name + ".json"), "w", encoding="utf-8"
    ) as f:
        json.dump(doc_json, f)

    # Create directories for images and tables
    out_img_path = os.path.join(output_path, file_name + "_images")
    out_tab_path = os.path.join(output_path, file_name + "_tables")
    os.makedirs(out_img_path, exist_ok=True)
    os.makedirs(out_tab_path, exist_ok=True)

    # Extract and save images
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img in doc.get_page_images(page_num):
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n - 1:  # RGB
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            pix.save(os.path.join(out_img_path, f"image_{page_num}_{xref}.png"))

    # Extract and save tables
    for page_num in range(len(doc)):
        page = doc[page_num]
        tables = page.find_tables()
        for i, table in enumerate(tables):
            table_data = table.extract()
            with open(
                os.path.join(out_tab_path, f"table_{page_num}_{i}.csv"), "w", newline=""
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)

    doc_text = doc_json_to_txt(doc_json)
    with open(
        os.path.join(output_path, file_name + ".txt"), "w", encoding="utf-8"
    ) as f:
        f.write(doc_text)

    return doc_text


def parse_arguments():
    """Parses command-line arguments using argparse."""

    parser = argparse.ArgumentParser(
        description="Parse PDF document and extract text and images."
    )

    # Required arguments
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
    args = parser.parse_args()

    # Perform some basic validation (optional, but recommended)
    if not os.path.exists(args.file_path):
        parser.error(f"Error: PDF file not found at '{args.file_path}'")

    if not os.path.isdir(args.out_path):  # Check if it is a directory
        try:
            os.makedirs(args.out_path, exist_ok=True)  # Try to create it
        except OSError as e:
            parser.error(
                f"Error: Output path '{args.out_path}' is not a valid directory or could not be created: {e}"
            )

    return args


if __name__ == "__main__":

    args = parse_arguments()
    doc_text = extract_from_pdf(args.file_path, args.out_path)
    # print(doc_text)
