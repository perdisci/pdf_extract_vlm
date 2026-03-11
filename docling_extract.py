"""
This script processes PDF documents and extracts text, images, and tables using Docling.

Usage:
  python3 docling_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/

Requirements:
  pip install docling
"""

import os
import argparse
import json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions


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
    pipeline_options.generate_page_images = True # Generate page images if needed

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    result = converter.convert(pdf_path)
    doc = result.document

    base_name = pdf_path.stem
    
    # 1. Save as JSON
    with open(output_path / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, indent=2)

    # 2. Save as Markdown (Docling's primary output)
    with open(output_path / f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())

    # 3. Save as Plain Text
    with open(output_path / f"{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown()) # Markdown is a good proxy for text with structure

    # 4. Extract and save images
    out_img_path = output_path / f"{base_name}_images"
    out_img_path.mkdir(exist_ok=True)
    
    for i, image_element in enumerate(doc.images):
        # Docling provides images as PIL Image objects
        img = image_element.image
        img_filename = f"image_{i}.png"
        img.save(out_img_path / img_filename)

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
