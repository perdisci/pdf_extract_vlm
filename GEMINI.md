# PDF Extract Project Overview

This project provides Python scripts to extract text, images, and tables from PDF documents. It includes two implementation options:
1. `pdf_extract.py`: Uses PyMuPDF and Tesseract OCR.
2. `docling_extract.py`: Uses IBM's Docling for advanced document conversion and analysis.

## Technologies

- **Python 3**
- **PyMuPDF (fitz):** For PDF parsing and extraction in `pdf_extract.py`.
- **Tesseract OCR / pytesseract:** For extracting text from images in `pdf_extract.py`.
- **Pillow (PIL):** For image processing.
- **Docling:** For comprehensive document conversion (PDF to Markdown, JSON, etc.) in `docling_extract.py`.

## Building and Running

### System Requirements

For `pdf_extract.py`, Tesseract OCR and multiple language packs are required:

```bash
sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-fra tesseract-ocr-ita tesseract-ocr-por tesseract-ocr-kor tesseract-ocr-chi-sim tesseract-ocr-rus tesseract-ocr-ukr tesseract-ocr-jpn
```

### Python Dependencies

Install the required Python libraries using a virtual environment:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install pymupdf pytesseract Pillow docling
```

### Usage

Always activate the virtual environment before running the scripts.

#### Using pdf_extract.py
```bash
source venv/bin/activate
python3 pdf_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/
```

#### Using docling_extract.py
```bash
source venv/bin/activate
python3 docling_extract.py -f /path/to/input_doc.pdf -o /out/dir/path/
```

### Output Structure

Both scripts generate output in the specified directory, including:
- JSON representation of the document structure.
- Plain text / Markdown content.
- Extracted images in an `_images` subdirectory.
- Extracted tables as CSV files in a `_tables` subdirectory.

## Development Conventions

- **Argument Parsing:** Both scripts use `argparse` for a consistent CLI interface.
- **Virtual Environment:** All development and execution should occur within the `venv`.
- **Image Handling:** `pdf_extract.py` uses manual preprocessing, while `docling_extract.py` leverages Docling's built-in pipeline options for OCR and scaling.
