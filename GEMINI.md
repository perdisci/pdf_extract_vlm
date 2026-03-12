# PDF Extract Project Overview

This project provides Python scripts to extract text, images, and tables from PDF documents. It includes two implementation options:
1. `pdf_extract.py`: Modernized script using PyMuPDF, `pymupdf4llm`, and `pymupdf-layout` for advanced layout analysis, with a **Hybrid OCR** pipeline (Tesseract OSD + RapidOCR) for image text extraction.
2. `docling_extract.py`: Uses IBM's Docling for advanced document conversion and analysis, also integrated with the **Hybrid OCR** pipeline for consistent image text extraction.

## Technologies

- **Python 3**
- **PyMuPDF (fitz):** For PDF parsing and extraction.
- **pymupdf4llm:** For layout-aware Markdown extraction in `pdf_extract.py`.
- **pymupdf-layout:** Integrated layout analysis for PyMuPDF.
- **Hybrid OCR Pipeline:** Combines **Tesseract** (for script and orientation detection) and **RapidOCR** (for high-fidelity text extraction) to handle multi-language documents and rotated images effectively.
- **Pillow (PIL):** For image processing.
- **Docling:** For comprehensive document conversion in `docling_extract.py`.
- **Pandas:** For table processing in `docling_extract.py`.

## Building and Running

### System Requirements

Both scripts use `RapidOCR` which typically does not require system-level Tesseract installations. However, for specific Docling OCR features or other Tesseract-based tasks, the following may still be useful:

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
pip install pymupdf pymupdf4llm pymupdf-layout pytesseract Pillow docling ollama
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

### Testing & Extra Tools

#### Ollama Multimodal Query
A test script is available to query an Ollama model with extracted images for further analysis (e.g., detail explanation, text extraction, and translation):

```bash
source venv/bin/activate
python3 tests/ollama_prompt_multimodal.py --image /path/to/extracted/image.png --model gemma3:27b-it-fp16
```

### Output Structure


Both scripts generate output in the specified directory, including:
- JSON representation of the document structure.
- Markdown and Plain text content.
- Extracted images in an `_images` subdirectory.
- Extracted tables as CSV files in a `_tables` subdirectory.

## Development Conventions

- **Argument Parsing:** Both scripts use `argparse` for a consistent CLI interface.
- **Virtual Environment:** All development and execution should occur within the `venv`.
- **Modernized Extraction:** `pdf_extract.py` now uses `pymupdf4llm` to produce high-quality, layout-aware Markdown, making it ideal for LLM processing.
