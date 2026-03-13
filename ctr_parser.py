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
import logging
import pymupdf.layout
import pymupdf
import pymupdf4llm
from pathlib import Path
import imagehash
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytesseract
from PIL import Image
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec
from ollama import Client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ImageAnalysisCache:
    def __init__(self, cache_file=".image_analysis_cache.json", threshold=0):
        self.cache_file = Path(cache_file)
        self.threshold = threshold
        self.cache = []  # List of dicts: {"hash": str, "results": dict, "path": str}
        self.load()

    def load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

    def save(self):
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_phash(self, image_path):
        """Computes the perceptual hash (pHash) for the image."""
        try:
            with Image.open(image_path) as img:
                return imagehash.phash(img)
        except Exception as e:
            logger.error(f"Failed to compute pHash for {image_path}: {e}")
            return None

    def find_match(self, current_hash, key):
        if current_hash is None:
            return None
        for entry in self.cache:
            stored_hash = imagehash.hex_to_hash(entry["hash"])
            distance = current_hash - stored_hash
            if distance <= self.threshold:
                return {"value": entry["results"].get(key), "distance": distance}
        return None

    def update(self, current_hash, key, value, image_path=None):
        if current_hash is None or value is None:
            return

        hash_str = str(current_hash)
        path_str = str(image_path) if image_path else "unknown"
        
        # Try to find an existing entry within the threshold to update
        for entry in self.cache:
            stored_hash = imagehash.hex_to_hash(entry["hash"])
            if (current_hash - stored_hash) <= self.threshold:
                entry["results"][key] = value
                # Optionally update path to the latest reference
                entry["path"] = path_str
                return

        # No existing entry found, add a new one
        self.cache.append({
            "hash": hash_str, 
            "results": {key: value},
            "path": path_str
        })


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
            self.engines[lang] = RapidOCR(
                params={"Rec.lang_type": lang, "Global.use_cls": True}
            )
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


def load_prompts(config_file="prompts.json"):
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load prompts from {config_file}. Error: {e}")
        return {}


def extract_ocr_from_image_file(image_path, hybrid_ocr):
    """
    Extracts text from an image file using Hybrid OCR (Tesseract + RapidOCR).
    """
    return hybrid_ocr.extract_text(image_path)


def extract_ollama_from_image_file(image_path, model, host, timeout, prompt_simple, retries=2):
    """
    Extracts description and text from an image file using Ollama with retries.
    """
    client = Client(host=host, timeout=timeout)

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
                logger.warning(
                    f"Ollama query failed for {image_path.name} (attempt {attempt + 1}/{retries + 1}): {e}. Retrying in 2s..."
                )
                time.sleep(2)
            else:
                logger.error(
                    f"Ollama query failed after {retries + 1} attempts for {image_path.name}: {e}"
                )
                return None


def get_image_links_from_markdown(md_text):
    """Extracts all image links from markdown text."""
    img_regex = r"!\[(.*?)\]\((.*?)\)"
    return re.findall(img_regex, md_text)

def process_image_text_for_md(md_text, output_path, image_text_map, mode="ocr"):
    """
    Finds image links in markdown and appends an invisible comment with extracted text.
    """
    md_content = md_text
    label = "OCR" if mode == "ocr" else "Ollama"

    for alt_text, img_rel_path in get_image_links_from_markdown(md_text):
        text = image_text_map.get(img_rel_path)
        if text:
            # Escape -- to avoid breaking the comment
            safe_text = text.replace("--", "- -")
            comment = f"\n<!-- Image Description ({label}): {safe_text} -->\n"
            target = f"![{alt_text}]({img_rel_path})"
            md_content = md_content.replace(target, target + comment, 1)

    return md_content


def process_image_text_for_txt(md_text, output_path, image_text_map, mode="ocr"):
    """
    Finds image links in markdown and replaces them with extracted text for the .txt version.
    """
    txt_content = md_text
    label = "OCR" if mode == "ocr" else "OLLAMA"

    for alt_text, img_rel_path in get_image_links_from_markdown(md_text):
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
    hybrid_ocr,
    image_cache,
    mode="ocr",
    model=None,
    ollama_host="http://localhost:11434",
    ollama_timeout=300,
    ollama_retries=2,
    max_threads=1,
):
    """
    Extracts text, images, and tables from a single PDF document using
    advanced PyMuPDF features (pymupdf4llm and layout analysis).

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path where parsed PDF output will be saved.
        hybrid_ocr: Instance of HybridOCR.
        image_cache: Instance of ImageAnalysisCache.
        mode: Image text extraction mode ('ocr' or 'ollama').
        model: Ollama model name (if mode is 'ollama').
        ollama_host: Ollama host URL.
        ollama_timeout: Ollama API timeout in seconds.
        ollama_retries: Number of Ollama API retries on failure.
        max_threads: Max threads for concurrent image processing.
    """
    total_start_time = time.time()
    pdf_path = Path(pdf_path).absolute()
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = pdf_path.stem
    out_img_dir_name = f"{base_name}_images"
    out_tab_path = output_path / f"{base_name}_tables"
    out_tab_path.mkdir(exist_ok=True)

    # 1. Advanced Layout-Aware Text/Markdown Extraction with Images
    logger.info(f"Extracting layout-aware markdown and images from: {pdf_path}")

    initial_parsing_start = time.time()
    # We change directory to output_path so that pymupdf4llm saves images
    # with relative paths in the markdown content.
    original_cwd = os.getcwd()
    try:
        os.chdir(output_path)
        md_text = pymupdf4llm.to_markdown(
            str(pdf_path),
            write_images=True,
            image_path=out_img_dir_name,
            image_format="png",
        )
    finally:
        os.chdir(original_cwd)
    
    initial_parsing_duration = time.time() - initial_parsing_start
    logger.info(f"Initial layout-aware parsing completed in {initial_parsing_duration:.2f}s")

    # Pre-calculate OCR or Ollama analysis for all images to avoid redundant processing
    image_text_map = {}
    matches = get_image_links_from_markdown(md_text)
    
    prompts = load_prompts()
    ollama_prompt = prompts.get("ollama_image_analysis", "")

    unique_image_paths = list(set([img_rel_path for _, img_rel_path in matches]))
    tasks_to_run = {}
    
    # Pre-process cache checks sequentially to deduplicate identical unseen images
    for img_rel_path in unique_image_paths:
        full_path = output_path / img_rel_path
        if not full_path.exists():
            continue
            
        phash = image_cache.get_phash(full_path)
        if phash is None:
            continue
            
        cached_result = image_cache.find_match(phash, mode)
        if cached_result:
            logger.info(f"Cache hit for {img_rel_path} (mode: {mode}, distance: {cached_result['distance']})")
            image_text_map[img_rel_path] = cached_result["value"]
        else:
            phash_str = str(phash)
            if phash_str not in tasks_to_run:
                tasks_to_run[phash_str] = {'paths': [], 'full_path': full_path, 'phash': phash}
            tasks_to_run[phash_str]['paths'].append(img_rel_path)

    def run_extraction(full_path):
        if mode == "ollama":
            logger.info(f"Querying Ollama ({model}) for {full_path.name}...")
            start_time = time.time()
            result = extract_ollama_from_image_file(
                full_path, model, ollama_host, ollama_timeout, ollama_prompt, ollama_retries
            )
            elapsed_time = time.time() - start_time
            if result is not None:
                logger.info(f"Ollama query successful for {full_path.name}; done in {elapsed_time:.2f}s")
                return result, "ollama"
            else:
                # Fallback to OCR if Ollama failed after all retries
                logger.warning(f"Ollama failed for {full_path.name}. Falling back to OCR...")
                result = extract_ocr_from_image_file(full_path, hybrid_ocr)
                return result, "ocr"
        else:
            logger.info(f"Performing OCR on {full_path.name}...")
            result = extract_ocr_from_image_file(full_path, hybrid_ocr)
            return result, "ocr"

    updates_made = False
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(run_extraction, task['full_path']): phash_str for phash_str, task in tasks_to_run.items()}
        for future in as_completed(futures):
            phash_str = futures[future]
            task = tasks_to_run[phash_str]
            try:
                result, update_mode = future.result()
                if result is not None:
                    for img_rel_path in task['paths']:
                        image_text_map[img_rel_path] = result
                if update_mode:
                    image_cache.update(task['phash'], update_mode, result, task['full_path'])
                    updates_made = True
            except Exception as exc:
                logger.error(f"Image processing generated an exception for phash {phash_str}: {exc}")

    # Batch cache writes
    if updates_made:
        image_cache.save()

    # Save the Markdown version with invisible comments
    md_with_text = process_image_text_for_md(md_text, output_path, image_text_map, mode)
    with open(output_path / f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(md_with_text)

    # Save the .txt version with extracted image text in place
    logger.info("Generating the text version...")
    txt_text = process_image_text_for_txt(md_text, output_path, image_text_map, mode)
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
    logger.info("Extracting tables...")
    for i, page in enumerate(doc):
        tabs = page.find_tables()
        for tab_index, tab in enumerate(tabs):
            table_data = tab.extract()
            with open(
                out_tab_path / f"page_{i}_table_{tab_index}.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)

    total_duration = time.time() - total_start_time
    logger.info(f"Extraction complete. Results saved in: {output_path}")
    logger.info(f"Total processing time: {total_duration:.2f}s")


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
    parser.add_argument(
        "--phash-th",
        type=int,
        default=5,
        help="Perceptual hash distance threshold for image matching (default: 5).",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=1,
        help="Maximum number of threads for image processing (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.file_path):
        logger.error(f"PDF file not found at '{args.file_path}'")
        exit(1)

    # Initialize dependencies
    hybrid_ocr_engine = HybridOCR()
    image_analysis_cache = ImageAnalysisCache()
    image_analysis_cache.threshold = args.phash_th

    extract_from_pdf(
        args.file_path,
        args.out_path,
        hybrid_ocr_engine,
        image_analysis_cache,
        args.mode,
        args.model,
        args.ollama_host,
        args.ollama_timeout,
        args.ollama_retries,
        args.max_threads,
    )
