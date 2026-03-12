"""
This script queries a local or remote Ollama model with an image and a prompt.
It uses the official 'ollama' Python library for efficient communication and
handling of multimodal data.

Usage:
    source venv/bin/activate
    python3 tests/ollama_prompt_multimodal.py --image /path/to/image.png --model gemma3:27b-it-fp16
"""

import argparse
import os
import sys
from pathlib import Path
from ollama import Client


def query_ollama_multimodal(image_path, model_name, host="http://localhost:11434"):
    """
    Sends a detailed multimodal query to Ollama.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    # Initialize the client (defaults to localhost)
    client = Client(host=host)

    prompt_short = """
        **Role**: You are a Senior Cyber Threat Intelligence (CTI) Analyst.

        Generate a detailed description of the content in the provided image extracted from a CTI technical report.

        Explain the image context and content in detail.

        If there is any text embedded in the image, extract the text and translate it if it is not in English.
        """

    prompt_simple = """
        **Role**: You are a Senior Cyber Threat Intelligence (CTI) Analyst.

        Generate a detailed description of the content in the provided image extracted from a CTI technical report.

        Explain the image context and content in detail.

        If there is any text embedded in the image, extract the text and translate it if it is not in English.

        If code is present, such as in a code snippet or code analysis window, extract all of the code text in detail, verbatim.

        If the image contains a system overview graph or flowchart, describe the directional flow of data/attacks.

        **Format Constraints**: Start the response with a line containing `### Image Category: <label>`, where <label> must be chosen among the following: Company Logo, Application Screeshot, Web Page, Code Snippet, Code Analysis, Traffic Analysis, C2 Infrastructure, Phishing Message, Text Document, Flow Chart, System Diagram, Data Table, or Unknown.
        """

    prompt_detailed = """
        **Role**: You are a Senior Cyber Threat Intelligence (CTI) Analyst.

        **Task**: Analyze the provided image from a threat report and generate a high-fidelity Markdown technical summary.

        **Output Requirements**:

        **Header**: Start the response with `### Image Category: <label>`, where label must be chosen among the following: Company Logo, OS Screeshot, Web Page, Code Snippet, Code Analysis, Traffic Analysis, C2 Infrastructure, Phishing Message, Text Document, Flow Chart, System Diagram, Data Table, or Unknown. Avoid generic labels like "Image" unless absolutely necessary.

        **Technical Description**: Provide a factual analysis of the image content. Identify entities (IP addresses, hostnames, file paths, registry keys, etc.) and their relationships. Use bolding for technical indicators.

        **Plain Text Extraction and Translation**: Extract all plain text verbatim. If the text is non-English, provide the original text followed by an English translation in a blockquote.

        **System Graphs and Flowcharts**: If the image contains a system overview graph or flowchart, describe the directional flow of data/attacks.

        **Code Text Extraction**: If code is present, such as in a code snippet or code analysis window, extract all of the code text in detail, verbatim, and report it into a syntax-highlighted code block, when possible. Follow the code text extraction with a "Functional Impact" section describing what the code likely does (e.g., "Persistence via registry modification", "String deobfuscation", "Privilege escalation exploit", etc.). Do NOT generate any new code. Use your OCR capabilities to only report exactly the code text that is present in the image itslef.

        **Strict Verbatim Code Rule**: You are a high-precision OCR engine. Do not 'fix', complete, or improve the code. If a line of code is partially cut off at the edge of the image, represent it exactly as it appears (e.g., use ... for obscured text) rather than guessing the intended syntax. Your goal is character-level fidelity to the image source.

        Extract the code exactly as it appears in the image. 
        - Do NOT add comments that aren't in the image.
        - Do NOT complete brackets or functions if they are cut off.
        - If a character is ambiguous (e.g., '0' vs 'O'), provide your best guess but do not invent new logic.
        - based ONLY on the text you just transcribed, describe the functionality.

        **Constraints**:
        - Refer to the source as "the image".
        - NO introductory fluff ("In this image...", "Sure, I can help") or follow-up questions ("Whould you like me to...").
        - Use exclusively standard Markdown formatting (headings, tables for data, lists for steps).
        """

    print(f"Querying model '{model_name}' with image: {image_path.name}...")

    try:
        # # The ollama library handles file reading and base64 encoding internally
        # # when a path or bytes are passed to the 'images' parameter.
        # response = client.chat(
        #     model=model_name,
        #     messages=[
        #         {
        #             'role': 'user',
        #             'content': prompt,
        #             'images': [str(image_path)],
        #         },
        #     ],
        #     stream=False
        # )

        # Use 'generate' for single-shot precision
        response = client.generate(
            model=model_name,
            prompt=prompt_simple,
            images=[image_path],
            stream=False,
            options={
                "temperature": 0.1,  # Critical for OCR accuracy
                # 'top_p': 0.1,        # Limits character 'guessing'
                # 'seed': 42,          # Ensures reproducible results
                # 'num_ctx': 4096      # Larger context for dense reports
            },
        )

        print("\n--- Response ---")
        # print(response['message']['content'])
        print(response["response"])
        print("----------------\n")

    except Exception as e:
        print(f"An error occurred while querying Ollama: {e}")
        sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Query Ollama with a multimodal prompt."
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to the image file."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gemma3:27b-it-fp16",
        help="Name of the Ollama model to use (default: gemma3:27b-it-fp16).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    query_ollama_multimodal(args.image, args.model, args.host)
