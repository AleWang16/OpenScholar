import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


AR5IV_HTML_BASE_URL = "https://ar5iv.labs.arxiv.org/html"


def remove_citations(text: str) -> str:
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", text)).replace(" |", "").replace("]", "")


def split_data_into_chunks(
    text: str,
    chunk_sz: Optional[int],
    min_chunk_sz: int,
    keep_last: bool,
) -> List[str]:
    # Mirrors retriever/src/data.py::split_data_into_chunks.
    if chunk_sz is None or chunk_sz <= 0:
        return [text]

    words = text.split()
    if not words:
        return []

    n_words = len(words) if keep_last else len(words) - len(words) % chunk_sz
    chunks = [" ".join(words[i : i + chunk_sz]) for i in range(0, n_words, chunk_sz)]

    if len(chunks) > 1 and len(chunks[-1].split(" ")) < min_chunk_sz:
        last_chunk = chunks.pop()
        chunks[-1] += " " + last_chunk

    return chunks


def extract_ar5iv_paragraphs_from_html(html_text: str) -> List[str]:
    # Mirrors src/use_search_apis.py::parsing_paragraph selectors.
    soup = BeautifulSoup(html_text, "html.parser")
    nodes = soup.find_all(class_="ltx_para", id=re.compile(r"^S\d+\.+(p|S)"))
    paragraphs: List[str] = []
    for node in nodes:
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def load_html(
    html_file: Optional[str],
    html_url: Optional[str],
    arxiv_id: Optional[str],
    timeout: int,
) -> Tuple[str, str]:
    source = ""
    if arxiv_id:
        html_url = f"{AR5IV_HTML_BASE_URL}/{arxiv_id}"

    if html_url:
        response = requests.get(html_url, timeout=timeout)
        response.raise_for_status()
        source = html_url
        return response.text, source

    if not html_file:
        raise ValueError("One of --html_file, --html_url, or --arxiv_id must be provided.")

    with open(html_file, "r", encoding="utf-8") as f:
        html_text = f.read()
    source = os.path.abspath(html_file)
    return html_text, source


def paragraphs_to_chunks(
    paragraphs: List[str],
    chunk_size: int,
    min_chunk_size: int,
    keep_last_chunk: bool,
    strip_citations: bool,
) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    global_chunk_id = 0

    for paragraph_id, paragraph in enumerate(paragraphs):
        text = remove_citations(paragraph) if strip_citations else paragraph
        paragraph_chunks = split_data_into_chunks(
            text=text,
            chunk_sz=chunk_size if chunk_size > 0 else None,
            min_chunk_sz=min_chunk_size,
            keep_last=keep_last_chunk,
        )
        for chunk_id_in_paragraph, chunk_text in enumerate(paragraph_chunks):
            if not chunk_text:
                continue
            chunks.append(
                {
                    "id": global_chunk_id,
                    "paragraph_id": paragraph_id,
                    "chunk_id_in_paragraph": chunk_id_in_paragraph,
                    "text": chunk_text,
                }
            )
            global_chunk_id += 1

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert paper HTML (ar5iv format) into passages/chunks."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--html_file", type=str, help="Path to a local HTML file")
    source_group.add_argument("--html_url", type=str, help="URL to a paper HTML page")
    source_group.add_argument("--arxiv_id", type=str, help="ArXiv id; uses ar5iv HTML endpoint")

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="Words per chunk. Set 0 to keep each paragraph as one chunk.",
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=0,
        help="If last chunk is shorter than this, merge it into the previous chunk.",
    )
    parser.add_argument(
        "--drop_last_chunk",
        action="store_true",
        help="Drop incomplete trailing chunks instead of keeping them.",
    )
    parser.add_argument(
        "--strip_citations",
        action="store_true",
        help="Remove inline [n] citation markers from text.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path. Prints JSON to stdout if omitted.",
    )
    args = parser.parse_args()

    if args.chunk_size < 0:
        raise ValueError("--chunk_size must be >= 0.")
    if args.min_chunk_size < 0:
        raise ValueError("--min_chunk_size must be >= 0.")
    if args.chunk_size == 0 and args.min_chunk_size > 0:
        raise ValueError("--min_chunk_size requires --chunk_size > 0.")

    html_text, source = load_html(
        html_file=args.html_file,
        html_url=args.html_url,
        arxiv_id=args.arxiv_id,
        timeout=args.timeout,
    )
    paragraphs = extract_ar5iv_paragraphs_from_html(html_text)
    chunks = paragraphs_to_chunks(
        paragraphs=paragraphs,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        keep_last_chunk=not args.drop_last_chunk,
        strip_citations=args.strip_citations,
    )

    payload = {
        "source": source,
        "num_paragraphs": len(paragraphs),
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
