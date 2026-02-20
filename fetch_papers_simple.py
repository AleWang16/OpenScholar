import argparse
import json
import os
import re
from typing import Any, Dict, List
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,year,abstract,url,authors.name,citationCount,externalIds"


def search_semantic_scholar(
    query: str, api_key: str, limit: int, min_citation_count: int, timeout: int
) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "limit": limit,
        "minCitationCount": min_citation_count,
        "sort": "citationCount:desc",
        "fields": S2_FIELDS,
    }
    headers = {"x-api-key": api_key}
    response = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])


def extract_arxiv_paragraphs(arxiv_id: str, timeout: int) -> List[str]:
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    nodes = soup.find_all(class_="ltx_para", id=re.compile(r"^S\d+\.+(p|S)"))

    paragraphs: List[str] = []
    for node in nodes:
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def to_base_ctx(paper: Dict[str, Any]) -> Dict[str, Any]:
    abstract = paper.get("abstract") or ""
    authors = paper.get("authors") or []
    return {
        "semantic_scholar_id": paper.get("paperId"),
        "type": "ss_abstract",
        "year": paper.get("year"),
        "authors": authors,
        "title": paper.get("title"),
        "text": abstract,
        "url": paper.get("url"),
        "citation_counts": paper.get("citationCount"),
        "abstract": abstract,
    }


def build_ctxs(
    papers: List[Dict[str, Any]],
    include_arxiv_paragraphs: bool,
    max_paragraphs_per_paper: int,
    timeout: int,
) -> List[Dict[str, Any]]:
    ctxs: List[Dict[str, Any]] = []
    for paper in papers:
        ctxs.append(to_base_ctx(paper))

        if not include_arxiv_paragraphs:
            continue

        external_ids = paper.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        if not arxiv_id:
            continue

        try:
            paragraphs = extract_arxiv_paragraphs(arxiv_id, timeout=timeout)
        except Exception:
            continue

        for paragraph in paragraphs[:max_paragraphs_per_paper]:
            ctxs.append(
                {
                    "semantic_scholar_id": paper.get("paperId"),
                    "type": "ar5iv_paragraph",
                    "year": paper.get("year"),
                    "authors": paper.get("authors") or [],
                    "title": paper.get("title"),
                    "text": paragraph,
                    "url": paper.get("url"),
                    "citation_counts": paper.get("citationCount"),
                    "abstract": paper.get("abstract") or "",
                }
            )
    return ctxs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple paper fetcher: Semantic Scholar search + optional ar5iv paragraphs."
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Number of papers to fetch")
    parser.add_argument(
        "--min_citation_count",
        type=int,
        default=10,
        help="Minimum citation count filter for Semantic Scholar",
    )
    parser.add_argument(
        "--include_arxiv_paragraphs",
        action="store_true",
        help="Also fetch ar5iv paragraphs when ArXiv ID is available",
    )
    parser.add_argument(
        "--max_paragraphs_per_paper",
        type=int,
        default=3,
        help="Max ar5iv paragraphs per paper when enabled",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout (seconds)")
    parser.add_argument("--s2_api_key", default=None, help="Optional override for S2 API key")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    api_key = args.s2_api_key or os.environ.get("S2_API_KEY")
    if not api_key:
        raise ValueError("Missing S2 API key. Set S2_API_KEY or pass --s2_api_key.")

    papers = search_semantic_scholar(
        query=args.query,
        api_key=api_key,
        limit=args.limit,
        min_citation_count=args.min_citation_count,
        timeout=args.timeout,
    )
    ctxs = build_ctxs(
        papers=papers,
        include_arxiv_paragraphs=args.include_arxiv_paragraphs,
        max_paragraphs_per_paper=args.max_paragraphs_per_paper,
        timeout=args.timeout,
    )
    output_payload = {"query": args.query, "ctxs": ctxs}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
