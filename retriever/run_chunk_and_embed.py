"""
High-level overview:
1) Load paper records from JSON/JSONL (single file or directory).
2) Normalize each record into a top-level linear `text` field:
   - Use existing `text` if present.
   - Otherwise, linearize S2ORC-style `pdf_parse`/`pdf-parse` by combining abstract,
     body paragraphs, and referenced table/figure entries.
3) Write normalized records to JSONL as `{"id": ..., "text": ...}`.
4) Run the existing retriever pipeline:
   - Chunk text via `src.data.fast_load_jsonl_shard`.
   - Generate embeddings via `src.embed.generate_passage_embeddings`.

This script is a thin orchestration layer that preserves original metadata during
conversion but emits chunk/embedding-ready text records for retrieval.
"""

import argparse
import json
import os
import re
from typing import Dict, Iterable, List


class AttrDict(dict):
    """Dict with attribute access, compatible with existing retriever helpers."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def parse_shard_ids(shard_ids_arg: str, num_shards: int) -> List[int]:
    if not shard_ids_arg:
        return list(range(num_shards))
    shard_ids = [int(x.strip()) for x in shard_ids_arg.split(",") if x.strip()]
    if not shard_ids:
        raise ValueError("No valid shard ids parsed from --shard_ids.")
    for shard_id in shard_ids:
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"Shard id {shard_id} is out of range [0, {num_shards - 1}].")
    return shard_ids


def _strip_html(raw: str) -> str:
    # Lightweight HTML cleanup for table content embedded in ref_entries.
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ref_entry_to_text(ref_entry: Dict) -> str:
    if not isinstance(ref_entry, dict):
        return ""

    pieces: List[str] = []

    # Include type/number context so injected refs remain interpretable in plain text.
    if isinstance(ref_entry.get("type_str"), str):
        pieces.append(ref_entry["type_str"].strip().upper())
    if isinstance(ref_entry.get("fig_num"), str) and ref_entry["fig_num"].strip():
        pieces.append(ref_entry["fig_num"].strip())

    if isinstance(ref_entry.get("text"), str) and ref_entry["text"].strip():
        pieces.append(ref_entry["text"].strip())

    content = ref_entry.get("content")
    if isinstance(content, str) and content.strip():
        content_text = content.strip()
        # if content_text:
        #     pieces.append(content_text)
        pieces.append(content_text)

    return " ".join(pieces).strip()


def _inline_ref_spans(paragraph_text: str, ref_spans: List[Dict], ref_entries: Dict) -> str:
    if not paragraph_text:
        return ""
    if not isinstance(ref_spans, list) or not ref_spans:
        return paragraph_text.strip()

    spans = []
    for span in ref_spans:
        if not isinstance(span, dict):
            continue
        start = span.get("start")
        end = span.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        # Ignore malformed offsets from parser output.
        if start < 0 or end < start:
            continue
        spans.append(span)

    if not spans:
        return paragraph_text.strip()

    spans.sort(key=lambda x: (x["start"], x["end"]))
    n = len(paragraph_text)
    cursor = 0
    out_parts: List[str] = []

    for span in spans:
        start = min(max(span["start"], 0), n)
        end = min(max(span["end"], start), n)
        if start < cursor:
            continue

        out_parts.append(paragraph_text[cursor:start])
        original_span_text = paragraph_text[start:end]
        out_parts.append(original_span_text)

        ref_id = span.get("ref_id")
        if isinstance(ref_id, str) and isinstance(ref_entries, dict) and ref_id in ref_entries:
            injected = _ref_entry_to_text(ref_entries[ref_id])
            if injected:
                # Keep the original inline marker, then append resolved table/figure text.
                out_parts.append(f" [{ref_id}: {injected}]")

        cursor = end

    if cursor < n:
        out_parts.append(paragraph_text[cursor:])

    merged = "".join(out_parts)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def convert_record_to_linear_text(record: Dict) -> Dict:
    """
    Converts S2ORC-like paper JSON into a linearized record with top-level `text`,
    preserving metadata fields (paper_id, title, authors, etc.).
    """
    if not isinstance(record, dict):
        return record

    # Respect records already normalized to a top-level text field.
    if isinstance(record.get("text"), str) and record["text"].strip():
        return record

    pdf_parse = record.get("pdf_parse")
    # Support both common key variants seen in converted S2ORC exports.
    if not isinstance(pdf_parse, dict):
        pdf_parse = record.get("pdf-parse")
    if not isinstance(pdf_parse, dict):
        return record

    ref_entries = pdf_parse.get("ref_entries", {})
    used_ref_ids = set()

    parts: List[str] = []

    abstract_items = pdf_parse.get("abstract", [])
    if isinstance(abstract_items, list):
        for item in abstract_items:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

    body_items = pdf_parse.get("body_text", [])
    if isinstance(body_items, list):
        for item in body_items:
            if not isinstance(item, dict):
                continue
            paragraph_text = item.get("text")
            if not isinstance(paragraph_text, str) or not paragraph_text.strip():
                continue
            paragraph_with_refs = _inline_ref_spans(
                paragraph_text=paragraph_text,
                ref_spans=item.get("ref_spans", []),
                ref_entries=ref_entries,
            )
            if paragraph_with_refs:
                parts.append(paragraph_with_refs)
            for span in item.get("ref_spans", []):
                if not isinstance(span, dict):
                    continue
                ref_id = span.get("ref_id")
                if isinstance(ref_id, str) and ref_id in ref_entries:
                    used_ref_ids.add(ref_id)

    # Preserve unreferenced tables/figures by appending them at the end.
    if isinstance(ref_entries, dict):
        for ref_id, ref_entry in ref_entries.items():
            if ref_id in used_ref_ids:
                continue
            ref_text = _ref_entry_to_text(ref_entry)
            if ref_text:
                parts.append(f"[{ref_id}: {ref_text}]")

    if not parts and isinstance(record.get("abstract"), str) and record["abstract"].strip():
        parts.append(record["abstract"].strip())

    linear_text = "\n\n".join(parts).strip()
    if not linear_text:
        return record

    converted = dict(record)
    converted["text"] = linear_text
    return converted


def iter_json_records(path: str) -> Iterable[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if not path.endswith(".json"):
        raise ValueError(f"Unsupported file extension for {path}. Expected .json or .jsonl")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict):
                    yield item
            return
        yield payload
        return

    raise ValueError(f"Unsupported JSON structure in {path}.")


def normalize_to_jsonl(
    input_path: str,
    output_dir: str,
    text_key: str,
    id_key: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "normalized_input.jsonl")

    if os.path.isdir(input_path):
        input_files = [
            os.path.join(input_path, name)
            for name in sorted(os.listdir(input_path))
            if name.endswith(".json") or name.endswith(".jsonl")
        ]
    else:
        input_files = [input_path]

    if not input_files:
        raise ValueError(f"No .json/.jsonl files found under {input_path}.")

    written = 0
    auto_id = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for file_path in input_files:
            for record in iter_json_records(file_path):
                # Convert nonlinear parser output into one linear passage before chunking.
                converted = convert_record_to_linear_text(record)
                text = converted.get(text_key)
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                record_id = converted.get(id_key)
                if record_id is None:
                    # Preserve paper identity when generic id is absent.
                    record_id = converted.get("paper_id", auto_id)
                out.write(json.dumps({"id": record_id, "text": text}, ensure_ascii=False) + "\n")
                written += 1
                auto_id += 1

    if written == 0:
        raise ValueError(
            f"No valid records were written. Check --text_key ('{text_key}') and input format."
        )

    print(f"Wrote {written} normalized records to {output_path}")
    return output_path


def build_embedding_args(args, raw_data_path: str, shard_ids: List[int]) -> AttrDict:
    emb_args = AttrDict()
    emb_args.raw_data_path = raw_data_path
    emb_args.shard_ids = shard_ids
    emb_args.num_shards = args.num_shards
    emb_args.chunk_size = args.chunk_size
    emb_args.min_chunk_sz = args.min_chunk_size
    emb_args.keep_last_chunk = not args.drop_last_chunk
    emb_args.passages_dir = args.passages_dir

    emb_args.model_name_or_path = args.model_name_or_path
    emb_args.tokenizer = args.tokenizer if args.tokenizer else args.model_name_or_path
    emb_args.per_gpu_batch_size = args.per_gpu_batch_size
    emb_args.passage_maxlength = args.passage_maxlength
    emb_args.no_fp16 = args.no_fp16
    emb_args.no_title = args.no_title
    emb_args.lowercase = args.lowercase
    emb_args.normalize_text = args.normalize_text
    emb_args.use_saved_if_exists = args.use_saved_if_exists

    emb_args.embedding_dir = args.embedding_dir
    emb_args.prefix = args.prefix
    return emb_args


def run_chunk_stage(embedding_args: AttrDict) -> None:
    from src.data import fast_load_jsonl_shard

    print("Stage 1/2: chunking with src.data.fast_load_jsonl_shard")
    for shard_id in embedding_args.shard_ids:
        passages = fast_load_jsonl_shard(embedding_args, shard_id)
        num_passages = 0 if passages is None else len(passages)
        print(f"  shard={shard_id}: {num_passages} chunked passages")


def run_embedding_stage(embedding_args: AttrDict) -> None:
    from src.embed import generate_passage_embeddings

    print("Stage 2/2: embedding with src.embed.generate_passage_embeddings")
    cfg = AttrDict()
    cfg.model = AttrDict({"sparse_retriever": None})
    cfg.datastore = AttrDict({"embedding": embedding_args})
    generate_passage_embeddings(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Faithful chunk+embed runner using retriever/src/data.py and retriever/src/embed.py "
            "for processed JSON/JSONL inputs."
        )
    )

    parser.add_argument("--input_path", type=str, required=True, help="Input .json/.jsonl file or directory.")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory for normalized data.")

    parser.add_argument("--text_key", type=str, default="text", help="Text field in input JSON records.")
    parser.add_argument("--id_key", type=str, default="id", help="ID field in input JSON records.")

    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards.")
    parser.add_argument(
        "--shard_ids",
        type=str,
        default="",
        help="Comma-separated shard ids to process. Default: all shards [0..num_shards-1].",
    )
    parser.add_argument("--chunk_size", type=int, default=256, help="Words per chunk.")
    parser.add_argument("--min_chunk_size", type=int, default=0, help="Minimum final chunk size before merge.")
    parser.add_argument("--drop_last_chunk", action="store_true", help="Drop trailing incomplete chunk.")

    parser.add_argument("--model_name_or_path", type=str, default="akariasai/pes2o_contriever")
    parser.add_argument("--tokenizer", type=str, default="", help="Optional tokenizer path/name.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=256)
    parser.add_argument("--passage_maxlength", type=int, default=256)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_title", action="store_true")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize_text", action="store_true")
    parser.add_argument("--use_saved_if_exists", action="store_true")

    parser.add_argument("--prefix", type=str, default="passages")
    parser.add_argument(
        "--passages_dir",
        type=str,
        default="",
        help="Where chunked passage shard files are stored. Default: <work_dir>/passages",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        default="",
        help="Where embedding shard files are stored. Default: <work_dir>/embeddings",
    )
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be > 0.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0 to match retriever chunking flow.")
    if args.min_chunk_size < 0:
        raise ValueError("--min_chunk_size must be >= 0.")

    if not args.passages_dir:
        args.passages_dir = os.path.join(args.work_dir, "passages")
    if not args.embedding_dir:
        args.embedding_dir = os.path.join(args.work_dir, "embeddings")

    # print(args)
    shard_ids = parse_shard_ids(args.shard_ids, args.num_shards)
    normalized_data_path = normalize_to_jsonl(
        input_path=args.input_path,
        output_dir=os.path.join(args.work_dir, "normalized_jsonl"),
        text_key=args.text_key,
        id_key=args.id_key,
    )

    if (
        "sentence-transformers" not in args.model_name_or_path
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is required for this model path with the current faithful embed flow "
            "(src/embed.py moves batches to CUDA). Use a sentence-transformers model or run on GPU."
        )

    embedding_args = build_embedding_args(args, normalized_data_path, shard_ids)
    run_chunk_stage(embedding_args)
    run_embedding_stage(embedding_args)

    print("Done.")


if __name__ == "__main__":
    main()
