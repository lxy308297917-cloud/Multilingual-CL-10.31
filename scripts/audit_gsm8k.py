"""Independently check GSM8K strict-match samples against the saved summary."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")


def single(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"{description}: expected one file, found {len(paths)}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result_path = single(list(args.run_dir.glob("results/**/results_*.json")), "summary")
    sample_path = single(list(result_path.parent.glob("samples_gsm8k_locked_*.jsonl")), "samples")
    summary = json.loads(result_path.read_text())["results"]["gsm8k_locked"]
    rows: dict[str, dict[int, dict]] = defaultdict(dict)
    with sample_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row["filter"]
            if kind not in ("strict-match", "flexible-extract"):
                raise ValueError(f"unexpected filter: {kind}")
            doc_id = row["doc_id"]
            if doc_id in rows[kind]:
                raise ValueError(f"duplicate {kind} doc_id: {doc_id}")
            rows[kind][doc_id] = row
    expected = set(range(1319))
    if any(set(rows[kind]) != expected for kind in ("strict-match", "flexible-extract")):
        raise ValueError("missing or extra test samples")
    if summary["sample_len"] != 1319:
        raise ValueError("incorrect summary sample count")

    scores = {}
    parse_failures = 0
    for kind in ("strict-match", "flexible-extract"):
        correct = 0
        for doc_id in sorted(expected):
            row = rows[kind][doc_id]
            if row["exact_match"] not in (0, 1, 0.0, 1.0):
                raise ValueError(f"nonbinary score: {kind} {doc_id}")
            if kind == "strict-match":
                raw = row["resps"][0][0]
                match = STRICT.search(raw)
                parsed = match.group(1) if match else "[invalid]"
                if row["filtered_resps"] != [parsed]:
                    raise ValueError(f"strict extraction mismatch: {doc_id}")
                parse_failures += match is None
                target = re.sub(r"(?s).*#### ", "", row["target"])
                normalized = lambda value: re.sub(r"\.$", "", value.replace(",", "").replace("$", "")).lower()
                independent = int(normalized(parsed) == normalized(target))
                if independent != row["exact_match"]:
                    raise ValueError(f"strict score mismatch: {doc_id}")
                other = rows["flexible-extract"][doc_id]
                if row["doc_hash"] != other["doc_hash"] or row["prompt_hash"] != other["prompt_hash"]:
                    raise ValueError(f"filter variants use different question/prompt: {doc_id}")
            correct += row["exact_match"]
        score = correct / 1319
        if abs(score - summary[f"exact_match,{kind}"]) > 1e-12:
            raise ValueError(f"{kind}: samples differ from summary")
        scores[kind] = {"correct": int(correct), "total": 1319, "accuracy": score}
    intersections = {'both_correct': 0, 'strict_only': 0, 'flexible_only': 0, 'both_wrong': 0}
    parse_failures_flexible_correct = 0
    for doc_id in sorted(expected):
        strict = int(rows['strict-match'][doc_id]['exact_match'])
        flexible = int(rows['flexible-extract'][doc_id]['exact_match'])
        name = ('both_correct' if strict and flexible else
                'strict_only' if strict else 'flexible_only' if flexible else 'both_wrong')
        intersections[name] += 1
        if rows['strict-match'][doc_id]['filtered_resps'] == ['[invalid]'] and flexible:
            parse_failures_flexible_correct += 1
    print(json.dumps({
        "status": "validated", "run_dir": str(args.run_dir),
        "strict_match_parse_failures": parse_failures,
        "parse_failures_flexible_correct": parse_failures_flexible_correct,
        "strict_flexible_intersection": intersections, "scores": scores,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
