"""Retrieval evaluation: does the search find the right page?

Metrics:
- Hit rate@k: share of questions where an expected page appears in the top-k results.
- MRR (Mean Reciprocal Rank): 1/rank of the first correct result, averaged (1.0 = always ranked first).

Usage:  python -m evaluation.evaluate --top-k 5
"""
import argparse
import json
from pathlib import Path

from app.rag import retrieve


def first_correct_rank(results: list[dict], expected_source: str, expected_pages: list[int]) -> int | None:
    for rank, r in enumerate(results, start=1):
        if r["source"] == expected_source and r["page"] in expected_pages:
            return rank
    return None


def evaluate(dataset: list[dict], top_k: int) -> dict:
    ranks = []
    for item in dataset:
        results = retrieve(item["question"], top_k)
        rank = first_correct_rank(results, item["expected_source"], item["expected_pages"])
        ranks.append(rank)
        status = f"found at rank {rank}" if rank else "NOT FOUND"
        print(f"- {item['question'][:70]:<70} {status}")
    n = len(ranks)
    return {
        "questions": n,
        f"hit_rate@{top_k}": round(sum(r is not None for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.json")))
    args = parser.parse_args()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    print(json.dumps(evaluate(data, args.top_k), indent=2))
