import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from rag_kit.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality on a golden QA set")
    parser.add_argument("dataset", help="JSONL file with question, expected_source, expected_keywords")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="data/eval_report.json")
    args = parser.parse_args()

    report = asyncio.run(evaluate(Path(args.dataset), args.top_k))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {output_path}")


async def evaluate(dataset_path: Path, top_k: int) -> Dict:
    service = RAGService()
    cases = _load_cases(dataset_path)
    results = []

    for case in cases:
        question = case["question"]
        hits = service.search(question, top_k)
        response = await service.ask(question, top_k)
        expected_source = case.get("expected_source", "")
        expected_keywords = case.get("expected_keywords", [])

        rank = _first_source_rank(hits, expected_source)
        keyword_score = _keyword_coverage(response.answer, expected_keywords)
        results.append(
            {
                "question": question,
                "expected_source": expected_source,
                "expected_keywords": expected_keywords,
                "hit": rank is not None,
                "rank": rank,
                "mrr": 0.0 if rank is None else 1.0 / rank,
                "answer_keyword_coverage": keyword_score,
                "trace_id": response.trace_id,
                "hits": [
                    {
                        "rank": index,
                        "score": round(hit.score, 4),
                        "vector_score": round(hit.vector_score, 4),
                        "bm25_score": round(hit.bm25_score, 4),
                        "source": hit.chunk.metadata.get("source"),
                        "chunk_index": hit.chunk.metadata.get("chunk_index"),
                    }
                    for index, hit in enumerate(hits, start=1)
                ],
            }
        )

    return {
        "summary": {
            "cases": len(results),
            "hit_at_k": round(mean([1.0 if item["hit"] else 0.0 for item in results]), 4) if results else 0.0,
            "mrr": round(mean([item["mrr"] for item in results]), 4) if results else 0.0,
            "answer_keyword_coverage": round(mean([item["answer_keyword_coverage"] for item in results]), 4)
            if results
            else 0.0,
            "top_k": top_k,
        },
        "cases": results,
    }


def _load_cases(path: Path) -> List[Dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _first_source_rank(hits, expected_source: str):
    if not expected_source:
        return None
    for index, hit in enumerate(hits, start=1):
        source = hit.chunk.metadata.get("source", "")
        if expected_source in source:
            return index
    return None


def _keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0
    matched = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
    return matched / len(expected_keywords)


if __name__ == "__main__":
    main()

