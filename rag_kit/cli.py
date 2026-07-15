import argparse
import asyncio
from pathlib import Path

from rag_kit.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Local hybrid RAG knowledge base")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build index from a file or directory")
    ingest_parser.add_argument("path", help="File or directory path")

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Number of retrieved chunks")

    args = parser.parse_args()
    service = RAGService()

    if args.command == "ingest":
        response = service.ingest(Path(args.path))
        print(response.model_dump_json(indent=2))
    elif args.command == "ask":
        response = asyncio.run(service.ask(args.question, args.top_k))
        print(response.answer)
        print("\nSources:")
        for source in response.sources:
            print(f"- {source.score:.4f} {source.source} {source.preview}")


if __name__ == "__main__":
    main()

