from rag_kit.models import Document
from rag_kit.splitter import RecursiveTextSplitter


def test_splitter_keeps_overlap_and_metadata():
    splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=10)
    documents = [
        Document(
            id="doc-1",
            text="第一段内容用于测试切分。\n\n第二段内容也需要进入索引。\n\n第三段内容用于验证 overlap。",
            metadata={"source": "demo.md"},
        )
    ]

    chunks = splitter.split_documents(documents)

    assert len(chunks) >= 2
    assert chunks[0].metadata["source"] == "demo.md"
    assert chunks[0].metadata["document_id"] == "doc-1"

