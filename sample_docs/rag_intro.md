# RAG Kit 示例知识库

RAG Kit 是一个 Python 知识库问答项目。它将文档加载、文本切分、Embedding、向量检索、BM25 关键词检索和大模型生成串成一条完整链路。

项目默认使用轻量 Hashing Embedding，因此不需要下载模型也能完成本地演示。向量索引会持久化到 `data/index`，包含 chunk 元数据和向量矩阵。

混合检索使用两个分数：向量相似度负责语义召回，BM25 负责关键词精确匹配。最终分数默认按 `0.65 * vector_score + 0.35 * bm25_score` 融合，并返回 Top-K 片段。

如果配置 `RAG_LLM_API_KEY`，系统会调用 DeepSeek 或其他 OpenAI-compatible 服务，通过 `/v1/chat/completions` 生成答案。未配置 Key 时，系统返回检索片段组成的抽取式答案，方便离线运行和调试。

FastAPI 提供 `/ingest`、`/ingest/file`、`/query`、`/query/stream` 和 `/health` 接口。其中 `/query/stream` 使用 Server-Sent Events 输出流式 token，并在结束时返回 sources 事件。

