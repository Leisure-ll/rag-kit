# RAG Kit

一个可本地运行的 Python RAG 知识库项目，覆盖文档导入、文本切分、Embedding、向量检索、BM25 混合检索、Top-K 召回、LLM 生成和 SSE 流式输出。

默认不需要大模型 Key：未配置 `RAG_LLM_API_KEY` 时，会返回基于检索片段的抽取式回答；配置 DeepSeek 或其他 OpenAI-compatible API 后，会自动使用流式大模型回答。

## Features

- 支持 `.txt`、`.md`、`.pdf`、`.docx` 文档导入
- Recursive chunking，支持 chunk overlap
- 本地持久化向量索引：`data/index/vectors.npy` + `chunks.json`
- Hybrid Search：Hashing Embedding 向量召回 + BM25 关键词召回
- 可选 `sentence-transformers` 语义向量模型
- FastAPI 接口：健康检查、导入、问答、SSE 流式问答
- CLI：命令行导入和问答

## Quick Start

```powershell
cd $env:USERPROFILE\Desktop\rag-kit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

导入示例文档：

```powershell
python -m rag_kit.cli ingest .\sample_docs
```

命令行问答：

```powershell
python -m rag_kit.cli ask "RAG Kit 支持哪些检索方式？"
```

启动 API：

```powershell
uvicorn rag_kit.api:app --reload --host 127.0.0.1 --port 8000
```

访问：

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## API Examples

导入本地目录：

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest -F "path=sample_docs"
```

普通问答：

```powershell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG Kit 的混合检索怎么做？\",\"top_k\":5}"
```

SSE 流式问答：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/query/stream `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG Kit 的链路是什么？\",\"top_k\":5}"
```

## Use DeepSeek

复制配置文件并填入 Key：

```powershell
Copy-Item .env.example .env
```

`.env` 示例：

```env
RAG_LLM_API_KEY=sk-...
RAG_LLM_BASE_URL=https://api.deepseek.com
RAG_LLM_MODEL=deepseek-chat
```

项目使用 OpenAI-compatible `/v1/chat/completions` 协议，因此也可以切到其他兼容服务。

## Optional Semantic Embedding

默认 Hashing Embedding 适合演示和离线运行。如果要提升语义召回效果：

```powershell
pip install -r requirements-optional.txt
```

然后在 `.env` 中配置：

```env
RAG_EMBEDDING_BACKEND=sentence-transformers
RAG_SENTENCE_TRANSFORMER_MODEL=BAAI/bge-small-zh-v1.5
```

## Architecture

```text
Documents
  -> Loader(txt/md/pdf/docx)
  -> RecursiveTextSplitter
  -> Embedding + LocalVectorStore
  -> BM25Index
  -> HybridRetriever(vector_score * 0.65 + bm25_score * 0.35)
  -> Prompt Builder
  -> DeepSeek/OpenAI-compatible LLM or local extractive fallback
  -> JSON/SSE response with sources
```

## Project Structure

```text
rag_kit/
  api.py          FastAPI routes
  service.py      RAG orchestration
  loaders.py      document loaders
  splitter.py     recursive chunking
  embeddings.py   hashing and optional sentence-transformers embeddings
  vector_store.py local vector index persistence
  bm25.py         keyword retrieval
  retriever.py    hybrid ranking
  llm.py          OpenAI-compatible chat client and fallback
  cli.py          command line entry
```

