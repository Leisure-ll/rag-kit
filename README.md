# RAG Kit

一个可本地运行、也可 Docker 一键启动的 Python RAG 知识库项目，覆盖文档导入、文本切分、Embedding、向量检索、BM25 混合检索、Top-K 召回、LLM 生成和 SSE 流式输出。

默认不需要大模型 Key：未配置 `RAG_LLM_API_KEY` 时，会返回基于检索片段的抽取式回答；配置 DeepSeek 或其他 OpenAI-compatible API 后，会自动使用流式大模型回答。

## Features

- Docker Compose 启动知识库基础设施：Elasticsearch + MinIO
- 原始文件存 MinIO，文档/chunk 元数据存 SQLite/MySQL，文本/BM25/向量索引存 Elasticsearch
- 支持 `.txt`、`.md`、`.pdf`、`.docx` 文档导入
- Recursive chunking，支持 chunk overlap
- 本地模式持久化向量索引：`data/index/vectors.npy` + `chunks.json`
- Hybrid Search：Hashing Embedding 向量召回 + BM25 关键词召回
- 可选 `sentence-transformers` 语义向量模型
- FastAPI 接口：健康检查、导入、问答、SSE 流式问答
- Web Console：导入、问答、流式输出、召回分数、文档/chunk 查看
- Query Trace：每次问答落库，记录 trace_id、latency、Top-K、hybrid/vector/BM25 分数
- Offline Eval：golden questions 评测 hit@k、MRR、答案关键词覆盖率
- CLI：命令行导入和问答

## Docker Demo

面试演示优先用 Docker 启动 MinIO，再用本机 Python API 连接 MinIO + SQLite 知识库数据库。这样即使 DockerHub 拉 Python/MySQL/Elasticsearch 镜像不稳定，也能演示真实外部存储。

```powershell
cd $env:USERPROFILE\Desktop\rag-kit
docker compose up -d minio
```

服务地址：

- Web Console: http://127.0.0.1:8000
- API Swagger: http://127.0.0.1:8000/docs
- API Health: http://127.0.0.1:8000/health
- MinIO Console: http://127.0.0.1:9001
- 元数据 SQLite: `data/rag_kit.db`

MinIO 登录：

```text
username: rag_minio
password: rag_minio_password
```

启动本机 API，并让它连接 Docker 里的 Elasticsearch 和 MinIO：

```powershell
Copy-Item .env.external.example .env
.\.venv\Scripts\Activate.ps1
uvicorn rag_kit.api:app --host 127.0.0.1 --port 8000
```

导入示例知识库：

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest -F "path=sample_docs"
```

查看存储统计：

```powershell
curl.exe http://127.0.0.1:8000/stats
```

查看文档、chunk 和检索打分：

```powershell
curl.exe http://127.0.0.1:8000/documents
curl.exe http://127.0.0.1:8000/chunks
curl.exe -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG Kit 的知识库数据分别存在哪里？\",\"top_k\":5}"
```

查看最近问答 Trace：

```powershell
curl.exe http://127.0.0.1:8000/traces
curl.exe http://127.0.0.1:8000/traces/<trace_id>
```

运行离线评测：

```powershell
python -m rag_kit.eval eval/golden_qa.jsonl --top-k 5
```

输出指标包括：

```text
hit_at_k
mrr
answer_keyword_coverage
```

普通问答：

```powershell
curl.exe -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG Kit 的知识库数据分别存在哪里？\",\"top_k\":5}"
```

SSE 流式问答：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/query/stream `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG Kit 的检索链路是什么？\",\"top_k\":5}"
```

停止：

```powershell
docker compose down
```

清空数据卷：

```powershell
docker compose down -v
```

如果当前网络可以正常拉 DockerHub 镜像，也可以直接把 API 一起容器化：

```powershell
docker compose --profile api up -d --build
```

如果想把元数据也换成 MySQL：

```powershell
docker compose -f docker-compose.yml -f docker-compose.mysql.yml --profile api up -d --build
```

## Storage Design

默认 Docker 演示模式下使用真实知识库存储：

```text
MinIO
  bucket: rag-documents
  保存原始 txt/md/pdf/docx 文件

SQLite
  database: data/rag_kit.db
  documents 表：文档来源、MinIO object key、文档元数据
  chunks 表：chunk 文本、chunk_index、source、page、metadata、vector_json

Elasticsearch 可选增强版
  index: rag_chunks
  text 字段：BM25 关键词检索
  vector 字段：dense_vector 向量检索
  metadata 字段：source/page/chunk_index 等召回溯源信息
```

默认演示版从 SQLite 读取 chunk 和 vector，做向量召回 + BM25 融合；Elasticsearch 增强版会分别从 Elasticsearch 做向量召回和 BM25 召回。最终都按默认权重融合：

```text
hybrid_score = 0.65 * vector_score + 0.35 * bm25_score
```

## Local Quick Start

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
  -> MinIO raw file storage
  -> MySQL metadata/chunk records
  -> Embedding + Elasticsearch dense_vector
  -> Elasticsearch BM25 text index
  -> HybridRetriever(vector_score * 0.65 + bm25_score * 0.35)
  -> Prompt Builder
  -> DeepSeek/OpenAI-compatible LLM or local extractive fallback
  -> JSON/SSE response with sources
  -> Query Trace + Offline Eval
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
  metadata_store.py MySQL metadata storage
  object_store.py   MinIO raw file storage
  elastic_store.py  Elasticsearch BM25/vector storage
  trace_store.py    query trace persistence
  eval.py           offline retrieval/answer evaluation
```
