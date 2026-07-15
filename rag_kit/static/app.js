const $ = (selector) => document.querySelector(selector);

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${text}`);
  }
  return response.json();
}

function renderStats(stats) {
  $("#statusLine").textContent = `${stats.storage_backend || "unknown"} storage · LLM ${stats.llm_enabled ? "enabled" : "fallback"}`;
  const entries = Object.entries(stats).filter(([key]) => key !== "status");
  $("#stats").innerHTML = entries
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd>`)
    .join("");
}

function renderDocuments(documents) {
  $("#documents").innerHTML = documents.length
    ? documents
        .map(
          (doc) => `
          <article class="row">
            <div class="path">${escapeHtml(doc.source || "")}</div>
            <div class="preview">object: ${escapeHtml(doc.object_key || "-")}</div>
          </article>
        `,
        )
        .join("")
    : "<p>No documents.</p>";
}

function renderChunks(chunks) {
  $("#chunks").innerHTML = chunks.length
    ? chunks
        .map(
          (chunk) => `
          <article class="chunk">
            <div>
              <span class="metric">chunk <strong>${escapeHtml(String(chunk.chunk_index ?? "-"))}</strong></span>
              <span class="metric">page <strong>${escapeHtml(String(chunk.page ?? "-"))}</strong></span>
            </div>
            <div class="path">${escapeHtml(chunk.source || "")}</div>
            <div class="preview">${escapeHtml((chunk.text || "").slice(0, 280))}</div>
          </article>
        `,
        )
        .join("")
    : "<p>No chunks.</p>";
}

function renderDebug(hits) {
  $("#debugHits").innerHTML = hits.length
    ? hits
        .map(
          (hit) => `
          <article class="hit">
            <div>
              <span class="metric">hybrid <strong>${hit.score}</strong></span>
              <span class="metric">vector <strong>${hit.vector_score}</strong></span>
              <span class="metric">bm25 <strong>${hit.bm25_score}</strong></span>
              <span class="metric">chunk <strong>${escapeHtml(String(hit.chunk_index ?? "-"))}</strong></span>
            </div>
            <div class="path">${escapeHtml(hit.source || "")}</div>
            <div class="preview">${escapeHtml(hit.preview || "")}</div>
          </article>
        `,
        )
        .join("")
    : "<p>No hits.</p>";
}

async function refresh() {
  const [stats, documents, chunks] = await Promise.all([
    requestJson("/stats"),
    requestJson("/documents"),
    requestJson("/chunks?limit=10"),
  ]);
  renderStats(stats);
  renderDocuments(documents);
  renderChunks(chunks);
}

async function ingestPath(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const result = await requestJson("/ingest", { method: "POST", body: form });
  $("#ingestResult").textContent = JSON.stringify(result, null, 2);
  await refresh();
}

async function ingestFile(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  if (!form.get("file") || !form.get("file").name) {
    $("#ingestResult").textContent = "Choose a file first.";
    return;
  }
  const result = await requestJson("/ingest/file", { method: "POST", body: form });
  $("#ingestResult").textContent = JSON.stringify(result, null, 2);
  await refresh();
}

async function ask(event) {
  event.preventDefault();
  const payload = queryPayload();
  const [answer, hits] = await Promise.all([
    requestJson("/query", jsonPost(payload)),
    requestJson("/search", jsonPost(payload)),
  ]);
  $("#answer").textContent = answer.answer;
  renderDebug(hits);
}

async function streamAnswer() {
  const payload = queryPayload();
  $("#answer").textContent = "";
  renderDebug(await requestJson("/search", jsonPost(payload)));

  const response = await fetch("/query/stream", jsonPost(payload));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const event of events) {
      const lines = event.split("\n");
      const data = lines.find((line) => line.startsWith("data: "));
      const type = lines.find((line) => line.startsWith("event: "));
      if (!data || type?.includes("sources") || data.includes("[DONE]")) continue;
      $("#answer").textContent += data.slice(6);
    }
  }
}

function queryPayload() {
  const form = new FormData($("#queryForm"));
  return {
    question: form.get("question"),
    top_k: Number(form.get("topK") || 5),
  };
}

function jsonPost(payload) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return map[char];
  });
}

$("#refreshBtn").addEventListener("click", refresh);
$("#pathForm").addEventListener("submit", ingestPath);
$("#fileForm").addEventListener("submit", ingestFile);
$("#queryForm").addEventListener("submit", ask);
$("#streamBtn").addEventListener("click", streamAnswer);

refresh().catch((error) => {
  $("#statusLine").textContent = error.message;
});

