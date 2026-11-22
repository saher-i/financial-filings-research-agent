const $ = (selector) => document.querySelector(selector);
const state = { status: null, initialized: false, chunkOffset: 0,
  variants: [], tuningMethod: "structure", lastComplete: null,
  document: null, selectedChunk: null, documentRequest: 0,
  embeddingOffset: 0, qdrantCursor: null };

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let payload;
  try { payload = await response.json(); } catch { throw new Error(`Request failed (${response.status})`); }
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

async function refreshStatus() {
  try {
    const payload = await request("/api/dev/status");
    const previous = state.status;
    state.status = payload;
    for (const strategy of ["structure", "semantic"]) {
      const info = payload.chunk_sets[strategy];
      const pill = $(`#${strategy}-status`);
      pill.textContent = info.ready ? "READY" : "NOT BUILT";
      pill.classList.toggle("ready", info.ready);
      $(`#${strategy}-count`).textContent = info.ready ? `${info.chunks.toLocaleString()} chunks` : "No chunks yet";
      const button = document.querySelector(`[data-build="${strategy}"]`);
      button.disabled = payload.job.state === "running" || (strategy === "semantic" && !payload.embedding_key_available);
      button.title = strategy === "semantic" && !payload.embedding_key_available ? "Set FIREWORKS_API_KEY before starting the server" : "";
      document.querySelector(`[data-inspect="${strategy}"]`).disabled = !info.ready;
    }
    renderVectorStatus(payload);
    if (!state.initialized) {
      await refreshVariants();
      state.initialized = true;
      if (payload.job.state !== "running") {
        await loadEmbeddings();
        await loadQdrantPoints();
      }
    }
    const message = $("#job-message");
    if (payload.job.stage === "chunking" && payload.job.state === "running") {
      message.textContent = `BUILDING ${payload.job.strategy.toUpperCase()} · ${payload.job.message}`;
    } else if (payload.job.stage === "chunking" && payload.job.state === "failed") {
      message.textContent = `BUILD FAILED · ${payload.job.message}`;
    } else if (payload.job.stage === "chunking" && payload.job.state === "complete") {
      message.textContent = `BUILD COMPLETE · ${payload.job.strategy} chunk set is ready.`;
    } else if (!payload.embedding_key_available) {
      message.textContent = "Semantic builds require FIREWORKS_API_KEY when starting this developer server.";
    } else {
      message.textContent = "Builds run only when requested; opening this page makes no model calls.";
    }
    const log = $("#build-log");
    log.hidden = payload.job.stage !== "chunking" || !payload.job.message || payload.job.state === "idle";
    $("#build-log-text").textContent = payload.job.message || "";
    $("#build-trial").disabled = payload.job.state === "running" ||
      (state.tuningMethod === "semantic" && !payload.embedding_key_available);
    if (payload.job.state === "complete" && state.lastComplete !== payload.job.run_id) {
      state.lastComplete = payload.job.run_id;
      if (payload.job.stage === "chunking") {
        await refreshVariants();
        if (payload.job.variant_id) {
          $("#chunk-strategy").value = payload.job.variant_id;
          $("#chunk-strategy").dispatchEvent(new Event("change"));
          $("#tune-message").textContent = `Saved ${payload.job.variant_id}. Inspect it under Inside the chunks.`;
        }
      } else if (payload.job.stage === "embedding") {
        $("#embedding-strategy").value = payload.job.strategy;
        state.embeddingOffset = 0;
        await loadEmbeddings();
      } else if (payload.job.stage === "indexing") {
        $("#qdrant-strategy").value = payload.job.strategy;
        state.qdrantCursor = null;
        await loadQdrantPoints();
      }
    }
    const current = $("#chunk-strategy").value;
    if (!previous || previous.chunk_sets[current]?.chunks !== payload.chunk_sets[current]?.chunks ||
        (previous.job.state === "running" && payload.job.state === "complete")) {
      renderChunkOverview();
      await loadChunks();
    }
  } catch (error) {
    $("#job-message").textContent = error.message;
  }
}

function renderVectorStatus(payload) {
  const running = payload.job.state === "running";
  for (const strategy of ["structure", "semantic"]) {
    const chunks = payload.chunk_sets[strategy];
    const { embedding, index } = payload.vectors[strategy];
    $(`#${strategy}-vector-chunks`).textContent = chunks.ready ? `${chunks.chunks.toLocaleString()} ready` : "Not built";
    $(`#${strategy}-vector-embedding`).textContent = embedding.ready ?
      `${embedding.count.toLocaleString()} · ${embedding.dimension} dimensions` :
      embedding.count ? "Outdated — regenerate" : "Not generated";
    $(`#${strategy}-vector-index`).textContent = index.ready ?
      `${index.count.toLocaleString()} points ready` : index.count ? "Outdated — rebuild" : "Not built";
    const pill = $(`#${strategy}-vector-status`);
    pill.textContent = index.ready ? "INDEXED" : embedding.ready ? "EMBEDDED" : "WAITING";
    pill.classList.toggle("ready", index.ready);
    document.querySelector(`[data-generate-embedding="${strategy}"]`).disabled =
      running || !chunks.ready || !payload.embedding_key_available;
    document.querySelector(`[data-build-index="${strategy}"]`).disabled = running || !embedding.ready;
    for (const [selector, ready] of [["data-open-embeddings", embedding.ready],
                                     ["data-open-qdrant", index.ready && !running]]) {
      const link = document.querySelector(`[${selector}="${strategy}"]`);
      link.classList.toggle("is-disabled", !ready);
      link.setAttribute("aria-disabled", String(!ready));
      link.tabIndex = ready ? 0 : -1;
    }
  }
  const message = $("#vector-message");
  if (payload.job.stage === "embedding" || payload.job.stage === "indexing") {
    const label = payload.job.stage === "embedding" ? "EMBEDDING" : "QDRANT INDEX";
    message.textContent = `${label} ${payload.job.state.toUpperCase()} · ${payload.job.message}`;
  } else if (!payload.embedding_key_available) {
    message.textContent = "Set FIREWORKS_API_KEY before starting this server to generate embeddings.";
  } else {
    message.textContent = "Choose a chunk set, then generate embeddings and build its Qdrant index.";
  }
}

function vectorItem(item, source) {
  const card = node("article", "vector-item");
  card.append(node("strong", "", item.chunk_id));
  const meta = source === "qdrant" ?
    `${item.ticker} · FY${item.fiscal_year} · ${item.filename} · PDF p.${item.pages.join(", ")} · point ${item.point_id} · ${item.dimension} values` :
    `${item.dimension} values · finished chunk embedding`;
  card.append(node("small", "", meta));
  if (source === "qdrant") card.append(node("p", "", item.text.slice(0, 260)));
  card.append(node("code", "", `[${item.values.slice(0, 12).join(", ")}${item.values.length > 12 ? ", …" : ""}]`));
  const details = node("details", "");
  details.append(node("summary", "", `Show all ${item.dimension} values`));
  details.append(node("pre", "", JSON.stringify(item.values)));
  card.append(details);
  return card;
}

async function loadEmbeddings() {
  const strategy = $("#embedding-strategy").value;
  const list = $("#embedding-list");
  list.replaceChildren();
  if (!state.status?.vectors[strategy].embedding.ready) {
    $("#embedding-caption").textContent = "Generate embeddings for this chunk set first.";
    $("#more-embeddings").hidden = true;
    return;
  }
  try {
    const payload = await request(`/api/dev/embeddings?${new URLSearchParams({strategy, offset: String(state.embeddingOffset)})}`);
    $("#embedding-caption").textContent = `${payload.total.toLocaleString()} vectors · showing ${payload.offset + 1}–${payload.offset + payload.items.length}`;
    for (const item of payload.items) list.append(vectorItem(item, "embedding"));
    $("#more-embeddings").hidden = payload.offset + payload.items.length >= payload.total;
  } catch (error) {
    $("#embedding-caption").textContent = error.message;
    $("#more-embeddings").hidden = true;
  }
}

async function loadQdrantPoints() {
  const strategy = $("#qdrant-strategy").value;
  const list = $("#qdrant-list");
  list.replaceChildren();
  if (!state.status?.vectors[strategy].index.ready) {
    $("#qdrant-caption").textContent = "Build the Qdrant index for this chunk set first.";
    $("#more-qdrant").hidden = true;
    return;
  }
  try {
    const params = new URLSearchParams({strategy});
    if (state.qdrantCursor) params.set("cursor", state.qdrantCursor);
    const payload = await request(`/api/dev/qdrant-points?${params}`);
    $("#qdrant-caption").textContent = `${payload.collection} · ${payload.count.toLocaleString()} stored points`;
    for (const item of payload.points) list.append(vectorItem(item, "qdrant"));
    state.qdrantCursor = payload.next_cursor;
    $("#more-qdrant").hidden = !payload.next_cursor;
  } catch (error) {
    $("#qdrant-caption").textContent = error.message;
    $("#more-qdrant").hidden = true;
  }
}

function variantLabel(identifier) {
  const item = state.variants.find((variant) => variant.id === identifier);
  if (!item) return identifier;
  return `${item.strategy === "structure" ? "Structure" : "Semantic"} · ${item.default ? "baseline" : `target ${item.target}, overlap ${item.overlap}`}`;
}

function currentChunkInfo() {
  const identifier = $("#chunk-strategy").value;
  if (identifier === "structure" || identifier === "semantic") return state.status?.chunk_sets[identifier];
  return state.variants.find((item) => item.id === identifier)?.summary;
}

async function refreshVariants() {
  const payload = await request("/api/dev/variants");
  state.variants = payload.variants;
  const select = $("#chunk-strategy");
  const previous = select.value || "structure";
  select.replaceChildren(new Option("Structure baseline", "structure"), new Option("Semantic baseline", "semantic"));
  for (const item of state.variants.filter((variant) => !variant.default && variant.summary.ready)) {
    select.add(new Option(variantLabel(item.id), item.id));
  }
  select.value = [...select.options].some((option) => option.value === previous) ? previous : "structure";
  renderTrialTable();
}

function renderTrialTable() {
  const body = $("#trial-rows");
  body.replaceChildren();
  const ready = state.variants.filter((item) => item.summary.ready);
  const baselines = ready.filter((item) => item.default).length;
  $("#trial-count").textContent = `${baselines} baseline · ${ready.length - baselines} saved trials`;
  for (const item of state.variants) {
    const row = node("tr", "");
    const name = node("td", "", item.strategy === "structure" ? "Structure aware" : "Semantic");
    name.append(node("small", "", item.default ? "Default set" : `Saved trial · ${item.id}`));
    row.append(name,
      node("td", "", item.target.toLocaleString()),
      node("td", "", item.maximum.toLocaleString()),
      node("td", "", item.overlap.toLocaleString()),
      node("td", "", item.summary.ready ? item.summary.chunks.toLocaleString() : "—"),
      node("td", "", item.summary.ready ? "Ready" : "Not built"));
    const action = node("td", "");
    const inspect = node("button", "secondary", "View chunks →");
    inspect.type = "button";
    inspect.disabled = !item.summary.ready;
    inspect.addEventListener("click", () => {
      $("#chunk-strategy").value = item.default ? item.strategy : item.id;
      $("#chunk-strategy").dispatchEvent(new Event("change"));
      $("#inspect-section").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    action.append(inspect);
    row.append(action);
    body.append(row);
  }
}

function renderChunkOverview() {
  const info = currentChunkInfo();
  const overview = $("#chunk-overview");
  const breakdown = $("#document-breakdown");
  const select = $("#chunk-document");
  const selectedFile = select.value;
  overview.replaceChildren();
  breakdown.replaceChildren();
  select.replaceChildren(new Option("All filings", ""));
  if (!info?.ready) {
    overview.append(node("p", "chunk-empty", "Build this chunk set to inspect its passages."));
    return;
  }
  const statistics = [
    ["CHUNKS", info.chunks.toLocaleString()],
    ["MEDIAN LENGTH", `${info.median_characters.toLocaleString()} chars`],
    ["WITH TABLES", info.table_chunks.toLocaleString()],
    ["SPANNING PAGES", info.cross_page_chunks.toLocaleString()],
  ];
  for (const [label, value] of statistics) {
    const card = node("div", "overview-stat");
    card.append(node("span", "", label), node("strong", "", value));
    overview.append(card);
  }
  for (const document of info.documents) {
    select.add(new Option(document.filename, document.filename));
    const button = node("button", "document-row");
    button.type = "button";
    button.append(node("strong", "", document.filename));
    button.append(node("span", "", `${document.chunks} chunks · ${document.pages} pages · ${document.table_chunks} with tables`));
    button.addEventListener("click", () => {
      select.value = document.filename;
      state.chunkOffset = 0;
      loadChunks();
    });
    breakdown.append(button);
  }
  if ([...select.options].some((option) => option.value === selectedFile)) select.value = selectedFile;
}

async function showDocument(filename, page, chunkId = null) {
  const identifier = $("#chunk-strategy").value;
  const requestId = ++state.documentRequest;
  const params = new URLSearchParams({ variant: identifier, filename, page: String(page) });
  $("#pdf-help").textContent = "Loading source page…";
  try {
    const documentPage = await request(`/api/dev/document-page?${params}`);
    if (requestId !== state.documentRequest) return;
    state.document = documentPage;
    state.selectedChunk = chunkId;
    $("#pdf-title").textContent = filename;
    $("#pdf-page-label").textContent = `${page} / ${documentPage.page_count}`;
    $("#pdf-prev").disabled = page <= 1;
    $("#pdf-next").disabled = page >= documentPage.page_count;
    const original = $("#pdf-original");
    original.href = `/api/dev/filings/${encodeURIComponent(filename)}#page=${page}`;
    original.hidden = false;
    const canvas = $("#pdf-canvas");
    canvas.replaceChildren();
    const sheet = node("div", "pdf-sheet");
    const image = node("img", "pdf-image");
    image.src = `/api/dev/document-image?${new URLSearchParams({ filename, page: String(page) })}`;
    image.alt = `${filename}, physical PDF page ${page}`;
    const overlay = node("div", "pdf-overlay");
    sheet.append(image, overlay);
    canvas.append(sheet);
    for (const mark of documentPage.marks.filter((item) => item.id === chunkId)) {
      for (const box of mark.boxes) {
        const region = node("div", "pdf-mark");
        region.title = `Source of selected chunk: ${mark.preview}`;
        region.style.left = `${100 * box[0] / documentPage.width}%`;
        region.style.top = `${100 * box[1] / documentPage.height}%`;
        region.style.width = `${100 * (box[2] - box[0]) / documentPage.width}%`;
        region.style.height = `${100 * (box[3] - box[1]) / documentPage.height}%`;
        overlay.append(region);
      }
    }
    selectDocumentChunk(chunkId);
    $("#pdf-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    if (requestId === state.documentRequest) $("#pdf-help").textContent = error.message;
  }
}

function selectDocumentChunk(chunkId) {
  state.selectedChunk = chunkId;
  document.querySelectorAll(".chunk-card").forEach((card) => {
    card.classList.toggle("selected", Boolean(chunkId) && card.dataset.chunkId === chunkId);
  });
  const mark = state.document?.marks.find((item) => item.id === chunkId);
  if (!state.document) return;
  $("#pdf-help").textContent = mark ? `Selected ${chunkId} · ${mark.preview}` :
    "The selected chunk has no source text on this page. Use the page arrows to return to its highlighted page.";
}

async function loadChunks() {
  const identifier = $("#chunk-strategy").value;
  const info = currentChunkInfo();
  const list = $("#chunk-list");
  const paging = $("#chunk-pagination");
  list.replaceChildren();
  if (!info?.ready) {
    list.append(node("p", "chunk-empty", "No chunk file exists for this strategy yet."));
    paging.hidden = true;
    return;
  }
  const params = new URLSearchParams({ offset: String(state.chunkOffset), limit: "8" });
  params.set(identifier === "structure" || identifier === "semantic" ? "strategy" : "variant", identifier);
  if ($("#chunk-document").value) params.set("filename", $("#chunk-document").value);
  if ($("#chunk-page").value) params.set("page", $("#chunk-page").value);
  if ($("#chunk-query").value.trim()) params.set("query", $("#chunk-query").value.trim());
  try {
    const payload = await request(`/api/dev/chunks?${params}`);
    if (!payload.chunks.length) list.append(node("p", "chunk-empty", "No chunks match these filters."));
    for (const chunk of payload.chunks) {
      const card = node("article", "chunk-card");
      card.dataset.chunkId = chunk.id;
      const header = node("div", "chunk-card-header");
      header.append(node("span", "chunk-id", chunk.id));
      header.append(node("span", "chunk-meta", `${chunk.characters} characters · ${chunk.kinds.join(" + ")} · PDF p.${chunk.pages.join(", ")}`));
      card.append(header, node("p", "", chunk.text));
      const link = node("a", "", `Open original filing at page ${chunk.pages[0]} ↗`);
      link.href = `/api/dev/filings/${encodeURIComponent(chunk.filename)}#page=${chunk.pages[0]}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.append(link);
      const show = node("button", "secondary show-on-pdf", "Show on PDF →");
      show.type = "button";
      show.addEventListener("click", () => showDocument(chunk.filename, chunk.pages[0], chunk.id));
      card.append(show);
      list.append(card);
    }
    selectDocumentChunk(state.selectedChunk);
    paging.hidden = payload.total <= payload.limit;
    $("#chunk-page-label").textContent = `${payload.total ? payload.offset + 1 : 0}–${Math.min(payload.offset + payload.limit, payload.total)} of ${payload.total.toLocaleString()} matching chunks`;
    $("#previous-chunks").disabled = payload.offset === 0;
    $("#next-chunks").disabled = payload.offset + payload.limit >= payload.total;
  } catch (error) {
    list.append(node("p", "chunk-empty", error.message));
    paging.hidden = true;
  }
}

async function build(strategy) {
  $("#job-message").textContent = "";
  try {
    await request("/api/dev/build", { method: "POST", body: JSON.stringify({ strategy }) });
    await refreshStatus();
  } catch (error) {
    $("#job-message").textContent = error.message;
  }
}

function updateTuningControls() {
  const target = Number($("#target-range").value);
  const maximumInput = $("#maximum-range");
  if (Number(maximumInput.value) < target) maximumInput.value = String(target);
  $("#target-value").textContent = target.toLocaleString();
  $("#maximum-value").textContent = Number(maximumInput.value).toLocaleString();
  $("#overlap-value").textContent = Number($("#overlap-range").value).toLocaleString();
  const canBuild = state.status?.job.state !== "running" &&
    (state.tuningMethod !== "semantic" || !!state.status?.embedding_key_available);
  $("#build-trial").disabled = !canBuild;
  if (state.tuningMethod === "semantic" && !state.status?.embedding_key_available) {
    $("#tune-message").textContent = "Set FIREWORKS_API_KEY and restart the server before building a semantic trial.";
  }
}

async function buildTrial() {
  const button = $("#build-trial");
  button.disabled = true;
  $("#tune-message").textContent = "Starting saved trial…";
  try {
    await request("/api/dev/variant/build", { method: "POST", body: JSON.stringify({
      strategy: state.tuningMethod,
      target: Number($("#target-range").value),
      maximum: Number($("#maximum-range").value),
      overlap: Number($("#overlap-range").value),
    }) });
    $("#tune-message").textContent = "Build started. The original chunk sets remain available while this runs.";
    await refreshStatus();
  } catch (error) {
    $("#tune-message").textContent = error.message;
    updateTuningControls();
  }
}

document.querySelectorAll("[data-build]").forEach((button) => button.addEventListener("click", () => build(button.dataset.build)));
async function startVectorStage(stage, strategy) {
  try {
    await request(stage === "embedding" ? "/api/dev/embedding/build" : "/api/dev/index/build",
      { method: "POST", body: JSON.stringify({ strategy }) });
    await refreshStatus();
  } catch (error) {
    $("#vector-message").textContent = error.message;
  }
}
document.querySelectorAll("[data-generate-embedding]").forEach((button) =>
  button.addEventListener("click", () => startVectorStage("embedding", button.dataset.generateEmbedding)));
document.querySelectorAll("[data-build-index]").forEach((button) =>
  button.addEventListener("click", () => startVectorStage("indexing", button.dataset.buildIndex)));
document.querySelectorAll("[data-open-embeddings]").forEach((link) => link.addEventListener("click", (event) => {
  if (link.getAttribute("aria-disabled") === "true") return event.preventDefault();
  $("#embedding-strategy").value = link.dataset.openEmbeddings;
  state.embeddingOffset = 0;
  loadEmbeddings();
}));
document.querySelectorAll("[data-open-qdrant]").forEach((link) => link.addEventListener("click", (event) => {
  if (link.getAttribute("aria-disabled") === "true") return event.preventDefault();
  $("#qdrant-strategy").value = link.dataset.openQdrant;
  state.qdrantCursor = null;
  loadQdrantPoints();
}));
$("#embedding-strategy").addEventListener("change", () => { state.embeddingOffset = 0; loadEmbeddings(); });
$("#qdrant-strategy").addEventListener("change", () => { state.qdrantCursor = null; loadQdrantPoints(); });
$("#more-embeddings").addEventListener("click", () => { state.embeddingOffset += 5; loadEmbeddings(); });
$("#more-qdrant").addEventListener("click", loadQdrantPoints);
document.querySelectorAll("[data-method]").forEach((button) => button.addEventListener("click", () => {
  state.tuningMethod = button.dataset.method;
  document.querySelectorAll("[data-method]").forEach((choice) => {
    const selected = choice.dataset.method === state.tuningMethod;
    choice.classList.toggle("selected", selected);
    choice.setAttribute("aria-pressed", String(selected));
  });
  $("#tune-message").textContent = "";
  updateTuningControls();
}));
for (const selector of ["#target-range", "#maximum-range", "#overlap-range"]) {
  $(selector).addEventListener("input", updateTuningControls);
}
$("#build-trial").addEventListener("click", buildTrial);
document.querySelectorAll("[data-inspect]").forEach((button) => button.addEventListener("click", () => {
  $("#chunk-strategy").value = button.dataset.inspect;
  $("#chunk-strategy").dispatchEvent(new Event("change"));
  $("#inspect-section").scrollIntoView({ behavior: "smooth", block: "start" });
}));
$("#chunk-strategy").addEventListener("change", () => {
  state.chunkOffset = 0;
  state.document = null;
  state.selectedChunk = null;
  state.documentRequest++;
  $("#pdf-title").textContent = "Select a chunk";
  $("#pdf-help").textContent = "Choose “Show on PDF” to mark a chunk in its source filing.";
  $("#pdf-page-label").textContent = "—";
  $("#pdf-canvas").replaceChildren();
  $("#pdf-original").hidden = true;
  $("#pdf-prev").disabled = true;
  $("#pdf-next").disabled = true;
  renderChunkOverview();
  loadChunks();
});
$("#pdf-prev").addEventListener("click", () => {
  if (state.document) showDocument(state.document.filename, state.document.page - 1, state.selectedChunk);
});
$("#pdf-next").addEventListener("click", () => {
  if (state.document) showDocument(state.document.filename, state.document.page + 1, state.selectedChunk);
});
$("#inspect-chunks").addEventListener("click", () => { state.chunkOffset = 0; loadChunks(); });
$("#previous-chunks").addEventListener("click", () => { state.chunkOffset = Math.max(0, state.chunkOffset - 8); loadChunks(); });
$("#next-chunks").addEventListener("click", () => { state.chunkOffset += 8; loadChunks(); });
refreshStatus();
setInterval(() => { if (state.status?.job.state === "running") refreshStatus(); }, 2500);
