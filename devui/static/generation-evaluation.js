const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let poller = null;
let rendered = null;
let retryBusy = false;
const citedIds = (answer) => [...new Set(String(answer || '').match(/\[S\d+\]/g) || [])];
const sourceButtons = (ids, questionId) => ids.map((id) =>
  `<button type="button" class="generation-source-link" data-question="${escapeHtml(questionId)}" data-source="${escapeHtml(id.slice(1, -1))}">${escapeHtml(id)}</button>`).join(' ');

function linkedText(value, ids, questionId) {
  const known = new Set(ids);
  return String(value || '').split(/(\[S\d+\])/g).map((part) =>
    known.has(part) ? sourceButtons([part], questionId) : escapeHtml(part)).join('');
}

function retryControl(scores, metric, questionId) {
  const error = scores.errors?.[metric];
  if (!error || !questionId) return '';
  const label = metric === 'faithfulness' ? 'Faithfulness' : 'Response relevancy';
  return `<div class="metric-retry"><p role="note">${escapeHtml(label)} could not be scored: ${escapeHtml(error)}</p>
    <button type="button" class="metric-retry-button" data-question="${escapeHtml(questionId)}" data-metric="${metric}" ${retryBusy ? 'disabled' : ''}>Retry ${escapeHtml(label)} only <span aria-hidden="true">↻</span></button></div>`;
}

function renderFaithfulness(scores, ids, questionId) {
  const claims = scores.faithfulness_claims || [];
  const supported = claims.filter((claim) => claim.verdict).length;
  const score = scores.faithfulness == null ? '—' : scores.faithfulness.toFixed(3);
  const explanation = claims.length
    ? `${supported} supported ÷ ${claims.length} claims = ${score}`
    : 'Claim breakdown is unavailable for this saved run.';
  const ordered = [...claims].sort((left, right) => Number(left.verdict) - Number(right.verdict));
  return `<details class="metric-analysis" data-metric="faithfulness"><summary><span><strong>Faithfulness analysis</strong><small>Does each claim follow from the gathered evidence?</small></span><b>${score}</b></summary>
    <div class="analysis-body"><p class="analysis-formula">${escapeHtml(explanation)}</p>
    <p class="analysis-help">RAGAS splits the answer into claims and judges each against the saved SQL and PDF evidence. A perfect score means the judge accepted every claim; it is not proof that every claim is correct.</p>
    ${retryControl(scores, 'faithfulness', questionId)}
    ${ordered.map((claim) => `<div class="claim-item ${claim.verdict ? 'supported' : 'unsupported'}"><span class="claim-status">${claim.verdict ? 'Supported' : 'Needs review'}</span><p>${linkedText(claim.statement, ids, questionId)}</p><small>${linkedText(claim.reason, ids, questionId)}</small>
      ${ids.length ? `<details class="claim-evidence"><summary>Open cited evidence</summary><div><span>Sources cited in the saved answer; RAGAS does not assign one to this claim.</span>${sourceButtons(ids, questionId)}</div></details>` : ''}</div>`).join('')}</div></details>`;
}

function renderRelevancy(scores, questionId) {
  const score = scores.response_relevancy == null ? '—' : scores.response_relevancy.toFixed(3);
  const questions = scores.response_relevancy_questions || [];
  return `<details class="metric-analysis" data-metric="response_relevancy"><summary><span><strong>Response relevancy analysis</strong><small>How closely does the answer address the question?</small></span><b>${score}</b></summary>
    <div class="analysis-body"><p class="analysis-help">Official RAGAS requests three generated questions by default, compares each with the original question using Nomic embeddings, and averages their similarities. The judge can repeat nearly identical questions; these are not guaranteed to be distinct paraphrases. This measures focus, not factual accuracy.</p>
    ${retryControl(scores, 'response_relevancy', questionId)}
    ${questions.length ? `<p class="analysis-formula">Average of ${questions.length} question similarities = ${score}</p><ol class="relevancy-questions">${questions.map((item) =>
      `<li><span>${escapeHtml(item.question)}</span><strong>${item.similarity == null ? '—' : item.similarity.toFixed(3)}</strong></li>`).join('')}</ol>`
      : '<p class="analysis-empty">The generated questions were not saved for this run. Rerun the two-question test to see the breakdown.</p>'}</div></details>`;
}

function render(result) {
  const key = JSON.stringify(result);
  if (key === rendered) return;
  const openCases = new Set([...document.querySelectorAll('.generation-case[open]')].map((node) => node.dataset.question));
  const openAnalyses = new Set([...document.querySelectorAll('.generation-case[open] .metric-analysis[open]')].map((node) => `${node.closest('.generation-case').dataset.question}:${node.dataset.metric}`));
  rendered = key;
  const names = {faithfulness:'RAGAS Faithfulness', response_relevancy:'RAGAS Response Relevancy'};
  $('scores').innerHTML = Object.entries(result.summary).map(([name, metric]) =>
    `<article class="score-card"><small>${names[name]}</small><strong>${metric.mean == null ? '—' : metric.mean.toFixed(3)}</strong><p>Mean across ${metric.scored}/${result.total} scored answers</p></article>`).join('');
  $('cases').innerHTML = result.rows.map((row) => {
    const s = row.scores;
    const ids = citedIds(row.answer);
    const faith = s.faithfulness == null ? '—' : s.faithfulness.toFixed(3);
    const relevancy = s.response_relevancy == null ? '—' : s.response_relevancy.toFixed(3);
    return `<details class="case-card generation-case" data-question="${escapeHtml(row.id)}"><summary><span class="case-id">${escapeHtml(row.id)}</span><span class="case-title">${escapeHtml(row.question)}</span><span class="case-rank">Faithfulness ${faith} · Relevancy ${relevancy}</span></summary>
      <div class="case-body"><p class="answer-label">Saved answer</p><p class="saved-answer">${linkedText(row.answer || 'No generated answer', ids, row.id)}</p>
      <div class="analysis-grid">${renderFaithfulness(s, ids, row.id)}${renderRelevancy(s, row.id)}</div>
      ${Object.entries(s.errors || {}).length ? `<p class="evidence-label">Unavailable: ${escapeHtml(Object.entries(s.errors).map(([name, reason]) => `${name}: ${reason}`).join(' · '))}</p>` : ''}</div></details>`;
  }).join('');
  for (const node of document.querySelectorAll('.generation-case')) {
    if (openCases.has(node.dataset.question)) node.open = true;
    for (const analysis of node.querySelectorAll('.metric-analysis')) {
      if (openAnalyses.has(`${node.dataset.question}:${analysis.dataset.metric}`)) analysis.open = true;
    }
  }
}

async function openSource(questionId, sourceId) {
  const params = new URLSearchParams({question_id: questionId, source_id: sourceId});
  const response = await fetch(`/api/dev/generation-evaluation/source?${params}`, {cache: 'no-store'});
  if (!response.ok) throw new Error('This saved source is unavailable.');
  const {source} = await response.json();
  if (!source) throw new Error('This saved source is unavailable.');
  const dialog = document.createElement('dialog');
  dialog.className = 'generation-source-dialog';
  const page = Number((Array.isArray(source.pages) && source.pages.length ? source.pages[0] : source.page)) || 1;
  const title = source.type === 'pdf' ? `${source.filename || 'Filing'} · PDF page ${page}` : 'Financial database · SQL result';
  const fullPdf = source.type === 'pdf' && source.filename
    ? `<a href="/filings/${encodeURIComponent(source.filename)}#page=${page}" target="_blank" rel="noopener">Open full PDF ↗</a>` : '';
  dialog.innerHTML = `<header><div><span class="source-id">[${escapeHtml(sourceId)}]</span><strong>${escapeHtml(title)}</strong></div><div>${fullPdf}<button type="button" class="close-source" aria-label="Close source preview">×</button></div></header><div class="generation-source-content"></div>`;
  const content = dialog.querySelector('.generation-source-content');
  if (source.type === 'pdf') {
    content.classList.add('pdf-source');
    content.innerHTML = `<div class="generation-pdf-page"><div class="generation-pdf-sheet"><img alt="${escapeHtml(title)}" src="/api/chat/document-image?${new URLSearchParams({filename: source.filename, page: String(page)})}"><div class="generation-pdf-overlay"></div></div></div><div class="generation-pdf-excerpt"><h3>Cited passage</h3><p>${escapeHtml(source.text || 'No passage text saved.')}</p></div>`;
  } else {
    const columns = Array.isArray(source.columns) ? source.columns : [];
    const rows = Array.isArray(source.rows) ? source.rows : [];
    const cells = rows.slice(0, 20).map((row) => `<tr>${columns.map((column, index) =>
      `<td>${escapeHtml(Array.isArray(row) ? row[index] : row?.[column])}</td>`).join('')}</tr>`).join('');
    content.innerHTML = `${source.sql ? `<pre class="generation-sql">${escapeHtml(source.sql)}</pre>` : ''}<div class="generation-table-wrap"><table><thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join('')}</tr></thead><tbody>${cells}</tbody></table></div>${rows.length > 20 ? `<p>Showing 20 of ${rows.length} rows.</p>` : ''}`;
  }
  document.body.append(dialog);
  dialog.querySelector('.close-source').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  dialog.addEventListener('close', () => dialog.remove(), {once: true});
  dialog.showModal();
  if (source.type === 'pdf' && source.chunk_id && source.filename) {
    try {
      const highlightParams = new URLSearchParams({filename: source.filename, page: String(page), chunk_id: source.chunk_id});
      const highlights = await fetch(`/api/chat/document-highlights?${highlightParams}`, {cache: 'no-store'});
      if (!highlights.ok || !dialog.open) return;
      const data = await highlights.json();
      const overlay = dialog.querySelector('.generation-pdf-overlay');
      for (const box of data.boxes || []) {
        if (!Array.isArray(box) || box.length !== 4 || !data.width || !data.height) continue;
        const mark = document.createElement('div');
        mark.className = 'generation-pdf-mark';
        mark.style.left = `${100 * box[0] / data.width}%`;
        mark.style.top = `${100 * box[1] / data.height}%`;
        mark.style.width = `${100 * (box[2] - box[0]) / data.width}%`;
        mark.style.height = `${100 * (box[3] - box[1]) / data.height}%`;
        overlay.append(mark);
      }
    } catch { /* The cited passage remains visible alongside the PDF page. */ }
  }
}

$('cases').addEventListener('click', (event) => {
  const retry = event.target.closest('.metric-retry-button');
  if (retry) {
    retryMetric(retry.dataset.question, retry.dataset.metric);
    return;
  }
  const button = event.target.closest('.generation-source-link');
  if (!button) return;
  openSource(button.dataset.question, button.dataset.source).catch((error) => {
    $('job-message').textContent = error.message;
  });
});

async function refresh() {
  const response = await fetch('/api/dev/generation-evaluation', {cache:'no-store'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load evaluation');
  const ready = data.job.state !== 'running' && data.answer_run_available && data.key_available && data.ragas_available;
  retryBusy = data.job.state === 'running';
  $('pilot-button').disabled = !ready;
  $('full-button').disabled = !ready || !data.pilot_ready;
  $('job-message').textContent = data.job.state === 'running'
    ? `${data.job.message} (${data.job.done}/${data.job.total})`
    : data.job.message || (!data.answer_run_available ? 'Generate answers in Benchmark Evaluation first.'
      : !data.ragas_available ? 'RAGAS is not installed. Install starter/requirements.txt and restart the server.'
      : !data.key_available ? 'Set FIREWORKS_API_KEY and restart the server.'
      : !data.pilot_ready ? 'Start with two saved questions. All ten unlock after both RAGAS metrics succeed on both.'
      : 'Pilot passed; you can now score all ten saved answers.');
  renderEvalProgress($('job-message'), data.job);
  if (data.result) render(data.result);
  for (const button of document.querySelectorAll('.metric-retry-button')) button.disabled = retryBusy;
  if (data.job.state === 'running' && !poller) poller = setInterval(() => refresh().catch(showError), 1200);
  if (data.job.state !== 'running' && poller) { clearInterval(poller); poller = null; }
}
function showError(error) { $('job-message').textContent = error.message; if (poller) { clearInterval(poller); poller = null; } }
async function start(mode) {
  $('pilot-button').disabled = true;
  $('full-button').disabled = true;
  try {
    const response = await fetch('/api/dev/generation-evaluation/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not start RAGAS evaluation');
    await refresh();
  } catch (error) { showError(error); await refresh(); }
}
async function retryMetric(questionId, metric) {
  if (retryBusy) return;
  retryBusy = true;
  for (const button of document.querySelectorAll('.metric-retry-button')) button.disabled = true;
  $('job-message').textContent = `Retrying ${metric === 'faithfulness' ? 'Faithfulness' : 'Response Relevancy'} for ${questionId}…`;
  try {
    const response = await fetch('/api/dev/generation-evaluation/retry', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question_id: questionId, metric}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not retry this metric');
    await refresh();
  } catch (error) {
    retryBusy = false;
    showError(error);
    for (const button of document.querySelectorAll('.metric-retry-button')) button.disabled = false;
  }
}
$('pilot-button').addEventListener('click', () => start('pilot'));
$('full-button').addEventListener('click', () => start('full'));
refresh().catch(showError);
