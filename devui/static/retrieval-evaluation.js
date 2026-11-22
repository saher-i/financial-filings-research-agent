const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const methodNames = {cited:'Cited evidence', retrieved:'All agent-retrieved evidence'};
let lastResultKey = null;
let poller = null;

function renderScores(result) {
  const scoreCard = ([name, score]) =>
    `<article class="score-card ${name === 'cited' ? 'hybrid' : ''}"><small>${methodNames[name] || name}</small><div class="score-pair"><div><small>Context precision</small><strong>${score.context_precision == null ? '—' : score.context_precision.toFixed(3)}</strong><p>${score.precision_scored} of ${result.question_count} questions scored</p></div><div><small>Context recall</small><strong>${score.context_recall == null ? '—' : score.context_recall.toFixed(3)}</strong><p>${score.recall_scored} of ${result.question_count} questions scored</p></div></div></article>`;
  $('scores').innerHTML = Object.entries(result.summary).map(scoreCard).join('');
}

function renderHit(hit, rank, judgment, questionId, method) {
  const label = judgment ? (judgment.relevant ? 'Relevant' : 'Not relevant') : 'Judgment unavailable';
  const state = judgment ? (judgment.relevant ? 'relevant' : 'irrelevant') : 'unavailable';
  return `<article class="result-hit" id="retrieval-${escapeHtml(questionId)}-${escapeHtml(method)}-${rank + 1}">
    <div class="hit-source"><span>#${rank + 1}</span><span class="judgment-badge ${state}">${label}</span></div>
    <div class="hit-source">[${escapeHtml(hit.source_id)}] ${hit.type === 'sql' ? 'Financial database · SQL result' : `${escapeHtml(hit.filename)} · PDF p.${(hit.pages || []).join(', ')}`}</div>
    <div class="hit-source">${hit.type === 'sql' ? 'Source' : 'Chunk'} ID: ${escapeHtml(hit.id)}</div>
    ${judgment?.reason ? `<details class="judge-reason"><summary>Why this label?</summary><p>${escapeHtml(judgment.reason)}</p></details>` : ''}
    <details class="chunk-details"><summary>Read evidence</summary><p>${escapeHtml(hit.text)}</p></details>
  </article>`;
}

function renderCalculation(metric, hits, questionId, method) {
  const judgments = metric.passage_judgments;
  let precision = '<p>The source labels were not saved for this run.</p>';
  if (Array.isArray(judgments) && judgments.length === hits.length) {
    let useful = 0;
    const terms = [];
    judgments.forEach((item, index) => {
      if (item.relevant) { useful += 1; terms.push(`${useful}/${index + 1}`); }
    });
    precision = useful
      ? `<p>At each relevant source, count relevant sources so far ÷ its displayed order. (${terms.join(' + ')}) ÷ ${useful} = <strong>${metric.context_precision?.toFixed(3) ?? '—'}</strong>.</p>`
      : `<p>0 relevant sources → precision ${metric.context_precision?.toFixed(3) ?? '—'}.</p>`;
  }
  const claims = metric.recall_claims;
  let recall = '<p>This run saved the recall score but not RAGAS’s claim-by-claim judgments. Rerun the pilot to see the calculation.</p>';
  if (Array.isArray(claims) && claims.length) {
    const supported = claims.filter(item => item.attributed).length;
    const rankById = new Map(hits.map((hit, index) => [hit.id, index + 1]));
    recall = `<p>${supported} supported reference claims ÷ ${claims.length} checked claims = <strong>${metric.context_recall?.toFixed(3) ?? '—'}</strong>.</p>
      <p>Supporting source or chunk IDs are a separate LLM evidence annotation; they do not change the RAGAS recall score.</p>
      ${metric.claim_chunk_link_error ? `<p>Evidence links unavailable: ${escapeHtml(metric.claim_chunk_link_error)}.</p>` : ''}
      <details class="recall-claims"><summary>See checked claims</summary><ul>${claims.map(item =>
        `<li><strong>${item.attributed ? 'Supported' : 'Not supported'}</strong> · ${escapeHtml(item.statement)}
          <div class="claim-chunk-links">${Array.isArray(item.supporting_chunk_ids)
            ? (item.supporting_chunk_ids.length ? `Supporting evidence: ${item.supporting_chunk_ids.map(id => {
                const rank = rankById.get(id);
                return rank ? `<a href="#retrieval-${escapeHtml(questionId)}-${escapeHtml(method)}-${rank}" title="${escapeHtml(id)}">#${rank}</a> <code>${escapeHtml(id)}</code>` : '';
              }).join(' · ')}` : 'No supporting source identified')
            : 'Evidence links not saved for this run'}</div>
          ${item.reason ? `<details class="judge-reason"><summary>Why?</summary><p>${escapeHtml(item.reason)}</p></details>` : ''}</li>`).join('')}</ul></details>`;
  }
  return `<details class="score-breakdown" open><summary>Score calculation</summary>
    <div><h4>Context Precision</h4>${precision}<h4>Context Recall</h4>${recall}</div></details>`;
}

function renderCases(result) {
  const initial = !document.querySelector('.case-card');
  const openCases = new Set([...document.querySelectorAll('.case-card[open]')].map(node => node.dataset.question));
  $('cases').innerHTML = result.rows.map((row, index) => {
    const target = row.case;
    const missing = Object.entries(row.metrics).flatMap(([method, metric]) =>
      ['context_precision', 'context_recall'].filter(name => metric[name] == null).map(name => `${method} ${name === 'context_precision' ? 'precision' : 'recall'}`));
    const retry = missing.length
      ? `<button class="primary retrieval-retry" type="button" data-question="${escapeHtml(target.question_id)}">Retry unavailable scores ↻</button>` : '';
    const ranks = Object.entries(row.metrics)
      .map(([name, metric]) => `${methodNames[name] || name} P ${metric.context_precision == null ? '—' : metric.context_precision.toFixed(2)} · R ${metric.context_recall == null ? '—' : metric.context_recall.toFixed(2)}`).join(' · ');
    const renderColumns = (entries) => entries.map(([name, hits]) => {
      const judgments = row.metrics[name].passage_judgments;
      const counts = Array.isArray(judgments) ? `${judgments.filter(item => item.relevant).length}/${hits.length} sources relevant` : 'Per-source labels unavailable for this run';
      return `<section class="method-column"><h3 class="method-title">${methodNames[name] || name} · precision ${row.metrics[name].context_precision == null ? '—' : row.metrics[name].context_precision.toFixed(3)} · recall ${row.metrics[name].context_recall == null ? '—' : row.metrics[name].context_recall.toFixed(3)}</h3>
      <p class="evidence-label">${counts}</p>
      ${Object.entries(row.metrics[name].errors || {}).map(([metric, error]) => `<p class="evidence-label">${escapeHtml(metric)} unavailable: ${escapeHtml(error)}</p>`).join('')}
      ${renderCalculation(row.metrics[name], hits, target.question_id, name)}
      ${hits.length ? hits.map((hit, rank) => renderHit(hit, rank, judgments?.[rank], target.question_id, name)).join('') : '<p class="empty-note">No evidence in this set.</p>'}</section>`;
    }).join('');
    const baseline = Object.entries(row.hits);
    return `<details class="case-card" data-question="${escapeHtml(target.question_id)}" ${openCases.has(target.question_id) || (index === 0 && initial) ? 'open' : ''}><summary><span class="case-id">${escapeHtml(target.question_id)}</span><span class="case-title">${escapeHtml(target.question)}</span><span class="case-rank">${escapeHtml(ranks)}</span></summary>
      <div class="case-body"><p class="evidence-label">Cited evidence is in first-citation order. All agent-retrieved evidence includes uncited sources in tool-call exposure order. Separate searches do not form one global ranking. RAGAS labels usefulness and checks support for expected claims in each set.</p>
      ${retry}<div class="method-columns agent-result-list">${renderColumns(baseline)}</div></div></details>`;
  }).join('');
}

async function refresh() {
  const response = await fetch('/api/dev/retrieval-evaluation', {cache:'no-store'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load evaluation status');
  $('readiness').textContent = data.answer_run_available
    ? `Saved benchmark answers available. ${data.judge_key_available ? 'Judge ready.' : 'Add FIREWORKS_API_KEY to .env and restart the server.'}`
    : 'Generate benchmark answers first.';
  $('run-button').disabled = data.job.state === 'running' || !data.answer_run_available || !data.judge_key_available;
  $('job-message').textContent = data.job.message || '';
  if (data.job.state === 'running') $('job-message').textContent += ` (${data.job.done || 0}/${data.job.total || 10})`;
  renderEvalProgress($('job-message'), data.job);
  if (data.job.state === 'running' && data.result) $('job-message').textContent += ' · Results below are from the previous completed run.';
  const resultKey = data.result ? JSON.stringify(data.result) : null;
  if (data.result && resultKey !== lastResultKey) {
    lastResultKey = resultKey;
    const currentMetrics = data.result.metric_implementation === 'ragas.metrics.collections.ContextPrecision/ContextRecall'
      && data.result.ranking_scope === 'saved_cited_and_agent_exposure_v2';
    if (currentMetrics) {
      renderScores(data.result);
      renderCases(data.result);
    } else {
      $('scores').innerHTML = '<p class="empty-note">This saved run measured an older search setup. Run retrieval evaluation again to score cited and retrieved evidence from the saved answers.</p>';
      $('cases').innerHTML = '';
    }
  }
  for (const button of document.querySelectorAll('.retrieval-retry')) button.disabled = data.job.state === 'running' || !data.judge_key_available;
  if (data.result && data.job.state !== 'running') $('job-message').textContent =
    data.result.ranking_scope === 'saved_cited_and_agent_exposure_v2'
      ? `Showing cited and retrieved evidence from saved answers · ${data.job.message || 'previous completed run'}`
      : 'The saved run used an older evaluation setup. Run retrieval evaluation again.';
  if (data.job.state === 'running' && !poller) poller = setInterval(() => refresh().catch(showError), 1200);
  if (data.job.state !== 'running' && poller) { clearInterval(poller); poller = null; }
}

function showError(error) { $('job-message').textContent = error.message; if (poller) { clearInterval(poller); poller = null; } }

$('eval-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('run-button').disabled = true;
  try {
    const response = await fetch('/api/dev/retrieval-evaluation/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question_count:Number($('question-count').value)})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not start evaluation');
    await refresh();
  } catch (error) { showError(error); $('run-button').disabled = false; }
});
$('cases').addEventListener('click', async (event) => {
  const button = event.target.closest('.retrieval-retry');
  if (!button) return;
  button.disabled = true;
  try {
    const response = await fetch('/api/dev/retrieval-evaluation/retry', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question_id:button.dataset.question})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not retry this question');
    await refresh();
  } catch (error) { showError(error); button.disabled = false; }
});
refresh().catch(showError);
