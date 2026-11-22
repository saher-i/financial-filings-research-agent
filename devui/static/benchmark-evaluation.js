const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let poller = null;
let lastResult = '';
let lastAnswers = '';

function renderAnswers(rows) {
  const openCases = new Set([...document.querySelectorAll('#answer-cases .case-card[open]')].map(node => node.dataset.question));
  $('answer-cases').innerHTML = rows.map(row => {
    const status = row.error ? 'generation error' : (row.answer_status || (row.answer ? 'answered' : 'no answer'));
    const diagnostic = row.error || row.failure_reason;
    return `<details class="case-card" data-question="${escapeHtml(row.id)}" ${openCases.has(row.id) ? 'open' : ''}>
      <summary><span class="case-id">${escapeHtml(row.id)}</span><span class="case-title">${escapeHtml(row.question)}</span><span class="case-rank">${escapeHtml(status)}${row.elapsed_seconds != null ? ` · ${escapeHtml(row.elapsed_seconds)}s` : ''}</span></summary>
      <div class="case-body"><p><strong>Saved agent answer</strong><br>${escapeHtml(row.answer || 'No answer was saved.')}</p>
      ${diagnostic ? `<p class="evidence-label">${escapeHtml(diagnostic)}</p>` : ''}
      <button class="primary answer-retry" type="button" data-question="${escapeHtml(row.id)}">Retry this answer ↻</button></div></details>`;
  }).join('') || '<p>No generated answers yet.</p>';
}

function render(result) {
  const openCases = new Set([...document.querySelectorAll('.case-card[open]')].map(node => node.dataset.question));
  const counts = result.counts;
  $('scores').innerHTML = ['correct','partial','incorrect','unavailable'].map(status =>
    `<article class="score-card"><small>${status}</small><strong>${counts[status]}</strong></article>`).join('');
  $('cases').innerHTML = result.rows.map(row => {
    const detail = row.method === 'llm_judge'
      ? row.status === 'unavailable'
        ? `<p class="evidence-label">Judge: ${escapeHtml(row.model)} · Failed at ${escapeHtml(row.error_stage || 'judging')} with ${escapeHtml(row.reason || 'unknown error')}${row.error_detail ? `: ${escapeHtml(row.error_detail)}` : ''}. Retry this judgment below.</p>`
        : `<p class="evidence-label">Judge: ${escapeHtml(row.model)} · Coverage ${escapeHtml(row.dimensions?.coverage)} · Grounding ${escapeHtml(row.dimensions?.faithfulness)}</p>
           <ul>${(row.findings || []).map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
      : `<p class="evidence-label">${escapeHtml(row.reason)} · expected ${escapeHtml(row.expected_numeric)}${row.expected_entity ? ` · entity ${escapeHtml(row.expected_entity)} (${row.entity_match ? 'matched' : 'missing'})` : ''}</p>`;
    const retry = row.status === 'unavailable' && row.answer
      ? `<button class="primary benchmark-retry" type="button" data-question="${escapeHtml(row.id)}">Retry this judgment ↻</button>` : '';
    const statusLabel = row.status === 'unavailable' && row.answer ? 'judgment unavailable' : row.status;
    return `<details class="case-card" data-question="${escapeHtml(row.id)}" ${openCases.has(row.id) ? 'open' : ''}><summary><span class="case-id">${escapeHtml(row.id)}</span><span class="case-title">${escapeHtml(row.question)}</span><span class="case-rank">${escapeHtml(row.method)} · ${escapeHtml(statusLabel)}</span></summary>
      <div class="case-body"><p><strong>Saved answer</strong><br>${escapeHtml(row.answer || 'No answer')}</p>
      <p><strong>Supplied expected answer</strong><br>${escapeHtml(row.expected_answer)}</p>${detail}${retry}</div></details>`;
  }).join('');
}

async function refresh() {
  const response = await fetch('/api/dev/benchmark-evaluation', {cache:'no-store'});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Could not load benchmark status');
  const generating = data.answer_job.state === 'running';
  const grading = data.job.state === 'running';
  const answerKey = JSON.stringify(data.answer_rows || []);
  if (answerKey !== lastAnswers) { lastAnswers = answerKey; renderAnswers(data.answer_rows || []); }
  $('generate-button').disabled = generating || grading || !data.key_available;
  $('generate-message').textContent = data.answer_job.message || (data.answer_run_available ? 'A saved answer run is available.' : 'Ready to generate the ten answers.');
  renderEvalProgress($('generate-message'), data.answer_job);
  $('run-button').disabled = grading || generating || !data.answer_run_available || !data.key_available;
  $('job-message').textContent = data.job.message || (!data.answer_run_available ? 'Generate the ten answers first.' :
    !data.key_available ? 'Set FIREWORKS_API_KEY for the LLM judge.' : 'Ready to grade saved answers.');
  renderEvalProgress($('job-message'), data.job);
  const currentResult = data.result?.answer_run_id === data.answer_run_id ? data.result : null;
  const key = currentResult ? JSON.stringify(currentResult) : '';
  if (!currentResult && lastResult) { lastResult = ''; $('scores').innerHTML = ''; $('cases').innerHTML = ''; }
  if (key && key !== lastResult) { lastResult = key; render(currentResult); }
  for (const button of document.querySelectorAll('.benchmark-retry')) button.disabled = grading || generating;
  for (const button of document.querySelectorAll('.answer-retry')) button.disabled = grading || generating || !data.key_available;
  if (data.result && data.answer_run_id !== data.result.answer_run_id && data.job.state !== 'running')
    $('job-message').textContent = 'These scores belong to an older answer run. Grade the current saved answers again.';
  if ((grading || generating) && !poller) poller = setInterval(() => refresh().catch(showError), 1200);
  if (!grading && !generating && poller) { clearInterval(poller); poller = null; }
}
function showError(error) { $('job-message').textContent = error.message; $('generate-message').textContent = error.message; if (poller) { clearInterval(poller); poller = null; } }
$('generate-button').addEventListener('click', async () => {
  $('generate-button').disabled = true;
  try {
    const response = await fetch('/api/dev/benchmark-evaluation/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:$('answer-model').value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not start answer generation');
    await refresh();
  } catch (error) { showError(error); $('generate-button').disabled = false; }
});
$('answer-cases').addEventListener('click', async event => {
  const button = event.target.closest('.answer-retry');
  if (!button) return;
  button.disabled = true;
  try {
    const response = await fetch('/api/dev/benchmark-evaluation/retry-answer', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question_id:button.dataset.question, model:$('answer-model').value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not retry this answer');
    await refresh();
  } catch (error) { showError(error); button.disabled = false; }
});
$('run-button').addEventListener('click', async () => {
  $('run-button').disabled = true;
  try {
    const response = await fetch('/api/dev/benchmark-evaluation/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not start benchmark grading');
    await refresh();
  } catch (error) { showError(error); $('run-button').disabled = false; }
});
$('cases').addEventListener('click', async event => {
  const button = event.target.closest('.benchmark-retry');
  if (!button) return;
  button.disabled = true;
  try {
    const response = await fetch('/api/dev/benchmark-evaluation/retry', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question_id:button.dataset.question})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not retry this judgment');
    await refresh();
  } catch (error) { showError(error); button.disabled = false; }
});
initModelPicker('answer-model', 'answer-model-refresh', 'answer-model-status');
refresh().catch(showError);
