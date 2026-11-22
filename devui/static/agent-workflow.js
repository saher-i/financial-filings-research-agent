const runsNode = document.getElementById('runs');
const detailNode = document.getElementById('detail');
const refreshButton = document.getElementById('refresh');
let selectedId = null;

function element(tag, className = '', value = '') {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value) node.textContent = value;
  return node;
}

function duration(ms) {
  if (ms === undefined || ms === null) return '—';
  const value = Number(ms);
  if (!Number.isFinite(value)) return '—';
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
}

function chip(value, good = false) {
  return element('span', `workflow-chip${good ? ' good' : ''}`, value);
}

function details(title, body) {
  const box = element('details', 'workflow-card');
  box.append(element('summary', '', title));
  const pre = element('pre', '', typeof body === 'string' ? body : JSON.stringify(body, null, 2));
  box.append(pre);
  return box;
}

function executionGraph(run, completions) {
  const figure = element('figure', 'execution-graph');
  figure.append(element('figcaption', '', 'Execution graph · this query'));
  const path = element('div', 'execution-path');
  const nodes = [{label:'Question', detail:'User request', stage:'query'},
    ...(run.events || []).filter(event => event.phase === 'start').map(event => ({
      label:event.message || event.stage || 'Agent step',
      detail:duration(completions.get(event.id)?.duration_ms),
      stage:event.stage || 'step',
    })),
    {label:run.answer ? 'Answer' : 'No answer', detail:run.answer_status || 'complete', stage:'answer'}];
  for (const [index, item] of nodes.entries()) {
    if (index) path.append(element('span', 'execution-arrow', '→'));
    const node = element('div', `execution-node stage-${item.stage}`);
    node.append(element('span', 'execution-node-number', String(index + 1).padStart(2, '0')),
                element('strong', '', item.label), element('small', '', item.detail));
    path.append(node);
  }
  figure.append(path);
  return figure;
}

function showRun(run) {
  selectedId = run.id;
  detailNode.replaceChildren();
  detailNode.append(element('p', 'workflow-muted', new Date(run.created_at).toLocaleString()));
  detailNode.append(element('h2', '', run.question));
  const toolCalls = Array.isArray(run.trace) ? run.trace : [];
  const kinds = new Set(toolCalls.map(call => call.tool));
  const usedFilings = kinds.has('search_filings') || kinds.has('read_pages');
  const route = kinds.has('query_financials') && usedFilings ? 'SQL + filings'
    : kinds.has('query_financials') ? 'SQL' : usedFilings ? 'Filings' : 'No tool';
  const meta = element('div', 'workflow-meta');
  meta.append(chip(`Route: ${route}`, true), chip(`${duration(run.total_ms)} total`),
              ...(run.model ? [chip(`Model: ${run.model.split('/').pop()}`)] : []),
              chip(`${toolCalls.length} tool call${toolCalls.length === 1 ? '' : 's'}`),
              chip(run.answer_status || 'answered'));
  const completions = new Map((run.events || []).filter(event => event.phase === 'complete')
    .map(event => [event.id, event]));
  detailNode.append(meta, executionGraph(run, completions), element('h3', '', 'Executed steps'));
  const steps = element('ol', 'workflow-steps');
  let toolIndex = 0;
  for (const event of (run.events || []).filter(item => item.phase === 'start')) {
    const row = element('li', 'workflow-step');
    const title = element('div', 'workflow-step-title');
    title.append(element('span', '', event.message || event.stage || 'Agent step'));
    title.append(element('time', '', duration(completions.get(event.id)?.duration_ms)));
    row.append(title);
    if (event.stage === 'sql' || event.stage === 'filings' || event.stage === 'pages' || event.stage === 'calculation') {
      const call = toolCalls[toolIndex++];
      if (call) {
        row.append(element('p', 'workflow-step-detail',
          `${call.tool}${call.source_ids?.length ? ` → ${call.source_ids.map(id => `[${id}]`).join(', ')}` : ''}${call.error ? ` · ${call.error}` : ''}`));
        row.append(details('View tool arguments', call.arguments || {}));
        if (call.rewrite) row.append(details('View keyword query reconstruction', call.rewrite));
        if (call.expanded_pages?.length) row.append(details('View opened table pages', call.expanded_pages));
        if (call.result) row.append(details('View calculated result', call.result));
      }
    } else {
      row.append(element('p', 'workflow-step-detail', 'Model turn; the next tool call shows its observable choice.'));
    }
    steps.append(row);
  }
  if (!steps.children.length) steps.append(element('li', 'workflow-step', 'No step events were saved.'));
  detailNode.append(steps, element('h3', '', 'Answer'));
  detailNode.append(element('p', '', run.answer || 'No answer was returned.'));
  if (run.rejected_draft) {
    detailNode.append(details('Rejected draft · inspect its source IDs', run.rejected_draft));
  }
  detailNode.append(element('h3', '', `Evidence · ${(run.sources || []).length}`));
  for (const source of (run.sources || [])) {
    const location = source.type === 'sql' ? 'Financial database' :
      `${source.filename || 'Filing'} · PDF p.${(source.pages || []).join(', ')}`;
    detailNode.append(details(`[${source.id}] ${location}`,
      source.type === 'sql' ? {sql:source.sql, rows:source.rows} : source.text || ''));
  }
  document.querySelectorAll('.workflow-run').forEach(node =>
    node.classList.toggle('active', node.dataset.id === selectedId));
}

async function loadRuns() {
  refreshButton.disabled = true;
  try {
    const response = await fetch('/api/dev/agent-workflow', {cache:'no-store'});
    if (!response.ok) throw new Error('Could not load local traces.');
    const payload = await response.json();
    const runs = Array.isArray(payload.runs) ? payload.runs : [];
    runsNode.replaceChildren();
    if (!runs.length) {
      runsNode.append(element('p', 'workflow-muted', 'No chat runs yet. Ask a question in 10K research agent chat, then refresh.'));
      return;
    }
    for (const run of runs) {
      const button = element('button', 'workflow-run');
      button.type = 'button';
      button.dataset.id = run.id;
      button.append(element('strong', '', run.question));
      button.append(element('small', '', `${new Date(run.created_at).toLocaleString()} · ${duration(run.total_ms)}`));
      button.addEventListener('click', () => showRun(run));
      runsNode.append(button);
    }
    showRun(runs.find(run => run.id === selectedId) || runs[0]);
  } catch (error) {
    runsNode.replaceChildren(element('p', 'workflow-muted', error.message));
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener('click', loadRuns);
loadRuns();
