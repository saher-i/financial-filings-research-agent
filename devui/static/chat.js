const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const sendButton = document.getElementById('send');
const newChatButton = document.getElementById('new-chat');
const conversation = document.getElementById('conversation');
const statusLine = document.getElementById('status');
const welcome = document.getElementById('welcome');
const recentChats = document.getElementById('recent-chats');
const modelSelect = document.getElementById('chat-model');
const modelStatus = document.getElementById('model-status');
let selectedChatId = sessionStorage.getItem('active-chat-id') || null;

function showLatestTurn() {
  requestAnimationFrame(() => {
    conversation.scrollTo({top: conversation.scrollHeight, behavior: 'smooth'});
  });
}

async function loadChatModels() {
  try {
    const response = await fetch('/api/chat/models', {cache:'no-store'});
    if (!response.ok) throw new Error('Could not load agent models.');
    const data = await response.json();
    const devLink = document.getElementById('dev-console-link');
    if (devLink) devLink.hidden = !data.developer_mode;
    const models = Array.isArray(data.models) ? data.models : [];
    if (!models.length) throw new Error('No agent models are available.');
    const preferred = sessionStorage.getItem('chat-model-v2') || data.default;
    modelSelect.replaceChildren();
    for (const model of models) {
      const option = element('option', '', model.label || model.id.split('/').pop());
      option.value = model.id;
      modelSelect.append(option);
    }
    modelSelect.value = models.some(item => item.id === preferred) ? preferred : data.default;
    modelStatus.textContent = '';
  } catch (error) {
    const fallback = element('option', '', 'Server default');
    fallback.value = '';
    modelSelect.replaceChildren(fallback);
    modelSelect.value = '';
    modelStatus.textContent = error.message || 'Using the server default model.';
  }
}

modelSelect.addEventListener('change', () => sessionStorage.setItem('chat-model-v2', modelSelect.value));

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function sourceId(source, index) {
  if (typeof source.id === 'string' && /^\[S\d+\]$/.test(source.id)) return source.id;
  if (typeof source.id === 'string' && /^S\d+$/.test(source.id)) return `[${source.id}]`;
  return `[S${index + 1}]`;
}

function appendInline(container, value, known, exchangeId) {
  const text = String(value);
  const pattern = /(\[S\d+\]|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|<br\s*\/?>)/gi;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) container.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (known.has(token)) {
      const link = element('a', '', token);
      link.href = `#source-${exchangeId}-${token.slice(2, -1)}`;
      link.setAttribute('aria-label', `Go to source ${token}`);
      link.addEventListener('click', () => {
        const source = document.getElementById(`source-${exchangeId}-${token.slice(2, -1)}`);
        if (source) {
          for (let parent = source.parentElement; parent; parent = parent.parentElement) {
            if (parent.tagName === 'DETAILS') parent.open = true;
          }
          source.open = true;
        }
      });
      container.append(link);
    } else if (/^<br/i.test(token)) {
      container.append(document.createElement('br'));
    } else if (token.startsWith('**')) {
      const strong = element('strong');
      appendInline(strong, token.slice(2, -2), known, exchangeId);
      container.append(strong);
    } else if (token.startsWith('*')) {
      const emphasis = element('em');
      appendInline(emphasis, token.slice(1, -1), known, exchangeId);
      container.append(emphasis);
    } else if (token.startsWith('`')) {
      container.append(element('code', '', token.slice(1, -1)));
    } else {
      container.append(document.createTextNode(token));
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) container.append(document.createTextNode(text.slice(cursor)));
}

function isTableRule(line) {
  return /^\s*\|?\s*:?-{3,}/.test(line);
}

function tableCells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());
}

function renderAnswerText(container, answer, sources, exchangeId) {
  const known = new Set(sources.map(sourceId));
  const lines = String(answer).replace(/\r/g, '').split('\n');
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { index += 1; continue; }
    if (line.startsWith('|') && index + 1 < lines.length && isTableRule(lines[index + 1])) {
      const table = element('table');
      const header = element('tr');
      for (const cell of tableCells(lines[index])) {
        const th = element('th');
        appendInline(th, cell, known, exchangeId);
        header.append(th);
      }
      table.append(header);
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith('|')) {
        const row = element('tr');
        for (const cell of tableCells(lines[index])) {
          const td = element('td');
          appendInline(td, cell, known, exchangeId);
          row.append(td);
        }
        table.append(row);
        index += 1;
      }
      container.append(table);
      continue;
    }
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) {
      const title = element('h3');
      appendInline(title, heading[1], known, exchangeId);
      container.append(title);
      index += 1;
      continue;
    }
    const bullet = line.match(/^[-*•]\s+(.+)$/);
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (bullet || numbered) {
      const list = element(bullet ? 'ul' : 'ol');
      while (index < lines.length) {
        const itemLine = lines[index].trim();
        const item = itemLine.match(bullet ? /^[-*•]\s+(.+)$/ : /^\d+[.)]\s+(.+)$/);
        if (!item) break;
        const li = element('li');
        appendInline(li, item[1], known, exchangeId);
        list.append(li);
        index += 1;
      }
      container.append(list);
      continue;
    }
    const paragraph = [];
    while (index < lines.length && lines[index].trim() &&
           !/^#{1,3}\s|^[-*•]\s|^\d+[.)]\s/.test(lines[index].trim()) &&
           !(lines[index].trim().startsWith('|') && index + 1 < lines.length && isTableRule(lines[index + 1]))) {
      paragraph.push(lines[index].trim().replace(/^>\s?/, ''));
      index += 1;
    }
    if (!paragraph.length) { index += 1; continue; }
    const p = element('p');
    appendInline(p, paragraph.join(' '), known, exchangeId);
    container.append(p);
  }
}

function renderSqlTable(container, source) {
  if (!Array.isArray(source.columns) || !Array.isArray(source.rows) || !source.columns.length) return;
  const table = element('table', 'source-table');
  const head = element('thead');
  const headerRow = element('tr');
  source.columns.forEach((column) => headerRow.append(element('th', '', column)));
  head.append(headerRow);
  const body = element('tbody');
  source.rows.slice(0, 20).forEach((row) => {
    const tr = element('tr');
    source.columns.forEach((column, index) => {
      const value = Array.isArray(row) ? row[index] : row?.[column];
      tr.append(element('td', '', value == null ? '—' : value));
    });
    body.append(tr);
  });
  table.append(head, body);
  container.append(table);
  if (source.rows.length > 20) container.append(element('p', '', `Showing 20 of ${source.rows.length} rows.`));
}

async function openFilingPreview(source, page) {
  const dialog = element('dialog', 'filing-preview');
  const header = element('div', 'filing-preview-header');
  const title = element('div');
  title.append(element('strong', '', source.filename || 'Annual filing'));
  const pageLine = element('div', 'filing-preview-page-line');
  pageLine.append(element('span', '', `PDF page ${page}`));
  const fullPdf = element('a', 'filing-preview-full', 'Open full PDF ↗');
  fullPdf.href = `/filings/${encodeURIComponent(source.filename)}#page=${page}`;
  fullPdf.target = '_blank';
  fullPdf.rel = 'noopener';
  pageLine.append(fullPdf);
  title.append(pageLine);
  const close = element('button', 'filing-preview-close', '×');
  close.type = 'button';
  close.setAttribute('aria-label', 'Close filing preview');
  close.addEventListener('click', () => dialog.close());
  header.append(title, close);
  const body = element('div', 'filing-preview-body');
  const pageWrap = element('div', 'filing-preview-page');
  const sheet = element('div', 'filing-preview-sheet');
  const image = element('img', 'filing-preview-image');
  image.alt = `${source.filename}, PDF page ${page}`;
  image.src = `/api/chat/document-image?${new URLSearchParams({filename:source.filename, page:String(page)})}`;
  const overlay = element('div', 'filing-preview-overlay');
  sheet.append(image, overlay);
  pageWrap.append(sheet);
  const excerpt = element('aside', 'filing-preview-excerpt');
  excerpt.append(element('h3', '', 'Highlighted cited passage'));
  const highlighted = element('mark', '', source.text || 'The cited passage is unavailable for this source.');
  excerpt.append(highlighted);
  body.append(pageWrap, excerpt);
  dialog.append(header, body);
  document.body.append(dialog);
  dialog.addEventListener('close', () => dialog.remove(), {once:true});
  dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  dialog.showModal();
  if (source.chunk_id) {
    try {
      const params = new URLSearchParams({filename:source.filename, page:String(page), chunk_id:source.chunk_id});
      const response = await fetch(`/api/chat/document-highlights?${params}`, {cache:'no-store'});
      if (!response.ok || !dialog.open) return;
      const data = await response.json();
      for (const box of data.boxes || []) {
        if (!Array.isArray(box) || box.length !== 4 || !data.width || !data.height) continue;
        const mark = element('div', 'filing-preview-mark');
        mark.style.left = `${100 * box[0] / data.width}%`;
        mark.style.top = `${100 * box[1] / data.height}%`;
        mark.style.width = `${100 * (box[2] - box[0]) / data.width}%`;
        mark.style.height = `${100 * (box[3] - box[1]) / data.height}%`;
        overlay.append(mark);
      }
    } catch { /* The companion excerpt still identifies the cited text. */ }
  }
}

function appendSourceRows(container, sources, exchangeId, allSources) {
  sources.forEach((source) => {
    const index = allSources.indexOf(source);
    const id = sourceId(source, index);
    const details = element('details', 'source');
    details.id = `source-${exchangeId}-${id.slice(2, -1)}`;
    const summary = element('summary');
    summary.append(element('span', 'source-index', id));
    const pdf = source.type === 'pdf' || Boolean(source.filename);
    const pages = Array.isArray(source.pages) ? source.pages : source.page ? [source.page] : [];
    const title = pdf
      ? `${source.filename || 'Filing'}${pages.length ? ` · PDF p.${pages.join(', ')}` : ''}`
      : 'Financial database · SQL result';
    summary.append(element('span', 'source-label', title));
    const content = element('div', 'source-content');
    if (pdf) {
      if (source.text) content.append(element('p', '', source.text));
      if (source.filename) {
        const preview = element('button', 'source-preview-button', 'Preview cited page ↗');
        preview.type = 'button';
        preview.addEventListener('click', () => openFilingPreview(source, Number(pages[0]) || 1));
        content.append(preview);
      }
    } else {
      if (source.sql) content.append(element('pre', '', source.sql));
      renderSqlTable(content, source);
    }
    details.append(summary, content);
    container.append(details);
  });
}

function renderSources(container, sources, exchangeId, answer) {
  if (!sources.length) return;
  const citedIds = new Set((String(answer).match(/\[S\d+\]/g) || []));
  const cited = sources.filter((source, index) => citedIds.has(sourceId(source, index)));
  const other = sources.filter((source) => !cited.includes(source));
  if (cited.length) {
    const citedGroup = element('details', 'source-group');
    citedGroup.append(element('summary', 'sources-title', `Cited sources · ${cited.length}`));
    appendSourceRows(citedGroup, cited, exchangeId, sources);
    container.append(citedGroup);
  }
  if (other.length) {
    const more = element('details', 'other-sources');
    const summary = element('summary', 'sources-title',
      `${cited.length ? 'Other retrieved evidence' : 'Retrieved evidence'} · ${other.length}`);
    more.append(summary);
    appendSourceRows(more, other, exchangeId, sources);
    container.append(more);
  }
}

function createExchange(question, scroll = true) {
  welcome.classList.add('has-conversation');
  const exchange = element('article', 'exchange');
  const exchangeId = conversation.children.length + 1;
  const questionCard = element('div', 'question-card');
  questionCard.append(element('p', '', question));
  const answerBlock = element('div', 'answer-block');
  const header = element('div', 'assistant-header');
  const modelLabel = element('span', 'assistant-model');
  header.append(element('span', 'assistant-icon', '10K'), element('span', '', '10K research agent'), modelLabel);
  const activity = element('p', 'activity is-running', 'Choosing sources');
  const activityLog = element('div', 'activity-log');
  const answerText = element('div', 'answer-text');
  const actions = element('div', 'answer-actions');
  const evidence = element('div', 'sources');
  answerBlock.append(header, activity, activityLog, answerText, actions, evidence);
  exchange.append(questionCard, answerBlock);
  conversation.append(exchange);
  if (scroll) showLatestTurn();
  return {exchange, exchangeId, activity, activityLog, answerText, actions, evidence, modelLabel,
          steps:new Map(), activeEvents:new Map(), completedEvents:new Set(), timer:null, startedAt:null};
}

function formatDuration(milliseconds) {
  const value = Number(milliseconds);
  if (!Number.isFinite(value) || value < 0) return null;
  return value < 1000 ? `${Math.round(value)}ms` : `${(value / 1000).toFixed(1)}s`;
}

function progressStage(stage) {
  if (stage === 'planning') return {key:'plan', label:'Choose sources'};
  if (['sql', 'filings', 'pages'].includes(stage)) return {key:'evidence', label:'Gather evidence'};
  if (stage === 'calculation') return {key:'calculation', label:'Calculate'};
  if (stage === 'verification') return {key:'verification', label:'Verify answer'};
  return {key:'answer', label:'Compose answer'};
}

function renderProgressStage(group) {
  const running = group.active.size > 0;
  const activeMs = running && group.activeSince != null ? performance.now() - group.activeSince : 0;
  const duration = formatDuration(group.durationMs + activeMs);
  group.node.textContent = `${running ? '●' : group.failed ? '!' : '✓'} ${group.label}${duration ? ` · ${duration}` : ''}`;
  group.node.classList.toggle('is-active', running);
  group.node.classList.toggle('has-error', !running && group.failed);
}

function showProgress(view, progress) {
  const stage = progressStage(progress?.stage);
  const id = String(progress?.id || `${stage.key}-${view.completedEvents.size}`);
  let group = view.steps.get(stage.key);
  if (!group) {
    group = {label:stage.label, node:element('span'), durationMs:0,
             active:new Set(), activeSince:null, failed:false};
    view.steps.set(stage.key, group);
    view.activityLog.append(group.node);
  }
  if (progress?.phase === 'complete') {
    if (view.completedEvents.has(id)) return;
    view.completedEvents.add(id);
    group.active.delete(id);
    view.activeEvents.delete(id);
    group.durationMs += Math.max(0, Number(progress?.duration_ms) || 0);
    group.failed ||= progress?.status === 'error';
    group.activeSince = group.active.size ? performance.now() : null;
  } else if (!view.activeEvents.has(id)) {
    if (!group.active.size) group.activeSince = performance.now();
    group.active.add(id);
    view.activeEvents.set(id, stage.key);
  }
  renderProgressStage(group);
}

function finishExchange(view, result, scroll = true) {
  if (view.timer) clearInterval(view.timer);
  view.timer = null;
  for (const group of view.steps.values()) {
    group.active.clear();
    group.activeSince = null;
    renderProgressStage(group);
  }
  view.activity.classList.remove('is-running');
  if (result.model) view.modelLabel.textContent = `· ${result.model.split('/').pop()}`;
  const duration = formatDuration(result.total_ms);
  const label = !result.answer_status || result.answer_status === 'answered'
    ? 'Answer ready' : 'Answer needs verification';
  view.activity.textContent = `${label}${duration ? ` · ${duration} total` : ''}`;
  const sources = Array.isArray(result.sources) ? result.sources : [];
  renderAnswerText(view.answerText, result.answer || '', sources, view.exchangeId);
  const copy = element('button', 'copy-button', 'Copy answer');
  copy.type = 'button';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(result.answer || '');
      copy.textContent = 'Copied';
      setTimeout(() => { copy.textContent = 'Copy answer'; }, 1800);
    } catch { copy.textContent = 'Could not copy'; }
  });
  view.actions.append(copy);
  renderSources(view.evidence, sources, view.exchangeId, result.answer || '');
  if (scroll) requestAnimationFrame(() => {
    view.exchange.querySelector('.answer-block')?.scrollIntoView({behavior:'smooth', block:'start'});
  });
}

function parseEvent(block) {
  const lines = block.split('\n');
  const event = lines.find((line) => line.startsWith('event:'))?.slice(6).trim();
  const data = lines.filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
  return event && data ? {event, data: JSON.parse(data)} : null;
}

async function loadRecentChats() {
  try {
    const response = await fetch('/api/chat/history', {cache:'no-store'});
    if (!response.ok) throw new Error('History unavailable');
    const payload = await response.json();
    recentChats.replaceChildren();
    if (!payload.chats?.length) {
      recentChats.append(element('p', 'recent-empty', 'Your questions will appear here.'));
      return;
    }
    for (const chat of payload.chats) {
      const row = element('div', 'recent-chat-row');
      const button = element('button', 'recent-chat', chat.question);
      button.type = 'button';
      button.title = chat.question;
      button.classList.toggle('active', chat.id === selectedChatId);
      button.addEventListener('click', () => openRecentChat(chat.id));
      const remove = element('button', 'delete-chat', '×');
      remove.type = 'button';
      remove.title = `Delete chat: ${chat.question}`;
      remove.setAttribute('aria-label', `Delete chat: ${chat.question}`);
      remove.addEventListener('click', () => deleteChat(chat.id, chat.question));
      row.append(button, remove);
      recentChats.append(row);
    }
  } catch {
    recentChats.replaceChildren(element('p', 'recent-empty', 'Restart the server to load recent chats.'));
  }
}

async function deleteChat(id, title) {
  if (sendButton.disabled || !window.confirm(`Delete this chat and all its answers?\n\n${title}`)) return;
  try {
    const response = await fetch(`/api/chat/conversations/${encodeURIComponent(id)}`, {method:'DELETE'});
    if (!response.ok) throw new Error('Could not delete this chat.');
    if (selectedChatId === id) {
      conversation.replaceChildren();
      welcome.classList.remove('has-conversation');
      selectedChatId = null;
      sessionStorage.removeItem('active-chat-id');
    }
    statusLine.textContent = '';
    await loadRecentChats();
  } catch (error) {
    statusLine.classList.add('error');
    statusLine.textContent = error.message;
  }
}

async function openRecentChat(id) {
  if (sendButton.disabled) return;
  try {
    const response = await fetch(`/api/chat/conversations/${encodeURIComponent(id)}`, {cache:'no-store'});
    if (!response.ok) throw new Error('Could not open this chat.');
    const {conversation: saved} = await response.json();
    conversation.replaceChildren();
    for (const chat of saved.turns || []) {
      const view = createExchange(chat.question, false);
      for (const event of (chat.events || []).filter(item => item.phase === 'complete')) showProgress(view, event);
      finishExchange(view, chat, false);
    }
    selectedChatId = id;
    sessionStorage.setItem('active-chat-id', id);
    statusLine.textContent = '';
    await loadRecentChats();
    showLatestTurn();
  } catch (error) {
    if (selectedChatId === id) {
      selectedChatId = null;
      sessionStorage.removeItem('active-chat-id');
    }
    statusLine.classList.add('error');
    statusLine.textContent = error.message;
  }
}

async function ask(question) {
  questionInput.value = '';
  sendButton.disabled = true;
  newChatButton.disabled = true;
  questionInput.disabled = true;
  modelSelect.disabled = true;
  statusLine.classList.remove('error');
  statusLine.textContent = '';
  const view = createExchange(question);
  view.startedAt = performance.now();
  view.timer = setInterval(() => {
    view.activity.textContent = `Working · ${formatDuration(performance.now() - view.startedAt)} total`;
    for (const group of view.steps.values()) if (group.active.size) renderProgressStage(group);
  }, 250);
  const answerStyle = document.querySelector('input[name="answer-style"]:checked')?.value || 'concise';
  const model = modelSelect.value;
  let finalResult = null;
  try {
    const response = await fetch('/api/chat/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question, answer_style:answerStyle, ...(model ? {model} : {}),
        ...(selectedChatId ? {conversation_id:selectedChatId} : {})})
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || 'The assistant could not answer this question.');
    }
    if (!response.body) throw new Error('The browser could not read the answer stream.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream:!done});
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const event = parseEvent(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (event?.event === 'progress') showProgress(view, event.data);
        if (event?.event === 'answer') finalResult = event.data;
        if (event?.event === 'error') throw new Error(event.data.error || 'The assistant could not answer.');
      }
      if (done) break;
    }
    if (!finalResult) throw new Error('The answer stream ended before an answer arrived.');
    finishExchange(view, finalResult);
    selectedChatId = finalResult.conversation_id || null;
    if (selectedChatId) sessionStorage.setItem('active-chat-id', selectedChatId);
    await loadRecentChats();
    statusLine.textContent = '';
  } catch (error) {
    if (view.timer) clearInterval(view.timer);
    view.timer = null;
    view.activity.classList.remove('is-running');
    view.activity.textContent = 'Could not complete this answer';
    statusLine.classList.add('error');
    statusLine.textContent = error.message || 'The assistant could not answer this question.';
  } finally {
    sendButton.disabled = false;
    newChatButton.disabled = false;
    questionInput.disabled = false;
    modelSelect.disabled = false;
    questionInput.focus();
  }
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) ask(question);
});

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

newChatButton.addEventListener('click', () => {
  conversation.replaceChildren();
  welcome.classList.remove('has-conversation');
  selectedChatId = null;
  sessionStorage.removeItem('active-chat-id');
  loadRecentChats();
  statusLine.textContent = '';
  questionInput.value = '';
  questionInput.focus();
});

loadRecentChats().then(() => { if (selectedChatId) openRecentChat(selectedChatId); });
loadChatModels();
