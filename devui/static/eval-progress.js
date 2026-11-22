/* Shared progress indicator for developer evaluation jobs. */
window.renderEvalProgress = function renderEvalProgress(statusNode, job) {
  if (!statusNode) return;
  let container = statusNode.nextElementSibling;
  if (!container?.classList.contains('eval-progress')) {
    container = document.createElement('div');
    container.className = 'eval-progress';
    container.innerHTML = '<div class="eval-progress-track"><span></span></div><small></small>';
    statusNode.insertAdjacentElement('afterend', container);
  }
  const running = job?.state === 'running';
  container.hidden = !running;
  if (!running) return;
  const done = Math.max(0, Number(job.done) || 0);
  const total = Number(job.total) || 0;
  const fill = container.querySelector('.eval-progress-track span');
  container.classList.toggle('indeterminate', total <= 0);
  fill.style.width = total > 0 ? `${Math.min(100, Math.round(done / total * 100))}%` : '';
  container.querySelector('small').textContent = total > 0
    ? `${Math.min(done, total)} of ${total} complete · working on the next step`
    : 'Working…';
};
