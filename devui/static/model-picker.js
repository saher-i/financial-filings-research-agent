async function initModelPicker(selectId, refreshId, statusId) {
  const select = document.getElementById(selectId);
  const button = document.getElementById(refreshId);
  const status = document.getElementById(statusId);
  if (!select || !button || !status) return;

  function populate(data) {
    const chosen = select.value || sessionStorage.getItem('evaluation-model-v2') || data.default;
    const models = [...(data.models || [])];
    if (!models.some(item => item.id === data.default)) {
      models.unshift({id: data.default, label: `${data.default.split('/').pop()} (default)`});
    }
    select.replaceChildren();
    for (const item of models) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.label;
      select.append(option);
    }
    select.value = models.some(item => item.id === chosen) ? chosen : data.default;
    status.textContent = models.length > 1
      ? `${models.length} tool-capable models available`
      : 'Using the default model. Refresh to discover other Fireworks models.';
  }

  select.addEventListener('change', () => sessionStorage.setItem('evaluation-model-v2', select.value));
  button.addEventListener('click', async () => {
    button.disabled = true;
    status.textContent = 'Checking Fireworks models…';
    try {
      const response = await fetch('/api/dev/models/refresh', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not load models');
      const current = await (await fetch('/api/dev/models', {cache: 'no-store'})).json();
      populate(current);
    } catch (error) {
      status.textContent = error.message || 'Could not load models.';
    } finally {
      button.disabled = false;
    }
  });
  try {
    const response = await fetch('/api/dev/models', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load models');
    populate(data);
  } catch (error) {
    status.textContent = error.message || 'Could not load models.';
  }
}
