const $ = (selector) => document.querySelector(selector);
const numberFormat = new Intl.NumberFormat('ru-RU');

function formatBytes(bytes) {
  if (bytes < 1024) return `${numberFormat.format(bytes)} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} КБ`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} МБ`;
  return `${(bytes / 1024 ** 3).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} ГБ`;
}

function formatMetric(value) {
  return typeof value === 'number'
    ? value.toLocaleString('ru-RU', { maximumFractionDigits: 3 })
    : String(value);
}

function sendForm(form, onUploadProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', form.action);
    request.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onUploadProgress) onUploadProgress(event.loaded / event.total);
    });
    request.addEventListener('load', () => {
      let data;
      try { data = JSON.parse(request.responseText); }
      catch { reject(new Error('Сервер вернул неожиданный ответ.')); return; }
      if (request.status >= 200 && request.status < 300) resolve(data);
      else reject(new Error(data.error || 'Не удалось обработать запрос.'));
    });
    request.addEventListener('error', () => reject(new Error('Не удалось связаться с приложением.')));
    request.send(new FormData(form));
  });
}

function setStatus(message, isError = false) {
  const status = $('#activity-status');
  status.textContent = message;
  status.classList.toggle('error-message', isError);
}

function createCell(tag, text) {
  const cell = document.createElement(tag);
  cell.textContent = text == null ? '—' : String(text);
  return cell;
}

function renderPreview(dataset) {
  const header = $('#preview-head');
  const body = $('#preview-body');
  header.replaceChildren();
  body.replaceChildren();
  const keys = dataset.columns.map((column) => column.name).slice(0, 8);
  const headerRow = document.createElement('tr');
  keys.forEach((name) => headerRow.append(createCell('th', name)));
  header.append(headerRow);

  dataset.preview.forEach((row) => {
    const tableRow = document.createElement('tr');
    keys.forEach((name) => tableRow.append(createCell('td', row[name])));
    body.append(tableRow);
  });
}

function renderSelect(dataset) {
  const select = $('#target-select');
  select.replaceChildren();
  dataset.targets.forEach((target) => {
    const option = new Option(`${target.name} · ${target.kind === 'regression' ? 'числовой прогноз' : 'категории'}`, target.name);
    select.add(option);
  });
  const taxiTarget = dataset.targets.find((target) => target.name === 'fare_amount');
  if (taxiTarget) select.value = taxiTarget.name;
  $('#train-form button').disabled = dataset.targets.length === 0;
  if (!dataset.targets.length) {
    const option = new Option('Нет подходящих столбцов', '');
    select.add(option);
  }
}

function renderHistogram(container, chart) {
  container.replaceChildren();
  if (!chart || !chart.values.length) {
    $('#distribution-panel').hidden = true;
    return;
  }
  $('#distribution-panel').hidden = false;
  $('#distribution-title').textContent = chart.title;
  $('#distribution-caption').textContent = `Распределение по ${chart.sample_rows.toLocaleString('ru-RU')} строкам выборки.`;
  const maxValue = Math.max(...chart.values, 1);
  chart.values.forEach((value, index) => {
    const item = document.createElement('li');
    item.className = 'histogram-bar';
    item.title = `${chart.labels[index]}: ${numberFormat.format(value)}`;
    const count = createCell('span', numberFormat.format(value));
    count.className = 'histogram-count';
    const fill = document.createElement('span');
    fill.className = 'histogram-fill';
    fill.style.height = `${Math.max(4, (value / maxValue) * 82)}px`;
    fill.setAttribute('aria-hidden', 'true');
    const label = createCell('span', chart.labels[index]);
    label.className = 'histogram-label';
    item.append(count, fill, label);
    container.append(item);
  });
}

function renderCategories(container, chart) {
  container.replaceChildren();
  if (!chart || !chart.values.length) {
    $('#category-panel').hidden = true;
    return;
  }
  $('#category-panel').hidden = false;
  $('#category-title').textContent = chart.title;
  $('#category-caption').textContent = `Частота значений по ${chart.sample_rows.toLocaleString('ru-RU')} строкам выборки.`;
  const maxValue = Math.max(...chart.values, 1);
  chart.values.forEach((value, index) => {
    const item = document.createElement('li');
    item.className = 'category-row';
    const label = createCell('span', chart.labels[index]);
    label.className = 'category-label';
    label.title = chart.labels[index];
    const track = document.createElement('span');
    track.className = 'category-track';
    const fill = document.createElement('span');
    fill.className = 'category-fill';
    fill.style.width = `${Math.max(2, (value / maxValue) * 100)}%`;
    fill.setAttribute('aria-hidden', 'true');
    track.append(fill);
    const count = createCell('span', numberFormat.format(value));
    count.className = 'category-count';
    item.append(label, track, count);
    container.append(item);
  });
}

function renderDataset(dataset) {
  $('#results').hidden = false;
  $('#dataset-name').textContent = dataset.filename;
  $('#dataset-meta').textContent = `${dataset.seconds} с на чтение · ${formatBytes(dataset.size_bytes)} · анализ всех строк`;
  $('#metric-rows').textContent = numberFormat.format(dataset.rows);
  $('#metric-columns').textContent = numberFormat.format(dataset.column_count);
  $('#metric-size').textContent = formatBytes(dataset.size_bytes);
  $('#metric-missing').textContent = numberFormat.format(dataset.missing_cells);
  $('#sample-size').textContent = `Выборка: ${numberFormat.format(dataset.sample_rows)} строк`;
  renderPreview(dataset);
  renderSelect(dataset);
  renderHistogram($('#distribution-chart'), dataset.distribution);
  renderCategories($('#category-chart'), dataset.categories);
  $('#training-status').textContent = '';
  $('#model-result').hidden = true;
  $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderModel(model) {
  if (!model) return;
  const result = $('#model-result');
  result.hidden = false;
  $('#result-task').textContent = `${model.task} · «${model.target}» · ${numberFormat.format(model.rows)} строк`;
  const labels = {
    mae: 'MAE', rmse: 'RMSE', r2: 'R²', accuracy: 'Accuracy', f1: 'F1', classes: 'Классов',
  };
  const metrics = $('#model-metrics');
  metrics.replaceChildren();
  Object.entries(model.metrics).forEach(([key, value]) => {
    const card = document.createElement('div');
    card.className = 'model-metric';
    const name = createCell('span', labels[key] || key);
    const number = createCell('strong', formatMetric(value));
    card.append(name, number);
    metrics.append(card);
  });

  const list = $('#importance-list');
  list.replaceChildren();
  const maximum = Math.max(...model.features.map((feature) => feature.value), 0.0001);
  model.features.forEach((feature) => {
    const row = document.createElement('li');
    row.className = 'importance-row';
    const name = createCell('span', feature.name);
    name.className = 'importance-name';
    name.title = feature.name;
    const track = document.createElement('span');
    track.className = 'importance-track';
    const fill = document.createElement('span');
    fill.className = 'importance-fill';
    fill.style.width = `${(feature.value / maximum) * 100}%`;
    fill.setAttribute('aria-hidden', 'true');
    track.append(fill);
    const value = createCell('span', formatMetric(feature.value));
    value.className = 'importance-value';
    row.append(name, track, value);
    list.append(row);
  });
  $('#training-status').textContent = `Проверка на отложенной выборке: ${numberFormat.format(model.test_rows)} строк.`;
}

function showError(error) {
  setStatus(error.message || 'Произошла ошибка.', true);
}

const uploadForm = $('#upload-form');
const fileInput = $('#dataset-file');
const dropArea = $('#drop-area');
const fileCaption = $('#file-caption');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) fileCaption.textContent = `${file.name} · ${formatBytes(file.size)}`;
});

['dragenter', 'dragover'].forEach((eventName) => dropArea.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropArea.classList.add('is-dragging');
}));
['dragleave', 'drop'].forEach((eventName) => dropArea.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropArea.classList.remove('is-dragging');
}));
dropArea.addEventListener('drop', (event) => {
  const files = event.dataTransfer.files;
  if (!files.length) return;
  fileInput.files = files;
  fileInput.dispatchEvent(new Event('change', { bubbles: true }));
});

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) { fileInput.reportValidity(); return; }
  const button = uploadForm.querySelector('button[type="submit"]');
  button.disabled = true;
  setStatus('Подготовка файла…');
  try {
    const response = await sendForm(uploadForm, (progress) => {
      setStatus(progress < 1
        ? `Загрузка файла: ${Math.round(progress * 100)}%`
        : 'Файл получен. Читаю данные по частям…');
    });
    renderDataset(response.dataset);
    setStatus(`Готово: обработано ${numberFormat.format(response.dataset.rows)} строк.`);
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
});

const sampleForm = $('#sample-form');
sampleForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = sampleForm.querySelector('button[type="submit"]');
  button.disabled = true;
  setStatus('Загружаю демонстрационный набор…');
  try {
    const response = await sendForm(sampleForm);
    renderDataset(response.dataset);
    setStatus(`Готово: обработано ${numberFormat.format(response.dataset.rows)} строк.`);
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
  }
});

const trainForm = $('#train-form');
trainForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = trainForm.querySelector('button[type="submit"]');
  button.disabled = true;
  $('#training-status').textContent = 'Обучаю модель и проверяю её на отложенных данных…';
  try {
    const response = await sendForm(trainForm);
    renderModel(response.model);
  } catch (error) {
    $('#training-status').textContent = error.message || 'Не удалось обучить модель.';
  } finally {
    button.disabled = false;
  }
});

try {
  const state = JSON.parse($('#bootstrap-data').textContent);
  if (state.dataset) renderDataset(state.dataset);
  if (state.model) renderModel(state.model);
} catch {
  setStatus('Не удалось восстановить состояние страницы.', true);
}
