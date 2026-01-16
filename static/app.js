const statusEl = document.getElementById('chart-status');

if (!window.klinecharts) {
  statusEl.textContent = 'Charting library missing. Add /static/vendor/klinecharts.min.js.';
  throw new Error('Missing klinecharts bundle');
}

const chart = klinecharts.init('chart');
chart.createIndicator('VOL', false, { id: 'volume_pane' });

const priceEl = document.getElementById('symbol-price');
const changeEl = document.getElementById('symbol-change');
const timeframesEl = document.getElementById('timeframes');
const chartTypeSelect = document.getElementById('chart-type');
const themeToggle = document.getElementById('theme-toggle');
const fullscreenToggle = document.getElementById('fullscreen-toggle');
const uploadTrigger = document.getElementById('upload-trigger');
const uploadInput = document.getElementById('csv-upload');
const saveDrawingsButton = document.getElementById('save-drawings');

let latestTimestamp = null;

const indicatorState = {};
const indicatorConfig = {
  sma: { name: 'MA', pane: 'candle_pane' },
  ema: { name: 'EMA', pane: 'candle_pane' },
  boll: { name: 'BOLL', pane: 'candle_pane' },
  rsi: { name: 'RSI', pane: 'indicator_pane' },
  macd: { name: 'MACD', pane: 'indicator_pane' },
  stochastic: { name: 'KDJ', pane: 'indicator_pane' },
};

function parseTimestamp(value) {
  if (!value) {
    return null;
  }
  const [datePart, timePart = '00:00:00'] = value.split(' ');
  const [year, month, day] = datePart.split('-').map(Number);
  const [hour, minute, second] = timePart.split(':').map(Number);
  return new Date(Date.UTC(year, month - 1, day, hour, minute, second));
}

function formatDate(value) {
  return value.toISOString().slice(0, 10);
}

function setStatus(message) {
  statusEl.textContent = message || '';
  statusEl.classList.toggle('visible', Boolean(message));
}

function updateHeader(candles) {
  if (!candles.length) {
    priceEl.textContent = '$--';
    changeEl.textContent = '--';
    changeEl.className = 'symbol-change';
    return;
  }
  const last = candles[candles.length - 1];
  const prev = candles[candles.length - 2] || last;
  const change = last.close - prev.close;
  const changePct = prev.close ? (change / prev.close) * 100 : 0;
  const sign = change >= 0 ? '+' : '';

  priceEl.textContent = `$${last.close.toFixed(2)}`;
  changeEl.textContent = `${sign}${change.toFixed(2)} (${sign}${changePct.toFixed(2)}%)`;
  changeEl.className = `symbol-change ${change >= 0 ? 'positive' : 'negative'}`;
}

function setActiveRange(target) {
  const buttons = timeframesEl.querySelectorAll('button');
  buttons.forEach((button) => button.classList.remove('active'));
  target.classList.add('active');
}

function computeRange(label) {
  if (label === 'MAX') {
    return { start: null, end: null };
  }
  if (!latestTimestamp) {
    return { start: null, end: null };
  }
  const endDate = parseTimestamp(latestTimestamp) || new Date();
  const end = formatDate(endDate);

  if (label === 'YTD') {
    const startDate = new Date(Date.UTC(endDate.getUTCFullYear(), 0, 1));
    return { start: formatDate(startDate), end };
  }

  const daysMap = {
    '1D': 1,
    '5D': 5,
    '1M': 30,
    '6M': 180,
    '1Y': 365,
  };
  const days = daysMap[label] || 30;
  const startDate = new Date(endDate);
  startDate.setUTCDate(startDate.getUTCDate() - (days - 1));
  return { start: formatDate(startDate), end };
}

async function loadCandles(range) {
  const params = new URLSearchParams();
  if (range?.start) {
    params.set('start', range.start);
  }
  if (range?.end) {
    params.set('end', range.end);
  }

  const response = await fetch(`/api/candles${params.toString() ? `?${params}` : ''}`);
  if (!response.ok) {
    throw new Error('Failed to load candles');
  }
  const candles = await response.json();
  if (!candles.length) {
    setStatus('No data for selected range.');
    chart.applyNewData([]);
    updateHeader([]);
    return;
  }

  latestTimestamp = candles[candles.length - 1].timestamp;
  const data = candles.map((candle) => {
    const date = parseTimestamp(candle.timestamp) || new Date();
    return {
      timestamp: date.getTime(),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
    };
  });

  setStatus('');
  chart.applyNewData(data);
  updateHeader(candles);
}

function toggleIndicator(key, enabled) {
  const config = indicatorConfig[key];
  if (!config) {
    return;
  }
  if (enabled) {
    if (indicatorState[key]) {
      return;
    }
    const id = chart.createIndicator(config.name, false, { id: config.pane });
    indicatorState[key] = { id, pane: config.pane };
  } else {
    const existing = indicatorState[key];
    if (!existing || typeof chart.removeIndicator !== 'function') {
      indicatorState[key] = null;
      return;
    }
    try {
      chart.removeIndicator(existing.id, existing.pane);
    } catch (error) {
      try {
        chart.removeIndicator(existing.id);
      } catch (err) {
        console.warn('Failed to remove indicator', err);
      }
    }
    indicatorState[key] = null;
  }
}

function setChartType(type) {
  if (typeof chart.setStyles !== 'function') {
    return;
  }
  const mapping = {
    candle: 'candle_solid',
    line: 'line',
    area: 'area',
    ohlc: 'ohlc',
  };
  const candleType = mapping[type] || 'candle_solid';
  try {
    chart.setStyles({
      candle: {
        type: candleType,
      },
    });
  } catch (error) {
    console.warn('Chart type not supported', error);
  }
}

function startDrawing(name) {
  if (typeof chart.createOverlay !== 'function') {
    return;
  }
  try {
    chart.createOverlay({ name });
  } catch (error) {
    try {
      chart.createOverlay(name);
    } catch (err) {
      console.warn('Overlay not supported', err);
    }
  }
}

async function saveDrawings() {
  if (typeof chart.getOverlays !== 'function') {
    return;
  }
  const overlays = chart.getOverlays();
  await fetch('/api/drawings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data: overlays }),
  });
}

async function loadDrawings() {
  if (typeof chart.createOverlay !== 'function') {
    return;
  }
  const response = await fetch('/api/drawings');
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  if (!payload.data || !Array.isArray(payload.data)) {
    return;
  }
  payload.data.forEach((overlay) => {
    try {
      chart.createOverlay(overlay);
    } catch (error) {
      console.warn('Failed to restore overlay', error);
    }
  });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('theme', theme);
  themeToggle.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
}

async function handleTimeframeClick(event) {
  const button = event.target.closest('button[data-range]');
  if (!button) {
    return;
  }
  setActiveRange(button);
  const range = computeRange(button.dataset.range);
  try {
    await loadCandles(range);
  } catch (error) {
    console.error(error);
    setStatus('Failed to load data.');
  }
}

async function init() {
  try {
    await loadCandles();
    await loadDrawings();
    const defaultButton = timeframesEl.querySelector('button[data-range="1D"]');
    if (defaultButton) {
      setActiveRange(defaultButton);
      await loadCandles(computeRange('1D'));
    }
  } catch (error) {
    console.error(error);
    setStatus('Failed to load data.');
  }
}

Array.from(document.querySelectorAll('[data-indicator]')).forEach((input) => {
  input.addEventListener('change', (event) => {
    toggleIndicator(event.target.dataset.indicator, event.target.checked);
  });
});

Array.from(document.querySelectorAll('[data-draw]')).forEach((button) => {
  button.addEventListener('click', () => {
    startDrawing(button.dataset.draw);
  });
});

chartTypeSelect.addEventListener('change', (event) => {
  setChartType(event.target.value);
});

timeframesEl.addEventListener('click', handleTimeframeClick);

saveDrawingsButton.addEventListener('click', saveDrawings);

if (typeof chart.subscribeAction === 'function') {
  chart.subscribeAction('overlay', () => {
    saveDrawings().catch(() => undefined);
  });
}

window.addEventListener('beforeunload', () => {
  saveDrawings().catch(() => undefined);
});

fullscreenToggle.addEventListener('click', async () => {
  if (!document.fullscreenElement) {
    await document.documentElement.requestFullscreen();
  } else {
    await document.exitFullscreen();
  }
});

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.dataset.theme || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

uploadTrigger.addEventListener('click', () => {
  uploadInput.click();
});

uploadInput.addEventListener('change', async () => {
  if (!uploadInput.files.length) {
    return;
  }
  const formData = new FormData();
  formData.append('file', uploadInput.files[0]);
  setStatus('Uploading CSV...');
  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      throw new Error('Upload failed');
    }
    await response.json();
    await loadCandles();
    setStatus('Upload complete.');
  } catch (error) {
    console.error(error);
    setStatus('CSV upload failed.');
  } finally {
    uploadInput.value = '';
  }
});

applyTheme(document.documentElement.dataset.theme || 'dark');
init();
