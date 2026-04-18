/* FraudSense — main.js */

// Static mode: pre-built JSON from /data/; Live: FastAPI /api/
const STATIC = window.FRAUDSENSE_STATIC === true;
const BASE   = window.FRAUDSENSE_BASE   || '';

function apiUrl(endpoint) {
  return STATIC ? `${BASE}/data/${endpoint}.json` : `${BASE}/api/${endpoint}`;
}

const COLORS = {
  indigo: '#818cf8', green: '#34d399', amber: '#fcd34d',
  red: '#f87171', blue: '#38bdf8', purple: '#c084fc',
};
const PALETTE = Object.values(COLORS);

const PLOTLY_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font: { color: '#d1d5db', family: 'Inter, system-ui, sans-serif', size: 11 },
};
const AXIS = { gridcolor: '#1f2937', zerolinecolor: '#374151', color: '#9ca3af' };

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadMetrics();
  loadFraudRings();
  loadShap();
  renderSplitBar();
  document.getElementById('predict-form').addEventListener('submit', handlePredict);
});

// ── Static split bar (no API needed) ─────────────────────────────────────────
function renderSplitBar() {
  const trace = {
    type: 'bar', orientation: 'h',
    x: [413378, 88581, 88581],
    y: ['Train', 'Val', 'Test'],
    marker: { color: [COLORS.indigo, COLORS.amber, COLORS.green] },
    text: ['413,378 (70%)', '88,581 (15%)', '88,581 (15%)'],
    textposition: 'inside',
    insidetextanchor: 'middle',
  };
  Plotly.newPlot('chart-split-bar', [trace], {
    ...PLOTLY_BASE,
    xaxis: { ...AXIS, title: 'Transactions', tickformat: ',d' },
    yaxis: { ...AXIS },
    height: 160,
    margin: { t: 10, r: 20, b: 40, l: 50 },
  }, { responsive: true, displayModeBar: false });
}

// ── Metrics ───────────────────────────────────────────────────────────────────
async function loadMetrics() {
  const data = await fetchJSON(apiUrl('metrics'));
  if (!data) return;

  setText('stat-auc',      fmt(data.ensemble_test_auc));
  setText('stat-prauc',    fmt(data.ensemble_pr_auc));
  setText('val-lgbm-auc',  fmt(data.lgbm_test_auc));
  setText('val-gnn-auc',   fmt(data.gnn_test_auc));
  setText('val-ens-auc',   fmt(data.ensemble_test_auc));
  setText('val-ens-prauc', fmt(data.ensemble_pr_auc));

  renderAucBar(data.model_comparison);
  renderRoc(data);
  renderPR(data);
  renderThresholdMetrics(data);
}

function renderAucBar(cmp) {
  if (!cmp) return;
  const trace1 = {
    type: 'bar', name: 'ROC-AUC',
    x: cmp.models, y: cmp.auc,
    marker: { color: COLORS.indigo },
    text: cmp.auc.map(v => v.toFixed(4)),
    textposition: 'outside',
  };
  const trace2 = {
    type: 'bar', name: 'PR-AUC',
    x: cmp.models, y: cmp.pr_auc || cmp.auc.map(() => 0),
    marker: { color: COLORS.green },
    text: (cmp.pr_auc || []).map(v => v.toFixed(4)),
    textposition: 'outside',
  };
  Plotly.newPlot('chart-auc-bar', [trace1, trace2], {
    ...PLOTLY_BASE,
    barmode: 'group',
    yaxis: { ...AXIS, range: [0, 1.08], title: 'Score' },
    xaxis: { ...AXIS },
    legend: { x: 0.6, y: 1, bgcolor: 'transparent' },
    height: 260,
    margin: { t: 30, r: 20, b: 40, l: 55 },
  }, { responsive: true, displayModeBar: false });
}

function renderRoc(data) {
  const models = [
    { key: 'roc_lgbm', label: 'LightGBM', color: COLORS.indigo },
    { key: 'roc_gnn',  label: 'GNN (GAT)', color: COLORS.green },
    { key: 'roc_ensemble', label: 'Ensemble', color: COLORS.amber },
  ];
  const traces = models.filter(m => data[m.key]).map(m => ({
    type: 'scatter', mode: 'lines', name: m.label,
    x: data[m.key].fpr, y: data[m.key].tpr,
    line: { color: m.color, width: 2 },
  }));
  traces.push({
    type: 'scatter', mode: 'lines', name: 'Random',
    x: [0, 1], y: [0, 1],
    line: { color: '#374151', dash: 'dash', width: 1 },
    showlegend: false,
  });
  Plotly.newPlot('chart-roc', traces, {
    ...PLOTLY_BASE,
    xaxis: { ...AXIS, title: 'False Positive Rate', range: [0, 1], dtick: 0.2 },
    yaxis: { ...AXIS, title: 'True Positive Rate', range: [0, 1], dtick: 0.2 },
    legend: { x: 0.58, y: 0.18, bgcolor: 'transparent' },
    height: 260,
    margin: { t: 10, r: 20, b: 45, l: 55 },
  }, { responsive: true, displayModeBar: false });
}

function renderPR(data) {
  const models = [
    { key: 'pr_lgbm', label: 'LightGBM', color: COLORS.indigo },
    { key: 'pr_gnn',  label: 'GNN (GAT)', color: COLORS.green },
    { key: 'pr_ensemble', label: 'Ensemble', color: COLORS.amber },
  ];
  const traces = models.filter(m => data[m.key]).map(m => ({
    type: 'scatter', mode: 'lines', name: m.label,
    x: data[m.key].recall, y: data[m.key].precision,
    line: { color: m.color, width: 2 },
  }));
  // Baseline: random classifier PR = fraud_rate
  traces.push({
    type: 'scatter', mode: 'lines', name: 'Random (3.5%)',
    x: [0, 1], y: [0.035, 0.035],
    line: { color: '#374151', dash: 'dash', width: 1 },
    showlegend: true,
  });
  Plotly.newPlot('chart-pr', traces, {
    ...PLOTLY_BASE,
    xaxis: { ...AXIS, title: 'Recall', range: [0, 1], dtick: 0.2 },
    yaxis: { ...AXIS, title: 'Precision', range: [0, 1], dtick: 0.2 },
    legend: { x: 0.55, y: 0.98, bgcolor: 'transparent' },
    height: 260,
    margin: { t: 10, r: 20, b: 45, l: 55 },
  }, { responsive: true, displayModeBar: false });
}

function renderThresholdMetrics(data) {
  if (!data.precision) return;
  const metrics = ['Precision', 'Recall', 'F1'];
  const values  = [data.precision, data.recall, data.f1];
  const colors  = [COLORS.blue, COLORS.green, COLORS.amber];
  const trace = {
    type: 'bar',
    x: metrics, y: values,
    marker: { color: colors },
    text: values.map(v => v.toFixed(4)),
    textposition: 'outside',
  };
  Plotly.newPlot('chart-threshold-metrics', [trace], {
    ...PLOTLY_BASE,
    yaxis: { ...AXIS, range: [0, 1.1], title: 'Score' },
    xaxis: { ...AXIS },
    annotations: [{
      x: 0.5, y: 1.08, xref: 'paper', yref: 'paper',
      text: `Threshold = ${data.best_threshold?.toFixed(2)}`,
      showarrow: false, font: { color: '#9ca3af', size: 11 },
    }],
    height: 230,
    margin: { t: 35, r: 20, b: 40, l: 55 },
  }, { responsive: true, displayModeBar: false });
}

// ── Fraud Rings ───────────────────────────────────────────────────────────────
async function loadFraudRings() {
  const data = await fetchJSON(apiUrl('fraud-rings'));
  if (!data || !data.nodes || data.nodes.length === 0) {
    document.getElementById('fraud-ring-graph').innerHTML =
      '<p style="color:#6b7280;padding:20px">No fraud ring data available.</p>';
    return;
  }

  setText('stat-rings', data.summary?.length ?? '—');
  renderRingTable(data.summary);
  renderRingGraphPlotly(data);

  if (data.summary?.length > 0) {
    const top = data.summary[0];
    setText('top-ring-rate', (top.avg_fraud_rate * 100).toFixed(1) + '%');
    setText('top-ring-mult', (top.avg_fraud_rate / 0.035).toFixed(1));
  }
}

function renderRingTable(summary) {
  if (!summary?.length) return;
  const rows = summary.map((r, i) => `
    <tr>
      <td><span class="badge-ring" style="background:${PALETTE[i % PALETTE.length]}"></span>#${r.rank}</td>
      <td>${r.size} cards</td>
      <td style="color:${rateColor(r.avg_fraud_rate)};font-weight:700">${(r.avg_fraud_rate * 100).toFixed(1)}%</td>
      <td style="color:#6b7280;font-size:0.75rem">${(r.avg_fraud_rate / 0.035).toFixed(1)}× base</td>
    </tr>`).join('');
  document.getElementById('ring-summary-table').innerHTML = `
    <table>
      <thead><tr><th>Ring</th><th>Size</th><th>Fraud Rate</th><th>vs Baseline</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function rateColor(r) {
  if (r >= 0.15) return '#f87171';
  if (r >= 0.08) return '#fcd34d';
  if (r >= 0.05) return '#34d399';
  return '#9ca3af';
}

function renderRingGraphPlotly(data) {
  // Use D3 force simulation purely for layout — no DOM manipulation
  const nodes = data.nodes.map(d => ({ ...d }));
  const links = data.links.map(d => ({ ...d }));

  // Place community centers in a circle so dense inter-community
  // links don't pull everything into one blob
  const allGroups = [...new Set(nodes.map(n => n.group))];
  const RADIUS = 220;
  const centerOf = {};
  allGroups.forEach((g, i) => {
    const angle = (2 * Math.PI * i) / allGroups.length;
    centerOf[g] = { x: RADIUS * Math.cos(angle), y: RADIUS * Math.sin(angle) };
  });

  // Initialise positions near each community's target center
  nodes.forEach(n => {
    const c = centerOf[n.group];
    n.x = c.x + (Math.random() - 0.5) * 30;
    n.y = c.y + (Math.random() - 0.5) * 30;
  });

  const sim = d3.forceSimulation(nodes)
    .force('link',    d3.forceLink(links).id(d => d.id).distance(30).strength(0.2))
    .force('charge',  d3.forceManyBody().strength(-300))
    .force('collide', d3.forceCollide(14))
    .force('cx',      d3.forceX(n => centerOf[n.group].x).strength(0.35))
    .force('cy',      d3.forceY(n => centerOf[n.group].y).strength(0.35))
    .stop();

  for (let i = 0; i < 400; i++) sim.tick();

  // Build lookup after simulation mutates source/target to node objects
  const nodeById = {};
  nodes.forEach(n => nodeById[n.id] = n);

  // Edge trace
  const ex = [], ey = [];
  links.forEach(l => {
    const s = l.source, t = l.target;
    const sx = typeof s === 'object' ? s.x : nodeById[s]?.x;
    const sy = typeof s === 'object' ? s.y : nodeById[s]?.y;
    const tx = typeof t === 'object' ? t.x : nodeById[t]?.x;
    const ty = typeof t === 'object' ? t.y : nodeById[t]?.y;
    if (sx != null) { ex.push(sx, tx, null); ey.push(sy, ty, null); }
  });

  const edgeTrace = {
    type: 'scatter', mode: 'lines',
    x: ex, y: ey,
    line: { color: '#1f2937', width: 0.8 },
    hoverinfo: 'none', showlegend: false,
  };

  // One trace per community group
  const groups = [...new Set(nodes.map(n => n.group))];
  const nodeTraces = groups.map((g, i) => {
    const gNodes = nodes.filter(n => n.group === g);
    return {
      type: 'scatter', mode: 'markers',
      name: `Ring ${i + 1}`,
      x: gNodes.map(n => n.x),
      y: gNodes.map(n => n.y),
      text: gNodes.map(n =>
        `<b>${n.label}</b><br>Fraud rate: ${(n.fraud_rate * 100).toFixed(1)}%`),
      hoverinfo: 'text',
      marker: {
        size: gNodes.map(n => Math.max(7, Math.min(22, n.fraud_rate * 120 + 7))),
        color: PALETTE[i % PALETTE.length],
        opacity: 0.85,
        line: { color: '#080b14', width: 1 },
      },
    };
  });

  Plotly.newPlot('fraud-ring-graph', [edgeTrace, ...nodeTraces], {
    ...PLOTLY_BASE,
    xaxis: { visible: false, zeroline: false, showgrid: false },
    yaxis: { visible: false, zeroline: false, showgrid: false },
    height: 460,
    margin: { t: 10, r: 20, b: 10, l: 20 },
    hovermode: 'closest',
    legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(13,17,23,0.8)',
              bordercolor: '#1f2937', borderwidth: 1, font: { size: 11 } },
  }, { responsive: true, displayModeBar: false });
}

// ── SHAP ──────────────────────────────────────────────────────────────────────
async function loadShap() {
  const data = await fetchJSON(apiUrl('shap'));
  if (!data) return;
  renderShapImportance(data.feature_importance);
  renderShapWaterfall(data.waterfall_fraud, 'chart-shap-waterfall-fraud');
}

function renderShapImportance(imp) {
  if (!imp) return;
  const n = imp.features.length;
  const trace = {
    type: 'bar', orientation: 'h',
    x: [...imp.importance].reverse(),
    y: [...imp.features].reverse(),
    marker: {
      color: [...imp.importance].reverse().map(v => `rgba(129,140,248,${0.4 + 0.6 * v / Math.max(...imp.importance)})`),
    },
  };
  Plotly.newPlot('chart-shap-importance', [trace], {
    ...PLOTLY_BASE,
    xaxis: { ...AXIS, title: 'Mean |SHAP|' },
    yaxis: { ...AXIS, automargin: true },
    height: Math.max(300, n * 18 + 60),
    margin: { t: 10, r: 30, b: 40, l: 140 },
  }, { responsive: true, displayModeBar: false });
}

function renderShapWaterfall(wf, elId) {
  if (!wf) return;
  const trace = {
    type: 'bar', orientation: 'h',
    x: wf.shap_values,
    y: wf.features.map((f, i) => `${f} = ${wf.feature_values[i]}`),
    marker: { color: wf.shap_values.map(v => v > 0 ? '#f87171' : '#34d399') },
    text: wf.shap_values.map(v => (v > 0 ? '+' : '') + v.toFixed(3)),
    textposition: 'outside',
  };
  Plotly.newPlot(elId, [trace], {
    ...PLOTLY_BASE,
    xaxis: { ...AXIS, title: 'SHAP value (impact on fraud probability)', zeroline: true },
    yaxis: { ...AXIS, automargin: true },
    height: Math.max(280, wf.features.length * 28 + 60),
    margin: { t: 10, r: 60, b: 45, l: 200 },
  }, { responsive: true, displayModeBar: false });
}

// ── Prediction ────────────────────────────────────────────────────────────────
async function handlePredict(e) {
  e.preventDefault();
  document.getElementById('predict-placeholder').classList.add('d-none');
  document.getElementById('predict-output').classList.add('d-none');
  document.getElementById('predict-error').classList.add('d-none');

  if (STATIC) {
    document.getElementById('predict-error').textContent = 'Models are not live right now.';
    document.getElementById('predict-error').classList.remove('d-none');
    return;
  }

  const payload = {
    TransactionAmt: parseFloat(document.getElementById('input-amount').value),
    hour:           parseInt(document.getElementById('input-hour').value),
    day_of_week:    parseInt(document.getElementById('input-dow').value),
    ProductCD:      document.getElementById('input-product').value,
    P_emaildomain:  document.getElementById('input-email').value,
  };

  const res = await fetch(`${BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Prediction failed.' }));
    document.getElementById('predict-error').textContent = err.detail;
    document.getElementById('predict-error').classList.remove('d-none');
    return;
  }

  const d = await res.json();
  const prob = d.fraud_probability;

  setText('fraud-prob', (prob * 100).toFixed(1) + '%');
  setText('score-lgbm', (d.lgbm_score * 100).toFixed(1) + '%');
  setText('score-gnn',  d.gnn_score != null ? (d.gnn_score * 100).toFixed(1) + '%' : 'N/A');

  const badge = document.getElementById('risk-badge');
  badge.textContent = d.risk_level;
  badge.className = 'risk-badge risk-' + d.risk_level.toLowerCase().replace(' ', '-');

  renderGauge(prob);
  document.getElementById('predict-output').classList.remove('d-none');
}

function renderGauge(prob) {
  const color = prob >= 0.8 ? '#f87171' : prob >= 0.6 ? '#fb923c' : prob >= 0.4 ? '#fcd34d' : '#34d399';
  Plotly.newPlot('chart-risk-gauge', [{
    type: 'indicator', mode: 'gauge+number',
    value: prob * 100,
    number: { suffix: '%', font: { color: '#f3f4f6', size: 26 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#374151', tickfont: { color: '#9ca3af' } },
      bar: { color },
      bgcolor: '#0d1117', bordercolor: '#1f2937',
      steps: [
        { range: [0, 20],  color: 'rgba(16,185,129,0.08)' },
        { range: [20, 40], color: 'rgba(16,185,129,0.04)' },
        { range: [40, 60], color: 'rgba(245,158,11,0.06)' },
        { range: [60, 80], color: 'rgba(249,115,22,0.08)' },
        { range: [80, 100],color: 'rgba(239,68,68,0.10)' },
      ],
    },
  }], {
    paper_bgcolor: 'transparent',
    font: { color: '#f3f4f6' },
    height: 160,
    margin: { t: 10, r: 20, b: 10, l: 20 },
  }, { responsive: true, displayModeBar: false });
}

// ── Utils ─────────────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function fmt(v) { return v != null ? v.toFixed(4) : '—'; }
