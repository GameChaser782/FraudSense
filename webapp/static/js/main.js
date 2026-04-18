/* FraudSense — main.js
   Fetches data from FastAPI and renders:
   - AUC bar chart
   - ROC curve comparison
   - Fraud ring D3 force-directed graph
   - SHAP feature importance + waterfall
   - Live prediction with gauge
*/

// Static mode: fetch pre-built JSON from /data/; Live mode: fetch from FastAPI /api/
const STATIC = window.FRAUDSENSE_STATIC === true;
const BASE   = window.FRAUDSENSE_BASE   || '';

function apiUrl(endpoint) {
  // endpoint e.g. 'metrics', 'shap', 'fraud-rings'
  return STATIC
    ? `${BASE}/data/${endpoint}.json`
    : `${BASE}/api/${endpoint}`;
}

const PLOTLY_DARK = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:          { color: '#e2e8f0', family: 'Inter, system-ui, sans-serif', size: 11 },
  xaxis:         { gridcolor: '#2d3348', zerolinecolor: '#2d3348' },
  yaxis:         { gridcolor: '#2d3348', zerolinecolor: '#2d3348' },
  margin:        { t: 20, r: 20, b: 40, l: 40 },
};

const ACCENT_COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#38bdf8'];

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadMetrics();
  loadFraudRings();
  loadShap();
  document.getElementById('predict-form').addEventListener('submit', handlePredict);
});

// ── Metrics ──────────────────────────────────────────────────────────────────
async function loadMetrics() {
  const data = await fetchJSON(apiUrl('metrics'));
  if (!data) return;

  document.getElementById('stat-auc').textContent  = data.ensemble_test_auc?.toFixed(4) ?? '—';
  document.getElementById('stat-f1').textContent   = data.f1?.toFixed(4) ?? '—';
  document.getElementById('metric-precision').textContent = data.precision?.toFixed(4) ?? '—';
  document.getElementById('metric-recall').textContent    = data.recall?.toFixed(4)    ?? '—';
  document.getElementById('metric-f1').textContent        = data.f1?.toFixed(4)        ?? '—';

  renderAucBar(data.model_comparison);
  renderRoc(data);
}

function renderAucBar(cmp) {
  if (!cmp) return;
  const trace = {
    type: 'bar',
    x: cmp.models,
    y: cmp.auc,
    marker: { color: ACCENT_COLORS },
    text: cmp.auc.map(v => v.toFixed(4)),
    textposition: 'outside',
  };
  Plotly.newPlot('chart-auc-bar', [trace], {
    ...PLOTLY_DARK,
    yaxis: { ...PLOTLY_DARK.yaxis, range: [0.85, 1.0], title: 'AUC' },
    margin: { t: 30, r: 20, b: 40, l: 55 },
    height: 250,
  }, { responsive: true, displayModeBar: false });
}

function renderRoc(data) {
  const models = [
    { key: 'roc_lgbm',     label: 'LightGBM',  color: ACCENT_COLORS[0] },
    { key: 'roc_gnn',      label: 'GNN (GAT)', color: ACCENT_COLORS[1] },
    { key: 'roc_ensemble', label: 'Ensemble',  color: ACCENT_COLORS[2] },
  ];

  const traces = models
    .filter(m => data[m.key])
    .map(m => ({
      type: 'scatter',
      mode: 'lines',
      name: m.label,
      x: data[m.key].fpr,
      y: data[m.key].tpr,
      line: { color: m.color, width: 2 },
    }));

  // Diagonal baseline
  traces.push({
    type: 'scatter', mode: 'lines', name: 'Random',
    x: [0, 1], y: [0, 1],
    line: { color: '#475569', dash: 'dash', width: 1 },
    showlegend: false,
  });

  Plotly.newPlot('chart-roc', traces, {
    ...PLOTLY_DARK,
    xaxis: { ...PLOTLY_DARK.xaxis, title: 'FPR', range: [0, 1] },
    yaxis: { ...PLOTLY_DARK.yaxis, title: 'TPR', range: [0, 1] },
    legend: { x: 0.6, y: 0.15, bgcolor: 'transparent' },
    height: 250,
    margin: { t: 20, r: 20, b: 45, l: 55 },
  }, { responsive: true, displayModeBar: false });
}

// ── Fraud Rings (D3) ──────────────────────────────────────────────────────────
async function loadFraudRings() {
  const data = await fetchJSON(apiUrl('fraud-rings'));
  if (!data) return;

  document.getElementById('stat-rings').textContent = data.summary?.length ?? '—';
  renderRingTable(data.summary);
  renderRingGraph(data);
}

function renderRingTable(summary) {
  if (!summary || !summary.length) return;
  const rows = summary.map((r, i) => `
    <tr>
      <td><span class="badge-ring" style="background:${ACCENT_COLORS[i % ACCENT_COLORS.length]}"></span>${r.rank}</td>
      <td>${r.size} cards</td>
      <td style="color:${fraudRateColor(r.avg_fraud_rate)};font-weight:600">${(r.avg_fraud_rate * 100).toFixed(1)}%</td>
    </tr>
  `).join('');
  document.getElementById('ring-summary-table').innerHTML = `
    <table>
      <thead><tr><th>Rank</th><th>Size</th><th>Fraud Rate</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function fraudRateColor(rate) {
  if (rate >= 0.7) return '#ef4444';
  if (rate >= 0.5) return '#f97316';
  if (rate >= 0.3) return '#f59e0b';
  return '#22c55e';
}

function renderRingGraph(data) {
  const container = document.getElementById('fraud-ring-graph');
  const W = container.clientWidth  || 700;
  const H = container.clientHeight || 480;

  // Map group → color
  const groups = [...new Set(data.nodes.map(n => n.group))];
  const colorMap = Object.fromEntries(groups.map((g, i) => [g, ACCENT_COLORS[i % ACCENT_COLORS.length]]));

  const svg = d3.select('#fraud-ring-graph').append('svg')
    .attr('width', W).attr('height', H);

  const sim = d3.forceSimulation(data.nodes)
    .force('link',   d3.forceLink(data.links).id(d => d.id).distance(60).strength(0.5))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(W / 2, H / 2))
    .force('collide', d3.forceCollide(18));

  const link = svg.append('g')
    .selectAll('line')
    .data(data.links)
    .join('line')
    .attr('stroke', '#2d3348')
    .attr('stroke-width', d => Math.min(Math.sqrt(d.weight), 4));

  const node = svg.append('g')
    .selectAll('circle')
    .data(data.nodes)
    .join('circle')
    .attr('r', 9)
    .attr('fill', d => colorMap[d.group])
    .attr('stroke', '#0f1117')
    .attr('stroke-width', 2)
    .call(d3.drag()
      .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag',  (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on('end',   (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }));

  // Tooltip
  const tooltip = d3.select('body').append('div')
    .style('position', 'absolute')
    .style('background', '#1a1d27')
    .style('border', '1px solid #2d3348')
    .style('border-radius', '8px')
    .style('padding', '8px 12px')
    .style('font-size', '12px')
    .style('color', '#e2e8f0')
    .style('pointer-events', 'none')
    .style('opacity', 0);

  node.on('mouseover', (event, d) => {
    tooltip.transition().duration(150).style('opacity', 1);
    tooltip.html(`<strong>${d.label}</strong><br/>Fraud rate: ${(d.fraud_rate * 100).toFixed(1)}%`)
      .style('left', (event.pageX + 12) + 'px')
      .style('top',  (event.pageY - 28) + 'px');
  }).on('mouseout', () => tooltip.transition().duration(200).style('opacity', 0));

  sim.on('tick', () => {
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
  });
}

// ── SHAP ─────────────────────────────────────────────────────────────────────
async function loadShap() {
  const data = await fetchJSON(apiUrl('shap'));
  if (!data) return;

  renderShapImportance(data.feature_importance);
  renderShapWaterfall(data.waterfall_fraud, 'chart-shap-waterfall-fraud');
}

function renderShapImportance(imp) {
  if (!imp) return;
  const trace = {
    type: 'bar', orientation: 'h',
    x: [...imp.importance].reverse(),
    y: [...imp.features].reverse(),
    marker: { color: '#6366f1' },
  };
  Plotly.newPlot('chart-shap-importance', [trace], {
    ...PLOTLY_DARK,
    xaxis: { ...PLOTLY_DARK.xaxis, title: 'Mean |SHAP|' },
    height: 340,
    margin: { t: 10, r: 20, b: 40, l: 130 },
  }, { responsive: true, displayModeBar: false });
}

function renderShapWaterfall(wf, elId) {
  if (!wf) return;
  const colors = wf.shap_values.map(v => v > 0 ? '#ef4444' : '#22c55e');
  const trace = {
    type: 'bar', orientation: 'h',
    x: wf.shap_values,
    y: wf.features.map((f, i) => `${f} = ${wf.feature_values[i]}`),
    marker: { color: colors },
    text: wf.shap_values.map(v => v.toFixed(3)),
    textposition: 'outside',
  };
  Plotly.newPlot(elId, [trace], {
    ...PLOTLY_DARK,
    xaxis: { ...PLOTLY_DARK.xaxis, title: 'SHAP value', zeroline: true },
    height: 300,
    margin: { t: 10, r: 50, b: 40, l: 200 },
    annotations: [{
      x: 0.5, y: 1.05, xref: 'paper', yref: 'paper',
      text: `Base value: ${wf.base_value?.toFixed(3)}`,
      showarrow: false, font: { color: '#94a3b8', size: 10 },
    }],
  }, { responsive: true, displayModeBar: false });
}

// ── Live Prediction ───────────────────────────────────────────────────────────
async function handlePredict(e) {
  e.preventDefault();

  const payload = {
    TransactionAmt:  parseFloat(document.getElementById('input-amount').value),
    hour:            parseInt(document.getElementById('input-hour').value),
    day_of_week:     parseInt(document.getElementById('input-dow').value),
    ProductCD:       document.getElementById('input-product').value,
    P_emaildomain:   document.getElementById('input-email').value,
  };

  document.getElementById('predict-placeholder').classList.add('d-none');
  document.getElementById('predict-output').classList.add('d-none');
  document.getElementById('predict-error').classList.add('d-none');

  // Prediction is not available in static mode (no backend)
  if (STATIC) {
    document.getElementById('predict-error').textContent =
      'Live prediction is not available in the static demo. Clone the repo and run locally.';
    document.getElementById('predict-error').classList.remove('d-none');
    return;
  }
  const res = await fetch(`${BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    document.getElementById('predict-error').textContent = err.detail || 'Prediction failed.';
    document.getElementById('predict-error').classList.remove('d-none');
    return;
  }

  const data = await res.json();
  const prob  = data.fraud_probability;
  const level = data.risk_level;

  document.getElementById('fraud-prob').textContent   = (prob * 100).toFixed(1) + '%';
  document.getElementById('score-lgbm').textContent   = (data.lgbm_score * 100).toFixed(1) + '%';
  document.getElementById('score-gnn').textContent    = data.gnn_score != null
    ? (data.gnn_score * 100).toFixed(1) + '%' : 'N/A';

  const badge = document.getElementById('risk-badge');
  badge.textContent = level;
  badge.className   = 'risk-badge risk-' + level.toLowerCase().replace(' ', '-');

  renderGauge(prob);
  document.getElementById('predict-output').classList.remove('d-none');
}

function renderGauge(prob) {
  const trace = {
    type: 'indicator', mode: 'gauge+number',
    value: prob * 100,
    number: { suffix: '%', font: { color: '#e2e8f0', size: 28 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: '#475569' },
      bar:  { color: gaugeColor(prob) },
      bgcolor: '#1a1d27',
      bordercolor: '#2d3348',
      steps: [
        { range: [0, 20],  color: 'rgba(34,197,94,0.15)'   },
        { range: [20, 40], color: 'rgba(34,197,94,0.08)'   },
        { range: [40, 60], color: 'rgba(245,158,11,0.1)'   },
        { range: [60, 80], color: 'rgba(249,115,22,0.12)'  },
        { range: [80, 100],color: 'rgba(239,68,68,0.15)'   },
      ],
    },
  };
  Plotly.newPlot('chart-risk-gauge', [trace], {
    paper_bgcolor: 'transparent',
    font: { color: '#e2e8f0' },
    height: 160,
    margin: { t: 10, r: 20, b: 10, l: 20 },
  }, { responsive: true, displayModeBar: false });
}

function gaugeColor(prob) {
  if (prob >= 0.8) return '#ef4444';
  if (prob >= 0.6) return '#f97316';
  if (prob >= 0.4) return '#f59e0b';
  return '#22c55e';
}

// ── Utils ─────────────────────────────────────────────────────────────────────
async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
