const state = {
  raw: null,
  filtered: null,
  charts: {},
  symbols: [],
  selectedSymbols: new Set(),
};

const palette = [
  "#ff6b3d",
  "#2ec4b6",
  "#f6c945",
  "#8d7bff",
  "#4dd0e1",
  "#f06292",
  "#aed581",
];

const $ = (id) => document.getElementById(id);

function formatPct(value) {
  if (value === null || value === undefined) return "--";
  return `${value.toFixed(2)}%`;
}

function formatMoney(value) {
  if (value === null || value === undefined) return "--";
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function parseSeries(series) {
  return series.map((pt) => ({ x: new Date(pt.x), y: pt.y }));
}

function computeDrawdown(series) {
  let peak = -Infinity;
  return series.map((pt) => {
    peak = Math.max(peak, pt.y);
    const dd = peak ? ((pt.y - peak) / peak) * 100 : 0;
    return { x: pt.x, y: dd };
  });
}

function computeDailyReturns(series) {
  if (series.length < 2) return [];
  const daily = [];
  for (let i = 1; i < series.length; i++) {
    const prev = series[i - 1].y;
    const curr = series[i].y;
    daily.push(((curr - prev) / prev) * 100);
  }
  return daily;
}

function histogram(data, buckets = 20) {
  if (!data.length) return { labels: [], counts: [] };
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const step = span / buckets;
  const counts = new Array(buckets).fill(0);
  data.forEach((val) => {
    let idx = Math.floor((val - min) / step);
    if (idx === buckets) idx = buckets - 1;
    counts[idx] += 1;
  });
  const labels = counts.map((_, i) => (min + step * i).toFixed(2));
  return { labels, counts };
}

function zoomConfig(mode = "x") {
  return {
    zoom: {
      wheel: { enabled: true },
      pinch: { enabled: true },
      mode,
    },
    pan: {
      enabled: true,
      mode,
    },
  };
}

function buildChart(id, config) {
  if (state.charts[id]) {
    state.charts[id].destroy();
  }
  const ctx = $(id).getContext("2d");
  state.charts[id] = new Chart(ctx, config);
}

function renderMetadata(session, orders) {
  $("metaAccount").textContent = session.account_id || "--";
  $("metaName").textContent = session.name || "--";
  $("metaCreated").textContent = session.created_at
    ? new Date(session.created_at).toLocaleString()
    : "--";
  $("metaOrders").textContent = orders ? `${orders.total} total` : "--";
}

function fmt2(value) {
  return value != null ? value.toFixed(2) : "--";
}

function renderMetrics(metrics, benchmark) {
  $("metricReturn").textContent = formatPct(metrics.total_return_pct);
  $("metricAnnualized").textContent = formatPct(metrics.annualized_return);
  $("metricSharpe").textContent = `${fmt2(metrics.sharpe_ratio)} / ${fmt2(metrics.sortino_ratio)}`;
  $("metricSortino").textContent = `Volatility ${fmt2(metrics.volatility)}%`;
  $("metricDrawdown").textContent = formatPct(metrics.max_drawdown_pct);
  $("metricCalmar").textContent = `Calmar ${fmt2(metrics.calmar_ratio)}`;
  $("metricWinRate").textContent = formatPct(metrics.win_rate);
  $("metricProfitFactor").textContent = `Profit Factor ${fmt2(metrics.profit_factor)}`;
  $("metricTrades").textContent = metrics.total_trades != null ? `${metrics.total_trades}` : "--";
  $("metricAvgTrade").textContent = `Avg PnL ${formatMoney(metrics.avg_trade_pnl)}`;

  if (benchmark && benchmark._comparison) {
    const alpha = benchmark._comparison.alpha ?? null;
    const out = benchmark._comparison.outperformance ? "Outperformed SPY" : "Underperformed SPY";
    $("metricAlpha").textContent = `Alpha vs SPY: ${formatPct(alpha)}`;
    $("metricOutperformance").textContent = out;
  } else {
    $("metricAlpha").textContent = "Alpha vs SPY: --";
    $("metricOutperformance").textContent = "SPY data not available";
  }
}

function renderTradeTable(trades) {
  const rows = trades
    .map((trade) => {
      const pnlColor = trade.pnl >= 0 ? "style='color:#2ec4b6'" : "style='color:#ff6b3d'";
      return `
        <tr>
          <td>${trade.symbol}</td>
          <td>${new Date(trade.entry_time).toLocaleString()}</td>
          <td>${new Date(trade.exit_time).toLocaleString()}</td>
          <td>${trade.quantity}</td>
          <td ${pnlColor}>${formatMoney(trade.pnl)}</td>
          <td ${pnlColor}>${formatPct(trade.pnl_pct)}</td>
          <td>${trade.duration_hours.toFixed(2)}</td>
          <td>${trade.bracket_exit_type || "STANDARD"}</td>
        </tr>`;
    })
    .join("");
  $("tradeRows").innerHTML = rows || `<tr><td colspan="8">No trades found.</td></tr>`;
}

function renderSymbols(symbols) {
  const filters = $("symbolFilters");
  filters.innerHTML = "";
  const entries = Object.keys(symbols).sort();
  state.symbols = entries;
  state.selectedSymbols = new Set(entries.slice(0, 3));
  if (entries.includes("SPY")) {
    state.selectedSymbols.add("SPY");
  }
  entries.forEach((symbol, idx) => {
    const btn = document.createElement("button");
    btn.className = "symbol-toggle";
    btn.textContent = symbol;
    if (state.selectedSymbols.has(symbol)) btn.classList.add("active");
    btn.style.borderColor = "transparent";
    btn.addEventListener("click", () => {
      if (state.selectedSymbols.has(symbol)) {
        state.selectedSymbols.delete(symbol);
        btn.classList.remove("active");
      } else {
        state.selectedSymbols.add(symbol);
        btn.classList.add("active");
      }
      renderSymbolChart();
      renderNormalizedSymbolChart();
    });
    filters.appendChild(btn);
  });
}

function renderEquityChart() {
  const equitySeries = parseSeries(state.filtered.portfolio.total_value);
  const datasets = [
    {
      label: "Portfolio",
      data: equitySeries,
      borderColor: palette[0],
      backgroundColor: "rgba(255, 107, 61, 0.15)",
      tension: 0.3,
      fill: true,
    },
  ];

  const spySeries = state.filtered.symbols.SPY ? parseSeries(state.filtered.symbols.SPY) : null;
  if (spySeries && spySeries.length) {
    const scale = equitySeries.length ? equitySeries[0].y / spySeries[0].y : 1;
    const normalizedSpy = spySeries.map((pt) => ({ x: pt.x, y: pt.y * scale }));
    datasets.push({
      label: "SPY (buy & hold)",
      data: normalizedSpy,
      borderColor: palette[1],
      borderDash: [6, 4],
      tension: 0.25,
      fill: false,
    });
  }

  buildChart("equityChart", {
    type: "line",
    data: { datasets },
    options: {
      plugins: {
        legend: { labels: { color: "#c9d2ef" } },
        zoom: zoomConfig("x"),
      },
      scales: {
        x: { type: "time", time: { tooltipFormat: "MMM d HH:mm" }, ticks: { color: "#8d96b3" } },
        y: { ticks: { color: "#8d96b3" } },
      },
    },
  });

  const ddSeries = computeDrawdown(equitySeries);
  buildChart("drawdownChart", {
    type: "line",
    data: {
      datasets: [
        {
          label: "Drawdown %",
          data: ddSeries,
          borderColor: palette[5],
          backgroundColor: "rgba(255, 107, 61, 0.15)",
          tension: 0.25,
          fill: true,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false }, zoom: zoomConfig("x") },
      scales: {
        x: { type: "time", ticks: { color: "#8d96b3" } },
        y: { ticks: { color: "#8d96b3", callback: (v) => `${v.toFixed(0)}%` } },
      },
    },
  });

  const dailyReturns = computeDailyReturns(equitySeries);
  const hist = histogram(dailyReturns, 20);
  buildChart("returnsChart", {
    type: "bar",
    data: {
      labels: hist.labels,
      datasets: [
        {
          label: "Daily Return %",
          data: hist.counts,
          backgroundColor: "rgba(46, 196, 182, 0.7)",
        },
      ],
    },
    options: {
      plugins: { legend: { display: false }, zoom: zoomConfig("xy") },
      scales: {
        x: { ticks: { color: "#8d96b3" } },
        y: { ticks: { color: "#8d96b3" } },
      },
    },
  });
}

function renderSymbolChart() {
  const datasets = [];
  let colorIndex = 0;
  state.selectedSymbols.forEach((symbol) => {
    const series = state.filtered.symbols[symbol];
    if (!series) return;
    datasets.push({
      label: symbol,
      data: parseSeries(series),
      borderColor: palette[colorIndex % palette.length],
      tension: 0.25,
      fill: false,
    });
    colorIndex += 1;
  });

  buildChart("symbolChart", {
    type: "line",
    data: { datasets },
    options: {
      plugins: { legend: { labels: { color: "#c9d2ef" } }, zoom: zoomConfig("x") },
      scales: {
        x: { type: "time", ticks: { color: "#8d96b3" } },
        y: { ticks: { color: "#8d96b3" } },
      },
    },
  });
}

function renderNormalizedSymbolChart() {
  const datasets = [];
  let colorIndex = 0;
  state.selectedSymbols.forEach((symbol) => {
    const series = state.filtered.symbols[symbol];
    if (!series || series.length === 0) return;
    const firstPrice = series[0].y;
    datasets.push({
      label: symbol,
      data: parseSeries(series).map((pt) => ({
        x: pt.x,
        y: ((pt.y - firstPrice) / firstPrice) * 100,
      })),
      borderColor: palette[colorIndex % palette.length],
      tension: 0.25,
      fill: false,
    });
    colorIndex += 1;
  });

  buildChart("normalizedChart", {
    type: "line",
    data: { datasets },
    options: {
      plugins: { legend: { labels: { color: "#c9d2ef" } }, zoom: zoomConfig("x") },
      scales: {
        x: { type: "time", ticks: { color: "#8d96b3" } },
        y: {
          ticks: {
            color: "#8d96b3",
            callback: (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`,
          },
        },
      },
    },
  });
}

function filterSeries(series, start, end) {
  return series.filter((pt) => {
    const t = new Date(pt.x);
    return (!start || t >= start) && (!end || t <= end);
  });
}

function recomputeMetrics(equitySeries, trades) {
  if (!equitySeries.length) return {};
  const values = equitySeries.map((pt) => pt.y);
  const first = values[0];
  const last = values[values.length - 1];
  const totalReturn = ((last - first) / first) * 100;

  let peak = -Infinity;
  let maxDD = 0;
  for (const v of values) {
    peak = Math.max(peak, v);
    const dd = peak ? ((v - peak) / peak) * 100 : 0;
    maxDD = Math.min(maxDD, dd);
  }

  const dailyReturns = [];
  for (let i = 1; i < values.length; i++) {
    dailyReturns.push(((values[i] - values[i - 1]) / values[i - 1]) * 100);
  }
  const mean = dailyReturns.length ? dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length : 0;
  const variance = dailyReturns.length
    ? dailyReturns.reduce((a, b) => a + (b - mean) ** 2, 0) / dailyReturns.length
    : 0;
  const std = Math.sqrt(variance);
  const sharpe = std ? (mean / std) * Math.sqrt(252) : null;
  const downReturns = dailyReturns.filter((r) => r < 0);
  const downStd = Math.sqrt(
    downReturns.length ? downReturns.reduce((a, b) => a + b ** 2, 0) / downReturns.length : 0
  );
  const sortino = downStd ? (mean / downStd) * Math.sqrt(252) : null;

  const startDate = new Date(equitySeries[0].x);
  const endDate = new Date(equitySeries[equitySeries.length - 1].x);
  const years = (endDate - startDate) / (365.25 * 24 * 3600 * 1000);
  const annualizedReturn = years > 0 ? (Math.pow(last / first, 1 / years) - 1) * 100 : totalReturn;
  const calmar = maxDD ? annualizedReturn / Math.abs(maxDD) : null;

  const winning = trades.filter((t) => t.pnl > 0);
  const losing = trades.filter((t) => t.pnl < 0);
  const winRate = trades.length ? (winning.length / trades.length) * 100 : null;
  const avgTradePnl = trades.length ? trades.reduce((a, t) => a + t.pnl, 0) / trades.length : null;
  const grossProfit = winning.reduce((a, t) => a + t.pnl, 0);
  const grossLoss = Math.abs(losing.reduce((a, t) => a + t.pnl, 0));
  const profitFactor = grossLoss ? grossProfit / grossLoss : null;

  return {
    total_return_pct: totalReturn,
    annualized_return: annualizedReturn,
    sharpe_ratio: sharpe,
    sortino_ratio: sortino,
    max_drawdown_pct: maxDD,
    calmar_ratio: calmar,
    volatility: std * Math.sqrt(252),
    win_rate: winRate,
    profit_factor: profitFactor,
    total_trades: trades.length,
    avg_trade_pnl: avgTradePnl,
  };
}

function recomputeBenchmark(equitySeries, spySeries) {
  if (!spySeries || !spySeries.length || !equitySeries.length) return {};
  const startDate = new Date(equitySeries[0].x);
  const endDate = new Date(equitySeries[equitySeries.length - 1].x);
  const filteredSpy = spySeries.filter((pt) => {
    const t = new Date(pt.x);
    return t >= startDate && t <= endDate;
  });
  if (!filteredSpy.length) return {};
  const spyReturn = ((filteredSpy[filteredSpy.length - 1].y - filteredSpy[0].y) / filteredSpy[0].y) * 100;
  const portfolioReturn = ((equitySeries[equitySeries.length - 1].y - equitySeries[0].y) / equitySeries[0].y) * 100;
  return { _comparison: { alpha: portfolioReturn - spyReturn, outperformance: portfolioReturn > spyReturn } };
}

function applyDateFilter(raw, start, end) {
  if (!start && !end) return raw;
  const filteredValue = filterSeries(raw.portfolio.total_value, start, end);
  const filteredCash = filterSeries(raw.portfolio.cash, start, end);
  const filteredSymbols = {};
  for (const [sym, series] of Object.entries(raw.symbols)) {
    filteredSymbols[sym] = filterSeries(series, start, end);
  }
  const filteredTrades = raw.trades.filter((t) => {
    const entry = new Date(t.entry_time);
    return (!start || entry >= start) && (!end || entry <= end);
  });
  return {
    ...raw,
    portfolio: { total_value: filteredValue, cash: filteredCash },
    symbols: filteredSymbols,
    trades: filteredTrades,
    metrics: recomputeMetrics(filteredValue, filteredTrades),
    benchmark: recomputeBenchmark(filteredValue, filteredSymbols.SPY),
  };
}

function applyAndRender() {
  if (!state.raw) return;
  const fromVal = $("dateFrom").value;
  const toVal = $("dateTo").value;
  const start = fromVal ? new Date(fromVal) : null;
  const end = toVal ? new Date(toVal + "T23:59:59") : null;
  state.filtered = applyDateFilter(state.raw, start, end);
  const d = state.filtered;
  renderMetadata(d.session, d.orders);
  renderMetrics(d.metrics, d.benchmark || {});
  renderTradeTable(d.trades);
  renderSymbols(d.symbols);
  renderEquityChart();
  renderSymbolChart();
  renderNormalizedSymbolChart();
}

function renderAll(data) {
  state.raw = data;
  const firstTs = data.portfolio.total_value[0]?.x;
  const lastTs = data.portfolio.total_value[data.portfolio.total_value.length - 1]?.x;
  if (firstTs) $("dateFrom").value = firstTs.slice(0, 10);
  if (lastTs) $("dateTo").value = lastTs.slice(0, 10);
  applyAndRender();
}

async function loadSessionList() {
  const db = $("dbName").value.trim();
  const sel = $("sessionId");
  sel.innerHTML = '<option value="">-- loading --</option>';
  try {
    const url = db ? `/api/sessions?db=${encodeURIComponent(db)}` : "/api/sessions";
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to list sessions");
    const sessions = await res.json();
    sel.innerHTML = '<option value="">-- select a session --</option>';
    sessions.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.session_id;
      const name = s.name.length > 22 ? s.name.slice(0, 22) + "…" : s.name;
      const d = s.created_at ? new Date(s.created_at) : null;
      const date = d
        ? `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`
        : "";
      opt.textContent = date ? `${name} (${date})` : name;
      sel.appendChild(opt);
    });
    if (sessions.length > 0) sel.value = sessions[0].session_id;
  } catch (err) {
    sel.innerHTML = '<option value="">-- error loading sessions --</option>';
  }
}

async function loadSession() {
  const sessionId = $("sessionId").value.trim();
  if (!sessionId) return;
  $("status").textContent = "Loading session data...";
  try {
    const db = $("dbName").value.trim();
    const url = db
      ? `/api/session/${sessionId}?db=${encodeURIComponent(db)}`
      : `/api/session/${sessionId}`;
    const res = await fetch(url);
    if (!res.ok) {
      const msg = await res.json();
      throw new Error(msg.error || "Failed to load session");
    }
    const data = await res.json();
    renderAll(data);
    $("status").textContent = `Loaded ${sessionId}`;
  } catch (err) {
    $("status").textContent = `Error: ${err.message}`;
  }
}

function exportEquityCsv() {
  if (!state.filtered) return;
  const rows = [["timestamp", "total_value", "cash"]];
  const values = state.filtered.portfolio.total_value;
  const cash = state.filtered.portfolio.cash;
  for (let i = 0; i < values.length; i++) {
    rows.push([values[i].x, values[i].y, cash[i] ? cash[i].y : ""]);
  }
  const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], {
    type: "text/csv",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "equity_curve.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function exportJson() {
  if (!state.filtered) return;
  const blob = new Blob([JSON.stringify(state.filtered, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "session_analysis.json";
  a.click();
  URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-animate]").forEach((el, idx) => {
    setTimeout(() => el.classList.add("visible"), idx * 120);
  });

  $("loadBtn").addEventListener("click", loadSession);
  $("dbName").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadSessionList();
  });
  $("dbName").addEventListener("change", loadSessionList);
  $("exportEquity").addEventListener("click", exportEquityCsv);
  $("exportJson").addEventListener("click", exportJson);
  $("dateFrom").addEventListener("change", applyAndRender);
  $("dateTo").addEventListener("change", applyAndRender);
  $("resetEquity").addEventListener("click", () => state.charts["equityChart"]?.resetZoom());
  $("resetDrawdown").addEventListener("click", () => state.charts["drawdownChart"]?.resetZoom());
  $("resetReturns").addEventListener("click", () => state.charts["returnsChart"]?.resetZoom());
  $("resetSymbol").addEventListener("click", () => state.charts["symbolChart"]?.resetZoom());
  $("resetNormalized").addEventListener("click", () => state.charts["normalizedChart"]?.resetZoom());

  const params = new URLSearchParams(window.location.search);
  const sessionId = params.get("session_id");
  const db = params.get("db");
  if (db) $("dbName").value = db;

  loadSessionList().then(() => {
    if (sessionId) {
      $("sessionId").value = sessionId;
      loadSession();
    }
  });
});
