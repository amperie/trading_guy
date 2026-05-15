const state = {
  raw: null,
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
    });
    filters.appendChild(btn);
  });
}

function renderEquityChart() {
  const equitySeries = parseSeries(state.raw.portfolio.total_value);
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

  const spySeries = state.raw.symbols.SPY ? parseSeries(state.raw.symbols.SPY) : null;
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
    const series = state.raw.symbols[symbol];
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

function renderAll(data) {
  state.raw = data;
  renderMetadata(data.session, data.orders);
  renderMetrics(data.metrics, data.benchmark || {});
  renderTradeTable(data.trades);
  renderSymbols(data.symbols);
  renderEquityChart();
  renderSymbolChart();
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
  if (!state.raw) return;
  const rows = [["timestamp", "total_value", "cash"]];
  const values = state.raw.portfolio.total_value;
  const cash = state.raw.portfolio.cash;
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
  if (!state.raw) return;
  const blob = new Blob([JSON.stringify(state.raw, null, 2)], {
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
  $("resetEquity").addEventListener("click", () => state.charts["equityChart"]?.resetZoom());
  $("resetDrawdown").addEventListener("click", () => state.charts["drawdownChart"]?.resetZoom());
  $("resetReturns").addEventListener("click", () => state.charts["returnsChart"]?.resetZoom());
  $("resetSymbol").addEventListener("click", () => state.charts["symbolChart"]?.resetZoom());

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
