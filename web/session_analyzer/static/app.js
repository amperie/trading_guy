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
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${value.toFixed(2)}%`;
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function formatNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(digits);
}

function parseSeries(series) {
  return (series || []).map((pt) => ({ x: new Date(pt.x), y: pt.y }));
}

function seriesMap(series) {
  const map = new Map();
  (series || []).forEach((pt) => map.set(pt.x, pt.y));
  return map;
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
  $("metaName").textContent = session.name || session._id || "--";
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
  if (entries.includes("SPY")) state.selectedSymbols.add("SPY");

  entries.forEach((symbol) => {
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
      renderCharts();
    });
    filters.appendChild(btn);
  });
}

function filterSeries(series, start, end) {
  return (series || []).filter((pt) => {
    const t = new Date(pt.x);
    return (!start || t >= start) && (!end || t <= end);
  });
}

function filterSignals(signals, start, end) {
  return (signals || []).filter((sig) => {
    const t = new Date(sig.x);
    return (!start || t >= start) && (!end || t <= end);
  });
}

function filterIndicators(indicators, start, end) {
  const out = { _config: indicators?._config || {} };
  for (const [symbol, payload] of Object.entries(indicators || {})) {
    if (symbol === "_config") continue;
    out[symbol] = {};
    for (const [key, series] of Object.entries(payload)) {
      out[symbol][key] = filterSeries(series, start, end);
    }
  }
  return out;
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
  for (const [sym, series] of Object.entries(raw.symbols || {})) {
    filteredSymbols[sym] = filterSeries(series, start, end);
  }
  const filteredTrades = raw.trades.filter((t) => {
    const entry = new Date(t.entry_time);
    return (!start || entry >= start) && (!end || entry <= end);
  });
  const filteredSignals = filterSignals(raw.signals, start, end);

  return {
    ...raw,
    portfolio: { total_value: filteredValue, cash: filteredCash },
    symbols: filteredSymbols,
    signals: filteredSignals,
    indicators: filterIndicators(raw.indicators, start, end),
    trades: filteredTrades,
    metrics: recomputeMetrics(filteredValue, filteredTrades),
    benchmark: recomputeBenchmark(filteredValue, filteredSymbols.SPY),
  };
}

function datasetBase(color, label, data, extra = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: `${color}22`,
    tension: 0.25,
    fill: false,
    ...extra,
  };
}

function signalTooltipLines(raw) {
  const lines = [
    `${raw.signalType} ${raw.symbol}`,
    `Time: ${new Date(raw.x).toLocaleString()}`,
  ];
  if (raw.price != null) lines.push(`Price: ${formatMoney(raw.price)}`);
  if (raw.strength != null) lines.push(`Strength: ${formatNum(raw.strength, 2)}`);
  const md = raw.metadata || {};
  if (md.rsi_value != null) lines.push(`RSI: ${formatNum(md.rsi_value, 2)}`);
  if (md.ma_50 != null && md.ma_200 != null) {
    lines.push(`MA50 / MA200: ${formatNum(md.ma_50, 2)} / ${formatNum(md.ma_200, 2)}`);
  }
  if (md.atr_percentile_50 != null) lines.push(`ATR p50: ${formatNum(md.atr_percentile_50, 4)}`);
  if (md.regime) lines.push(`Regime: ${md.regime}`);
  if (md.volatility_filtered != null) lines.push(`Vol filter pass: ${md.volatility_filtered ? "yes" : "no"}`);
  return lines;
}

function tooltipLabel(context) {
  const raw = context.raw || {};
  if (raw.signalType) return signalTooltipLines(raw);
  return `${context.dataset.label}: ${formatNum(raw.y ?? context.parsed.y, 2)}`;
}

function commonChartOptions(yTickCallback = null, yMin = null, yMax = null) {
  return {
    interaction: {
      mode: "nearest",
      intersect: false,
    },
    plugins: {
      legend: { labels: { color: "#c9d2ef" } },
      tooltip: { callbacks: { label: tooltipLabel } },
      zoom: zoomConfig("x"),
    },
    scales: {
      x: { type: "time", ticks: { color: "#8d96b3" } },
      y: {
        min: yMin,
        max: yMax,
        ticks: {
          color: "#8d96b3",
          callback: yTickCallback || undefined,
        },
      },
    },
  };
}

function buildSignalDataset(label, color, points, pointStyle = "circle") {
  return {
    type: "scatter",
    label,
    data: points.map((pt) => ({ ...pt, x: new Date(pt.x) })),
    borderColor: color,
    backgroundColor: color,
    pointRadius: 4,
    pointHoverRadius: 7,
    pointStyle,
    showLine: false,
  };
}

function signalPointsForSymbol(symbol, valueFn) {
  return (state.filtered.signals || [])
    .filter((sig) => sig.symbol === symbol)
    .map((sig) => {
      const y = valueFn(sig);
      if (y === null || y === undefined || Number.isNaN(y)) return null;
      return {
        x: sig.x,
        y,
        symbol: sig.symbol,
        signalType: sig.type,
        strength: sig.strength,
        price: sig.price,
        metadata: sig.metadata || {},
      };
    })
    .filter(Boolean);
}

function renderEquityChart() {
  const equityRaw = state.filtered.portfolio.total_value;
  const equitySeries = parseSeries(equityRaw);
  const datasets = [
    datasetBase(palette[0], "Portfolio", equitySeries, {
      backgroundColor: "rgba(255, 107, 61, 0.15)",
      fill: true,
      tension: 0.3,
    }),
  ];

  const spySeriesRaw = state.filtered.symbols.SPY;
  if (spySeriesRaw?.length) {
    const spySeries = parseSeries(spySeriesRaw);
    const scale = equitySeries.length ? equitySeries[0].y / spySeries[0].y : 1;
    datasets.push(
      datasetBase(palette[1], "SPY (buy & hold)", spySeries.map((pt) => ({ x: pt.x, y: pt.y * scale })), {
        borderDash: [6, 4],
      })
    );
  }

  const equityMap = seriesMap(equityRaw);
  const allSignalPoints = (state.filtered.signals || [])
    .map((sig) => {
      const y = equityMap.get(sig.x);
      if (y == null) return null;
      return {
        x: sig.x,
        y,
        symbol: sig.symbol,
        signalType: sig.type,
        strength: sig.strength,
        price: sig.price,
        metadata: sig.metadata || {},
      };
    })
    .filter(Boolean);

  datasets.push(buildSignalDataset("Signals", "#f6c945", allSignalPoints, "rectRot"));

  buildChart("equityChart", {
    type: "line",
    data: { datasets },
    options: commonChartOptions(),
  });

  const ddSeries = computeDrawdown(equitySeries);
  buildChart("drawdownChart", {
    type: "line",
    data: {
      datasets: [
        datasetBase(palette[5], "Drawdown %", ddSeries, {
          backgroundColor: "rgba(255, 107, 61, 0.15)",
          fill: true,
        }),
      ],
    },
    options: commonChartOptions((v) => `${v.toFixed(0)}%`),
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
  const cfg = state.filtered.indicators?._config || {};
  let colorIndex = 0;

  state.selectedSymbols.forEach((symbol) => {
    const series = state.filtered.symbols[symbol];
    if (!series?.length) return;

    const priceColor = palette[colorIndex % palette.length];
    datasets.push(datasetBase(priceColor, `${symbol} Price`, parseSeries(series)));

    const ind = state.filtered.indicators?.[symbol];
    if (ind?.ma_short?.length) {
      datasets.push(
        datasetBase("#ffd166", `${symbol} MA ${cfg.ma_short_period || 50}`, parseSeries(ind.ma_short), {
          borderDash: [4, 4],
          pointRadius: 0,
        })
      );
    }
    if (ind?.ma_long?.length) {
      datasets.push(
        datasetBase("#8ecae6", `${symbol} MA ${cfg.ma_long_period || 200}`, parseSeries(ind.ma_long), {
          borderDash: [8, 4],
          pointRadius: 0,
        })
      );
    }

    const buyPoints = signalPointsForSymbol(symbol, (sig) => sig.price).filter((sig) => sig.signalType === "BUY");
    const sellPoints = signalPointsForSymbol(symbol, (sig) => sig.price).filter((sig) => sig.signalType === "SELL");
    if (buyPoints.length) datasets.push(buildSignalDataset(`${symbol} BUY`, "#2ec4b6", buyPoints, "triangle"));
    if (sellPoints.length) datasets.push(buildSignalDataset(`${symbol} SELL`, "#ff6b3d", sellPoints, "rectRot"));

    colorIndex += 1;
  });

  buildChart("symbolChart", {
    type: "line",
    data: { datasets },
    options: commonChartOptions(),
  });
}

function renderNormalizedSymbolChart() {
  const datasets = [];
  let colorIndex = 0;

  state.selectedSymbols.forEach((symbol) => {
    const series = state.filtered.symbols[symbol];
    if (!series?.length) return;
    const firstPrice = series[0].y;
    const normalized = series.map((pt) => ({
      x: pt.x,
      y: ((pt.y - firstPrice) / firstPrice) * 100,
    }));

    datasets.push(datasetBase(palette[colorIndex % palette.length], symbol, parseSeries(normalized)));

    const signalPts = signalPointsForSymbol(symbol, (sig) => ((sig.price - firstPrice) / firstPrice) * 100);
    if (signalPts.length) {
      datasets.push(buildSignalDataset(`${symbol} Signals`, "#f6c945", signalPts, "circle"));
    }
    colorIndex += 1;
  });

  buildChart("normalizedChart", {
    type: "line",
    data: { datasets },
    options: commonChartOptions((v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`),
  });
}

function renderRsiChart() {
  const datasets = [];
  const cfg = state.filtered.indicators?._config || {};
  const oversold = cfg.rsi_oversold_threshold;
  const overbought = cfg.rsi_overbought_threshold;
  const symbols = Array.from(state.selectedSymbols).filter((symbol) => state.filtered.indicators?.[symbol]?.rsi?.length);

  if (symbols.length) {
    const anchor = state.filtered.indicators[symbols[0]].rsi;
    if (oversold != null) {
      datasets.push(
        datasetBase("#2ec4b6", `Oversold ${oversold}`, anchor.map((pt) => ({ x: new Date(pt.x), y: oversold })), {
          borderDash: [5, 5],
          pointRadius: 0,
        })
      );
    }
    if (overbought != null) {
      datasets.push(
        datasetBase("#ff6b3d", `Overbought ${overbought}`, anchor.map((pt) => ({ x: new Date(pt.x), y: overbought })), {
          borderDash: [5, 5],
          pointRadius: 0,
        })
      );
    }
  }

  let colorIndex = 0;
  symbols.forEach((symbol) => {
    datasets.push(
      datasetBase(palette[colorIndex % palette.length], `${symbol} RSI ${cfg.rsi_period || 14}`, parseSeries(state.filtered.indicators[symbol].rsi))
    );

    const buyPoints = signalPointsForSymbol(symbol, (sig) => sig.metadata?.rsi_value).filter((sig) => sig.signalType === "BUY");
    const sellPoints = signalPointsForSymbol(symbol, (sig) => sig.metadata?.rsi_value).filter((sig) => sig.signalType === "SELL");
    if (buyPoints.length) datasets.push(buildSignalDataset(`${symbol} BUY RSI`, "#2ec4b6", buyPoints, "triangle"));
    if (sellPoints.length) datasets.push(buildSignalDataset(`${symbol} SELL RSI`, "#ff6b3d", sellPoints, "rectRot"));
    colorIndex += 1;
  });

  buildChart("rsiChart", {
    type: "line",
    data: { datasets },
    options: commonChartOptions((v) => v.toFixed(0), 0, 100),
  });
}

function renderAtrChart() {
  const datasets = [];
  const cfg = state.filtered.indicators?._config || {};
  let colorIndex = 0;

  state.selectedSymbols.forEach((symbol) => {
    const ind = state.filtered.indicators?.[symbol];
    if (!ind?.atr?.length) return;

    datasets.push(datasetBase(palette[colorIndex % palette.length], `${symbol} ATR ${cfg.atr_period || 14}`, parseSeries(ind.atr)));
    if (ind.atr_percentile?.length) {
      datasets.push(
        datasetBase("#f6c945", `${symbol} ATR p${cfg.atr_percentile_level || 50}`, parseSeries(ind.atr_percentile), {
          borderDash: [6, 4],
        })
      );
    }

    const atrByTs = seriesMap(ind.atr);
    const signalPts = signalPointsForSymbol(symbol, (sig) => atrByTs.get(sig.x));
    if (signalPts.length) datasets.push(buildSignalDataset(`${symbol} ATR Signals`, "#ffffff", signalPts, "circle"));
    colorIndex += 1;
  });

  buildChart("atrChart", {
    type: "line",
    data: { datasets },
    options: commonChartOptions(),
  });
}

function renderCharts() {
  renderEquityChart();
  renderSymbolChart();
  renderNormalizedSymbolChart();
  renderRsiChart();
  renderAtrChart();
}

function applyAndRender() {
  if (!state.raw) return;
  const fromVal = $("dateFrom").value;
  const toVal = $("dateTo").value;
  const start = fromVal ? new Date(fromVal) : null;
  const end = toVal ? new Date(`${toVal}T23:59:59`) : null;
  state.filtered = applyDateFilter(state.raw, start, end);
  const d = state.filtered;
  renderMetadata(d.session, d.orders);
  renderMetrics(d.metrics, d.benchmark || {});
  renderTradeTable(d.trades);
  renderSymbols(d.symbols);
  renderCharts();
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
      const name = s.name.length > 22 ? `${s.name.slice(0, 22)}...` : s.name;
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
  $("resetEquity").addEventListener("click", () => state.charts.equityChart?.resetZoom());
  $("resetDrawdown").addEventListener("click", () => state.charts.drawdownChart?.resetZoom());
  $("resetReturns").addEventListener("click", () => state.charts.returnsChart?.resetZoom());
  $("resetSymbol").addEventListener("click", () => state.charts.symbolChart?.resetZoom());
  $("resetNormalized").addEventListener("click", () => state.charts.normalizedChart?.resetZoom());
  $("resetRsi").addEventListener("click", () => state.charts.rsiChart?.resetZoom());
  $("resetAtr").addEventListener("click", () => state.charts.atrChart?.resetZoom());

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
