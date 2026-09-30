"use strict";

const DATA_BASE = "https://symbolicarena-pages-1988054973082523.oss-cn-hongkong.aliyuncs.com/web/releases/release-20260930-v1/";
const SCHEMA = "symbolicarena-pages-v1";
const SPLITS = ["train", "valid", "id_test", "ood_test"];
const SPLIT_LABELS = { train: "训练", valid: "验证", id_test: "ID 测试", ood_test: "OOD 测试" };
const SPLIT_COLORS = { train: "#1c8f88", valid: "#db8656", id_test: "#4779b8", ood_test: "#865db3" };
const ALGORITHM_LABELS = { imcts: "iMCTS", qlattice: "QLattice", llmsr: "LLM-SR", drsr: "DrSR", e2esr: "E2ESR", tpsr: "TPSR" };

const elements = {
  algorithm: document.querySelector("#algorithm-select"),
  dataset: document.querySelector("#dataset-select"),
  seed: document.querySelector("#seed-select"),
  status: document.querySelector("#selection-status"),
  datasetInfo: document.querySelector("#dataset-info"),
  chart: document.querySelector("#run-chart"),
  chartTitle: document.querySelector("#chart-title"),
  dimension: document.querySelector("#chart-dimension"),
  plotNote: document.querySelector("#plot-note"),
  minute: document.querySelector("#minute-range"),
  minuteValue: document.querySelector("#minute-value"),
  minuteStatus: document.querySelector("#minute-status"),
  play: document.querySelector("#play-button"),
  speed: document.querySelector("#playback-speed"),
  equation: document.querySelector("#equation-text"),
  runSummary: document.querySelector("#run-summary"),
  metrics: {
    train: document.querySelector("#metric-train"),
    valid: document.querySelector("#metric-valid"),
    id_test: document.querySelector("#metric-id"),
    ood_test: document.querySelector("#metric-ood"),
  },
};

const state = {
  catalog: null,
  datasets: new Map(),
  runs: new Map(),
  dataset: null,
  run: null,
  condition: "clean",
  minute: 1,
  requestId: 0,
  controller: null,
  playing: false,
  playbackTimer: null,
  plotRunning: false,
  plotRequested: false,
};

function showStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
}

function reportError(error) {
  if (error.name === "AbortError") return;
  stopPlayback();
  showStatus(`读取失败：${error.message}`, true);
  console.error(error);
}

async function readJson(path, signal) {
  const response = await fetch(DATA_BASE + path, { signal });
  if (!response.ok) throw new Error(`${path} 返回 HTTP ${response.status}`);
  if (!path.endsWith(".gz")) return response.json();
  if (!window.DecompressionStream || !response.body) {
    throw new Error("当前浏览器无法读取压缩的实验资源");
  }
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).json();
}

async function cachedJson(cache, path, signal) {
  if (cache.has(path)) return cache.get(path);
  const value = await readJson(path, signal);
  if (value.schema !== SCHEMA) throw new Error(`${path} 的数据版本不一致`);
  cache.set(path, value);
  return value;
}

function selectedKey(algorithm, dataset, condition, seed) {
  return `${algorithm}:${dataset}:${condition}:${seed}`;
}

function availableDatasets(algorithm) {
  const indexes = new Set(
    state.catalog.runs.filter((run) => run.algorithm === algorithm).map((run) => run.dataset_index),
  );
  return state.catalog.datasets.filter((dataset) => indexes.has(dataset.index));
}

function setOptions(select, items, selected) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = String(item.value);
    option.textContent = item.label;
    select.append(option);
  }
  if (items.some((item) => String(item.value) === String(selected))) select.value = String(selected);
}

function setDatasetOptions(selected) {
  const algorithm = elements.algorithm.value;
  const datasets = availableDatasets(algorithm);
  setOptions(elements.dataset, datasets.map((dataset) => {
    const run = state.catalog.runs.find((item) => item.algorithm === algorithm && item.dataset_index === dataset.index);
    return { value: dataset.index, label: `${run.dataset_id} · ${dataset.name}` };
  }), selected);
}

function setSeedOptions(selected) {
  const seeds = state.catalog.runs
    .filter((run) => run.algorithm === elements.algorithm.value && run.dataset_index === elements.dataset.value && run.condition === state.condition)
    .map((run) => run.seed)
    .sort((left, right) => left - right);
  setOptions(elements.seed, seeds.map((seed) => ({ value: seed, label: String(seed) })), selected);
}

function selectedRunEntry() {
  const key = selectedKey(elements.algorithm.value, elements.dataset.value, state.condition, Number(elements.seed.value));
  const run = state.catalog.runs.find((entry) => selectedKey(entry.algorithm, entry.dataset_index, entry.condition, entry.seed) === key);
  if (!run) throw new Error(`实验记录不存在：${key}`);
  return run;
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.playbackTimer);
  elements.play.textContent = "▶ 播放";
}

async function loadSelection() {
  stopPlayback();
  state.requestId += 1;
  const requestId = state.requestId;
  state.controller?.abort();
  state.controller = new AbortController();
  const entry = selectedRunEntry();
  const datasetEntry = state.catalog.datasets.find((item) => item.index === entry.dataset_index);
  if (!datasetEntry) throw new Error(`数据集记录不存在：${entry.dataset_index}`);
  showStatus(`正在读取 ${entry.algorithm} / ${entry.dataset_id} / ${entry.condition} / ${entry.seed}…`);
  const [dataset, run] = await Promise.all([
    cachedJson(state.datasets, datasetEntry.path, state.controller.signal),
    cachedJson(state.runs, entry.path, state.controller.signal),
  ]);
  if (requestId !== state.requestId) return;
  if (
    run.identity.algorithm !== entry.algorithm
    || run.identity.dataset_index !== entry.dataset_index
    || run.identity.condition !== entry.condition
    || run.identity.seed !== entry.seed
  ) throw new Error("运行文件与目录索引不一致");
  if (dataset.dataset_index !== entry.dataset_index || run.timeline.length !== 180) {
    throw new Error("数据集或训练时间线格式不正确");
  }
  state.dataset = dataset;
  state.run = run;
  while (state.runs.size > 8) state.runs.delete(state.runs.keys().next().value);
  updateDatasetInfo();
  showStatus(`${entry.dataset_id} · ${entry.available_snapshots}/180 分钟记录 · ${entry.candidate_count} 个候选`);
  await renderMinute();
}

function updateDatasetInfo() {
  const dataset = state.dataset;
  const parts = [
    `数据集：${dataset.name}`,
    `特征：${dataset.feature_names.join("、")}`,
    `展示样本：${dataset.samples.split.length} 个`,
    `许可证：${dataset.license}`,
  ];
  if (dataset.citation?.title) parts.push(`数据来源：${dataset.citation.title}`);
  if (dataset.plot.pca) {
    const variance = dataset.plot.pca.explained_variance_ratio.map((value) => `${(value * 100).toFixed(1)}%`);
    parts.push(`PCA 解释方差：PC1 ${variance[0]}，PC2 ${variance[1]}`);
  }
  if (state.catalog.source_summary.formal_ready === false) {
    parts.push("归档的指标核验状态尚未标记完成");
  }
  elements.datasetInfo.replaceChildren(...parts.map((content) => {
    const line = document.createElement("span");
    line.textContent = content;
    return line;
  }));
  elements.chartTitle.textContent = `${state.run.identity.dataset_id} 的拟合结果`;
  elements.dimension.textContent = {
    line: "一维曲线",
    surface: "二维曲面",
    projection: "PCA 投影",
  }[dataset.plot.kind];
  elements.runSummary.textContent = `结束状态：${state.run.terminal.status} · 候选数量：${state.run.candidates.length} · 噪声系数：${state.run.noise.sigma}`;
}

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

function observedTraces(dataset, run, is3d) {
  const coordinates = dataset.plot.coordinates;
  const traces = [];
  for (const split of SPLITS) {
    const indexes = dataset.samples.split.flatMap((name, index) => name === split ? [index] : []);
    if (indexes.length === 0) continue;
    const trace = {
      type: is3d ? "scatter3d" : "scatter",
      mode: "markers",
      name: `${SPLIT_LABELS[split]}样本`,
      x: indexes.map((index) => coordinates[index][0]),
      marker: { size: is3d ? 3 : 6, color: SPLIT_COLORS[split], opacity: 0.78 },
    };
    if (is3d) {
      trace.y = indexes.map((index) => coordinates[index][1]);
      trace.z = indexes.map((index) => run.observed_y[index]);
    } else {
      trace.y = indexes.map((index) => run.observed_y[index]);
    }
    traces.push(trace);
  }
  return traces;
}

function plotLayout(dataset, is3d) {
  const base = {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: "#31524b", size: 11 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: -0.08 },
    uirevision: dataset.dataset_index,
  };
  if (is3d) {
    base.margin = { l: 0, r: 0, t: 0, b: 0 };
    base.scene = {
      xaxis: { title: dataset.plot.axis_labels[0] },
      yaxis: { title: dataset.plot.axis_labels[1] },
      zaxis: { title: dataset.plot.axis_labels[2] },
      bgcolor: "rgba(0,0,0,0)",
    };
  } else {
    base.margin = { l: 58, r: 20, t: 12, b: 56 };
    base.xaxis = { title: dataset.plot.axis_labels[0], gridcolor: "#e2eae3" };
    base.yaxis = { title: dataset.plot.axis_labels[1], gridcolor: "#e2eae3" };
  }
  return base;
}

function plotTraces(dataset, run, candidate) {
  const kind = dataset.plot.kind;
  const is3d = kind !== "line";
  const traces = observedTraces(dataset, run, is3d);
  const predictions = candidate?.plot;
  if (!predictions || !["ok", "partial"].includes(predictions.status)) return traces;
  if (predictions.sample_y.length !== dataset.samples.split.length) {
    throw new Error("候选预测数量与展示样本数量不一致");
  }
  if (kind === "line") {
    traces.unshift({
      type: "scatter", mode: "lines", name: "候选曲线",
      x: dataset.plot.grid.x, y: predictions.grid_y,
      line: { color: "#123f3b", width: 3 },
    });
  } else {
    if (kind === "surface") {
      const grid = dataset.plot.grid;
      const side = grid.x.length;
      if (predictions.grid_y.length !== side * grid.y.length) throw new Error("曲面网格数量不一致");
      const surface = [];
      for (let row = 0; row < grid.y.length; row += 1) {
        surface.push(predictions.grid_y.slice(row * side, (row + 1) * side));
      }
      traces.unshift({
        type: "surface", name: "候选曲面", x: grid.x, y: grid.y, z: surface,
        colorscale: [[0, "#cee8db"], [1, "#168981"]], opacity: 0.68, showscale: false,
      });
    }
    traces.push({
      type: "scatter3d", mode: "markers", name: "候选预测点",
      x: dataset.plot.coordinates.map((row) => row[0]),
      y: dataset.plot.coordinates.map((row) => row[1]),
      z: predictions.sample_y,
      marker: { color: "#173c39", size: 2.5, symbol: "diamond", opacity: 0.65 },
    });
  }
  return traces;
}

async function requestPlot() {
  state.plotRequested = true;
  if (state.plotRunning) return;
  state.plotRunning = true;
  try {
    while (state.plotRequested) {
      state.plotRequested = false;
      const dataset = state.dataset;
      const run = state.run;
      const frame = run.timeline[state.minute - 1];
      const candidate = frame.candidate === null ? null : run.candidates[frame.candidate];
      await Plotly.react(
        elements.chart,
        plotTraces(dataset, run, candidate),
        plotLayout(dataset, dataset.plot.kind !== "line"),
        { responsive: true, displaylogo: false, scrollZoom: false },
      );
    }
  } finally {
    state.plotRunning = false;
  }
}

async function renderMinute() {
  if (!state.dataset || !state.run) return;
  const frame = state.run.timeline[state.minute - 1];
  const candidate = frame.candidate === null ? null : state.run.candidates[frame.candidate];
  elements.minute.value = String(state.minute);
  elements.minuteValue.textContent = `第 ${state.minute} / 180 分钟`;
  elements.minuteStatus.textContent = frame.status === "missing" ? "该分钟缺少记录" : `记录状态：${frame.status}`;
  elements.equation.textContent = candidate?.equation || "当前分钟没有可用公式";
  for (const split of SPLITS) {
    elements.metrics[split].textContent = formatMetric(frame.metrics?.[split]?.r2);
  }
  const plot = candidate?.plot;
  elements.plotNote.textContent = !candidate
    ? "当前分钟没有可用候选。"
    : plot.status === "ok" ? "图形数值已预计算；训练样本显示本次运行实际使用的噪声标签。"
      : plot.status === "partial" ? "部分预测点无有限数值，图形中保留空缺。"
        : `当前候选无法绘图：${plot.reason}`;
  await requestPlot();
}

async function playbackStep() {
  if (!state.playing) return;
  if (state.minute >= 180) {
    stopPlayback();
    return;
  }
  state.minute += 1;
  await renderMinute();
  if (state.playing) state.playbackTimer = setTimeout(() => playbackStep().catch(reportError), Number(elements.speed.value));
}

for (const button of document.querySelectorAll(".noise-choice")) {
  button.addEventListener("click", () => {
    state.condition = button.dataset.condition;
    for (const option of document.querySelectorAll(".noise-choice")) {
      option.setAttribute("aria-pressed", String(option === button));
    }
    setSeedOptions(elements.seed.value);
    loadSelection().catch(reportError);
  });
}
elements.algorithm.addEventListener("change", () => {
  setDatasetOptions();
  setSeedOptions();
  loadSelection().catch(reportError);
});
elements.dataset.addEventListener("change", () => {
  setSeedOptions();
  loadSelection().catch(reportError);
});
elements.seed.addEventListener("change", () => loadSelection().catch(reportError));
elements.minute.addEventListener("input", () => {
  state.minute = Number(elements.minute.value);
  renderMinute().catch(reportError);
});
elements.play.addEventListener("click", () => {
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (!state.run) return;
  if (state.minute >= 180) state.minute = 1;
  state.playing = true;
  elements.play.textContent = "Ⅱ 暂停";
  playbackStep().catch(reportError);
});

async function initialize() {
  if (!window.Plotly) throw new Error("图形组件未能加载");
  const catalog = await readJson("catalog.json");
  if (catalog.schema !== SCHEMA || catalog.runs.length !== 6750 || catalog.datasets.length !== 50) {
    throw new Error("实验目录数量或版本不正确");
  }
  state.catalog = catalog;
  setOptions(elements.algorithm, catalog.algorithms.map((id) => ({ value: id, label: ALGORITHM_LABELS[id] || id })));
  setDatasetOptions();
  setSeedOptions();
  await loadSelection();
}

initialize().catch(reportError);
