"use strict";

const DATA_BASE = "https://symbolicarena-pages-1988054973082523.oss-cn-hongkong.aliyuncs.com/web/releases/release-20260930-v1/";
const SCHEMA = "symbolicarena-pages-v1";
const SPLITS = ["train", "valid", "id_test", "ood_test"];
const SPLIT_LABELS = { train: "Train", valid: "Validation", id_test: "ID test", ood_test: "OOD test" };
const SPLIT_COLORS = { train: "#1c8f88", valid: "#db8656", id_test: "#4779b8", ood_test: "#865db3" };
const ALGORITHM_LABELS = {
  drsr: "DrSR", dso: "DSO", e2esr: "E2ESR", fepysr: "FePySR", gplearn: "gplearn",
  imcts: "iMCTS", jaxsr: "JAXSR", llmsr: "LLM-SR", pyoperon: "PyOperon",
  pysr: "PySR", qlattice: "QLattice", ragsr: "RAG-SR", symbolfit: "SymbolFit",
  tpsr: "TPSR", udsr: "uDSR",
};

const elements = {
  algorithm: document.querySelector("#algorithm-select"),
  dataset: document.querySelector("#dataset-select"),
  seed: document.querySelector("#seed-select"),
  status: document.querySelector("#selection-status"),
  progressWrap: document.querySelector("#load-progress-wrap"),
  progress: document.querySelector("#load-progress"),
  progressStage: document.querySelector("#load-stage"),
  progressPercent: document.querySelector("#load-percent"),
  datasetInfo: document.querySelector("#dataset-info"),
  chart: document.querySelector("#run-chart"),
  qualityChart: document.querySelector("#quality-chart"),
  qualityCurrent: document.querySelector("#quality-current"),
  paperChart: document.querySelector("#paper-chart"),
  paperChartTitle: document.querySelector("#paper-chart-title"),
  paperCondition: document.querySelector("#paper-condition"),
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
  paperResults: null,
  datasets: new Map(),
  runs: new Map(),
  dataset: null,
  run: null,
  quality: null,
  condition: "clean",
  minute: 180,
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

function showProgress(stage, loaded, total) {
  if (!Number.isFinite(total) || total <= 0 || loaded < 0 || loaded > total) {
    throw new Error("Invalid loading progress");
  }
  const percent = Math.round((100 * loaded) / total);
  elements.progressWrap.hidden = false;
  elements.progressStage.textContent = stage;
  elements.progress.value = percent;
  elements.progressPercent.textContent = `${percent}%`;
}

function reportError(error) {
  if (error.name === "AbortError") return;
  stopPlayback();
  elements.play.disabled = true;
  elements.progressWrap.hidden = true;
  showStatus(`Could not load data: ${error.message}`, true);
  console.error(error);
}

async function readJson(path, signal, onBytes, expectedBytes = null) {
  const response = await fetch(DATA_BASE + path, { signal });
  if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
  const total = Number(response.headers.get("Content-Length"));
  if (!Number.isInteger(total) || total <= 0 || !response.body) {
    throw new Error(`Missing transfer length for ${path}`);
  }
  if (expectedBytes !== null && total !== expectedBytes) {
    throw new Error(`Transfer length differs from the catalog for ${path}`);
  }
  let received = 0;
  const countedStream = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      onBytes?.(chunk.byteLength, total, received);
      controller.enqueue(chunk);
    },
  }), { signal });
  if (path.endsWith(".gz") && !window.DecompressionStream) {
    throw new Error("This browser cannot read compressed experiment data");
  }
  const stream = path.endsWith(".gz") ? countedStream.pipeThrough(new DecompressionStream("gzip"), { signal }) : countedStream;
  const value = await new Response(stream).json();
  if (received !== total) throw new Error(`Incomplete transfer for ${path}`);
  return value;
}

async function cachedJson(cache, path, signal, expectedBytes, onBytes) {
  if (cache.has(path)) {
    onBytes(expectedBytes);
    return cache.get(path);
  }
  const value = await readJson(path, signal, onBytes, expectedBytes);
  if (value.schema !== SCHEMA) throw new Error(`${path} has an incompatible data version`);
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
  if (!run) throw new Error(`Run not found: ${key}`);
  return run;
}

function stopPlayback() {
  state.playing = false;
  clearTimeout(state.playbackTimer);
  elements.play.textContent = "▶ Play";
}

async function loadSelection() {
  stopPlayback();
  elements.play.disabled = true;
  state.requestId += 1;
  const requestId = state.requestId;
  state.controller?.abort();
  state.controller = new AbortController();
  const entry = selectedRunEntry();
  const datasetEntry = state.catalog.datasets.find((item) => item.index === entry.dataset_index);
  if (!datasetEntry) throw new Error(`Dataset not found: ${entry.dataset_index}`);
  showStatus(`Loading ${entry.algorithm} / ${entry.dataset_id} / ${entry.condition} / ${entry.seed}…`);
  const totalBytes = datasetEntry.bytes + entry.bytes;
  let loadedBytes = 0;
  const onBytes = (bytes) => {
    if (requestId !== state.requestId) return;
    loadedBytes += bytes;
    showProgress(`Loading ${entry.dataset_id} and its trajectory`, loadedBytes, totalBytes);
  };
  showProgress(`Loading ${entry.dataset_id} and its trajectory`, 0, totalBytes);
  const [dataset, run] = await Promise.all([
    cachedJson(state.datasets, datasetEntry.path, state.controller.signal, datasetEntry.bytes, onBytes),
    cachedJson(state.runs, entry.path, state.controller.signal, entry.bytes, onBytes),
  ]);
  if (requestId !== state.requestId) return;
  if (
    run.identity.algorithm !== entry.algorithm
    || run.identity.dataset_index !== entry.dataset_index
    || run.identity.condition !== entry.condition
    || run.identity.seed !== entry.seed
  ) throw new Error("The run does not match the catalog");
  if (dataset.dataset_index !== entry.dataset_index || run.timeline.length !== 180) {
    throw new Error("The dataset or timeline has an invalid format");
  }
  state.dataset = dataset;
  state.run = run;
  state.quality = {
    id: run.timeline.map((frame) => qualityFromNmse(frame.metrics?.id_test?.nmse)),
    ood: run.timeline.map((frame) => qualityFromNmse(frame.metrics?.ood_test?.nmse)),
  };
  state.minute = 180;
  while (state.runs.size > 8) state.runs.delete(state.runs.keys().next().value);
  updateDatasetInfo();
  showStatus(`${entry.dataset_id} · ${entry.available_snapshots}/180 checkpoints · ${entry.candidate_count} candidates`);
  await Promise.all([renderMinute(), renderPaperChart()]);
  if (requestId === state.requestId) {
    elements.progressWrap.hidden = true;
    elements.play.disabled = false;
  }
}

function updateDatasetInfo() {
  const dataset = state.dataset;
  const parts = [
    `Dataset: ${dataset.name}`,
    `Features: ${dataset.feature_names.join(", ")}`,
    `Displayed samples: ${dataset.samples.split.length}`,
    `License: ${dataset.license}`,
  ];
  if (dataset.citation?.title) parts.push(`Data source: ${dataset.citation.title}`);
  if (dataset.plot.pca) {
    const variance = dataset.plot.pca.explained_variance_ratio.map((value) => `${(value * 100).toFixed(1)}%`);
    parts.push(`PCA explained variance: PC1 ${variance[0]}, PC2 ${variance[1]}`);
  }
  if (state.catalog.source_summary.formal_ready === false) {
    parts.push("The source archive has not been marked formally verified.");
  }
  elements.datasetInfo.replaceChildren(...parts.map((content) => {
    const line = document.createElement("span");
    line.textContent = content;
    return line;
  }));
  elements.chartTitle.textContent = `Fit on ${state.run.identity.dataset_id}`;
  elements.dimension.textContent = {
    line: "1D curve",
    surface: "2D surface",
    projection: "PCA projection",
  }[dataset.plot.kind];
  elements.runSummary.textContent = `Final status: ${state.run.terminal.status} · Candidates: ${state.run.candidates.length} · Noise coefficient: ${state.run.noise.sigma}`;
}

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

function qualityFromNmse(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return null;
  const logError = Math.min(2, Math.max(-12, Math.log10(Math.max(value, 1e-12))));
  return ((2 - logError) / 14) * 100;
}

function qualityLabel(value) {
  return value === null ? "—" : value.toFixed(1);
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
      name: `${SPLIT_LABELS[split]} samples`,
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
    throw new Error("The prediction count does not match the displayed samples");
  }
  if (kind === "line") {
    traces.unshift({
      type: "scatter", mode: "lines", name: "Candidate curve",
      x: dataset.plot.grid.x, y: predictions.grid_y,
      line: { color: "#123f3b", width: 3 },
    });
  } else {
    if (kind === "surface") {
      const grid = dataset.plot.grid;
      const side = grid.x.length;
      if (predictions.grid_y.length !== side * grid.y.length) throw new Error("The surface grid has an invalid size");
      const surface = [];
      for (let row = 0; row < grid.y.length; row += 1) {
        surface.push(predictions.grid_y.slice(row * side, (row + 1) * side));
      }
      traces.unshift({
        type: "surface", name: "Candidate surface", x: grid.x, y: grid.y, z: surface,
        colorscale: [[0, "#cee8db"], [1, "#168981"]], opacity: 0.68, showscale: false,
      });
    }
    traces.push({
      type: "scatter3d", mode: "markers", name: "Candidate predictions",
      x: dataset.plot.coordinates.map((row) => row[0]),
      y: dataset.plot.coordinates.map((row) => row[1]),
      z: predictions.sample_y,
      marker: { color: "#173c39", size: 2.5, symbol: "diamond", opacity: 0.65 },
    });
  }
  return traces;
}

function qualityTraces(minute) {
  const minutes = Array.from({ length: 180 }, (_, index) => index + 1);
  const series = [
    { label: "ID", values: state.quality.id, color: "#1c8f88" },
    { label: "OOD", values: state.quality.ood, color: "#db8656" },
  ];
  return series.flatMap(({ label, values, color }) => [
    {
      type: "scatter", mode: "lines", name: label,
      x: minutes, y: values, line: { color, width: 2.5 },
      hovertemplate: `Minute %{x}<br>${label}: %{y:.1f}<extra></extra>`,
    },
    {
      type: "scatter", mode: "markers", name: `${label} at minute ${minute}`,
      x: [minute], y: [values[minute - 1]], showlegend: false,
      marker: { color, size: 12, line: { color: "#ffffff", width: 2 } },
      hovertemplate: `Minute %{x}<br>${label}: %{y:.1f}<extra></extra>`,
    },
  ]);
}

function qualityLayout(minute) {
  return {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: "#31524b", size: 11 },
    margin: { l: 52, r: 18, t: 12, b: 52 },
    xaxis: { title: "Training minute", range: [1, 180], dtick: 30, gridcolor: "#e2eae3" },
    yaxis: { title: "Score", range: [0, 105], gridcolor: "#e2eae3" },
    legend: { orientation: "h", x: 0, y: 1.16 },
    shapes: [{ type: "line", xref: "x", yref: "paper", x0: minute, x1: minute, y0: 0, y1: 1, line: { color: "#173c39", width: 1, dash: "dash" } }],
    uirevision: selectedKey(state.run.identity.algorithm, state.run.identity.dataset_index, state.run.identity.condition, state.run.identity.seed),
  };
}

async function renderPaperChart() {
  const algorithm = state.paperResults.algorithms.find((item) => item.id === elements.algorithm.value);
  if (!algorithm) throw new Error(`The paper has no profile for ${elements.algorithm.value}`);
  const values = algorithm[state.condition];
  if (!values || values.length !== 6) throw new Error("The paper profile has an invalid format");
  const conditionLabel = { clean: "Clean", noise001: "1% noise", noise005: "5% noise" }[state.condition];
  elements.paperChartTitle.textContent = `${algorithm.name}: published six-axis profile`;
  elements.paperCondition.textContent = conditionLabel;
  await Plotly.react(elements.paperChart, [{
    type: "bar", x: state.paperResults.axes, y: values,
    marker: { color: ["#1c8f88", "#4779b8", "#865db3", "#aa739c", "#db8656", "#b89b5b"] },
    text: values.map((value) => value.toFixed(2)), textposition: "outside",
    hovertemplate: "%{x}: %{y:.2f}<extra></extra>",
  }], {
    autosize: true,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: "#31524b", size: 11 },
    margin: { l: 45, r: 14, t: 20, b: 48 },
    yaxis: { title: "Published score", range: [0, 108], gridcolor: "#e2eae3" },
    xaxis: { title: "Evaluation axis" },
  }, { responsive: true, displaylogo: false });
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
      const minute = state.minute;
      const frame = run.timeline[minute - 1];
      const candidate = frame.candidate === null ? null : run.candidates[frame.candidate];
      await Promise.all([
        Plotly.react(
          elements.chart,
          plotTraces(dataset, run, candidate),
          plotLayout(dataset, dataset.plot.kind !== "line"),
          { responsive: true, displaylogo: false, scrollZoom: false },
        ),
        Plotly.react(
          elements.qualityChart,
          qualityTraces(minute),
          qualityLayout(minute),
          { responsive: true, displaylogo: false, scrollZoom: false },
        ),
      ]);
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
  elements.minuteValue.textContent = `Minute ${state.minute} / 180`;
  elements.minuteStatus.textContent = frame.status === "missing" ? "No checkpoint recorded" : `Checkpoint: ${frame.status}`;
  elements.equation.textContent = candidate?.equation || "No equation available at this minute";
  elements.qualityCurrent.textContent = `Minute ${state.minute} · ID ${qualityLabel(state.quality.id[state.minute - 1])} · OOD ${qualityLabel(state.quality.ood[state.minute - 1])}`;
  for (const split of SPLITS) {
    elements.metrics[split].textContent = formatMetric(frame.metrics?.[split]?.r2);
  }
  const plot = candidate?.plot;
  elements.plotNote.textContent = !candidate
    ? "No candidate is available at this minute."
    : plot.status === "ok" ? "Predictions were precomputed; training samples show the labels used in this run."
      : plot.status === "partial" ? "Some predictions are non-finite and appear as gaps."
        : `The current candidate cannot be plotted: ${plot.reason}`;
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
  if (state.minute >= 180) state.minute = 0;
  state.playing = true;
  elements.play.textContent = "Ⅱ Pause";
  playbackStep().catch(reportError);
});

async function initialize() {
  if (!window.Plotly) throw new Error("The chart library did not load");
  showProgress("Loading experiment catalog", 0, 1);
  const [catalog, paperResults] = await Promise.all([
    readJson("catalog.json", undefined, (_bytes, total, received) => showProgress("Loading experiment catalog", received, total)),
    fetch("./paper-results.json").then((response) => {
      if (!response.ok) throw new Error(`Paper results returned HTTP ${response.status}`);
      return response.json();
    }),
  ]);
  if (catalog.schema !== SCHEMA || catalog.runs.length !== 6750 || catalog.datasets.length !== 50) {
    throw new Error("The experiment catalog has an invalid size or version");
  }
  if (paperResults.schema !== "symbolicarena-paper-v1" || paperResults.algorithms.length !== 15 || paperResults.axes.length !== 6) {
    throw new Error("The paper results have an invalid size or version");
  }
  const paperIds = new Set(paperResults.algorithms.map((algorithm) => algorithm.id));
  if (paperIds.size !== catalog.algorithms.length || catalog.algorithms.some((id) => !paperIds.has(id))) {
    throw new Error("Paper methods do not match the experiment catalog");
  }
  for (const algorithm of paperResults.algorithms) {
    for (const condition of ["clean", "noise001", "noise005"]) {
      if (algorithm[condition].length !== 6 || algorithm[condition].some((value) => !Number.isFinite(value) || value < 0 || value > 100)) {
        throw new Error(`Invalid paper results for ${algorithm.id} / ${condition}`);
      }
    }
  }
  state.catalog = catalog;
  state.paperResults = paperResults;
  setOptions(elements.algorithm, catalog.algorithms.map((id) => ({ value: id, label: ALGORITHM_LABELS[id] || id })));
  setDatasetOptions();
  setSeedOptions();
  for (const button of document.querySelectorAll(".noise-choice")) button.disabled = false;
  await loadSelection();
}

initialize().catch(reportError);
