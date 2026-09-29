"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const PLOT = { left: 48, top: 20, width: 686, height: 312 };

const PRESETS = {
  quadratic: { seedOffset: 11, value: (x) => 0.55 * x * x - 0.45 * x + 0.8 },
  sine: { seedOffset: 29, value: (x) => 1.4 * Math.sin(x) + 0.2 * x - 0.2 },
  rational: { seedOffset: 47, value: (x) => 2.2 / (1 + x * x) + 0.12 * x - 0.7 },
};

const MODELS = [
  {
    id: "linear",
    name: "线性",
    basis: [(x) => x, () => 1],
    terms: ["x", ""],
  },
  {
    id: "quadratic",
    name: "二次多项式",
    basis: [(x) => x * x, (x) => x, () => 1],
    terms: ["x²", "x", ""],
  },
  {
    id: "cubic",
    name: "三次多项式",
    basis: [(x) => x * x * x, (x) => x * x, (x) => x, () => 1],
    terms: ["x³", "x²", "x", ""],
  },
  {
    id: "sine",
    name: "正弦组合",
    basis: [(x) => Math.sin(x), (x) => x, () => 1],
    terms: ["sin(x)", "x", ""],
  },
  {
    id: "rational",
    name: "有理组合",
    basis: [(x) => 1 / (1 + x * x), (x) => x, () => 1],
    terms: ["/(1+x²)", "x", ""],
  },
];

const state = {
  preset: "quadratic",
  noise: 0.1,
  seed: 1,
  samples: [],
  candidates: [],
  selectedId: null,
};

const chart = document.querySelector("#fit-chart");
const ranking = document.querySelector("#candidate-list");
const noiseInput = document.querySelector("#noise");

function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => {
    value = (Math.imul(1664525, value) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function standardNormal(random) {
  const u = Math.max(random(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

function generateSamples() {
  const preset = PRESETS[state.preset];
  const random = randomGenerator(20260929 + preset.seedOffset + state.seed * 101);

  return Array.from({ length: 40 }, (_, index) => {
    const x = -3 + (6 * index) / 39;
    return {
      x,
      y: preset.value(x) + state.noise * standardNormal(random),
      split: index % 4 === 3 ? "valid" : "train",
    };
  });
}

// 候选结构固定，仅对每个结构的系数做最小二乘拟合。
function fitCoefficients(samples, basis) {
  const size = basis.length;
  const matrix = Array.from({ length: size }, () => Array(size).fill(0));
  const result = Array(size).fill(0);

  for (const sample of samples) {
    const values = basis.map((term) => term(sample.x));
    for (let row = 0; row < size; row += 1) {
      result[row] += values[row] * sample.y;
      for (let column = 0; column < size; column += 1) {
        matrix[row][column] += values[row] * values[column];
      }
    }
  }

  // 微小正则项与主元交换让浏览器端示例在退化数据上也能稳定返回。
  for (let index = 0; index < size; index += 1) matrix[index][index] += 1e-9;
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    if (Math.abs(matrix[pivot][column]) < 1e-12) return null;
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    [result[column], result[pivot]] = [result[pivot], result[column]];

    const divisor = matrix[column][column];
    for (let index = column; index < size; index += 1) matrix[column][index] /= divisor;
    result[column] /= divisor;

    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = matrix[row][column];
      for (let index = column; index < size; index += 1) {
        matrix[row][index] -= factor * matrix[column][index];
      }
      result[row] -= factor * result[column];
    }
  }
  return result;
}

function predict(candidate, x) {
  return candidate.coefficients.reduce(
    (sum, coefficient, index) => sum + coefficient * candidate.basis[index](x),
    0,
  );
}

function rmse(candidate, samples) {
  const error = samples.reduce((sum, sample) => {
    const residual = predict(candidate, sample.x) - sample.y;
    return sum + residual * residual;
  }, 0);
  return Math.sqrt(error / samples.length);
}

function formatEquation(candidate) {
  return candidate.coefficients.map((coefficient, index) => {
    const sign = index === 0 ? (coefficient < 0 ? "−" : "") : (coefficient < 0 ? " − " : " + ");
    const magnitude = Math.abs(coefficient).toFixed(2);
    return `${sign}${magnitude}${candidate.terms[index]}`;
  }).join("");
}

function buildCandidates() {
  const training = state.samples.filter((sample) => sample.split === "train");
  const validation = state.samples.filter((sample) => sample.split === "valid");

  return MODELS.map((model) => {
    const coefficients = fitCoefficients(training, model.basis);
    if (!coefficients) return null;
    const candidate = { ...model, coefficients };
    candidate.trainRmse = rmse(candidate, training);
    candidate.validRmse = rmse(candidate, validation);
    candidate.score = candidate.validRmse + 0.02 * coefficients.length;
    candidate.equation = formatEquation(candidate);
    return candidate;
  }).filter(Boolean).sort((left, right) => left.score - right.score);
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  return element;
}

function drawChart(candidate) {
  chart.replaceChildren();
  const title = svgElement("title");
  title.textContent = `${candidate.name}：${candidate.equation}；训练与验证样本及拟合曲线`;
  chart.append(title);

  const curveSamples = Array.from({ length: 141 }, (_, index) => {
    const x = -3 + (6 * index) / 140;
    return { x, y: predict(candidate, x) };
  });
  const values = [...state.samples.map((sample) => sample.y), ...curveSamples.map((sample) => sample.y)];
  const bottom = Math.min(...values);
  const top = Math.max(...values);
  const padding = Math.max((top - bottom) * 0.14, 0.3);
  const minimum = bottom - padding;
  const maximum = top + padding;
  const mapX = (x) => PLOT.left + ((x + 3) / 6) * PLOT.width;
  const mapY = (y) => PLOT.top + ((maximum - y) / (maximum - minimum)) * PLOT.height;

  for (let step = 0; step <= 4; step += 1) {
    const y = PLOT.top + (PLOT.height * step) / 4;
    const value = maximum - ((maximum - minimum) * step) / 4;
    chart.append(svgElement("line", { x1: PLOT.left, y1: y, x2: PLOT.left + PLOT.width, y2: y, class: "chart-grid" }));
    const label = svgElement("text", { x: PLOT.left - 10, y: y + 4, "text-anchor": "end", class: "chart-label" });
    label.textContent = value.toFixed(1);
    chart.append(label);
  }
  for (const x of [-3, -2, -1, 0, 1, 2, 3]) {
    chart.append(svgElement("line", {
      x1: mapX(x), y1: PLOT.top, x2: mapX(x), y2: PLOT.top + PLOT.height,
      class: x === 0 ? "chart-zero" : "chart-grid",
    }));
    if (x % 3 === 0) {
      const label = svgElement("text", {
        x: mapX(x), y: PLOT.top + PLOT.height + 21,
        "text-anchor": "middle", class: "chart-label",
      });
      label.textContent = String(x);
      chart.append(label);
    }
  }

  const path = curveSamples.map((point, index) => `${index === 0 ? "M" : "L"}${mapX(point.x).toFixed(2)} ${mapY(point.y).toFixed(2)}`).join(" ");
  chart.append(svgElement("path", { d: path, class: "chart-curve" }));
  for (const sample of state.samples) {
    chart.append(svgElement("circle", {
      cx: mapX(sample.x), cy: mapY(sample.y), r: sample.split === "train" ? 5 : 6,
      class: sample.split === "train" ? "chart-train" : "chart-valid",
    }));
  }
}

function renderSelected() {
  const candidate = state.candidates.find((item) => item.id === state.selectedId);
  if (!candidate) return;
  document.querySelector("#selected-model-name").textContent = `${candidate.name}${candidate === state.candidates[0] ? " · 当前推荐" : " · 手动查看"}`;
  document.querySelector("#selected-equation").textContent = candidate.equation;
  document.querySelector("#train-rmse").textContent = candidate.trainRmse.toFixed(3);
  document.querySelector("#valid-rmse").textContent = candidate.validRmse.toFixed(3);
  for (const row of ranking.querySelectorAll(".candidate-row")) {
    row.setAttribute("aria-pressed", String(row.dataset.model === state.selectedId));
  }
  drawChart(candidate);
}

function renderRanking() {
  ranking.replaceChildren();
  const columns = document.createElement("div");
  columns.className = "candidate-column-head";
  for (const label of ["排名", "结构", "拟合后的公式", "验证 RMSE", "选择分数"]) {
    const cell = document.createElement("span");
    cell.textContent = label;
    columns.append(cell);
  }
  ranking.append(columns);

  state.candidates.forEach((candidate, index) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "candidate-row";
    row.dataset.model = candidate.id;
    row.setAttribute("aria-pressed", String(candidate.id === state.selectedId));
    row.setAttribute("aria-label", `${candidate.name}，验证 RMSE ${candidate.validRmse.toFixed(3)}，选择分数 ${candidate.score.toFixed(3)}，查看曲线`);

    const rank = document.createElement("span");
    rank.className = "candidate-rank";
    rank.textContent = String(index + 1).padStart(2, "0");
    const name = document.createElement("span");
    name.className = "candidate-name";
    name.textContent = candidate.name;
    if (index === 0) {
      const badge = document.createElement("em");
      badge.textContent = "推荐";
      name.append(badge);
    }
    const equation = document.createElement("span");
    equation.className = "candidate-equation";
    equation.textContent = candidate.equation;
    const error = document.createElement("span");
    error.className = "candidate-value";
    error.textContent = candidate.validRmse.toFixed(3);
    const score = document.createElement("span");
    score.className = "candidate-score";
    score.textContent = candidate.score.toFixed(3);
    row.append(rank, name, equation, error, score);
    row.addEventListener("click", () => {
      state.selectedId = candidate.id;
      renderSelected();
    });
    ranking.append(row);
  });
}

function renderAll() {
  state.samples = generateSamples();
  state.candidates = buildCandidates();
  state.selectedId = state.candidates[0]?.id ?? null;
  document.querySelector("#noise-value").textContent = state.noise.toFixed(2);
  for (const button of document.querySelectorAll(".dataset-button")) {
    button.setAttribute("aria-pressed", String(button.dataset.preset === state.preset));
  }
  renderRanking();
  renderSelected();
}

for (const button of document.querySelectorAll(".dataset-button")) {
  button.addEventListener("click", () => {
    state.preset = button.dataset.preset;
    renderAll();
  });
}
noiseInput.addEventListener("input", () => {
  state.noise = Number(noiseInput.value);
  renderAll();
});
document.querySelector("#resample").addEventListener("click", () => {
  state.seed += 1;
  renderAll();
});

renderAll();
