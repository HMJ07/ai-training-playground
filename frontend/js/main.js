import { renderTick } from "./scene.js";

const taskInput = document.getElementById("task-input");
const generateBtn = document.getElementById("generate-btn");
const generateStatus = document.getElementById("generate-status");
const taskResult = document.getElementById("task-result");
const taskNameEl = document.getElementById("task-name");
const taskKindBadge = document.getElementById("task-kind-badge");
const taskDescriptionEl = document.getElementById("task-description");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const viewerHint = document.getElementById("viewer-hint");
const legendEl = document.getElementById("legend");

const classificationPanel = document.getElementById("classification-panel");
const exampleText = document.getElementById("example-text");
const exampleLabel = document.getElementById("example-label");
const addExampleBtn = document.getElementById("add-example-btn");
const examplesStatus = document.getElementById("examples-status");
const predictText = document.getElementById("predict-text");
const predictBtn = document.getElementById("predict-btn");
const predictResult = document.getElementById("predict-result");

const mLabels = [1, 2, 3, 4].map((i) => document.getElementById(`m-label-${i}`));
const mValues = [1, 2, 3, 4].map((i) => document.getElementById(`m-${i}`));

const chartCanvas = document.getElementById("reward-chart");
const chartCtx = chartCanvas.getContext("2d");
const chartHistory = [];

let currentTaskId = null;
let currentTaskKind = null;
let socket = null;

async function generateTask() {
  const description = taskInput.value.trim();
  if (!description) {
    generateStatus.textContent = "Escribe primero qué quieres que aprenda.";
    return;
  }

  generateBtn.disabled = true;
  startBtn.disabled = true;
  taskResult.classList.add("hidden");
  classificationPanel.classList.add("hidden");
  chartHistory.length = 0;
  generateStatus.textContent = "Generando tarea con la IA...";

  try {
    const res = await fetch("/api/tasks/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description }),
    });
    const data = await res.json();

    if (data.error) {
      generateStatus.textContent = data.error;
      return;
    }

    currentTaskId = data.task.id;
    currentTaskKind = data.task.kind;
    taskNameEl.textContent = data.task.name;
    taskDescriptionEl.textContent = data.task.description;
    taskResult.classList.remove("hidden");
    generateStatus.textContent = "Tarea lista.";
    startBtn.disabled = false;

    setupForKind(data.task);
  } catch (err) {
    generateStatus.textContent = "No se pudo generar la tarea: " + err;
  } finally {
    generateBtn.disabled = false;
  }
}

function setupForKind(task) {
  if (task.kind === "classification") {
    taskKindBadge.textContent = "Clasificación de texto";
    viewerHint.textContent =
      "Cada punto es un ejemplo; el color es su clase. Al entrenar, verás cómo se agrupan por clase en el espacio.";
    classificationPanel.classList.remove("hidden");
    exampleLabel.innerHTML = task.labels.map((l) => `<option value="${l}">${l}</option>`).join("");
    examplesStatus.textContent = `${task.numSeedExamples} ejemplos semilla generados por la IA.`;
    predictResult.classList.add("hidden");
    setMetricLabels(["Época", "Precisión (val.)", "Nº ejemplos", ""]);
  } else {
    taskKindBadge.textContent = "Tarea de control físico";
    viewerHint.textContent = "";
    classificationPanel.classList.add("hidden");
    setMetricLabels(["Paso", "Episodio", "Última recompensa", "Media (100 ep.)"]);
    legendEl.classList.add("hidden");
  }
}

function updateLegend(legend) {
  if (!legend || legend.length === 0) {
    legendEl.classList.add("hidden");
    return;
  }
  legendEl.innerHTML = legend
    .map(
      (item) =>
        `<div class="legend-item"><span class="legend-swatch" style="background:${item.color}"></span>${item.label}</div>`
    )
    .join("");
  legendEl.classList.remove("hidden");
}

function setMetricLabels(labels) {
  labels.forEach((label, i) => {
    mLabels[i].textContent = label;
    mValues[i].textContent = "0";
  });
}

function connectSocket() {
  if (socket) return;
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "tick") {
      renderTick(msg.render);
      updateMetrics(msg.metrics);
      if (msg.render.legend) updateLegend(msg.render.legend);
    }
  };
  socket.onclose = () => {
    socket = null;
  };
}

function updateMetrics(metrics) {
  if (metrics.kind === "classification") {
    mValues[0].textContent = `${metrics.epoch}/${metrics.totalEpochs}`;
    mValues[1].textContent = (metrics.accuracy * 100).toFixed(1) + "%";
    mValues[2].textContent = metrics.numExamples;
    mValues[3].textContent = "";
    pushChartValue(metrics.accuracy);
  } else {
    mValues[0].textContent = metrics.step;
    mValues[1].textContent = metrics.episode;
    mValues[2].textContent = metrics.lastReturn.toFixed(2);
    mValues[3].textContent = metrics.meanReturn100.toFixed(2);
    pushChartValue(metrics.meanReturn100);
  }
}

function pushChartValue(value) {
  if (chartHistory.length === 0 || chartHistory[chartHistory.length - 1] !== value) {
    chartHistory.push(value);
    if (chartHistory.length > 200) chartHistory.shift();
    drawChart();
  }
}

function drawChart() {
  const { width, height } = chartCanvas;
  chartCtx.clearRect(0, 0, width, height);
  chartCtx.strokeStyle = "#4f8cff";
  chartCtx.lineWidth = 2;
  if (chartHistory.length < 2) return;

  const min = Math.min(...chartHistory);
  const max = Math.max(...chartHistory);
  const range = max - min || 1;

  chartCtx.beginPath();
  chartHistory.forEach((val, i) => {
    const x = (i / (chartHistory.length - 1)) * width;
    const y = height - ((val - min) / range) * height;
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  });
  chartCtx.stroke();
}

generateBtn.addEventListener("click", generateTask);

startBtn.addEventListener("click", async () => {
  if (!currentTaskId) return;
  connectSocket();
  const res = await fetch(`/api/train/start/${currentTaskId}`, { method: "POST" });
  const data = await res.json();
  if (data.error) {
    generateStatus.textContent = data.error;
    return;
  }
  startBtn.disabled = true;
  stopBtn.disabled = false;
});

stopBtn.addEventListener("click", async () => {
  await fetch("/api/train/stop", { method: "POST" });
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

addExampleBtn.addEventListener("click", async () => {
  const text = exampleText.value.trim();
  if (!text || !currentTaskId) return;
  const res = await fetch(`/api/tasks/${currentTaskId}/examples`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, label: exampleLabel.value }),
  });
  const data = await res.json();
  if (data.error) {
    examplesStatus.textContent = data.error;
    return;
  }
  examplesStatus.textContent = `${data.numExamples} ejemplos en total. Vuelve a entrenar para incluirlo.`;
  exampleText.value = "";
});

predictBtn.addEventListener("click", async () => {
  const text = predictText.value.trim();
  if (!text || !currentTaskId) return;
  const res = await fetch(`/api/tasks/${currentTaskId}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  predictResult.classList.remove("hidden");
  if (data.error) {
    predictResult.textContent = data.error;
    return;
  }
  const probs = Object.entries(data.probabilities || {})
    .map(([label, p]) => `${label}: ${(p * 100).toFixed(1)}%`)
    .join(" · ");
  predictResult.innerHTML = `<strong>${data.label}</strong><p>${probs}</p>`;
});
