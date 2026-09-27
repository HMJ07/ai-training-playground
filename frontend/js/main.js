import { renderTick } from "./scene.js";

const taskInput = document.getElementById("task-input");
const generateBtn = document.getElementById("generate-btn");
const generateStatus = document.getElementById("generate-status");
const taskResult = document.getElementById("task-result");
const taskNameEl = document.getElementById("task-name");
const taskDescriptionEl = document.getElementById("task-description");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");

let currentTaskId = null;

const mStep = document.getElementById("m-step");
const mEpisode = document.getElementById("m-episode");
const mLast = document.getElementById("m-last");
const mMean = document.getElementById("m-mean");

const chartCanvas = document.getElementById("reward-chart");
const chartCtx = chartCanvas.getContext("2d");
const returnHistory = [];

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
    taskNameEl.textContent = data.task.name;
    taskDescriptionEl.textContent = data.task.description;
    taskResult.classList.remove("hidden");
    generateStatus.textContent = "Tarea lista.";
    startBtn.disabled = false;
  } catch (err) {
    generateStatus.textContent = "No se pudo generar la tarea: " + err;
  } finally {
    generateBtn.disabled = false;
  }
}

function connectSocket() {
  if (socket) return;
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "tick") {
      renderTick(msg.render);
      updateMetrics(msg.metrics);
    }
  };
  socket.onclose = () => {
    socket = null;
  };
}

function updateMetrics(metrics) {
  mStep.textContent = metrics.step;
  mEpisode.textContent = metrics.episode;
  mLast.textContent = metrics.lastReturn.toFixed(2);
  mMean.textContent = metrics.meanReturn100.toFixed(2);

  if (returnHistory.length === 0 || returnHistory[returnHistory.length - 1] !== metrics.meanReturn100) {
    returnHistory.push(metrics.meanReturn100);
    if (returnHistory.length > 200) returnHistory.shift();
    drawChart();
  }
}

function drawChart() {
  const { width, height } = chartCanvas;
  chartCtx.clearRect(0, 0, width, height);
  chartCtx.strokeStyle = "#4f8cff";
  chartCtx.lineWidth = 2;
  if (returnHistory.length < 2) return;

  const min = Math.min(...returnHistory);
  const max = Math.max(...returnHistory);
  const range = max - min || 1;

  chartCtx.beginPath();
  returnHistory.forEach((val, i) => {
    const x = (i / (returnHistory.length - 1)) * width;
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
  await fetch(`/api/train/start/${currentTaskId}`, { method: "POST" });
  startBtn.disabled = true;
  stopBtn.disabled = false;
});

stopBtn.addEventListener("click", async () => {
  await fetch("/api/train/stop", { method: "POST" });
  startBtn.disabled = false;
  stopBtn.disabled = true;
});
