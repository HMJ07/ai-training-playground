import { renderTick } from "./scene.js";

const taskSelect = document.getElementById("task-select");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");

const mStep = document.getElementById("m-step");
const mEpisode = document.getElementById("m-episode");
const mLast = document.getElementById("m-last");
const mMean = document.getElementById("m-mean");

const chartCanvas = document.getElementById("reward-chart");
const chartCtx = chartCanvas.getContext("2d");
const returnHistory = [];

let socket = null;

async function loadTasks() {
  const res = await fetch("/api/tasks");
  const tasks = await res.json();
  taskSelect.innerHTML = "";
  for (const task of tasks) {
    const opt = document.createElement("option");
    opt.value = task.id;
    opt.textContent = `${task.name} - ${task.description}`;
    taskSelect.appendChild(opt);
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

startBtn.addEventListener("click", async () => {
  connectSocket();
  const taskId = taskSelect.value;
  await fetch(`/api/train/start/${taskId}`, { method: "POST" });
  startBtn.disabled = true;
  stopBtn.disabled = false;
});

stopBtn.addEventListener("click", async () => {
  await fetch("/api/train/stop", { method: "POST" });
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

loadTasks();
