const MODEL_URLS = [
  "https://teachablemachine.withgoogle.com/models/tTYdfh6E2/",
  "/static/my_model/",
];
const WEBCAM_WIDTH = 960;
const WEBCAM_HEIGHT = 720;
const GUESS_LEVEL_COUNT = 12;

const state = {
  photos: [],
  model: null,
  webcam: null,
  labelCount: 0,
  initialized: false,
  loopStarted: false,
  modelAvailable: false,
  live: { running: false, target: null },
  guess: { running: false, levelIndex: 0, target: null, queue: [], results: [] },
};

const liveCanvas = document.getElementById("liveCanvas");
const guessCanvas = document.getElementById("guessCanvas");

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Request failed");
  }
  return response.json();
}

function normalizeName(value) {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase("tr-TR")
    .replace(/\.[^/.]+$/, "")
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_À-ɏḀ-ỿЀ-ӿ֐-׿؀-ۿऀ-ॿğüşöçıİi]/gi, "")
    .trim();
}

function photoStem(photo) {
  return normalizeName(photo.filename || photo.image_path.split("/").pop() || "");
}

async function loadPhotos() {
  state.photos = await api("/api/photo-library");
  if (state.photos.length && !state.guess.target) {
    setGuessTarget(state.photos[0]);
  }
  resetGuessProgress();
}

function shuffledPhotos(photos) {
  const items = photos.slice();
  for (let index = items.length - 1; index > 0; index -= 1) {
    const randomIndex = Math.floor(Math.random() * (index + 1));
    const current = items[index];
    items[index] = items[randomIndex];
    items[randomIndex] = current;
  }
  return items;
}

function guessLevelLimit() {
  return Math.min(GUESS_LEVEL_COUNT, state.photos.length);
}

function buildGuessQueue() {
  state.guess.queue = shuffledPhotos(state.photos).slice(0, guessLevelLimit());
}

function guessProgressMarkup(status) {
  return '<span class="guess-progress-dot' + (status ? " is-" + status : "") + '"></span>';
}

function renderGuessProgress() {
  const container = document.getElementById("guessProgress");
  if (!container) return;
  const total = guessLevelLimit();
  let markup = "";
  for (let index = 0; index < total; index += 1) {
    markup += guessProgressMarkup(state.guess.results[index] || "");
  }
  container.innerHTML = markup;
}

function resetGuessProgress() {
  state.guess.results = [];
  renderGuessProgress();
}

// ── Live mode target ──────────────────────────────────────────────────────────
function setLiveTarget(photo) {
  state.live.target = photo;

  const img = document.getElementById("liveTargetImage");
  img.src = photo.image_path;
  img.style.display = "block";

  document.getElementById("livePlaceholder").style.display = "none";

  const labelBar = document.getElementById("liveLabelBar");
  labelBar.style.display = "flex";
  document.getElementById("liveMemeLabel").textContent = photoStem(photo).replace(/_/g, " ");
  document.getElementById("liveTargetTitle").textContent = photoStem(photo).replace(/_/g, " ");
}

// ── Guess mode target ─────────────────────────────────────────────────────────
function setGuessTarget(photo) {
  state.guess.target = photo;

  const img = document.getElementById("guessTargetImage");
  img.src = photo.image_path;
  img.style.display = "block";

  document.getElementById("guessPlaceholder").style.display = "none";

  const labelBar = document.getElementById("guessLabelBar");
  labelBar.style.display = "flex";
  document.getElementById("guessMemeLabel").textContent = photoStem(photo).replace(/_/g, " ");
  document.getElementById("guessLevelTitle").textContent = "Niveau " + (state.guess.levelIndex + 1);
}

// ── Badges ────────────────────────────────────────────────────────────────────
function showSuccess(mode, visible) {
  document.getElementById(mode + "SuccessBadge").classList.toggle("hidden", !visible);
}

// ── Hints ─────────────────────────────────────────────────────────────────────
function renderHints(mode, matched, probability, targetName) {
  const hintsId = mode === "live" ? "liveHints" : "guessHints";
  if (mode === "guess") return;
  const name = (targetName || "—").replace(/_/g, " ");
  const message = matched
    ? name + " — tiens la pose !"
    : "Plus proche : " + name + " (" + (probability * 100).toFixed(0) + "%)";

  document.getElementById(hintsId).innerHTML = '<span class="hint-chip">' + message + "</span>";
}

// ── Status bar ────────────────────────────────────────────────────────────────
function setStatus(live, text) {
  const user = document.querySelector(".topbar-user");
  if (!user) return;
  user.textContent = live ? "Inan" : text;
}

// ── Model init ────────────────────────────────────────────────────────────────
async function initModel() {
  if (state.initialized) return;
  let lastError = null;
  for (const baseUrl of MODEL_URLS) {
    try {
      const modelURL = baseUrl + "model.json";
      const metadataURL = baseUrl + "metadata.json";
      state.model = await tmPose.load(modelURL, metadataURL);
      state.labelCount = state.model.getTotalClasses();
      state.webcam = new tmPose.Webcam(WEBCAM_WIDTH, WEBCAM_HEIGHT, true);
      await state.webcam.setup();
      await state.webcam.play();
      state.initialized = true;
      state.modelAvailable = true;
      setStatus(true, "Caméra active");
      return;
    } catch (error) {
      lastError = error;
    }
  }
  state.modelAvailable = false;
  setStatus(false, "Modèle introuvable");
  throw lastError;
}

async function ensureLoop() {
  if (state.loopStarted) return;
  await initModel();
  state.loopStarted = true;
  window.requestAnimationFrame(loop);
}

async function loop() {
  if (state.webcam) {
    state.webcam.update();
    await predict();
  }
  window.requestAnimationFrame(loop);
}

// ── Draw ──────────────────────────────────────────────────────────────────────
function drawPoseToCanvas(canvas, pose) {
  const ctx = canvas.getContext("2d");
  canvas.width = WEBCAM_WIDTH;
  canvas.height = WEBCAM_HEIGHT;
  ctx.drawImage(state.webcam.canvas, 0, 0, canvas.width, canvas.height);
}

// ── Prediction helpers ────────────────────────────────────────────────────────
function predictionForTarget(predictions, targetPhoto) {
  if (!targetPhoto) return null;
  const target = photoStem(targetPhoto);
  return predictions.find((item) => normalizeName(item.className) === target) || null;
}

function photoForPrediction(prediction) {
  if (!prediction) return null;
  const predictedStem = normalizeName(prediction.className);
  return state.photos.find((item) => photoStem(item) === predictedStem) || null;
}

// ── Predict ───────────────────────────────────────────────────────────────────
async function predict() {
  if (!state.model || !state.webcam) return;
  const { pose, posenetOutput } = await state.model.estimatePose(state.webcam.canvas);
  const predictions = await state.model.predict(posenetOutput);
  predictions.sort((a, b) => b.probability - a.probability);
  const topPrediction = predictions[0] || null;
  const topPhoto = photoForPrediction(topPrediction);

  drawPoseToCanvas(liveCanvas, pose);
  drawPoseToCanvas(guessCanvas, pose);
  if (topPhoto) setLiveTarget(topPhoto);

  handleLiveMode(predictions);
  handleGuessMode(predictions);
}

// ── Live mode ─────────────────────────────────────────────────────────────────
function handleLiveMode(predictions) {
  if (!state.live.running) return;
  const match = predictions[0] || null;
  const probability = match ? match.probability : 0;
  const topPhoto = photoForPrediction(match);
  if (topPhoto) setLiveTarget(topPhoto);

  renderHints("live", Boolean(match), probability, match ? match.className : "—");
  showSuccess("live", Boolean(match));
}

// ── Guess mode ────────────────────────────────────────────────────────────────
function handleGuessMode(predictions) {
  if (!state.guess.running || !state.guess.target) return;
  const topPrediction = predictions[0] || null;
  const matched =
    Boolean(topPrediction) &&
    normalizeName(topPrediction.className) === photoStem(state.guess.target);
  renderHints("guess", matched, topPrediction ? topPrediction.probability : 0, photoStem(state.guess.target));

  if (matched) {
    completeGuessLevel("correct");
  }
}

function finishGuessMode() {
  state.guess.running = false;
  state.guess.target = null;
  document.getElementById("guessLevelTitle").textContent = "Tous les niveaux complétés !";
}

function advanceGuessLevel() {
  state.guess.levelIndex += 1;
  if (state.guess.levelIndex >= state.guess.queue.length) {
    finishGuessMode();
    return;
  }
  setGuessTarget(state.guess.queue[state.guess.levelIndex]);
  state.guess.running = true;
}

function completeGuessLevel(result) {
  state.guess.running = false;
  state.guess.results[state.guess.levelIndex] = result;
  renderGuessProgress();
  if (result === "correct") {
    showSuccess("guess", true);
  }
  setTimeout(function () {
    showSuccess("guess", false);
    advanceGuessLevel();
  }, result === "correct" ? 900 : 150);
}

// ── Start / stop ──────────────────────────────────────────────────────────────
async function startLiveMode() {
  setStatus(false, "Chargement...");
  await ensureLoop();
  state.live.running = true;
  showSuccess("live", false);
  const btn = document.getElementById("startLiveMode");
  btn.textContent = "◼ Arrêter";
  btn.onclick = stopLiveMode;
}

function stopLiveMode() {
  state.live.running = false;
  const btn = document.getElementById("startLiveMode");
  btn.textContent = "▶ Jouer";
  btn.onclick = function () { startLiveMode().catch(onModelError); };
}

async function startGuessMode() {
  setStatus(false, "Chargement...");
  await ensureLoop();
  if (!state.photos.length) throw new Error("Aucune photo trouvée.");
  buildGuessQueue();
  state.guess.running = true;
  state.guess.levelIndex = 0;
  state.guess.results = [];
  renderGuessProgress();
  setGuessTarget(state.guess.queue[0]);
  showSuccess("guess", false);
  const btn = document.getElementById("startGuessMode");
  btn.textContent = "◼ Arrêter";
  btn.onclick = stopGuessMode;
}

function stopGuessMode() {
  state.guess.running = false;
  const btn = document.getElementById("startGuessMode");
  btn.textContent = "▶ Jouer";
  btn.onclick = function () { startGuessMode().catch(onModelError); };
}

function onModelError() {
  setStatus(false, "Échec du modèle");
  alert("Impossible de charger le modèle Teachable Machine.");
}

// ── Screen switch ─────────────────────────────────────────────────────────────
function switchScreen(screenId) {
  document.querySelectorAll(".screen").forEach(function (el) { el.classList.remove("active"); });
  document.querySelectorAll(".mode-tab").forEach(function (el) { el.classList.remove("active"); });
  document.getElementById(screenId).classList.add("active");
  document.querySelector('[data-screen="' + screenId + '"]').classList.add("active");
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.querySelectorAll(".mode-tab").forEach(function (button) {
  button.addEventListener("click", function () { switchScreen(button.dataset.screen); });
});

document.getElementById("startLiveMode").addEventListener("click", function () {
  startLiveMode().catch(onModelError);
});

document.getElementById("startGuessMode").addEventListener("click", function () {
  startGuessMode().catch(onModelError);
});

document.getElementById("skipGuessLevel").addEventListener("click", function () {
  if (!state.guess.running || !state.guess.target) return;
  completeGuessLevel("skipped");
});

document.getElementById("liveTargetImage").style.display = "none";
document.getElementById("guessTargetImage").style.display = "none";

loadPhotos();
