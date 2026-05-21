const MODEL_URLS = [
  "https://teachablemachine.withgoogle.com/models/tTYdfh6E2/",
  "./static/my_model/",
];
const PHOTO_LIBRARY = [
  { filename: "angry.png", image_path: "./photos/angry.png" },
  { filename: "boring.png", image_path: "./photos/boring.png" },
  { filename: "bıkmış.png", image_path: "./photos/bıkmış.png" },
  { filename: "kurnaz.png", image_path: "./photos/kurnaz.png" },
  { filename: "magara_adami.png", image_path: "./photos/magara_adami.png" },
  { filename: "merhaba.png", image_path: "./photos/merhaba.png" },
  { filename: "ne_diyosun_be.png", image_path: "./photos/ne_diyosun_be.png" },
  { filename: "ordek.png", image_path: "./photos/ordek.png" },
  { filename: "perfect.png", image_path: "./photos/perfect.png" },
  { filename: "rainbow.png", image_path: "./photos/rainbow.png" },
  { filename: "scream.png", image_path: "./photos/scream.png" },
  { filename: "sus.png", image_path: "./photos/sus.png" }
];
const WEBCAM_WIDTH = 960;
const WEBCAM_HEIGHT = 720;
const MATCH_THRESHOLD = 0.9;
const HOLD_FRAMES = 4;

const state = {
  photos: [],
  model: null,
  webcam: null,
  labelCount: 0,
  initialized: false,
  loopStarted: false,
  modelAvailable: false,
  live: { running: false, target: null, holdFrames: 0 },
  guess: { running: false, levelIndex: 0, target: null, holdFrames: 0 },
};

const liveCanvas = document.getElementById("liveCanvas");
const guessCanvas = document.getElementById("guessCanvas");

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
  state.photos = PHOTO_LIBRARY;
  if (state.photos.length && !state.guess.target) {
    setGuessTarget(state.photos[0]);
  }
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
  state.guess.holdFrames = 0;

  const img = document.getElementById("guessTargetImage");
  img.src = photo.image_path;
  img.style.display = "block";

  document.getElementById("guessPlaceholder").style.display = "none";

  const labelBar = document.getElementById("guessLabelBar");
  labelBar.style.display = "flex";
  document.getElementById("guessMemeLabel").textContent = photoStem(photo).replace(/_/g, " ");
  document.getElementById("guessLevelTitle").textContent = "Niveau " + (state.guess.levelIndex + 1);
  document.getElementById("guessLevelCounter").textContent =
    (state.guess.levelIndex + 1) + " / " + state.photos.length;
}

// ── Badges ────────────────────────────────────────────────────────────────────
function showSuccess(mode, visible) {
  document.getElementById(mode + "SuccessBadge").classList.toggle("hidden", !visible);
}

// ── Hints ─────────────────────────────────────────────────────────────────────
function renderHints(mode, matched, probability, targetName) {
  const hintsId = mode === "live" ? "liveHints" : "guessHints";
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
  const matched = probability >= MATCH_THRESHOLD;
  const topPhoto = photoForPrediction(match);
  if (topPhoto) setLiveTarget(topPhoto);

  state.live.holdFrames = matched ? state.live.holdFrames + 1 : 0;
  renderHints("live", matched, probability, match ? match.className : "—");

  if (state.live.holdFrames >= HOLD_FRAMES) {
    state.live.running = false;
    state.live.holdFrames = 0;
    showSuccess("live", true);
    setTimeout(() => {
      showSuccess("live", false);
      state.live.running = true;
    }, 1500);
  }
}

// ── Guess mode ────────────────────────────────────────────────────────────────
function handleGuessMode(predictions) {
  if (!state.guess.running || !state.guess.target) return;
  const match = predictionForTarget(predictions, state.guess.target);
  const probability = match ? match.probability : 0;
  const matched = probability >= MATCH_THRESHOLD;
  state.guess.holdFrames = matched ? state.guess.holdFrames + 1 : 0;
  renderHints("guess", matched, probability, photoStem(state.guess.target));

  if (state.guess.holdFrames >= HOLD_FRAMES) {
    state.guess.running = false;
    state.guess.holdFrames = 0;
    showSuccess("guess", true);
    setTimeout(function () {
      showSuccess("guess", false);
      state.guess.levelIndex += 1;
      if (state.guess.levelIndex >= state.photos.length) {
        document.getElementById("guessHints").innerHTML =
          '<span class="hint-chip">Tous les niveaux complétés !</span>';
        return;
      }
      setGuessTarget(state.photos[state.guess.levelIndex]);
      state.guess.running = true;
    }, 900);
  }
}

// ── Start / stop ──────────────────────────────────────────────────────────────
async function startLiveMode() {
  setStatus(false, "Chargement...");
  await ensureLoop();
  state.live.running = true;
  state.live.holdFrames = 0;
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
  state.guess.running = true;
  state.guess.levelIndex = 0;
  state.guess.holdFrames = 0;
  setGuessTarget(state.photos[0]);
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

document.getElementById("liveTargetImage").style.display = "none";
document.getElementById("guessTargetImage").style.display = "none";

loadPhotos();
