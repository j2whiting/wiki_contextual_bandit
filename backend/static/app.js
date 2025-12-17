const API_BASE = "";

let displayTimestamps = {};
let rewardChart = null;
let hitRateChart = null;
let driftChart = null;
let tasteMapChart = null;
let deltaMapChart = null;
let isLoading = false;
let isReranking = false;
let seenPageIds = new Set();
let retryTimer = null;
let emptyPollDelayMs = 1200;
let emptyPollActive = false;
let lastAutoLoadAt = 0;
const AUTO_LOAD_COOLDOWN_MS = 1200;
let autoFillTimer = null;
let swipeDeckAvgPred = null;
let swipeDeckDeltaScale = null;

// Swipe deck state
let activeView = "swipe"; // "swipe" | "grid"
let swipeDeck = [];
let swipeIndex = 0;
let swipeFetching = false;
let swipeDeckRequestId = 0;
let currentSwipeItem = null;
let swipeShownAt = null;
let swipeDrag = null;

function isFeedActive() {
  return document.getElementById("feed-tab")?.classList.contains("active") ?? true;
}

function isSentinelNearBottom() {
  const sentinel = document.getElementById("feedSentinel");
  if (!sentinel) return false;
  const rect = sentinel.getBoundingClientRect();
  // If the sentinel is within the viewport + buffer, we should load more.
  return rect.top < window.innerHeight + 800;
}

function scheduleAutoFill(delayMs) {
  const ms = Math.max(0, Number(delayMs || 0));
  if (autoFillTimer) return;
  autoFillTimer = setTimeout(() => {
    autoFillTimer = null;
    maybeAutoFillFeed();
  }, ms);
}

function maybeAutoFillFeed() {
  if (!isFeedActive()) return;
  if (activeView !== "grid") return;
  if (isLoading) return;
  if (emptyPollActive) return;
  if (!isSentinelNearBottom()) return;

  const now = Date.now();
  const since = now - lastAutoLoadAt;
  if (since < AUTO_LOAD_COOLDOWN_MS) {
    // If the sentinel stays intersecting (common on large screens), the IO callback
    // won't fire again. Schedule a follow-up check right after the cooldown so we
    // don't stall at the bottom.
    scheduleAutoFill(AUTO_LOAD_COOLDOWN_MS - since + 25);
    return;
  }
  lastAutoLoadAt = now;
  loadBatch({ reset: false });
}

function computeAvgPredictedReward(items) {
  const arr = Array.isArray(items) ? items : [];
  let sum = 0;
  let n = 0;
  for (const it of arr) {
    const v = Number(it?.predicted_mean_reward);
    if (!Number.isFinite(v)) continue;
    sum += v;
    n += 1;
  }
  return n > 0 ? sum / n : null;
}

function computeDeltaScale(items, avg) {
  if (avg == null || !Number.isFinite(avg)) return null;
  const arr = Array.isArray(items) ? items : [];
  let maxAbs = 0;
  for (const it of arr) {
    const v = Number(it?.predicted_mean_reward);
    if (!Number.isFinite(v)) continue;
    const d = v - avg;
    const a = Math.abs(d);
    if (a > maxAbs) maxAbs = a;
  }
  // Keep a stable-ish floor so tiny deltas still render.
  const floor = 0.05;
  const cap = 0.5;
  if (!Number.isFinite(maxAbs) || maxAbs <= 0) return floor;
  return Math.max(floor, Math.min(cap, maxAbs));
}

function deltaBarWidths(delta, scale) {
  const d = Number(delta);
  if (!Number.isFinite(d)) return { pos: 0, neg: 0 };
  const s = Number(scale);
  const denom = Number.isFinite(s) && s > 0 ? s : 0.05;
  const w = Math.min(50, (Math.abs(d) / denom) * 50);
  return { pos: d > 0 ? w : 0, neg: d < 0 ? w : 0 };
}

function formatSigned(x, digits = 2) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "+0.00";
  const sign = v >= 0 ? "+" : "";
  return sign + v.toFixed(digits);
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  const contents = document.querySelectorAll(".tab-content");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.tab;

      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      contents.forEach((c) => {
        if (c.id === targetId) {
          c.classList.add("active");
        } else {
          c.classList.remove("active");
        }
      });

      if (targetId === "insights-tab") {
        loadInsights();
        loadModelDiffs();
      }
    });
  });
}

function resetFeed() {
  const container = document.getElementById("cardsContainer");
  if (container) {
    container.innerHTML = "";
  }
  displayTimestamps = {};
  seenPageIds = new Set();

  if (retryTimer) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  emptyPollDelayMs = 1200;
  emptyPollActive = false;

  // Allow immediate auto-fill after a reset.
  lastAutoLoadAt = 0;

  if (autoFillTimer) {
    clearTimeout(autoFillTimer);
    autoFillTimer = null;
  }
}

function setActiveView(view) {
  activeView = view === "grid" ? "grid" : "swipe";
  const swipeViewEl = document.getElementById("swipeView");
  const gridViewEl = document.getElementById("gridView");
  const swipeBtn = document.getElementById("viewSwipeBtn");
  const gridBtn = document.getElementById("viewGridBtn");

  if (swipeViewEl) swipeViewEl.classList.toggle("hidden", activeView !== "swipe");
  if (gridViewEl) gridViewEl.classList.toggle("hidden", activeView !== "grid");
  if (swipeBtn) swipeBtn.classList.toggle("active", activeView === "swipe");
  if (gridBtn) gridBtn.classList.toggle("active", activeView === "grid");

  // Kick initial loads for the chosen view
  if (activeView === "swipe") {
    ensureSwipeDeck();
  } else {
    loadBatch({ reset: true });
  }
}

async function fetchArticles(batchSize = 5) {
  const cacheBuster = Date.now();
  // Use a short alias endpoint to avoid aggressive blockers (Firefox extensions).
  const url = `${API_BASE}/r?batch_size=${encodeURIComponent(batchSize)}&_=${cacheBuster}`;
  const res = await fetch(url, { method: "GET", cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.items || [];
}

async function regenerateArticles(batchSize = 5) {
  const cacheBuster = Date.now();
  const url = `${API_BASE}/api/regenerate?batch_size=${encodeURIComponent(batchSize)}&_=${cacheBuster}`;
  // Use GET + cache-buster + no-store (works better with some Firefox blockers than POST).
  const res = await fetch(url, { method: "GET", cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.items || [];
}

function deckSourceLabel(item) {
  return item?.source_type === "neighbor"
    ? "neighbor"
    : item?.source_type === "search"
    ? "keywords"
    : "discovery";
}

function renderSwipeCard(item) {
  const deckEl = document.getElementById("swipeDeck");
  if (!deckEl) return;

  deckEl.innerHTML = "";
  if (!item) {
    const empty = document.createElement("div");
    empty.className = "status-text";
    empty.textContent = "Waiting for recommendations…";
    deckEl.appendChild(empty);
    currentSwipeItem = null;
    swipeShownAt = null;
    return;
  }

  currentSwipeItem = item;
  swipeShownAt = Date.now();

  const expected = Number(item.predicted_mean_reward ?? 0) || 0;
  const delta =
    swipeDeckAvgPred != null && Number.isFinite(swipeDeckAvgPred)
      ? expected - swipeDeckAvgPred
      : null;
  const deltaText = delta != null ? ` (Δ ${formatSigned(delta)})` : "";
  const widths = delta != null ? deltaBarWidths(delta, swipeDeckDeltaScale) : { pos: 0, neg: 0 };
  const deltaBar =
    delta != null
      ? `<div class="delta-bar-container swipe-delta" aria-hidden="true">
          <div class="delta-bar-neg" style="width: ${widths.neg}%;"></div>
          <div class="delta-bar-pos" style="width: ${widths.pos}%;"></div>
        </div>`
      : "";

  const card = document.createElement("article");
  card.className = "swipe-card";
  card.dataset.pageid = String(item.pageid);
  card.dataset.selectionType = item.selection_type;

  const thumb = item.thumbnail_url
    ? `<div class="swipe-thumb"><img src="${escapeHtml(item.thumbnail_url)}" alt="${escapeHtml(
        item.title
      )}" loading="lazy" /></div>`
    : `<div class="swipe-thumb"></div>`;

  card.innerHTML = `
    ${thumb}
    <div class="swipe-body">
      <div class="swipe-meta">
        <span>via ${escapeHtml(deckSourceLabel(item))}</span>
        <span>expected ${expected.toFixed(2)}${deltaText}</span>
      </div>
      ${deltaBar}
      <h2 class="swipe-title">${escapeHtml(item.title || "")}</h2>
      <p class="swipe-extract">${escapeHtml(truncateText(item.extract || "", 420))}</p>
    </div>
  `;

  attachSwipeGestures(card);
  deckEl.appendChild(card);
}

function attachSwipeGestures(cardEl) {
  const thresholdPx = 90;

  function onPointerDown(e) {
    if (!currentSwipeItem) return;
    swipeDrag = {
      startX: e.clientX,
      startY: e.clientY,
      dx: 0,
      dy: 0,
      active: true,
    };
    cardEl.setPointerCapture?.(e.pointerId);
  }

  function onPointerMove(e) {
    if (!swipeDrag?.active) return;
    swipeDrag.dx = e.clientX - swipeDrag.startX;
    swipeDrag.dy = e.clientY - swipeDrag.startY;
    const rot = Math.max(-12, Math.min(12, swipeDrag.dx / 18));
    cardEl.style.transform = `translate(${swipeDrag.dx}px, ${swipeDrag.dy}px) rotate(${rot}deg)`;
  }

  function onPointerUp() {
    if (!swipeDrag?.active) return;
    const dx = swipeDrag.dx || 0;
    swipeDrag.active = false;

    if (dx > thresholdPx) {
      swipeAction("like");
      return;
    }
    if (dx < -thresholdPx) {
      swipeAction("pass");
      return;
    }
    cardEl.style.transform = "";
  }

  cardEl.addEventListener("pointerdown", onPointerDown);
  cardEl.addEventListener("pointermove", onPointerMove);
  cardEl.addEventListener("pointerup", onPointerUp);
  cardEl.addEventListener("pointercancel", onPointerUp);
}

async function sendSwipeReward({ item, eventType, reward }) {
  const dwellMs = swipeShownAt != null ? Date.now() - swipeShownAt : null;
  try {
    await fetch(`${API_BASE}/api/reward`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pageid: item.pageid,
        reward,
        selection_type: item.selection_type,
        event_type: eventType,
        dwell_time_ms: dwellMs,
      }),
    });
  } catch (err) {
    console.error("Error sending reward:", err);
  }
}

async function swipeAction(action) {
  const item = currentSwipeItem;
  if (!item) return;

  // Animate offscreen
  const cardEl = document.querySelector("#swipeDeck .swipe-card");
  if (cardEl) {
    const dir = action === "like" ? 1 : action === "pass" ? -1 : 0;
    if (dir !== 0) {
      cardEl.style.transition = "transform 180ms ease, opacity 180ms ease";
      cardEl.style.transform = `translate(${dir * 420}px, -20px) rotate(${dir * 10}deg)`;
      cardEl.style.opacity = "0";
    }
  }

  if (action === "like") {
    await sendSwipeReward({ item, eventType: "swipe_like", reward: 1.0 });
  } else if (action === "pass") {
    await sendSwipeReward({ item, eventType: "swipe_pass", reward: 0.0 });
  } else if (action === "open") {
    await sendSwipeReward({ item, eventType: "open", reward: 1.0 });
    openArticleModal(item.pageid, item.title, item.url);
  }

  // Advance deck
  swipeIndex += 1;
  currentSwipeItem = null;
  swipeShownAt = null;
  renderSwipeCard(swipeDeck[swipeIndex] || null);
  ensureSwipeDeck();

  // Update insights/progress
  loadInsights();
}

async function ensureSwipeDeck() {
  if (activeView !== "swipe") return;
  if (isReranking) return;
  if (swipeFetching) return;

  const remaining = swipeDeck.length - swipeIndex;
  if (remaining >= 3) return;

  swipeFetching = true;
  const reqId = ++swipeDeckRequestId;
  try {
    const items = await fetchArticles(6);
    if (reqId !== swipeDeckRequestId) return;
    if (items.length > 0) {
      // Reset deck if we're empty; otherwise append
      if (remaining <= 0) {
        swipeDeck = items;
        swipeIndex = 0;
      } else {
        swipeDeck = swipeDeck.slice(swipeIndex).concat(items);
        swipeIndex = 0;
      }
      swipeDeckAvgPred = computeAvgPredictedReward(swipeDeck);
      swipeDeckDeltaScale = computeDeltaScale(swipeDeck, swipeDeckAvgPred);
      renderSwipeCard(swipeDeck[swipeIndex] || null);
    }
  } catch (err) {
    console.error(err);
  } finally {
    swipeFetching = false;
  }
}

async function rerankWithUpdatedModel() {
  if (isReranking) return;
  // Don't block swipe rerank on a grid fetch in-flight.
  if (activeView === "grid" && isLoading) return;

  isReranking = true;
  const reqId = ++swipeDeckRequestId; // invalidate any in-flight swipe fetch

  // Optimistically reset progress UI immediately; we'll sync with /api/state after.
  updateProgressBar(0);

  const statusEl = document.getElementById("feedStatus");
  const button = document.getElementById("loadBatchButton");
  const spinner = document.getElementById("loadingSpinner");

  try {
    if (activeView === "grid") {
      await loadBatch({ reset: true, regenerate: true });
      // Ensure progress/stat UI reflects the rerank immediately.
      loadInsights();
      return;
    }

    // Swipe view: regenerate + refresh swipe deck.
    if (button) button.disabled = true;
    if (spinner) spinner.classList.add("active");
    if (statusEl) statusEl.textContent = "Reranking with updated model…";

    const items = await regenerateArticles(6);
    if (reqId !== swipeDeckRequestId) return;
    if (items.length > 0) {
      swipeDeck = items;
      swipeIndex = 0;
      currentSwipeItem = null;
      swipeShownAt = null;
      swipeDeckAvgPred = computeAvgPredictedReward(swipeDeck);
      swipeDeckDeltaScale = computeDeltaScale(swipeDeck, swipeDeckAvgPred);
      renderSwipeCard(swipeDeck[swipeIndex] || null);
      ensureSwipeDeck();
      if (statusEl) {
        statusEl.textContent = "Swipe to explore. Like stuff you enjoy – the model will adapt.";
      }
    } else {
      swipeDeck = [];
      swipeIndex = 0;
      currentSwipeItem = null;
      swipeShownAt = null;
      swipeDeckAvgPred = null;
      swipeDeckDeltaScale = null;
      renderSwipeCard(null);
      if (statusEl) {
        statusEl.textContent =
          "Waiting for recommendations… (start the worker: `python -m backend.worker`)";
      }
    }

    loadInsights();
  } catch (err) {
    console.error("Error reranking:", err);
    // Re-sync (in case the optimistic reset was wrong).
    loadInsights();
    if (statusEl) statusEl.textContent = "Failed to rerank. Check backend logs.";
  } finally {
    isReranking = false;
    if (button && !isLoading) button.disabled = false;
    if (spinner && !isLoading && !emptyPollActive) spinner.classList.remove("active");
  }
}

async function loadBatch({ reset = false, regenerate = false } = {}) {
  const statusEl = document.getElementById("feedStatus");
  const button = document.getElementById("loadBatchButton");
  const spinner = document.getElementById("loadingSpinner");
  if (!statusEl) return;
  if (isLoading) return;
  if (!reset && emptyPollActive && retryTimer) return;

  // Only run grid fetch logic when grid is active.
  if (activeView !== "grid") {
    return;
  }

  if (reset) {
    resetFeed();
  }

  let shouldCheckAutoFill = false;

  isLoading = true;
  if (button) button.disabled = true;
  if (spinner) spinner.classList.add("active");
  statusEl.textContent =
    reset
      ? "Loading recommendations from Wikipedia..."
      : "Loading more recommendations...";

  try {
    const cacheBuster = Date.now();
    const url = regenerate
      ? `${API_BASE}/api/regenerate?batch_size=5&_=${cacheBuster}`
      : `${API_BASE}/r?batch_size=5&_=${cacheBuster}`;
    // Use GET + cache-buster + no-store (works better with some Firefox blockers than POST).
    const res = await fetch(url, { method: "GET", cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    const items = data.items || [];
    renderCards(items, { append: !reset });
    if (items.length === 0) {
      emptyPollActive = true;
      const count = document.getElementById("cardsContainer")?.children?.length ?? 0;
      statusEl.textContent =
        count === 0
          ? "Waiting for recommendations… (start the worker: `python -m backend.worker`)"
          : "No more queued items right now — waiting for more…";

      const delay = Math.min(emptyPollDelayMs, 12000);
      emptyPollDelayMs = Math.min(Math.floor(emptyPollDelayMs * 1.6), 12000);

      if (!retryTimer) {
        retryTimer = setTimeout(() => {
          retryTimer = null;
          loadBatch({ reset: false });
        }, delay);
      }
    } else {
      emptyPollActive = false;
      emptyPollDelayMs = 1200;
      if (retryTimer) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      statusEl.textContent =
        "Keep scrolling to load more. Click stuff you like – the model will adapt.";

      // If the page still isn't scrollable (or sentinel is still near), keep fetching.
      shouldCheckAutoFill = true;
    }
  } catch (err) {
    console.error(err);
    emptyPollActive = false;
    statusEl.textContent = "Failed to load articles. Check the backend logs.";
  } finally {
    isLoading = false;
    if (button) button.disabled = false;
    if (spinner && !emptyPollActive) spinner.classList.remove("active");

    if (shouldCheckAutoFill) {
      // Let layout settle, then check if we should fetch another batch.
      setTimeout(maybeAutoFillFeed, 0);
    }
  }
}

function renderCards(items, { append = false } = {}) {
  const container = document.getElementById("cardsContainer");
  if (!container) return;

  const batchAvgPred = computeAvgPredictedReward(items);
  const batchDeltaScale = computeDeltaScale(items, batchAvgPred);

  if (!append) {
    container.innerHTML = "";
    displayTimestamps = {};
    seenPageIds = new Set();
  }

  items.forEach((item) => {
    if (item?.pageid == null) return;
    if (seenPageIds.has(item.pageid)) return;
    seenPageIds.add(item.pageid);

    const card = document.createElement("article");
    card.className = "card";
    card.dataset.pageid = String(item.pageid);
    card.dataset.selectionType = item.selection_type;

    const predictedRaw = Number(item.predicted_mean_reward ?? 0);
    const predicted = Number.isFinite(predictedRaw) ? predictedRaw : 0;
    const delta =
      batchAvgPred != null && Number.isFinite(batchAvgPred) ? predicted - batchAvgPred : null;
    const deltaText = delta != null ? ` • Δ ${formatSigned(delta)}` : "";

    const sourceLabel =
      item.source_type === "neighbor"
        ? "neighbor"
        : item.source_type === "search"
        ? "keywords"
        : "discovery";

    const widths = delta != null ? deltaBarWidths(delta, batchDeltaScale) : { pos: 0, neg: 0 };

    const thumbUrl = item.thumbnail_url;
    const thumbHtml = thumbUrl
      ? `<div class="card-thumb"><img src="${escapeHtml(
          thumbUrl
        )}" alt="${escapeHtml(item.title)}" loading="lazy" /></div>`
      : "";

    card.innerHTML = `
      ${thumbHtml}
      <div class="card-header-line">
        <h2 class="card-title">${escapeHtml(item.title)}</h2>
      </div>
      <div class="expected-reward">
        <div class="delta-bar-container" aria-hidden="true">
          <div class="delta-bar-neg" style="width: ${widths.neg}%;"></div>
          <div class="delta-bar-pos" style="width: ${widths.pos}%;"></div>
        </div>
        <span class="reward-label">Expected: ${predicted.toFixed(2)}${deltaText}</span>
      </div>
      <p class="card-body">${escapeHtml(truncateText(item.extract, 360))}</p>
      <div class="card-footer">
        <div class="card-meta">
          via ${sourceLabel}
        </div>
        <div class="card-buttons">
          <button class="card-button view-btn">View article</button>
          <button class="card-button like-btn">Like</button>
          <button class="card-button skip-btn">Skip</button>
        </div>
      </div>
    `;

    container.appendChild(card);
    displayTimestamps[item.pageid] = Date.now();

    const viewBtn = card.querySelector(".view-btn");
    const likeBtn = card.querySelector(".like-btn");
    const skipBtn = card.querySelector(".skip-btn");

    if (viewBtn) {
      viewBtn.addEventListener("click", () => {
        handleInteraction(
          item.pageid,
          item.selection_type,
          "click",
          item.url,
          item.title
        );
      });
    }

    if (likeBtn) {
      likeBtn.addEventListener("click", () => {
        handleInteraction(item.pageid, item.selection_type, "like", null, null);
        // Visually mark as consumed since it will never be served again.
        card.style.opacity = "0.35";
      });
    }

    if (skipBtn) {
      skipBtn.addEventListener("click", () => {
        handleInteraction(item.pageid, item.selection_type, "skip", null, null);
        card.style.opacity = "0.4";
      });
    }
  });
}

function truncateText(text, maxLen) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + "...";
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function handleInteraction(pageid, selectionType, eventType, url, title) {
  const shownAt = displayTimestamps[pageid];
  const now = Date.now();
  const dwellMs = shownAt ? now - shownAt : null;

  let reward = 0.0;
  if (eventType === "click") {
    if (dwellMs != null) {
      if (dwellMs > 8000) {
        reward = 1.0;
      } else if (dwellMs > 3000) {
        reward = 0.7;
      } else {
        reward = 0.4;
      }
    } else {
      reward = 0.6;
    }
  } else if (eventType === "skip") {
    reward = 0.0;
  } else if (eventType === "like") {
    reward = 1.0;
  }

  try {
    await fetch(`${API_BASE}/api/reward`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pageid,
        reward,
        selection_type: selectionType,
        event_type: eventType,
        dwell_time_ms: dwellMs,
      }),
    });
  } catch (err) {
    console.error("Error sending reward:", err);
  }

  if (eventType === "click") {
    openArticleModal(pageid, title, url);
  }

  loadInsights();
}

// Article Modal Functions
function openArticleModal(pageid, title, wikiUrl) {
  const modal = document.getElementById("articleModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalContent = document.getElementById("modalContent");
  const modalWikiLink = document.getElementById("modalWikiLink");

  if (!modal) return;

  modalTitle.textContent = title || "Loading...";
  modalWikiLink.href = wikiUrl || "#";
  modalContent.innerHTML = '<div class="modal-loading">Loading article...</div>';

  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";

  // Fetch full article HTML
  fetchArticleContent(pageid);
}

async function fetchArticleContent(pageid) {
  const modalContent = document.getElementById("modalContent");
  const modalTitle = document.getElementById("modalTitle");

  try {
    const res = await fetch(`${API_BASE}/api/article/${pageid}`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();

    if (data.title) {
      modalTitle.innerHTML = data.title;
    }

    // Clean up Wikipedia HTML for better display
    let html = data.html || "";
    // Remove edit links
    html = html.replace(/<span class="mw-editsection">.*?<\/span>/g, "");
    // Fix relative links to point to Wikipedia
    html = html.replace(/href="\/wiki\//g, 'href="https://en.wikipedia.org/wiki/');
    html = html.replace(/src="\/\//g, 'src="https://');

    modalContent.innerHTML = html;
  } catch (err) {
    console.error("Error fetching article:", err);
    modalContent.innerHTML = '<div class="modal-error">Failed to load article. Try opening in Wikipedia.</div>';
  }
}

function closeArticleModal() {
  const modal = document.getElementById("articleModal");
  if (!modal) return;

  modal.classList.remove("active");
  modal.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function initModal() {
  const modal = document.getElementById("articleModal");
  const closeBtn = document.getElementById("modalClose");

  if (closeBtn) {
    closeBtn.addEventListener("click", closeArticleModal);
  }

  if (modal) {
    // Close on overlay click
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        closeArticleModal();
      }
    });

    // Close on Escape key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal.classList.contains("active")) {
        closeArticleModal();
      }
    });
  }
}

async function loadInsights() {
  try {
    const res = await fetch(`${API_BASE}/api/state`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const state = await res.json();
    updateStats(state);
    updateCharts(state);
    loadModelDiffs();
  } catch (err) {
    console.error("Error loading state:", err);
  }
}

const PROGRESS_MAX = 10; // rerank suggested after this many rewards

function updateStats(state) {
  const interactionsEl = document.getElementById("stat-interactions");
  const hitRateEl = document.getElementById("stat-hit-rate");
  const rollingRewardEl = document.getElementById("stat-rolling-reward");

  if (interactionsEl) {
    interactionsEl.textContent = state.interactions ?? 0;
  }
  if (hitRateEl) {
    const rate = (state.hit_rate ?? 0) * 100;
    hitRateEl.textContent = `${rate.toFixed(1)}%`;
  }
  if (rollingRewardEl) {
    const rolling = state.rolling_avg_reward ?? 0;
    rollingRewardEl.textContent = rolling.toFixed(3);
  }

  // Update progress bar
  updateProgressBar(state.rewards_since_last_rerank ?? 0);
}

function updateProgressBar(count) {
  const progressFill = document.getElementById("progressFill");
  const progressCount = document.getElementById("progressCount");
  const button = document.getElementById("loadBatchButton");

  if (progressFill) {
    const pct = Math.min(100, (count / PROGRESS_MAX) * 100);
    progressFill.style.width = `${pct}%`;

    // Glow effect when full
    if (count >= PROGRESS_MAX) {
      progressFill.classList.add("ready");
    } else {
      progressFill.classList.remove("ready");
    }
  }

  if (progressCount) {
    progressCount.textContent = `${count} / ${PROGRESS_MAX}`;
  }

  if (button) {
    if (count >= PROGRESS_MAX) {
      button.classList.add("pulse");
    } else {
      button.classList.remove("pulse");
    }
  }
}

function updateCharts(state) {
  // Hit rate over time chart
  const hitRateCtx = document.getElementById("hitRateChart");
  if (hitRateCtx) {
    if (hitRateChart) {
      hitRateChart.destroy();
    }

    const rollingData = state.rolling_hit_rates || [];
    const hrLabels = rollingData.map((p) => p.step);
    const hrRates = rollingData.map((p) => (p.reward ?? 0) * 100);

    hitRateChart = new Chart(hitRateCtx, {
      type: "line",
      data: {
        labels: hrLabels,
        datasets: [
          {
            label: "Hit rate %",
            data: hrRates,
            tension: 0.3,
            borderColor: "#22c55e",
            backgroundColor: "rgba(34, 197, 94, 0.15)",
            fill: true,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            title: { display: true, text: "Interaction", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
          },
          y: {
            title: { display: true, text: "Hit rate %", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
            suggestedMin: 0,
            suggestedMax: 100,
          },
        },
      },
    });
  }

  // Raw reward signal chart
  const history = state.reward_history || [];
  const labels = history.map((p) => p.step);
  const rewards = history.map((p) => p.reward);

  const rewardCtx = document.getElementById("rewardChart");
  if (rewardCtx) {
    if (rewardChart) {
      rewardChart.destroy();
    }

    rewardChart = new Chart(rewardCtx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Reward",
            data: rewards,
            tension: 0.1,
            borderColor: "#38bdf8",
            backgroundColor: "rgba(56, 189, 248, 0.1)",
            fill: true,
            pointRadius: 2,
            pointBackgroundColor: rewards.map((r) =>
              r > 0 ? "#22c55e" : "#64748b"
            ),
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            title: { display: true, text: "Update step", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
          },
          y: {
            title: { display: true, text: "Reward", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
            suggestedMin: 0,
            suggestedMax: 1,
          },
        },
      },
    });
  }
}

async function loadModelDiffs() {
  const driftCtx = document.getElementById("driftChart");
  if (!driftCtx) return;

  try {
    const res = await fetch(`${API_BASE}/api/snapshots?limit=80`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    const items = data.items || [];
    updateDriftChart(items);
  } catch (err) {
    console.error("Error loading snapshots:", err);
  }
}

function updateDriftChart(items) {
  const driftCtx = document.getElementById("driftChart");
  if (!driftCtx) return;

  const chronological = (items || []).slice().reverse();
  const labels = chronological.map((s, idx) => {
    const interactions = s.interactions ?? null;
    if (interactions != null) return String(interactions);
    return String(idx + 1);
  });
  const drift = chronological.map((s) => s.l2_delta_theta_mean ?? 0);

  if (driftChart) {
    driftChart.destroy();
  }

  driftChart = new Chart(driftCtx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Parameter drift (L2)",
          data: drift,
          tension: 0.25,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.15)",
          fill: true,
          pointRadius: 3,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          title: { display: true, text: "Interactions at snapshot", color: "#9ca3af" },
          ticks: { color: "#9ca3af" },
          grid: { color: "rgba(148, 163, 184, 0.1)" },
        },
        y: {
          title: { display: true, text: "||Δθ|| drift", color: "#9ca3af" },
          ticks: { color: "#9ca3af" },
          grid: { color: "rgba(148, 163, 184, 0.1)" },
          beginAtZero: true,
        },
      },
    },
  });
}

const TASTE_COLORS = [
  "#38bdf8",
  "#22c55e",
  "#f97316",
  "#a78bfa",
  "#fb7185",
  "#eab308",
  "#14b8a6",
  "#94a3b8",
];

async function buildTasteMap() {
  const statusEl = document.getElementById("tasteMapStatus");
  const btn = document.getElementById("buildTasteMapBtn");
  const canvas = document.getElementById("tasteMapChart");
  if (!statusEl || !canvas) return;

  if (btn) btn.disabled = true;
  statusEl.textContent = "Building taste map…";

  try {
    const url = `${API_BASE}/api/taste_map?limit=250&_=${Date.now()}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const items = data.items || [];
    if (!items.length) {
      if (tasteMapChart) {
        tasteMapChart.destroy();
        tasteMapChart = null;
      }
      statusEl.textContent = "No liked items yet. Like a few cards first, then try again.";
      return;
    }

    const byCluster = new Map();
    items.forEach((p) => {
      const c = Number.isFinite(p.cluster) ? p.cluster : 0;
      if (!byCluster.has(c)) byCluster.set(c, []);
      byCluster.get(c).push(p);
    });

    const datasets = Array.from(byCluster.entries()).map(([cluster, pts]) => ({
      label: `Cluster ${cluster}`,
      data: pts.map((p) => ({
        x: Number(p.x ?? 0),
        y: Number(p.y ?? 0),
        title: String(p.title ?? ""),
        pageid: Number(p.pageid ?? 0),
        url: String(p.url ?? ""),
      })),
      backgroundColor: TASTE_COLORS[Math.abs(cluster) % TASTE_COLORS.length],
      pointRadius: 4,
      pointHoverRadius: 7,
    }));

    if (tasteMapChart) {
      tasteMapChart.destroy();
    }

    tasteMapChart = new Chart(canvas, {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        onClick: (evt, elements) => {
          if (!tasteMapChart) return;
          const els =
            elements && elements.length
              ? elements
              : tasteMapChart.getElementsAtEventForMode(
                  evt,
                  "nearest",
                  { intersect: true },
                  true
                );
          if (!els || !els.length) return;
          const el = els[0];
          const ds = tasteMapChart.data.datasets?.[el.datasetIndex];
          const raw = ds?.data?.[el.index] || {};
          const url =
            raw.url ||
            (raw.pageid ? `https://en.wikipedia.org/?curid=${raw.pageid}` : null);
          if (url) {
            // Prefer same-tab navigation to avoid popup blockers (Firefox is stricter than Safari).
            // If the user holds Ctrl/Cmd, attempt a new tab.
            const nativeEvt = evt?.native || evt;
            const wantsNewTab = !!(nativeEvt && (nativeEvt.ctrlKey || nativeEvt.metaKey));
            if (wantsNewTab) {
              const w = window.open(url, "_blank", "noopener");
              if (!w) window.location.assign(url);
            } else {
              window.location.assign(url);
            }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw || {};
                const title = raw.title || "";
                const pid = raw.pageid ? ` (pageid=${raw.pageid})` : "";
                return `${title}${pid}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "PCA 1", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
          },
          y: {
            title: { display: true, text: "PCA 2", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
          },
        },
      },
    });

    statusEl.textContent = `Showing ${items.length} liked articles across ${data.k ?? "?"} clusters.`;
  } catch (err) {
    console.error("Error building taste map:", err);
    statusEl.textContent = "Failed to build taste map. Check backend logs.";
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function buildDeltaMap() {
  const statusEl = document.getElementById("deltaMapStatus");
  const btn = document.getElementById("buildDeltaMapBtn");
  const canvas = document.getElementById("deltaMapChart");
  if (!statusEl || !canvas) return;

  if (btn) btn.disabled = true;
  statusEl.textContent = "Building delta map…";

  try {
    if (typeof Chart === "undefined") {
      statusEl.textContent = "Chart.js is not available (blocked?).";
      throw new Error("Chart.js not available");
    }

    const url = `${API_BASE}/api/delta_map?limit=600&_=${Date.now()}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const items = data.items || [];
    if (!items.length) {
      if (deltaMapChart) {
        deltaMapChart.destroy();
        deltaMapChart = null;
      }
      statusEl.textContent =
        "No delta-map samples yet. Scroll/swipe a bit to log display events, then try again.";
      return;
    }

    const pos = [];
    const nonpos = [];
    items.forEach((p) => {
      const d = Number(p.delta_vs_batch_avg);
      const pt = {
        x: Number(p.x ?? 0),
        y: Number(p.y ?? 0),
        title: String(p.title ?? ""),
        pageid: Number(p.pageid ?? 0),
        url: String(p.url ?? ""),
        delta_vs_batch_avg: d,
        predicted_mean_reward: Number(p.predicted_mean_reward ?? 0),
        batch_predicted_mean_reward_avg: Number(p.batch_predicted_mean_reward_avg ?? 0),
      };
      if (Number.isFinite(d) && d > 0) {
        pos.push(pt);
      } else {
        nonpos.push(pt);
      }
    });

    const datasets = [
      {
        label: "Δ > 0",
        data: pos,
        backgroundColor: "rgba(34, 197, 94, 0.85)",
        pointRadius: 4,
        pointHoverRadius: 7,
      },
      {
        label: "Δ ≤ 0",
        data: nonpos,
        backgroundColor: "rgba(249, 115, 22, 0.85)",
        pointRadius: 4,
        pointHoverRadius: 7,
      },
    ];

    if (deltaMapChart) {
      deltaMapChart.destroy();
    }

    deltaMapChart = new Chart(canvas, {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        onClick: (evt, elements) => {
          if (!deltaMapChart) return;
          const els =
            elements && elements.length
              ? elements
              : deltaMapChart.getElementsAtEventForMode(
                  evt,
                  "nearest",
                  { intersect: true },
                  true
                );
          if (!els || !els.length) return;
          const el = els[0];
          const ds = deltaMapChart.data.datasets?.[el.datasetIndex];
          const raw = ds?.data?.[el.index] || {};
          const url =
            raw.url ||
            (raw.pageid ? `https://en.wikipedia.org/?curid=${raw.pageid}` : null);
          if (url) {
            const nativeEvt = evt?.native || evt;
            const wantsNewTab = !!(nativeEvt && (nativeEvt.ctrlKey || nativeEvt.metaKey));
            if (wantsNewTab) {
              const w = window.open(url, "_blank", "noopener");
              if (!w) window.location.assign(url);
            } else {
              window.location.assign(url);
            }
          }
        },
        plugins: {
          legend: { display: true, labels: { color: "#9ca3af" } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const raw = ctx.raw || {};
                const title = raw.title || "";
                const delta = Number(raw.delta_vs_batch_avg);
                const pred = Number(raw.predicted_mean_reward);
                const avg = Number(raw.batch_predicted_mean_reward_avg);
                const dTxt = Number.isFinite(delta) ? `Δ ${formatSigned(delta)}` : "Δ ?";
                const predTxt = Number.isFinite(pred) ? pred.toFixed(2) : "?";
                const avgTxt = Number.isFinite(avg) ? avg.toFixed(2) : "?";
                return `${title} — expected ${predTxt} (avg ${avgTxt}, ${dTxt})`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "PCA 1", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
          },
          y: {
            title: { display: true, text: "PCA 2", color: "#9ca3af" },
            ticks: { color: "#9ca3af" },
            grid: { color: "rgba(148, 163, 184, 0.1)" },
          },
        },
      },
    });

    statusEl.textContent = `Showing ${items.length} cards (Δ>0: ${pos.length}, Δ≤0: ${nonpos.length}).`;
  } catch (err) {
    console.error("Error building delta map:", err);
    statusEl.textContent = "Failed to build delta map. Check backend logs.";
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function buildEmbeddingQuality() {
  const btn = document.getElementById("buildEmbeddingQualityBtn");
  const statusEl = document.getElementById("embeddingQualityStatus");
  const statsEl = document.getElementById("embeddingQualityStats");
  const canvas = document.getElementById("embeddingQualityChart");
  if (!statusEl || !statsEl || !canvas) return;

  if (btn) btn.disabled = true;
  statusEl.textContent = "Computing…";
  statsEl.textContent = "";

  try {
    if (typeof Chart === "undefined") {
      statusEl.textContent = "Chart.js is not available (blocked?). Stats will still compute.";
      throw new Error("Chart.js not available");
    }

    const url = `${API_BASE}/api/embedding_quality?limit_per_class=250&include_scores=true&_=${Date.now()}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const nLiked = data.n_liked ?? 0;
    const nDisliked = data.n_disliked ?? 0;
    const gap = data.gap ?? 0;
    const auc = data.auc ?? 0;
    const d = data.cohens_d ?? 0;
    const meanLiked = data.mean_pred_liked ?? 0;
    const meanDisliked = data.mean_pred_disliked ?? 0;

    statsEl.textContent = `n_liked=${nLiked}  n_disliked=${nDisliked}  mean_like=${meanLiked.toFixed?.(
      4
    ) ?? meanLiked}  mean_dislike=${meanDisliked.toFixed?.(4) ?? meanDisliked}  gap=${gap.toFixed?.(
      4
    ) ?? gap}  auc=${auc.toFixed?.(3) ?? auc}  d=${d.toFixed?.(3) ?? d}`;

    const sLike = data.scores_liked || [];
    const sDis = data.scores_disliked || [];
    if (!Array.isArray(sLike) || !Array.isArray(sDis) || sLike.length < 2 || sDis.length < 2) {
      statusEl.textContent = "Need more liked + disliked samples (with embeddings) to plot.";
      return;
    }

    // Build simple histograms.
    const all = sLike.concat(sDis).map((x) => Number(x)).filter((x) => Number.isFinite(x));
    if (all.length < 2) {
      statusEl.textContent = "Not enough finite scores to plot (scores contain NaN/Infinity).";
      return;
    }

    let min = all[0];
    let max = all[0];
    for (let i = 1; i < all.length; i++) {
      const v = all[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const bins = 24;
    const width = max > min ? (max - min) / bins : 1;

    function hist(arr) {
      const counts = new Array(bins).fill(0);
      for (const v of arr) {
        const x = Number(v);
        if (!Number.isFinite(x)) continue;
        let idx = Math.floor((x - min) / width);
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        counts[idx] += 1;
      }
      return counts;
    }

    const hLike = hist(sLike);
    const hDis = hist(sDis);
    const labels = new Array(bins)
      .fill(0)
      .map((_, i) => Number(min + (i + 0.5) * width).toFixed(2));

    if (window.embeddingQualityChart) {
      window.embeddingQualityChart.destroy();
      window.embeddingQualityChart = null;
    }

    const ctx = canvas.getContext?.("2d");
    if (!ctx) {
      statusEl.textContent = "Could not create chart (no 2D canvas context).";
      return;
    }

    window.embeddingQualityChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Liked",
            data: hLike,
            borderColor: "#22c55e",
            backgroundColor: "rgba(34, 197, 94, 0.15)",
            fill: true,
            tension: 0.25,
            pointRadius: 0,
          },
          {
            label: "Disliked",
            data: hDis,
            borderColor: "#f97316",
            backgroundColor: "rgba(249, 115, 22, 0.15)",
            fill: true,
            tension: 0.25,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true, labels: { color: "#9ca3af" } },
        },
        scales: {
          x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(148, 163, 184, 0.1)" } },
          y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(148, 163, 184, 0.1)" } },
        },
      },
    });

    statusEl.textContent = "Done.";
  } catch (err) {
    console.error("Error computing embedding quality:", err);
    statusEl.textContent = "Failed. Check browser console.";
  } finally {
    if (btn) btn.disabled = false;
  }
}

function initInfiniteScroll() {
  const sentinel = document.getElementById("feedSentinel");
  if (!sentinel) return;

  if (!("IntersectionObserver" in window)) {
    window.addEventListener("scroll", () => maybeAutoFillFeed());
    window.addEventListener("resize", () => maybeAutoFillFeed());
    return;
  }

  const obs = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          // Defer to the common auto-fill logic (handles cooldown + scheduling).
          maybeAutoFillFeed();
          break;
        }
      }
    },
    { root: null, rootMargin: "800px" }
  );

  obs.observe(sentinel);

  // In case the sentinel starts in-view (e.g. large screens), auto-fill immediately.
  window.addEventListener("resize", () => maybeAutoFillFeed());
}

function init() {
  initTabs();
  initInfiniteScroll();
  initModal();

  const tasteBtn = document.getElementById("buildTasteMapBtn");
  if (tasteBtn) {
    tasteBtn.addEventListener("click", buildTasteMap);
  }

  const deltaBtn = document.getElementById("buildDeltaMapBtn");
  if (deltaBtn) {
    deltaBtn.addEventListener("click", buildDeltaMap);
  }

  const qualityBtn = document.getElementById("buildEmbeddingQualityBtn");
  if (qualityBtn) {
    qualityBtn.addEventListener("click", buildEmbeddingQuality);
  }

  const swipeBtn = document.getElementById("viewSwipeBtn");
  const gridBtn = document.getElementById("viewGridBtn");
  if (swipeBtn) swipeBtn.addEventListener("click", () => setActiveView("swipe"));
  if (gridBtn) gridBtn.addEventListener("click", () => setActiveView("grid"));

  const likeBtn = document.getElementById("likeBtn");
  const passBtn = document.getElementById("passBtn");
  const openBtn = document.getElementById("openBtn");
  if (likeBtn) likeBtn.addEventListener("click", () => swipeAction("like"));
  if (passBtn) passBtn.addEventListener("click", () => swipeAction("pass"));
  if (openBtn) openBtn.addEventListener("click", () => swipeAction("open"));

  const button = document.getElementById("loadBatchButton");
  if (button) {
    button.addEventListener("click", rerankWithUpdatedModel);
  }

  // Default to swipe view (mobile-first). Grid view remains available.
  setActiveView("swipe");
  loadInsights();
}

window.addEventListener("DOMContentLoaded", init);
