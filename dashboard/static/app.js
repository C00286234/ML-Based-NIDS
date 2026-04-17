/* =====================================================================
   NIDS Dashboard - app.js
   SSE-driven live feed, Chart.js charts, browser notifications
   ===================================================================== */

const SEVERITY_ORDER = { critical: 4, high: 3, medium: 2, low: 1 };
const SEVERITY_COLOR = {
  critical: "#f38ba8",
  high: "#fab387",
  medium: "#f9e2af",
  low: "#a6e3a1",
};

let attackChart = null;
let timelineChart = null;
const timelineData = { labels: [], benign: [], attack: [] };
const MAX_TIMELINE_POINTS = 60;
let feedScrolledAway = false;
let pendingCount = 0;
let sseConnected = false;
const severityCounts = { critical: 0, high: 0, medium: 0, low: 0 };
const talkerCounts = {};

function fmt(n) {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
function fmtUptime(s) {
  if (s < 60) return Math.floor(s) + "s";
  if (s < 3600) return Math.floor(s / 60) + "m " + Math.floor(s % 60) + "s";
  return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
}
function timeAgo(iso) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return Math.floor(diff) + "s ago";
  if (diff < 3600) return Math.floor(diff / 60) + "m ago";
  return Math.floor(diff / 3600) + "h ago";
}

function setConnected(ok) {
  const dot = document.querySelector("#connection-status .status-dot");
  const txt = document.querySelector("#connection-status .status-text");
  dot.className = "status-dot " + (ok ? "connected" : "disconnected");
  txt.textContent = ok ? "Connected" : "Disconnected";
}

async function pollStats() {
  try {
    const res = await fetch("/api/status");
    const d = await res.json();
    setConnected(d.connected);
    document.getElementById("total-predictions").textContent = fmt(d.metrics.total_predictions);
    document.getElementById("total-attacks").textContent = fmt(d.metrics.total_attacks);
    document.getElementById("attack-rate").textContent = d.attack_rate_percent + "% attack rate";
    document.getElementById("avg-latency").textContent = d.metrics.avg_prediction_time_ms.toFixed(1) + "ms";
    const up = d.metrics.uptime_seconds;
    document.getElementById("uptime").textContent = fmtUptime(up);
    document.getElementById("sb-uptime").textContent = fmtUptime(up);

    updateStatusDot("sb-api", d.health.status === "healthy");
    updateStatusDot("sb-models", d.health.models_loaded);
    document.getElementById("sb-api-label").textContent = d.health.status === "healthy" ? "OK" : d.health.status;
    document.getElementById("sb-models-label").textContent = d.health.models_loaded ? "OK" : "down";

    updateDoughnut(d.metrics.attack_distribution);
    pushTimelineBucket(d.metrics.total_predictions, d.metrics.total_attacks);
  } catch {
    setConnected(false);
  }
}

function updateStatusDot(id, ok) {
  const el = document.getElementById(id);
  el.className = "status-bar-dot " + (ok ? "ok" : "err");
}

const CATPPUCCIN_PALETTE = [
  "#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89b4fa",
  "#cba6f7", "#94e2d5", "#f5c2e7", "#74c7ec", "#b4befe",
  "#eba0ac", "#f2cdcd", "#89dceb", "#a6e3a1", "#cdd6f4",
];

function updateDoughnut(distribution) {
  const labels = Object.keys(distribution);
  const values = Object.values(distribution);
  if (labels.length === 0) return;

  const colors = labels.map((_, i) => CATPPUCCIN_PALETTE[i % CATPPUCCIN_PALETTE.length]);
  const ctx = document.getElementById("attack-chart").getContext("2d");

  if (attackChart) {
    attackChart.data.labels = labels;
    attackChart.data.datasets[0].data = values;
    attackChart.data.datasets[0].backgroundColor = colors;
    attackChart.update();
  } else {
    attackChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{ data: values, backgroundColor: colors, borderWidth: 2, borderColor: "#1e1e2e" }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right", labels: { color: "#cdd6f4", font: { size: 11 } } },
        },
      },
    });
  }
}

let _lastTotal = null;
let _lastAttacks = null;

function pushTimelineBucket(total, attacks) {
  if (_lastTotal === null) {
    _lastTotal = total;
    _lastAttacks = attacks;
    return;
  }
  const benignDelta = (total - attacks) - (_lastTotal - _lastAttacks);
  const attackDelta = attacks - _lastAttacks;
  _lastTotal = total;
  _lastAttacks = attacks;

  const now = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  timelineData.labels.push(now);
  timelineData.benign.push(Math.max(0, benignDelta));
  timelineData.attack.push(Math.max(0, attackDelta));

  if (timelineData.labels.length > MAX_TIMELINE_POINTS) {
    timelineData.labels.shift();
    timelineData.benign.shift();
    timelineData.attack.shift();
  }

  const ctx = document.getElementById("timeline-chart").getContext("2d");
  if (timelineChart) {
    timelineChart.data.labels = timelineData.labels;
    timelineChart.data.datasets[0].data = timelineData.benign;
    timelineChart.data.datasets[1].data = timelineData.attack;
    timelineChart.update("none");
  } else {
    timelineChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: timelineData.labels,
        datasets: [
          {
            label: "Benign",
            data: timelineData.benign,
            borderColor: "#a6e3a1",
            backgroundColor: "rgba(166,227,161,0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
          },
          {
            label: "Attack",
            data: timelineData.attack,
            borderColor: "#f38ba8",
            backgroundColor: "rgba(243,139,168,0.15)",
            fill: true,
            tension: 0.3,
            pointRadius: 2,
            pointBackgroundColor: "#f38ba8",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { color: "#6c7086", maxTicksLimit: 10 }, grid: { color: "rgba(205,214,244,0.06)" } },
          y: { beginAtZero: true, ticks: { color: "#6c7086" }, grid: { color: "rgba(205,214,244,0.06)" } },
        },
        plugins: {
          legend: { labels: { color: "#cdd6f4", font: { size: 11 } } },
        },
      },
    });
  }
}

function esc(s) {
  const el = document.createElement("span");
  el.textContent = s;
  return el.innerHTML;
}

function renderCard(d) {
  const card = document.createElement("div");
  card.className = "dcard " + d.severity + (d.prediction === "BENIGN" ? " benign" : "");
  card.dataset.id = d.id;

  const confPct = Math.round(d.confidence * 100);

  card.innerHTML = `
    <div class="stripe"></div>
    <div class="body">
      <div class="row1">
        <span class="attack">${esc(d.prediction)}</span>
        <span class="sev">${esc(d.severity)}</span>
      </div>
      <div class="confbar"><div class="conffill" style="width:${confPct}%"></div></div>
      <div class="flow">${esc(d.src_ip)} &rarr; ${esc(d.dst_ip)}</div>
    </div>
    <div class="rightcol">
      <div class="confpct">${confPct}%</div>
      <div class="time-ago">${timeAgo(d.timestamp)}</div>
    </div>
  `;

  card.addEventListener("click", () => toggleExpand(card, d));
  return card;
}

function toggleExpand(card, d) {
  if (card.classList.contains("expanded")) {
    card.classList.remove("expanded");
    const detail = card.querySelector(".detail");
    if (detail) detail.remove();
    return;
  }

  card.classList.add("expanded");

  const detail = document.createElement("div");
  detail.className = "detail";

  let shapHtml = "";
  if (d.top_features && d.top_features.length > 0) {
    shapHtml = `<div><div class="detail-title">Top SHAP contributions</div>`;
    d.top_features.forEach((f) => {
      const isPos = f.contribution >= 0;
      const pct = Math.min(Math.abs(f.contribution) * 100, 50);
      const cls = isPos ? "shap-fill-pos" : "shap-fill-neg";
      shapHtml += `
        <div class="shap-row">
          <div class="fname">${esc(f.feature)}</div>
          <div class="shap-bar"><div class="${cls}" style="width:${pct}%"></div></div>
          <div class="shap-value">${isPos ? "+" : ""}${f.contribution.toFixed(2)}</div>
        </div>`;
    });
    shapHtml += `</div>`;
  }

  let probHtml = "";
  if (d.probabilities) {
    const sorted = Object.entries(d.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    probHtml = `<div><div class="detail-title">Class probabilities</div><div class="probs">`;
    sorted.forEach(([cls, prob]) => {
      const pct = (prob * 100).toFixed(1);
      probHtml += `
        <div class="prob-row">
          <span class="pname">${esc(cls)}</span>
          <div class="prob-bar"><div class="prob-fill" style="width:${prob * 100}%"></div></div>
          <span class="prob-value">${pct}%</span>
        </div>`;
    });
    probHtml += `</div></div>`;
  }

  const agreeHtml = d.model_agreement
    ? `<div class="agree">&#10003; Random Forest and XGBoost agree</div>`
    : `<div class="agree dis">&#9888; Models disagree</div>`;

  detail.innerHTML = shapHtml + probHtml + agreeHtml;
  const body = card.querySelector(".body");
  body.parentNode.insertBefore(detail, body.nextSibling);
}

const MAX_FEED = 200;

function addCardToFeed(d) {
  const list = document.getElementById("feed-list");
  const empty = document.getElementById("feed-empty");
  if (empty) empty.remove();

  if (d.is_attack && d.severity) {
    severityCounts[d.severity] = (severityCounts[d.severity] || 0) + 1;
    updateSeverityBars();
  }
  if (d.src_ip && d.src_ip !== "\u2014") {
    talkerCounts[d.src_ip] = (talkerCounts[d.src_ip] || 0) + 1;
    updateTopTalkers();
  }

  const card = renderCard(d);

  if (feedScrolledAway) {
    list.prepend(card);
    pendingCount++;
    showNewPill();
  } else {
    list.prepend(card);
    list.scrollTop = 0;
  }

  while (list.children.length > MAX_FEED) {
    list.lastChild.remove();
  }
}

function initFeedScroll() {
  const list = document.getElementById("feed-list");
  list.addEventListener("scroll", () => {
    feedScrolledAway = list.scrollTop > 60;
    if (!feedScrolledAway) {
      pendingCount = 0;
      hideNewPill();
    }
  });
}

function showNewPill() {
  const pill = document.getElementById("new-pill");
  document.getElementById("new-pill-count").textContent = pendingCount;
  pill.classList.remove("hidden");
}

function hideNewPill() {
  document.getElementById("new-pill").classList.add("hidden");
}

function updateSeverityBars() {
  const container = document.getElementById("severity-bars");
  const total = Object.values(severityCounts).reduce((a, b) => a + b, 0);
  if (total === 0) return;

  let html = "";
  for (const sev of ["critical", "high", "medium", "low"]) {
    const count = severityCounts[sev] || 0;
    const pct = ((count / total) * 100).toFixed(1);
    html += `
      <div class="sev-row">
        <span class="sev-label">${sev}</span>
        <div class="sev-track"><div class="sev-fill" style="width:${pct}%;background:${SEVERITY_COLOR[sev]}"></div></div>
        <span class="sev-count">${count}</span>
      </div>`;
  }
  container.innerHTML = html;
}

function updateTopTalkers() {
  const container = document.getElementById("top-talkers");
  const sorted = Object.entries(talkerCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  if (sorted.length === 0) return;

  let html = "";
  sorted.forEach(([ip, count]) => {
    html += `<div class="talker-row"><span class="talker-ip">${esc(ip)}</span><span class="talker-count">${count}</span></div>`;
  });
  container.innerHTML = html;
}

function connectSSE() {
  const banner = document.getElementById("sse-banner");
  const source = new EventSource("/api/stream");

  source.onopen = () => {
    sseConnected = true;
    banner.classList.add("hidden");
    updateStatusDot("sb-sse", true);
    document.getElementById("sb-sse-label").textContent = "OK";
  };

  source.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.error) return;
      addCardToFeed(d);
      maybeNotify(d);
    } catch {
      // ignore parse errors
    }
  };

  source.onerror = () => {
    sseConnected = false;
    banner.classList.remove("hidden");
    updateStatusDot("sb-sse", false);
    document.getElementById("sb-sse-label").textContent = "down";
  };
}

const NOTIFY_COOLDOWN_MS = 3000;
let lastNotifyTime = 0;
let coalescedCount = 0;

function initNotifications() {
  if (!("Notification" in window)) return;
  if (Notification.permission === "granted") return;
  showNotifyButton();
}

function showNotifyButton() {
  const btn = document.getElementById("notify-btn");
  btn.classList.remove("hidden");
  btn.addEventListener("click", () => {
    if (Notification.permission === "denied") {
      alert(
        "Notifications are blocked for this site.\n\n" +
        "Click the lock/info icon in the address bar, find 'Notifications', " +
        "set it to 'Allow', and reload the page."
      );
      return;
    }
    Notification.requestPermission().then((perm) => {
      if (perm === "granted") {
        btn.classList.add("hidden");
        try {
          new Notification("NIDS Dashboard", {
            body: "Alerts enabled. You'll be notified of medium+ severity detections.",
          });
        } catch (e) {
          alert(
            "Permission granted, but the test notification didn't fire. " +
            "Check Windows Settings > System > Notifications and ensure your " +
            "browser is allowed to send notifications."
          );
        }
      } else if (perm === "denied") {
        alert(
          "Notifications were blocked. Reset the permission via the lock icon " +
          "in the address bar and try again."
        );
      }
    });
  });
}

function maybeNotify(d) {
  if (Notification.permission !== "granted") return;
  if ((SEVERITY_ORDER[d.severity] || 0) < SEVERITY_ORDER.medium) return;

  const now = Date.now();
  if (now - lastNotifyTime < NOTIFY_COOLDOWN_MS) {
    coalescedCount++;
    return;
  }

  lastNotifyTime = now;
  let body = `${d.src_ip} \u2192 ${d.dst_ip}  \u2022  ${Math.round(d.confidence * 100)}%`;
  if (coalescedCount > 0) {
    body += `\n+${coalescedCount} more detections`;
    coalescedCount = 0;
  }

  const n = new Notification(d.prediction, {
    body,
    icon: "/static/favicon.ico",
    tag: "nids-" + d.severity,
  });
  n.onclick = () => {
    window.focus();
    const card = document.querySelector(`.dcard[data-id="${d.id}"]`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
    n.close();
  };
}

async function seedFeed() {
  try {
    const res = await fetch("/api/recent_detections?limit=50");
    const records = await res.json();
    records.reverse().forEach((d) => addCardToFeed(d));
  } catch {
    // API not up yet - SSE will populate
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initFeedScroll();
  initNotifications();
  document.getElementById("new-pill").addEventListener("click", () => {
    const list = document.getElementById("feed-list");
    list.scrollTop = 0;
    pendingCount = 0;
    hideNewPill();
  });
  seedFeed();
  connectSSE();
  pollStats();
  setInterval(pollStats, 10000);
});
