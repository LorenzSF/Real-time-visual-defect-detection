"""Live XAI dashboard server for the streaming inference loop.

Implements PLAN.md §2.1 on top of the existing lightweight HTTP server.
The UI stays dependency-free on the frontend and consumes two JSON feeds:

* ``/api/bootstrap`` for static embedding-reference data
* ``/api/status`` for per-frame live metrics and artifacts
"""
from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Callable
from urllib.parse import unquote, urlparse


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>streaming_output</title>
  <style>
    :root {
      --bg: #edf1ea;
      --panel: #fffef8;
      --ink: #1e2623;
      --muted: #69756f;
      --line: #d5dbd5;
      --good: #1f9d55;
      --bad: #cb3a32;
      --warn: #d7a626;
      --accent: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 28%),
        linear-gradient(180deg, #eef4ef 0%, #f7f7f1 100%);
      color: var(--ink);
    }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 28px;
      letter-spacing: 0.01em;
    }
    .subtitle {
      color: var(--muted);
      margin-bottom: 16px;
    }
    .top-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1.12fr 0.88fr;
      gap: 12px;
      margin-bottom: 12px;
    }
    .bottom-grid {
      display: grid;
      grid-template-columns: 0.78fr 1.22fr;
      gap: 12px;
      margin-bottom: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 10px 28px rgba(17, 24, 21, 0.05);
      min-width: 0;
    }
    .card h2, .card h3 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .metric-value {
      font-size: 28px;
      font-weight: 700;
      line-height: 1.05;
    }
    .metric-sub {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }
    .score-value {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 12px;
    }
    .score-number {
      font-size: 38px;
      font-weight: 800;
      line-height: 1;
    }
    .score-threshold {
      color: var(--muted);
      font-size: 14px;
    }
    .gauge {
      position: relative;
      height: 20px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(31,157,85,0.16) 0%, rgba(215,166,38,0.18) 52%, rgba(203,58,50,0.18) 100%);
      border: 1px solid var(--line);
      overflow: hidden;
    }
    .gauge-fill {
      position: absolute;
      inset: 0 auto 0 0;
      border-radius: 999px;
    }
    .gauge-threshold {
      position: absolute;
      top: -3px;
      bottom: -3px;
      width: 2px;
      background: #1f2937;
      opacity: 0.85;
    }
    .gauge-labels {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .frame-shell {
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #eff3ee;
      overflow: hidden;
    }
    .frame-shell img {
      display: block;
      width: 100%;
      height: auto;
    }
    .empty {
      color: var(--muted);
      font-size: 14px;
    }
    .chart-box {
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fbfcf9;
      overflow: hidden;
    }
    svg {
      width: 100%;
      height: 100%;
      display: block;
    }
    .legend {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: -1px;
    }
    .legend .ref::before { background: #9aa59e; }
    .legend .live::before { background: linear-gradient(90deg, #1f9d55, #cb3a32); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
    .mono {
      font-family: Consolas, monospace;
      font-size: 12px;
      word-break: break-word;
    }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .pill-good {
      background: rgba(31,157,85,0.12);
      color: var(--good);
    }
    .pill-bad {
      background: rgba(203,58,50,0.12);
      color: var(--bad);
    }
    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    @media (max-width: 1180px) {
      .top-grid, .main-grid, .bottom-grid {
        grid-template-columns: 1fr 1fr;
      }
    }
    @media (max-width: 820px) {
      .top-grid, .main-grid, .bottom-grid {
        grid-template-columns: 1fr;
      }
      .frame-shell {
        min-height: 260px;
      }
      .chart-box {
        height: 220px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Streaming Inspection Dashboard</h1>
    <div class="subtitle">Live XAI monitoring for folder-based production simulation.</div>

    <section class="top-grid">
      <div class="card">
        <h2>Active Model</h2>
        <div class="metric-value" id="active-model">-</div>
        <div class="metric-sub" id="embedding-meta">Embedding: unavailable</div>
      </div>
      <div class="card">
        <h2>Inference Throughput</h2>
        <div class="metric-value" id="rolling-fps">0.00 FPS</div>
        <div class="metric-sub" id="mean-latency">Mean latency: 0.00 ms</div>
      </div>
      <div class="card">
        <h2>Anomaly Rate</h2>
        <div class="metric-value" id="anomaly-rate">0.0%</div>
        <div class="metric-sub" id="fail-count">0 anomalous decisions</div>
      </div>
      <div class="card">
        <h2>Frames Processed</h2>
        <div class="metric-value" id="frames-seen">0</div>
        <div class="metric-sub" id="decisions-emitted">0 decisions emitted</div>
      </div>
    </section>

    <section class="main-grid">
      <div class="card">
        <h2>Current Frame</h2>
        <div class="frame-shell" id="current-frame-shell">
          <div class="empty">Waiting for the first frame.</div>
        </div>
        <div class="status-row">
          <span id="current-frame-caption">Latest overlay updates each frame.</span>
          <span id="corruption-caption">Clean run</span>
        </div>
      </div>
      <div class="card">
        <h2>Anomaly Score</h2>
        <div class="score-value">
          <div class="score-number" id="score-number">0.0000</div>
          <div class="score-threshold" id="threshold-text">threshold 0.0000</div>
        </div>
        <div class="gauge">
          <div class="gauge-fill" id="gauge-fill"></div>
          <div class="gauge-threshold" id="gauge-threshold"></div>
        </div>
        <div class="gauge-labels">
          <span>0</span>
          <span id="gauge-max">0.0000</span>
        </div>
        <div class="status-row">
          <span id="processed-fps">Processed FPS: 0.00</span>
          <span id="decision-fps">Decision FPS: 0.00</span>
        </div>
      </div>
    </section>

    <section class="bottom-grid">
      <div class="card">
        <h2>Score History</h2>
        <div class="chart-box" id="score-history"></div>
        <div class="status-row">
          <span id="history-points">0 points</span>
          <span id="history-threshold">Threshold line shown in dark green.</span>
        </div>
      </div>
      <div class="card">
        <h2>Live Embedding Plot</h2>
        <div class="chart-box" id="embedding-plot"></div>
        <div class="legend">
          <span class="ref">Training reference cloud</span>
          <span class="live">Live frames colored by score</span>
        </div>
      </div>
    </section>

    <section class="card">
      <h3>Recent Decisions</h3>
      <table>
        <thead>
          <tr>
            <th>Frame</th>
            <th>Score</th>
            <th>Decision</th>
            <th>Defect Type</th>
            <th>Path</th>
          </tr>
        </thead>
        <tbody id="recent-decisions"></tbody>
      </table>
    </section>
  </main>

  <script>
    let bootstrapData = {
      enabled: false,
      projection: "pca",
      source: "unavailable",
      reference_points: [],
      axis: { min_x: -1, max_x: 1, min_y: -1, max_y: 1 },
      refresh_ms: 1000
    };

    function safeNumber(value, fallback = 0) {
      return Number.isFinite(Number(value)) ? Number(value) : fallback;
    }

    function formatNumber(value, digits = 2) {
      return safeNumber(value).toFixed(digits);
    }

    function formatPercent(value) {
      return (safeNumber(value) * 100).toFixed(1) + "%";
    }

    function scoreColor(score, threshold) {
      const denom = Math.max(safeNumber(threshold), 1e-6);
      const ratio = Math.max(0, Math.min(1, safeNumber(score) / denom));
      const r = Math.round(31 + ratio * (203 - 31));
      const g = Math.round(157 + ratio * (58 - 157));
      const b = Math.round(85 + ratio * (50 - 85));
      return `rgb(${r}, ${g}, ${b})`;
    }

    function fileUrl(relativePath, version) {
      const clean = String(relativePath || "").replaceAll("\\\\", "/");
      return "/session/" + clean + "?v=" + encodeURIComponent(String(version || 0));
    }

    function renderTopMetrics(data) {
      document.getElementById("active-model").textContent = data.active_model || "-";
      document.getElementById("embedding-meta").textContent =
        `Embedding: ${data.embedding_source || bootstrapData.source || "unavailable"} · ${data.embedding_projection || bootstrapData.projection || "pca"}`;
      document.getElementById("rolling-fps").textContent = `${formatNumber(data.rolling_fps_10, 2)} FPS`;
      document.getElementById("mean-latency").textContent =
        `Mean latency: ${formatNumber(data.mean_latency_ms, 2)} ms`;
      document.getElementById("anomaly-rate").textContent = formatPercent(data.anomaly_rate);
      document.getElementById("fail-count").textContent = `${safeNumber(data.fail_count, 0)} anomalous decisions`;
      document.getElementById("frames-seen").textContent = String(safeNumber(data.frames_seen, 0));
      document.getElementById("decisions-emitted").textContent =
        `${safeNumber(data.decisions_emitted, 0)} decisions emitted`;
    }

    function renderCurrentFrame(data) {
      const shell = document.getElementById("current-frame-shell");
      const currentPath = data.current_frame_path;
      if (!currentPath) {
        shell.innerHTML = `<div class="empty">Waiting for the first frame.</div>`;
      } else {
        shell.innerHTML =
          `<img src="${fileUrl(currentPath, data.current_frame_version)}" alt="Current frame with heatmap overlay">`;
      }
      document.getElementById("current-frame-caption").textContent =
        currentPath ? `Frame #${safeNumber(data.current_frame_version, 0)} · overlay refreshed live` : "Latest overlay updates each frame.";
      const corrType = data.corruption_type || "";
      document.getElementById("corruption-caption").textContent =
        corrType ? `${corrType} · severity ${safeNumber(data.severity, 0)}` : "Clean run";
    }

    function renderGauge(data) {
      const score = safeNumber(data.latest_score, 0);
      const threshold = Math.max(safeNumber(data.threshold, 0), 1e-6);
      const axisMax = Math.max(safeNumber(data.score_axis_max, threshold), threshold);
      const fillPct = Math.max(0, Math.min(100, (score / axisMax) * 100));
      const thresholdPct = Math.max(0, Math.min(100, (threshold / axisMax) * 100));
      document.getElementById("score-number").textContent = formatNumber(score, 4);
      document.getElementById("score-number").style.color = scoreColor(score, threshold);
      document.getElementById("threshold-text").textContent = `threshold ${formatNumber(threshold, 4)}`;
      document.getElementById("gauge-max").textContent = formatNumber(axisMax, 4);

      const fill = document.getElementById("gauge-fill");
      fill.style.width = `${fillPct}%`;
      fill.style.background = scoreColor(score, threshold);

      const marker = document.getElementById("gauge-threshold");
      marker.style.left = `calc(${thresholdPct}% - 1px)`;

      document.getElementById("processed-fps").textContent =
        `Processed FPS: ${formatNumber(data.processed_fps, 2)}`;
      document.getElementById("decision-fps").textContent =
        `Decision FPS: ${formatNumber(data.decision_fps, 2)}`;
    }

    function buildLineSvg(scores, threshold, axisMax) {
      const width = 800;
      const height = 260;
      const pad = 18;
      const innerW = width - pad * 2;
      const innerH = height - pad * 2;
      const yMax = Math.max(axisMax, threshold, 1e-6);
      const history = (scores || []).map((v) => safeNumber(v));

      const grid = [];
      for (let i = 0; i < 5; i++) {
        const y = pad + (innerH * i / 4);
        grid.push(`<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="#dbe2dc" stroke-width="1"/>`);
      }

      const thresholdY = pad + innerH - (Math.max(0, Math.min(threshold / yMax, 1)) * innerH);
      let polyline = "";
      if (history.length > 0) {
        polyline = history.map((value, idx) => {
          const x = pad + (history.length === 1 ? innerW / 2 : (innerW * idx / (history.length - 1)));
          const y = pad + innerH - (Math.max(0, Math.min(value / yMax, 1)) * innerH);
          return `${x},${y}`;
        }).join(" ");
      }

      return `
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="score history chart">
          <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfcf9"/>
          ${grid.join("")}
          <line x1="${pad}" y1="${thresholdY}" x2="${width - pad}" y2="${thresholdY}" stroke="#14532d" stroke-width="2" stroke-dasharray="6 4"/>
          ${polyline ? `<polyline fill="none" stroke="#0f766e" stroke-width="3" points="${polyline}"/>` : ""}
        </svg>
      `;
    }

    function renderScoreHistory(data) {
      const scores = data.score_history || [];
      const threshold = safeNumber(data.threshold, 0);
      const axisMax = Math.max(safeNumber(data.score_axis_max, 0), threshold, 1e-6);
      document.getElementById("score-history").innerHTML = buildLineSvg(scores, threshold, axisMax);
      document.getElementById("history-points").textContent = `${scores.length} points`;
    }

    function buildEmbeddingSvg(referencePoints, livePoints, axis) {
      const width = 800;
      const height = 260;
      const pad = 18;
      const innerW = width - pad * 2;
      const innerH = height - pad * 2;

      const minX = safeNumber(axis.min_x, -1);
      const maxX = safeNumber(axis.max_x, 1);
      const minY = safeNumber(axis.min_y, -1);
      const maxY = safeNumber(axis.max_y, 1);

      function sx(x) {
        const denom = Math.max(maxX - minX, 1e-6);
        return pad + ((x - minX) / denom) * innerW;
      }
      function sy(y) {
        const denom = Math.max(maxY - minY, 1e-6);
        return pad + innerH - ((y - minY) / denom) * innerH;
      }

      const grid = [];
      for (let i = 0; i < 5; i++) {
        const x = pad + (innerW * i / 4);
        const y = pad + (innerH * i / 4);
        grid.push(`<line x1="${x}" y1="${pad}" x2="${x}" y2="${height - pad}" stroke="#e1e7e1" stroke-width="1"/>`);
        grid.push(`<line x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}" stroke="#e1e7e1" stroke-width="1"/>`);
      }

      const refDots = (referencePoints || []).map((point) =>
        `<circle cx="${sx(safeNumber(point.x))}" cy="${sy(safeNumber(point.y))}" r="2.5" fill="#9aa59e" opacity="0.55">
          <title>${String(point.path || "reference")}</title>
        </circle>`
      ).join("");

      const liveDots = (livePoints || []).map((point) => {
        const color = scoreColor(point.score_ratio, 1.0);
        return `<circle cx="${sx(safeNumber(point.x))}" cy="${sy(safeNumber(point.y))}" r="4.2" fill="${color}" stroke="#13201b" stroke-width="0.5" opacity="0.92">
          <title>${String(point.path || "frame")} · score ${formatNumber(point.score, 4)}</title>
        </circle>`;
      }).join("");

      return `
        <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-label="embedding scatter plot">
          <rect x="0" y="0" width="${width}" height="${height}" fill="#fbfcf9"/>
          ${grid.join("")}
          ${refDots}
          ${liveDots}
        </svg>
      `;
    }

    function renderEmbedding(data) {
      const plot = document.getElementById("embedding-plot");
      if (!data.embedding_enabled) {
        plot.innerHTML = `<div class="empty" style="padding:16px;">Embedding projection unavailable for this session.</div>`;
        return;
      }
      const referencePoints = bootstrapData.reference_points || [];
      const livePoints = data.embedding_live_points || [];
      const axis = data.embedding_axis || bootstrapData.axis || { min_x: -1, max_x: 1, min_y: -1, max_y: 1 };
      plot.innerHTML = buildEmbeddingSvg(referencePoints, livePoints, axis);
    }

    function renderDecisions(data) {
      const body = document.getElementById("recent-decisions");
      const rows = data.recent_decisions || [];
      body.innerHTML = "";
      for (const row of rows) {
        const decision = row.pred_is_anomaly === 1
          ? `<span class="pill pill-bad">FAIL</span>`
          : `<span class="pill pill-good">PASS</span>`;
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${safeNumber(row.current_frame_version || row.frame_id || 0, 0)}</td>
          <td style="color:${scoreColor(row.score, Math.max(safeNumber(data.threshold, 0), 1e-6))}; font-weight:700;">${formatNumber(row.score, 4)}</td>
          <td>${decision}</td>
          <td>${row.defect_type || "-"}</td>
          <td class="mono">${row.path || ""}</td>
        `;
        body.appendChild(tr);
      }
    }

    async function loadBootstrap() {
      const response = await fetch("/api/bootstrap", { cache: "no-store" });
      bootstrapData = await response.json();
    }

    async function refresh() {
      const response = await fetch("/api/status", { cache: "no-store" });
      const data = await response.json();
      renderTopMetrics(data);
      renderCurrentFrame(data);
      renderGauge(data);
      renderScoreHistory(data);
      renderEmbedding(data);
      renderDecisions(data);
    }

    async function main() {
      await loadBootstrap();
      await refresh();
      setInterval(refresh, Math.max(100, safeNumber(bootstrapData.refresh_ms, 1000)));
    }

    main();
  </script>
</body>
</html>
"""


def _guess_content_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


class _DashboardHandler(BaseHTTPRequestHandler):
    server_version = "streaming_output/0.2"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            payload = json.dumps(self.server.status_provider(), indent=2).encode("utf-8")
            self._write_bytes(payload, "application/json; charset=utf-8")
            return
        if parsed.path == "/api/bootstrap":
            payload = json.dumps(self.server.bootstrap_provider(), indent=2).encode("utf-8")
            self._write_bytes(payload, "application/json; charset=utf-8")
            return
        if parsed.path.startswith("/session/"):
            relative = unquote(parsed.path[len("/session/"):])
            self._serve_session_file(relative)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _serve_session_file(self, relative_path: str) -> None:
        session_dir = self.server.session_dir.resolve()
        target = (session_dir / relative_path).resolve()
        try:
            target.relative_to(session_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid path")
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content = target.read_bytes()
        self._write_bytes(content, _guess_content_type(target))

    def _write_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        session_dir: Path,
        status_provider: Callable[[], dict],
        bootstrap_provider: Callable[[], dict],
    ) -> None:
        super().__init__(address, _DashboardHandler)
        self.session_dir = Path(session_dir)
        self.status_provider = status_provider
        self.bootstrap_provider = bootstrap_provider


class LiveDashboardServer:
    """Thin wrapper that runs the HTTP dashboard in a daemon thread."""

    def __init__(
        self,
        host: str,
        port: int,
        session_dir: Path,
        status_provider: Callable[[], dict],
        bootstrap_provider: Callable[[], dict],
    ) -> None:
        self.host = host
        self.port = int(port)
        self._server = _DashboardServer(
            (self.host, self.port),
            session_dir=Path(session_dir),
            status_provider=status_provider,
            bootstrap_provider=bootstrap_provider,
        )
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
