/**
 * IRIS Pipeline — React Frontend (IrisApp.jsx)
 * =============================================
 * Connects to the FastAPI backend at http://localhost:8000/api/process.
 * Sends an MP4 file + text query, renders the full JSON response
 * including a live knowledge-graph visualization of the L2 Asphodel graph.
 *
 * HOW TO RUN
 * ----------
 * Terminal 1 — FastAPI backend (from the IRIS project root):
 *     uvicorn api:app --reload --host 0.0.0.0 --port 8000
 *
 * Terminal 2 — React dev server (from iris-frontend/):
 *     npm run dev
 *
 * Open http://localhost:5173 (or 5174) in your browser.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import {
  Upload, Play, CheckCircle2, XCircle, Cpu, Layers, Clock,
  Activity, Zap, Database, Shield, BarChart3, FileVideo,
  AlertTriangle, Network, GitBranch, Info,
} from "lucide-react";

// ─── API Config ───────────────────────────────────────────────────────────────
const API_BASE_URL = import.meta.env.VITE_IRIS_API_URL || "http://localhost:8000";
const API_URL = `${API_BASE_URL.replace(/\/$/, "")}/api/process`;
const HEALTH_URL = `${API_BASE_URL.replace(/\/$/, "")}/health`;

// ─── Timing Labels ────────────────────────────────────────────────────────────
const TIMING_META = [
  { key: "charon_v",     label: "Charon-V",     icon: <Layers size={14} /> },
  { key: "action_score", label: "Action Score", icon: <Activity size={14} /> },
  { key: "l2_retrieval", label: "L2 Asphodel",  icon: <Database size={14} /> },
  { key: "elysium",      label: "L1 Elysium",   icon: <Cpu size={14} /> },
  { key: "aria",         label: "ARIA",          icon: <Zap size={14} /> },
  { key: "cerberus_v",   label: "Cerberus-V",   icon: <Shield size={14} /> },
  { key: "total",        label: "TOTAL",         icon: <Clock size={14} />, isTotal: true },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatMs(seconds) {
  if (seconds === undefined || seconds === null) return "—";
  return seconds < 1
    ? `${(seconds * 1000).toFixed(1)} ms`
    : `${seconds.toFixed(2)} s`;
}
function formatPct(ratio) {
  if (ratio === undefined || ratio === null) return "—";
  return `${(ratio * 100).toFixed(1)}%`;
}

// ─── Canvas Knowledge Graph ───────────────────────────────────────────────────
function KnowledgeGraph({ graphData }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const nodesRef = useRef([]);
  const edgesRef = useRef([]);
  const hoveredRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);

  const { nodes: rawNodes = [], edges: rawEdges = [] } = graphData || {};

  // Layout: place nodes in a circle, then simulate spring physics
  useEffect(() => {
    if (!rawNodes.length) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const W = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    const H = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    canvas.style.width = canvas.offsetWidth + "px";
    canvas.style.height = canvas.offsetHeight + "px";
    const cx = W / 2, cy = H / 2;
    const radius = Math.min(W, H) * 0.33;

    // Init node positions in a circle
    nodesRef.current = rawNodes.map((n, i) => {
      const angle = (i / rawNodes.length) * 2 * Math.PI - Math.PI / 2;
      return {
        ...n,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
        vx: 0,
        vy: 0,
      };
    });
    edgesRef.current = rawEdges;

    // Simple force simulation
    const K_REPEL = 8000;
    const K_SPRING = 0.03;
    const REST_LEN = radius * 0.75;
    const DAMP = 0.88;

    let frame = 0;
    const MAX_FRAMES = 200; // run layout for N frames then freeze

    function simulate() {
      const ns = nodesRef.current;
      if (frame < MAX_FRAMES) {
        // Repulsion
        for (let i = 0; i < ns.length; i++) {
          for (let j = i + 1; j < ns.length; j++) {
            const dx = ns[j].x - ns[i].x || 0.1;
            const dy = ns[j].y - ns[i].y || 0.1;
            const dist2 = dx * dx + dy * dy;
            const dist = Math.sqrt(dist2) || 0.1;
            const force = K_REPEL / dist2;
            ns[i].vx -= (dx / dist) * force;
            ns[i].vy -= (dy / dist) * force;
            ns[j].vx += (dx / dist) * force;
            ns[j].vy += (dy / dist) * force;
          }
        }
        // Spring attraction along edges
        for (const e of edgesRef.current) {
          const u = ns.find((n) => n.id === e.source);
          const v = ns.find((n) => n.id === e.target);
          if (!u || !v) continue;
          const dx = v.x - u.x;
          const dy = v.y - u.y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
          const stretch = dist - REST_LEN * (e.cross ? 1.6 : 1);
          const fx = (dx / dist) * K_SPRING * stretch;
          const fy = (dy / dist) * K_SPRING * stretch;
          u.vx += fx; u.vy += fy;
          v.vx -= fx; v.vy -= fy;
        }
        // Center gravity
        for (const n of ns) {
          n.vx += (cx - n.x) * 0.004;
          n.vy += (cy - n.y) * 0.004;
          n.vx *= DAMP; n.vy *= DAMP;
          n.x += n.vx; n.y += n.vy;
        }
        frame++;
      }
      draw();
      animRef.current = requestAnimationFrame(simulate);
    }

    function draw() {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, W, H);

      // Draw edges
      for (const e of edgesRef.current) {
        const u = nodesRef.current.find((n) => n.id === e.source);
        const v = nodesRef.current.find((n) => n.id === e.target);
        if (!u || !v) continue;
        const alpha = 0.15 + e.weight * 0.55;
        ctx.save();
        if (e.cross) {
          ctx.setLineDash([6, 8]);
          ctx.strokeStyle = `rgba(139,92,246,${alpha * 0.7})`;
        } else {
          ctx.setLineDash([]);
          ctx.strokeStyle = `rgba(59,130,246,${alpha})`;
        }
        ctx.lineWidth = Math.max(1, e.weight * 3) * window.devicePixelRatio;
        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);
        ctx.stroke();
        // Weight label on edge midpoint
        if (e.weight > 0.5) {
          ctx.font = `${10 * window.devicePixelRatio}px JetBrains Mono, monospace`;
          ctx.fillStyle = `rgba(148,163,184,${alpha})`;
          ctx.textAlign = "center";
          ctx.fillText(e.label, (u.x + v.x) / 2, (u.y + v.y) / 2 - 4 * window.devicePixelRatio);
        }
        ctx.restore();
      }

      // Draw nodes
      const DPR = window.devicePixelRatio;
      for (const n of nodesRef.current) {
        const isHovered = hoveredRef.current === n.id;
        const radius = (n.is_peak ? 22 : 16) * DPR;
        const score = n.action_score || 0;

        // Glow for peaks
        if (n.is_peak || isHovered) {
          ctx.save();
          ctx.shadowColor = n.is_peak ? "#22d3a3" : "#3b82f6";
          ctx.shadowBlur = 20 * DPR;
          ctx.beginPath();
          ctx.arc(n.x, n.y, radius + 4 * DPR, 0, Math.PI * 2);
          ctx.fillStyle = n.is_peak ? "rgba(34,211,163,0.12)" : "rgba(59,130,246,0.12)";
          ctx.fill();
          ctx.restore();
        }

        // Node circle — color by action_score
        const hue = Math.round(220 - score * 130); // blue→green
        const sat = 70 + score * 25;
        const lit = 45 + score * 20;
        ctx.save();
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(n.x - radius * 0.3, n.y - radius * 0.3, 0, n.x, n.y, radius);
        grad.addColorStop(0, `hsl(${hue},${sat}%,${lit + 20}%)`);
        grad.addColorStop(1, `hsl(${hue},${sat}%,${lit}%)`);
        ctx.fillStyle = grad;
        ctx.strokeStyle = n.is_peak ? "#22d3a3" : (isHovered ? "#3b82f6" : "rgba(255,255,255,0.15)");
        ctx.lineWidth = (n.is_peak ? 2.5 : 1.5) * DPR;
        ctx.fill();
        ctx.stroke();
        ctx.restore();

        // Frame label
        ctx.save();
        ctx.font = `bold ${9 * DPR}px JetBrains Mono, monospace`;
        ctx.fillStyle = "#e2e8f0";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`F${n.frame_idx}`, n.x, n.y);
        ctx.restore();

        // Timestamp below node
        ctx.save();
        ctx.font = `${8 * DPR}px Inter, sans-serif`;
        ctx.fillStyle = "rgba(148,163,184,0.75)";
        ctx.textAlign = "center";
        ctx.fillText(`${n.timestamp.toFixed(2)}s`, n.x, n.y + radius + 10 * DPR);
        ctx.restore();
      }
    }

    animRef.current = requestAnimationFrame(simulate);
    return () => cancelAnimationFrame(animRef.current);
  }, [rawNodes, rawEdges]);

  // Resize handler
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  // Mouse hover for tooltip
  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas || !nodesRef.current.length) return;
    const rect = canvas.getBoundingClientRect();
    const DPR = window.devicePixelRatio;
    const mx = (e.clientX - rect.left) * DPR;
    const my = (e.clientY - rect.top) * DPR;
    let found = null;
    for (const n of nodesRef.current) {
      const r = (n.is_peak ? 22 : 16) * DPR;
      const dx = mx - n.x, dy = my - n.y;
      if (dx * dx + dy * dy < r * r) { found = n; break; }
    }
    hoveredRef.current = found ? found.id : null;
    if (found) {
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        node: found,
      });
    } else {
      setTooltip(null);
    }
  }, []);

  if (!rawNodes.length) {
    return (
      <div className="graph-empty">
        <Network size={40} />
        <span>No graph data yet. Run the pipeline to see the knowledge graph.</span>
      </div>
    );
  }

  return (
    <div className="graph-container" style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        className="graph-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => { hoveredRef.current = null; setTooltip(null); }}
      />
      {tooltip && (
        <div
          className="graph-tooltip"
          style={{ left: tooltip.x + 16, top: tooltip.y - 20 }}
        >
          <div className="gt-title">Frame {tooltip.node.frame_idx}</div>
          <div className="gt-row"><span>Timestamp</span><span>{tooltip.node.timestamp.toFixed(3)}s</span></div>
          <div className="gt-row"><span>Action Score</span><span>{tooltip.node.action_score.toFixed(4)}</span></div>
          <div className="gt-row"><span>Persistence</span><span>{tooltip.node.persistence_value.toFixed(4)}</span></div>
          <div className="gt-row"><span>Residual Energy</span><span>{tooltip.node.residual_energy.toFixed(4)}</span></div>
          <div className="gt-row"><span>Entropy</span><span>{tooltip.node.entropy.toFixed(4)}</span></div>
          <div className="gt-row"><span>Is Peak</span><span className={tooltip.node.is_peak ? "gt-peak" : "gt-nopeak"}>{tooltip.node.is_peak ? "✓ YES" : "NO"}</span></div>
          {tooltip.node.caption && tooltip.node.caption !== "—" && (
            <div className="gt-caption">{tooltip.node.caption}</div>
          )}
        </div>
      )}
      <div className="graph-legend">
        <span className="gl-item"><span className="gl-dot peak" />Peak Frame</span>
        <span className="gl-item"><span className="gl-dot regular" />Retrieved Frame</span>
        <span className="gl-item"><span className="gl-edge solid" />Motion Edge</span>
        <span className="gl-item"><span className="gl-edge dashed" />Cross-Link</span>
      </div>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function DropZone({ onFileSelect, selectedFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFileSelect(f);
  }, [onFileSelect]);

  return (
    <div
      id="dropzone"
      className={`dropzone ${dragging ? "dragging" : ""} ${selectedFile ? "has-file" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button" tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
    >
      <input
        ref={inputRef} id="file-input" type="file"
        accept="video/mp4,video/quicktime,video/avi,video/webm,.mkv"
        style={{ display: "none" }}
        onChange={(e) => e.target.files[0] && onFileSelect(e.target.files[0])}
      />
      {selectedFile ? (
        <div className="dropzone-file-info">
          <FileVideo size={36} className="drop-icon accent" />
          <span className="file-name">{selectedFile.name}</span>
          <span className="file-size">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
          <span className="change-hint">Click or drag to change</span>
        </div>
      ) : (
        <div className="dropzone-empty">
          <Upload size={36} className="drop-icon" />
          <span className="drop-label">Drop your video here</span>
          <span className="drop-sub">MP4, MOV, AVI, MKV, WEBM · click to browse</span>
        </div>
      )}
    </div>
  );
}

function PulsingLoader() {
  return (
    <div id="loading-indicator" className="loader-wrap">
      <div className="loader-orb-ring"><div className="loader-orb" /></div>
      <div className="loader-text">
        <span className="loader-title">Processing Pipeline</span>
        <span className="loader-sub">
          Charon-V → Action Score → L2 Asphodel → L1 Elysium → ARIA → Cerberus-V
        </span>
        <div className="loader-stages">
          {["Decoding frames","Scoring actions","Graph retrieval","Cache population","Generating answer","Verifying claims"].map(
            (s, i) => <span key={i} className="loader-stage" style={{ animationDelay: `${i * 0.4}s` }}>{s}</span>
          )}
        </div>
      </div>
    </div>
  );
}

function CerberusGate({ verified, mocked }) {
  return (
    <div id="cerberus-gate" className={`cerberus-gate ${verified ? "verified" : "rejected"}`}>
      <div className="cerberus-icon">
        {verified ? <CheckCircle2 size={28} /> : <XCircle size={28} />}
      </div>
      <div className="cerberus-content">
        <span className="cerberus-label">Cerberus Gate</span>
        <span className={`cerberus-badge ${verified ? "badge-verified" : "badge-rejected"}`}>
          {verified ? "VERIFIED" : "CONTRADICTION DETECTED"}
        </span>
        {mocked && (
          <span className="cerberus-mocked">
            <AlertTriangle size={12} /> NLI fallback active
          </span>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, accent }) {
  return (
    <div className={`stat-card ${accent ? "accent-card" : ""}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-body">
        <span className="stat-label">{label}</span>
        <span className="stat-value">{value}</span>
      </div>
    </div>
  );
}

function TimingRow({ icon, label, value, isTotal }) {
  return (
    <div className={`timing-row ${isTotal ? "timing-total" : ""}`}>
      <span className="timing-icon">{icon}</span>
      <span className="timing-label">{label}</span>
      <span className="timing-value">{formatMs(value)}</span>
    </div>
  );
}

// ─── API Status Indicator ─────────────────────────────────────────────────────
function ApiStatusBadge() {
  const [status, setStatus] = useState("checking"); // "ok" | "error" | "checking"
  useEffect(() => {
    const check = async () => {
      try {
        await axios.get(HEALTH_URL, { timeout: 3000 });
        setStatus("ok");
      } catch {
        setStatus("error");
      }
    };
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={`api-status api-status--${status}`} title="Backend API status">
      <span className={`pulse-dot ${status === "ok" ? "pulse-green" : status === "error" ? "pulse-red" : "pulse-dim"}`} />
      {status === "ok" ? "API ONLINE" : status === "error" ? "API OFFLINE" : "CHECKING…"}
    </div>
  );
}

// ─── Graph Stats panel ────────────────────────────────────────────────────────
function GraphStats({ graphData }) {
  if (!graphData || !graphData.nodes.length) return null;
  const peaks = graphData.nodes.filter(n => n.is_peak).length;
  const avgScore = (graphData.nodes.reduce((a, n) => a + n.action_score, 0) / graphData.nodes.length).toFixed(3);
  return (
    <div className="graph-stats-row">
      <span className="gs-item"><GitBranch size={12} /> {graphData.nodes.length} nodes</span>
      <span className="gs-sep">·</span>
      <span className="gs-item">{graphData.edges.length} edges</span>
      <span className="gs-sep">·</span>
      <span className="gs-item gs-peak">{peaks} peaks</span>
      <span className="gs-sep">·</span>
      <span className="gs-item">avg score {avgScore}</span>
    </div>
  );
}

// ─── Debug Claims Panel ───────────────────────────────────────────────────────
function ClaimsPanel({ debugInfo }) {
  const [open, setOpen] = useState(false);
  if (!debugInfo) return null;
  const verified = debugInfo.verified_claims || [];
  const rejected = debugInfo.rejected_claims || [];
  if (!verified.length && !rejected.length) return null;
  return (
    <div className="claims-panel">
      <button className="claims-toggle" onClick={() => setOpen(o => !o)}>
        <Info size={13} /> Claim Analysis ({verified.length} verified, {rejected.length} rejected)
        <span className="claims-arrow">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="claims-body">
          {verified.map((c, i) => (
            <div key={`v${i}`} className="claim claim-ok"><CheckCircle2 size={12} />{c}</div>
          ))}
          {rejected.map((c, i) => (
            <div key={`r${i}`} className="claim claim-bad"><XCircle size={12} />{c}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Results Panel ────────────────────────────────────────────────────────────
function FrameDetailsCard({ frame }) {
  if (!frame) {
    return (
      <div className="frame-detail-card" style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "150px" }}>
        <span className="text-secondary" style={{ fontSize: "13px" }}>Click on any point in the Action Score timeline chart below to inspect frame details.</span>
      </div>
    );
  }

  return (
    <div className="frame-detail-card">
      <div className="frame-detail-thumb">
        {frame.thumbnail ? (
          <img src={frame.thumbnail} alt={`Frame ${frame.frame_idx}`} />
        ) : (
          <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100%", color: "var(--text-secondary)" }}>No Thumbnail</div>
        )}
      </div>
      <div className="frame-detail-info">
        <div className="frame-detail-header">
          <span className="frame-title">Frame {frame.frame_idx}</span>
          <span className={`frame-badge badge-${frame.is_peak ? "peak" : (frame.label === "HIGH_IMPORTANCE" ? "high" : (frame.label === "MEDIUM_IMPORTANCE" ? "medium" : (frame.label === "LOW_IMPORTANCE" ? "low" : "bg")))}`}>
            {frame.label}
          </span>
        </div>
        <div className="frame-card-time" style={{ fontSize: "11px", color: "var(--text-secondary)", marginBottom: "4px" }}>
          Timestamp: {frame.timestamp.toFixed(3)}s · Type: {frame.frame_type || "P"}
        </div>
        
        <div className="frame-metrics-grid" style={{ marginBottom: "8px" }}>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl" style={{ fontSize: "9px" }}>Action Score</div>
            <div className="frame-metric-val">{frame.action_score.toFixed(4)}</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl" style={{ fontSize: "9px" }}>Persistence</div>
            <div className="frame-metric-val">{frame.persistence_value.toFixed(4)}</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl" style={{ fontSize: "9px" }}>Residual Energy</div>
            <div className="frame-metric-val">{frame.residual_energy.toFixed(4)}</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl" style={{ fontSize: "9px" }}>Motion Mag</div>
            <div className="frame-metric-val">{frame.motion_magnitude?.toFixed(4) || "0.0000"}</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl" style={{ fontSize: "9px" }}>Entropy</div>
            <div className="frame-metric-val">{frame.entropy.toFixed(4)}</div>
          </div>
        </div>

        <div className="frame-reasons">
          <div className="frame-reasons-title">Pipeline Triage Decisions:</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {frame.reasons && frame.reasons.map((r, i) => (
              <div key={i} className="reason-item" style={{ fontSize: "12px" }}>
                <span style={{ color: frame.selected ? "var(--neon-green)" : "var(--neon-red)", marginRight: "6px" }}>▶</span> {r}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function SVGTimelineChart({ allFrames, selectedIdx, onPointClick }) {
  if (!allFrames || !allFrames.length) return null;
  
  const width = 1000;
  const height = 120;
  const paddingLeft = 30;
  const paddingRight = 10;
  const paddingTop = 10;
  const paddingBottom = 20;

  const scores = allFrames.map(f => f.action_score || 0);
  const maxScore = Math.max(...scores, 1.0); 

  const xScale = (width - paddingLeft - paddingRight) / (allFrames.length - 1 || 1);
  const yScale = (height - paddingTop - paddingBottom) / maxScore;

  const pathD = scores.map((s, i) => {
    const x = paddingLeft + i * xScale;
    const y = height - paddingBottom - s * yScale;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  return (
    <div className="svg-chart-wrapper">
      <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart" style={{ overflow: "visible" }}>
        <line x1={paddingLeft} y1={paddingTop} x2={width - paddingRight} y2={paddingTop} stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
        <line x1={paddingLeft} y1={(height - paddingTop - paddingBottom) / 2 + paddingTop} x2={width - paddingRight} y2={(height - paddingTop - paddingBottom) / 2 + paddingTop} stroke="rgba(255,255,255,0.05)" strokeDasharray="3,3" />
        <line x1={paddingLeft} y1={height - paddingBottom} x2={width - paddingRight} y2={height - paddingBottom} stroke="rgba(255,255,255,0.15)" />

        <text x={paddingLeft - 8} y={paddingTop + 4} fill="rgba(148,163,184,0.5)" fontSize="9" textAnchor="end">1.0</text>
        <text x={paddingLeft - 8} y={(height - paddingTop - paddingBottom) / 2 + paddingTop + 4} fill="rgba(148,163,184,0.5)" fontSize="9" textAnchor="end">0.5</text>
        <text x={paddingLeft - 8} y={height - paddingBottom + 4} fill="rgba(148,163,184,0.5)" fontSize="9" textAnchor="end">0.0</text>

        <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth="2.5" />

        {allFrames.map((f, i) => {
          const x = paddingLeft + i * xScale;
          const y = height - paddingBottom - f.action_score * yScale;
          const isSelected = selectedIdx === f.frame_idx;
          
          if (f.is_peak || isSelected) {
            return (
              <g key={i} onClick={() => onPointClick(f)} style={{ cursor: "pointer" }}>
                <circle
                  cx={x}
                  cy={y}
                  r={isSelected ? 6 : 4.5}
                  fill={isSelected ? "var(--neon-red)" : (f.is_peak ? "var(--neon-green)" : "var(--accent)")}
                  stroke="#fff"
                  strokeWidth={isSelected ? 2 : 1.2}
                />
                <title>{`Frame ${f.frame_idx}: Score ${f.action_score.toFixed(3)} (Label: ${f.label})`}</title>
              </g>
            );
          }
          if (i % 2 === 0) {
            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r="6"
                fill="transparent"
                style={{ cursor: "pointer" }}
                onClick={() => onPointClick(f)}
              />
            );
          }
          return null;
        })}
      </svg>
    </div>
  );
}

function ExtractedFramesGrid({
  allFrames,
  filterLabel,
  setFilterLabel,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder,
  onFrameClick,
  selectedIdx
}) {
  const [showGuide, setShowGuide] = useState(false);

  const filtered = allFrames.filter(f => {
    if (filterLabel === "ALL") return true;
    return f.label === filterLabel;
  });

  const sorted = [...filtered].sort((a, b) => {
    let valA = a[sortBy];
    let valB = b[sortBy];
    return sortOrder === "asc" ? valA - valB : valB - valA;
  });

  return (
    <div>
      <div className="filter-bar">
        <div className="filter-group">
          <span className="filter-label">Filter:</span>
          <select
            className="filter-select"
            value={filterLabel}
            onChange={(e) => setFilterLabel(e.target.value)}
          >
            <option value="ALL">All Frames</option>
            <option value="PEAK">PEAK</option>
            <option value="HIGH_IMPORTANCE">HIGH IMPORTANCE</option>
            <option value="MEDIUM_IMPORTANCE">MEDIUM IMPORTANCE</option>
            <option value="LOW_IMPORTANCE">LOW IMPORTANCE</option>
            <option value="BACKGROUND">BACKGROUND</option>
          </select>
        </div>

        <div className="filter-group">
          <span className="filter-label">Sort By:</span>
          <select
            className="filter-select"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          >
            <option value="frame_idx">Frame Index</option>
            <option value="action_score">Action Score</option>
            <option value="persistence_value">Persistence Value</option>
          </select>
        </div>

        <div className="filter-group">
          <span className="filter-label">Order:</span>
          <select
            className="filter-select"
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
          >
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>
        </div>

        <span className="range-guide-badge" onClick={() => setShowGuide(g => !g)} style={{ marginLeft: "auto", cursor: "pointer", textDecoration: "underline", display: "flex", alignItems: "center", gap: "4px" }}>
          <Info size={12} /> Label Range Logic {showGuide ? "▲" : "▼"}
        </span>
      </div>

      {showGuide && (
        <div className="glass-card" style={{ padding: "15px", marginBottom: "20px", fontSize: "12px", borderLeft: "4px solid var(--accent-2)" }}>
          <p style={{ fontWeight: 700, marginBottom: "8px" }}>Frame Label Assignment Logic</p>
          <ul style={{ paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "4px" }}>
            <li><strong>PEAK</strong>: Local action maximum and survived NMS (is_peak = True).</li>
            <li><strong>HIGH_IMPORTANCE</strong>: Action score <code>&gt;= 0.70</code>, not a PEAK.</li>
            <li><strong>MEDIUM_IMPORTANCE</strong>: Action score between <code>0.35</code> and <code>0.70</code>, not a PEAK.</li>
            <li><strong>LOW_IMPORTANCE</strong>: Action score between <code>0.10</code> and <code>0.35</code>, not a PEAK.</li>
            <li><strong>BACKGROUND</strong>: Action score <code>&lt; 0.10</code>, representing quiet static scenes.</li>
          </ul>
        </div>
      )}

      <div className="frames-grid">
        {sorted.map(f => {
          const isSelected = selectedIdx === f.frame_idx;
          return (
            <div
              key={f.frame_idx}
              className={`frame-card ${isSelected ? (f.is_peak ? "selected-peak" : "selected-valley") : ""}`}
              onClick={() => onFrameClick(f)}
            >
              <div className="frame-card-thumb">
                {f.thumbnail ? (
                  <img src={f.thumbnail} alt={`Frame ${f.frame_idx}`} />
                ) : (
                  <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100%", color: "var(--text-secondary)" }}>No Image</div>
                )}
                <span className="frame-card-idx">F{f.frame_idx}</span>
                <span className="frame-card-badge">
                  <span className={`frame-badge badge-${f.is_peak ? "peak" : (f.label === "HIGH_IMPORTANCE" ? "high" : (f.label === "MEDIUM_IMPORTANCE" ? "medium" : (f.label === "LOW_IMPORTANCE" ? "low" : "bg")))}`} style={{ transform: "scale(0.85)", transformOrigin: "top right" }}>
                    {f.label === "HIGH_IMPORTANCE" ? "HIGH" : (f.label === "MEDIUM_IMPORTANCE" ? "MEDIUM" : (f.label === "LOW_IMPORTANCE" ? "LOW" : (f.label === "BACKGROUND" ? "BG" : f.label)))}
                  </span>
                </span>
              </div>
              <div className="frame-card-body">
                <span className="frame-card-time">Time: {f.timestamp.toFixed(2)}s · Type: {f.frame_type || "P"}</span>
                <div className="frame-card-scores">
                  <div>
                    <div style={{ fontSize: "9px", color: "var(--text-secondary)" }}>ACTION</div>
                    <div style={{ fontWeight: 700, color: "#fff" }}>{f.action_score.toFixed(3)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: "9px", color: "var(--text-secondary)" }}>PERSIST</div>
                    <div style={{ fontWeight: 700, color: "#fff" }}>{f.persistence_value.toFixed(3)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: "9px", color: "var(--text-secondary)" }}>RESID</div>
                    <div style={{ fontWeight: 700, color: "#fff" }}>{f.residual_energy.toFixed(3)}</div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PipelineDebugTab({ debugInfo, timings }) {
  if (!debugInfo) return null;
  const thresholds = debugInfo.thresholds || {};

  const allFrames = debugInfo.all_frames || [];
  const keptPeaks = allFrames.filter(f => f.is_peak).map(f => f.frame_idx);
  const suppressedPeaks = allFrames.filter(f => f.nms_suppressed);

  return (
    <div className="debug-panel">
      <div className="debug-stage-card">
        <div className="debug-stage-header">
          <Layers size={18} style={{ color: "var(--accent)", marginRight: "8px" }} />
          <span className="debug-stage-title">Stage 1: Charon-V (Raw Signal Extraction)</span>
        </div>
        <div style={{ display: "flex", gap: "10px", marginBottom: "15px" }}>
          <span className="peak-tag">Total Decoded Frames: {allFrames.length}</span>
          <span className="peak-tag">I-Frames Count: {allFrames.filter(f => f.frame_type === "I").length}</span>
        </div>
        <div className="debug-grid-half">
          <div>
            <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Residual Energy Timeline</div>
            <SVGTimelineChartSimple allFrames={allFrames} activeKey="residual_energy" strokeColor="#3b82f6" />
          </div>
          <div>
            <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Motion Magnitude Timeline</div>
            <SVGTimelineChartSimple allFrames={allFrames} activeKey="motion_magnitude" strokeColor="#8b5cf6" />
          </div>
        </div>
      </div>

      <div className="debug-stage-card">
        <div className="debug-stage-header">
          <Activity size={18} style={{ color: "var(--accent)", marginRight: "8px" }} />
          <span className="debug-stage-title">Stage 2: ActionScore (Continuous Metric Formulation)</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "10px", marginBottom: "15px" }}>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl">Peak Prominence Floor</div>
            <div className="frame-metric-val">{thresholds.peak_prominence ?? "0.05"}</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl">Peak Distance</div>
            <div className="frame-metric-val">{thresholds.peak_distance ?? "5"} frames</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl">Persistence Threshold</div>
            <div className="frame-metric-val">{thresholds.persistence_threshold ?? "0.40"}</div>
          </div>
          <div className="frame-metric-box">
            <div className="frame-metric-lbl">Max Prominence Divisor</div>
            <div className="frame-metric-val">{thresholds.max_prominence ?? "0.50"}</div>
          </div>
        </div>
        <div className="debug-grid-half">
          <div>
            <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Blended Action Score Graph</div>
            <SVGTimelineChartSimple allFrames={allFrames} activeKey="action_score" strokeColor="var(--accent)" />
          </div>
          <div>
            <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Persistence Value Graph</div>
            <SVGTimelineChartSimple allFrames={allFrames} activeKey="persistence_value" strokeColor="var(--neon-green)" />
          </div>
        </div>
      </div>

      <div className="debug-stage-card">
        <div className="debug-stage-header">
          <GitBranch size={18} style={{ color: "var(--accent)", marginRight: "8px" }} />
          <span className="debug-stage-title">Stage 3: Non-Maximum Suppression (Peak Pruning)</span>
        </div>
        <div className="debug-grid-half">
          <div>
            <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Accepted Peaks ({keptPeaks.length})</div>
            <div className="peaks-list">
              {keptPeaks.map(p => <span key={p} className="peak-tag">Frame {p}</span>)}
            </div>
          </div>
          <div>
            <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Suppressed Local Peaks ({suppressedPeaks.length})</div>
            <div className="peaks-list">
              {suppressedPeaks.map(p => (
                <span key={p.frame_idx} className="suppressed-tag">
                  F{p.frame_idx} ➔ parent: F{p.nms_parent}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="debug-stage-card">
        <div className="debug-stage-header">
          <Database size={18} style={{ color: "var(--accent)", marginRight: "8px" }} />
          <span className="debug-stage-title">Stage 4: L2 Asphodel Retrieval Pool</span>
        </div>
        <p style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "10px" }}>
          Peak frames indexed in graph and ranked against query semantic embeddings.
        </p>
        <div className="peaks-list">
          {debugInfo.retrieved_frames && debugInfo.retrieved_frames.map(f => (
            <span key={f.frame_idx} className="peak-tag" style={{ borderStyle: "dashed" }}>
              Frame {f.frame_idx}: Rank score {f.pagerank_score?.toFixed(4) || "0.0"}
            </span>
          ))}
        </div>
      </div>

      <div className="debug-stage-card">
        <div className="debug-stage-header">
          <Zap size={18} style={{ color: "var(--accent)", marginRight: "8px" }} />
          <span className="debug-stage-title">Stage 5: ARIA LLM brain input/output (Exact Payload)</span>
        </div>
        <div style={{ display: "flex", gap: "15px", marginBottom: "15px", fontSize: "12px" }}>
          <span>Frames given to ARIA: <strong>[{debugInfo.frames_given_to_aria?.join(", ")}]</strong></span>
          <span>Context length: <strong>{debugInfo.context_length} chars</strong></span>
        </div>
        <div style={{ marginBottom: "10px" }}>
          <div style={{ fontSize: "11px", fontWeight: 700, textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "4px" }}>Exact Prompt</div>
          <div className="debug-payload-box" style={{ maxHeight: "80px", color: "#fff" }}>{debugInfo.aria_prompt}</div>
        </div>
        <div style={{ marginBottom: "10px" }}>
          <div style={{ fontSize: "11px", fontWeight: 700, textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "4px" }}>Exact Context Payload</div>
          <div className="debug-payload-box">{debugInfo.aria_context}</div>
        </div>
        <div>
          <div style={{ fontSize: "11px", fontWeight: 700, textTransform: "uppercase", color: "var(--text-secondary)", marginBottom: "4px" }}>Exact Answer Output</div>
          <div className="debug-payload-box" style={{ color: "var(--text-primary)" }}>{debugInfo.aria_response}</div>
        </div>
      </div>
    </div>
  );
}

function SVGTimelineChartSimple({ allFrames, activeKey, strokeColor }) {
  if (!allFrames || !allFrames.length) return null;
  const width = 500;
  const height = 80;
  const padding = 5;

  const points = allFrames.map(f => f[activeKey] ?? 0);
  const maxVal = Math.max(...points, 1e-4);

  const xScale = (width - padding * 2) / (allFrames.length - 1 || 1);
  const yScale = (height - padding * 2) / maxVal;

  const pathD = points.map((p, i) => {
    const x = padding + i * xScale;
    const y = height - padding - p * yScale;
    return `${i === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  return (
    <div className="svg-chart-wrapper" style={{ padding: "5px" }}>
      <svg viewBox={`0 0 ${width} ${height}`} className="svg-chart">
        <path d={pathD} fill="none" stroke={strokeColor || "var(--accent)"} strokeWidth="1.5" />
      </svg>
    </div>
  );
}

function CerberusDebugTab({ debugInfo }) {
  if (!debugInfo || !debugInfo.cerberus_result) return null;
  const cerb = debugInfo.cerberus_result;

  return (
    <div className="debug-stage-card">
      <div className="debug-stage-header">
        <Shield size={18} style={{ color: "var(--accent)", marginRight: "8px" }} />
        <span className="debug-stage-title">Hallucination Verification (Cerberus-V NLI Truth Gate)</span>
      </div>
      <div style={{ display: "flex", gap: "10px", marginBottom: "15px" }}>
        <span className="peak-tag">Extraction Mode: {debugInfo.cerberus_result.mode || "full_nli"}</span>
        <span className={`frame-badge ${cerb.is_verified ? "badge-verified" : "badge-rejected"}`} style={{ height: "fit-content" }}>
          {cerb.is_verified ? "VERIFIED" : "CONTRADICTIONS FOUND"}
        </span>
      </div>

      <div style={{ marginBottom: "15px" }}>
        <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Extracted Claims from ARIA ({debugInfo.cerberus_claims?.length || 0})</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          {debugInfo.cerberus_claims && debugInfo.cerberus_claims.map((c, i) => {
            const isRejected = cerb.rejected && cerb.rejected.includes(c);
            const isUnverifiable = cerb.unverifiable && cerb.unverifiable.includes(c);
            let color = "var(--neon-green)";
            let status = "✓ Verified";
            if (isRejected) {
              color = "var(--neon-red)";
              status = "✗ Contradicted";
            } else if (isUnverifiable) {
              color = "var(--neon-orange)";
              status = "? Unverifiable";
            }
            return (
              <div key={i} className="glass-card" style={{ padding: "10px", fontSize: "12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>{c}</span>
                <span style={{ color, fontWeight: 700, fontSize: "11px", textTransform: "uppercase" }}>{status}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <div style={{ fontSize: "12px", fontWeight: 700, marginBottom: "6px" }}>Evidence Triples Grounded in Facts ({debugInfo.cerberus_evidence?.length || 0})</div>
        <div className="debug-payload-box" style={{ maxHeight: "200px" }}>
          {debugInfo.cerberus_evidence && debugInfo.cerberus_evidence.join("\n")}
        </div>
      </div>
    </div>
  );
}

function ResultsPanel({
  data,
  activeTab,
  setActiveTab,
  selectedFrame,
  setSelectedFrame,
  filterLabel,
  setFilterLabel,
  sortBy,
  setSortBy,
  sortOrder,
  setSortOrder
}) {
  const timings = data.timings || {};
  const graphData = data.graph_data || { nodes: [], edges: [] };
  const debugInfo = data.debug_info || null;
  const allFrames = debugInfo?.all_frames || [];

  return (
    <div id="results-panel" className="results-panel">
      {/* Tabs Header */}
      <div className="tabs-header">
        <button className={`tab-btn ${activeTab === "interactive" ? "active" : ""}`} onClick={() => setActiveTab("interactive")}>
          <Network size={14} /> Interactive Graph
        </button>
        <button className={`tab-btn ${activeTab === "all_frames" ? "active" : ""}`} onClick={() => setActiveTab("all_frames")}>
          <Layers size={14} /> All Extracted Frames ({allFrames.length})
        </button>
        <button className={`tab-btn ${activeTab === "cerberus" ? "active" : ""}`} onClick={() => setActiveTab("cerberus")}>
          <Shield size={14} /> Cerberus NLI Gate
        </button>
        <button className={`tab-btn ${activeTab === "debug" ? "active" : ""}`} onClick={() => setActiveTab("debug")}>
          <Cpu size={14} /> Pipeline Debug Panel
        </button>
      </div>

      {/* Tab 1: Interactive Viewer */}
      {activeTab === "interactive" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div className="timeline-card">
            <div className="timeline-title">
              <Activity size={16} className="accent" /> Action Score Timeline &amp; Frame Inspector (Click Graph points)
            </div>
            <SVGTimelineChart
              allFrames={allFrames}
              selectedIdx={selectedFrame?.frame_idx}
              onPointClick={setSelectedFrame}
            />
          </div>

          <FrameDetailsCard frame={selectedFrame} />

          <section className="result-section">
            <h3 className="section-title">
              <Network size={16} className="accent" /> L2 Asphodel Knowledge Graph (Spatiotemporal)
            </h3>
            <GraphStats graphData={graphData} />
            <KnowledgeGraph graphData={graphData} />
          </section>

          <section className="result-section">
            <h3 className="section-title">
              <Database size={16} className="accent" /> Stage 4: Top Retrieved Frames (Graph Node Matches)
            </h3>
            <div className="retrieval-pool-grid">
              {debugInfo?.retrieved_frames && debugInfo.retrieved_frames.map(f => (
                <div key={f.frame_idx} className="retrieved-thumb-card">
                  <div className="retrieved-img-wrap">
                    {allFrames.find(af => af.frame_idx === f.frame_idx)?.thumbnail ? (
                      <img src={allFrames.find(af => af.frame_idx === f.frame_idx).thumbnail} alt={`F${f.frame_idx}`} />
                    ) : (
                      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100%", fontSize: "9px" }}>No Thumbnail</div>
                    )}
                  </div>
                  <div className="retrieved-text">
                    <span>Frame {f.frame_idx}</span>
                    <span>{f.timestamp.toFixed(1)}s</span>
                  </div>
                  <div style={{ fontSize: "9px", color: "var(--neon-green)", marginTop: "2px", fontWeight: "bold" }}>
                    Score: {f.action_score.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </section>

          <section className="result-section">
            <h3 className="section-title">
              <Zap size={16} className="accent" /> ARIA Output (Generative Answer)
            </h3>
            <div id="aria-answer" className="answer-box">
              {data.answer || <em>No answer generated.</em>}
            </div>
          </section>
        </div>
      )}

      {/* Tab 2: All Extracted Frames Grid */}
      {activeTab === "all_frames" && (
        <section className="result-section">
          <h3 className="section-title" style={{ marginBottom: "15px" }}>
            <Layers size={16} className="accent" /> All Decoded &amp; Extracted Frames
          </h3>
          <ExtractedFramesGrid
            allFrames={allFrames}
            filterLabel={filterLabel}
            setFilterLabel={setFilterLabel}
            sortBy={sortBy}
            setSortBy={setSortBy}
            sortOrder={sortOrder}
            setSortOrder={setSortOrder}
            onFrameClick={(f) => {
              setSelectedFrame(f);
              setActiveTab("interactive");
            }}
            selectedIdx={selectedFrame?.frame_idx}
          />
        </section>
      )}

      {/* Tab 3: Cerberus NLI Check */}
      {activeTab === "cerberus" && (
        <section className="result-section">
          <CerberusDebugTab debugInfo={debugInfo} />
        </section>
      )}

      {/* Tab 4: Pipeline Debug Panel */}
      {activeTab === "debug" && (
        <section className="result-section">
          <PipelineDebugTab debugInfo={debugInfo} timings={timings} />
        </section>
      )}

      {/* Summary stats */}
      <section className="result-section" style={{ marginTop: "30px", borderTop: "1px solid var(--border)", paddingTop: "20px" }}>
        <h3 className="section-title">
          <BarChart3 size={16} className="accent" /> Video Data Triage Summary
        </h3>
        <div className="stats-grid">
          <StatCard icon={<Layers size={20} />} label="Total Video Frames" value={data.frames_processed ?? "—"} accent />
          <StatCard icon={<Activity size={20} />} label="Continuous Peaks" value={data.peak_count ?? "—"} />
          <StatCard icon={<Database size={20} />} label="Compression Ratio" value={formatPct(data.compression_ratio)} />
          <StatCard icon={<BarChart3 size={20} />} label="Storage Reduction" value={`${(data.storage_reduction_factor ?? 1).toFixed(2)}×`} />
        </div>
      </section>

      {/* Latency details */}
      <section className="result-section">
        <h3 className="section-title">
          <Clock size={16} className="accent" /> Latency Telemetry
        </h3>
        <div id="timing-breakdown" className="timing-panel">
          {TIMING_META.map(({ key, label, icon, isTotal }) => (
            <TimingRow key={key} icon={icon} label={label} value={timings[key]} isTotal={isTotal} />
          ))}
        </div>
      </section>
    </div>
  );
}

export default function IrisApp() {
  const [videoFile, setVideoFile] = useState(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Observability Dashboard States
  const [activeTab, setActiveTab] = useState("interactive");
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [filterLabel, setFilterLabel] = useState("ALL");
  const [sortBy, setSortBy] = useState("frame_idx");
  const [sortOrder, setSortOrder] = useState("asc");

  const chessRules = false;
  const canRun = videoFile && query.trim() && !loading;

  const handleRun = async () => {
    if (!canRun) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("query", query.trim());

    try {
      const response = await axios.post(API_URL, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 600000, 
      });
      setResult(response.data);
      
      // Default initializations on new result
      setActiveTab("interactive");
      setSelectedFrame(response.data.debug_info?.all_frames?.[0] || null);
      setFilterLabel("ALL");
      setSortBy("frame_idx");
      setSortOrder("asc");
    } catch (err) {
      let detail = "Unknown error communicating with the IRIS API.";
      if (err?.code === "ECONNABORTED") {
        detail = "Request timed out — the pipeline is still running. Try a shorter video.";
      } else if (err?.code === "ERR_NETWORK" || err?.message === "Network Error") {
        detail =
          "Network Error: Cannot reach the backend.\n\nMake sure the FastAPI server is running:\n  uvicorn api:app --reload --host 0.0.0.0 --port 8000";
      } else {
        detail = err?.response?.data?.detail || err?.message || detail;
      }
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <div className="bg-grid" aria-hidden />
      <div className="glow-orb glow-1" aria-hidden />
      <div className="glow-orb glow-2" aria-hidden />

      <header className="header">
        <div className="logo-lockup">
          <div className="logo-icon"><Cpu size={28} /></div>
          <div>
            <h1 className="logo-title">IRIS</h1>
            <p className="logo-sub">Intelligent Retrieval &amp; Inference System</p>
          </div>
        </div>
        <div className="header-right">
          <ApiStatusBadge />
          <div className="header-badge">
            <span className="pulse-dot" />
            LIVE PIPELINE
          </div>
        </div>
      </header>

      <main className="main-layout">
        <aside className="control-panel glass-card">
          <h2 className="panel-title"><Upload size={16} /> Input Configuration</h2>

          <DropZone onFileSelect={setVideoFile} selectedFile={videoFile} />

          <div className="query-wrap">
            <label htmlFor="query-input" className="query-label">Natural Language Query</label>
            <textarea
              id="query-input"
              className="query-input"
              placeholder="e.g. What is the main action happening in the video?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={4}
              disabled={loading}
            />
          </div>

          <button
            id="run-pipeline-btn"
            className={`run-btn ${canRun ? "run-btn-active" : ""}`}
            onClick={handleRun}
            disabled={!canRun}
          >
            <Play size={18} />
            {loading ? "Running…" : "Run IRIS Pipeline"}
          </button>

          <div className="pipeline-legend">
            {["Charon-V","Action Score","L2 Asphodel","L1 Elysium","ARIA","Cerberus-V"].map((s, i, arr) => (
              <span key={s} className="legend-stage">
                {s}{i < arr.length - 1 && <span className="legend-arrow">→</span>}
              </span>
            ))}
          </div>
        </aside>

        <section className="output-panel">
          {!loading && !result && !error && (
            <div className="empty-state glass-card">
              <Network size={56} className="empty-icon" />
              <p className="empty-title">Awaiting Input</p>
              <p className="empty-sub">
                Upload a video and enter a query to run the full IRIS pipeline.
                The knowledge graph will appear here.
              </p>
            </div>
          )}

          {loading && <PulsingLoader />}

          {error && !loading && (
            <div id="error-panel" className="error-card glass-card">
              <XCircle size={32} className="error-icon" />
              <p className="error-title">Pipeline Error</p>
              <pre className="error-detail">{error}</pre>
              <p className="error-hint">
                Ensure the backend is running:<br />
                <code>uvicorn api:app --reload --host 0.0.0.0 --port 8000</code>
              </p>
            </div>
          )}

          {result && !loading && (
            <ResultsPanel
              data={result}
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              selectedFrame={selectedFrame}
              setSelectedFrame={setSelectedFrame}
              filterLabel={filterLabel}
              setFilterLabel={setFilterLabel}
              sortBy={sortBy}
              setSortBy={setSortBy}
              sortOrder={sortOrder}
              setSortOrder={setSortOrder}
            />
          )}
        </section>
      </main>

      <footer className="footer">
        IRIS · Charon-V · Action Score · L2 Asphodel · L1 Elysium · ARIA · Cerberus-V
      </footer>
    </div>
  );
}
