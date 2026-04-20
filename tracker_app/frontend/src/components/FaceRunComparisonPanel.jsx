import React, { useState } from "react";
import { BarChart2, Trash2, ChevronUp, ChevronDown } from "lucide-react";

const COLS = [
  { key: "model", header: "Model", type: "text" },
  { key: "media", header: "Media", type: "text" },
  { key: "faces_detected", header: "Faces", higher: true },
  { key: "identified_faces", header: "Identified", higher: true },
  { key: "unknown_faces", header: "Unknown", higher: false },
  { key: "frames_processed", header: "Frames", higher: true },
  { key: "avg_match_confidence", header: "Avg Match", higher: true, format: (v) => number(v, 3) },
  { key: "inference_time_ms", header: "Inference ms", higher: false, format: (v) => number(v, 2) },
  { key: "enrolled_count", header: "Gallery", higher: true },
];

function number(value, digits = 2) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function bestIdx(runs, key, higher) {
  if (runs.length < 2) return -1;
  const vals = runs.map((r) => r.metrics?.[key] ?? null);
  if (vals.every((v) => v === null || v === undefined)) return -1;
  return vals.reduce((best, v, i) => {
    if (v === null || v === undefined) return best;
    if (best === -1) return i;
    return higher ? (v > vals[best] ? i : best) : v < vals[best] ? i : best;
  }, -1);
}

export default function FaceRunComparisonPanel({ runs, onClear }) {
  const [sortCol, setSortCol] = useState("createdAt");
  const [sortAsc, setSortAsc] = useState(false);

  if (!runs || runs.length === 0) {
    return (
      <div
        style={{
          marginTop: 24,
          border: "1px solid var(--border)",
          borderRadius: "12px",
          background: "var(--bg2)",
          padding: "18px 16px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <BarChart2 size={14} color="var(--accent)" />
          <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "1px", color: "var(--text2)" }}>
            FACE RUN EVALUATION
          </span>
        </div>
        <div style={{ marginTop: 10, fontSize: 12, color: "var(--text3)", lineHeight: 1.6 }}>
          No completed face runs yet. Start a run and each result will be added here for side-by-side comparison.
        </div>
      </div>
    );
  }

  const handleSort = (key) => {
    if (sortCol === key) setSortAsc((a) => !a);
    else {
      setSortCol(key);
      setSortAsc(false);
    }
  };

  const sorted = [...runs].sort((a, b) => {
    const av = sortCol in a ? a[sortCol] : a.metrics?.[sortCol];
    const bv = sortCol in b ? b[sortCol] : b.metrics?.[sortCol];
    const aVal = av ?? (typeof av === "string" ? "" : -Infinity);
    const bVal = bv ?? (typeof bv === "string" ? "" : -Infinity);
    if (aVal < bVal) return sortAsc ? -1 : 1;
    if (aVal > bVal) return sortAsc ? 1 : -1;
    return 0;
  });

  const bests = Object.fromEntries(COLS.map((c) => [c.key, bestIdx(runs, c.key, c.higher)]));

  return (
    <div
      style={{
        marginTop: 24,
        border: "1px solid var(--border)",
        borderRadius: "12px",
        background: "var(--bg2)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          padding: "12px 16px",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg3)",
        }}
      >
        <BarChart2 size={14} color="var(--accent)" />
        <span style={{ fontSize: "12px", fontWeight: 700, letterSpacing: "1px", color: "var(--text2)" }}>
          FACE RUN EVALUATION
        </span>
        <span
          style={{
            marginLeft: 6,
            padding: "1px 8px",
            borderRadius: "20px",
            background: "var(--accent-dim)",
            border: "1px solid var(--accent)",
            fontSize: "11px",
            color: "var(--accent)",
            fontFamily: "var(--font-mono)",
          }}
        >
          {runs.length} run{runs.length > 1 ? "s" : ""}
        </span>
        <span style={{ marginLeft: "auto", fontSize: "10px", color: "var(--text3)" }}>
          Higher is better except unknowns and inference time
        </span>
        <button
          onClick={onClear}
          title="Clear history"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "4px",
            padding: "4px 10px",
            borderRadius: "6px",
            background: "transparent",
            border: "1px solid var(--border)",
            color: "var(--text3)",
            fontSize: "11px",
            cursor: "pointer",
          }}
        >
          <Trash2 size={11} /> Clear
        </button>
      </div>

      <div style={{ maxHeight: 320, overflow: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
          <thead>
            <tr>
              <th style={headerStyle(false)}>Status</th>
              {COLS.map((col) => {
                const active = sortCol === col.key;
                return (
                  <th key={col.key} onClick={() => handleSort(col.key)} style={headerStyle(active)}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
                      {col.header}
                      {active ? (sortAsc ? <ChevronUp size={10} /> : <ChevronDown size={10} />) : null}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((run, rowIdx) => {
              const origIdx = runs.indexOf(run);
              return (
                <tr
                  key={run.id}
                  style={{
                    borderBottom: rowIdx < sorted.length - 1 ? "1px solid var(--border)" : "none",
                    background: rowIdx % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent",
                  }}
                >
                  <td style={cellStyle(false)}>
                    <span style={{ color: run.status === "done" ? "var(--green)" : run.status === "error" ? "var(--red)" : "var(--text3)" }}>
                      {String(run.status || "unknown").toUpperCase()}
                    </span>
                  </td>
                  {COLS.map((col) => {
                    const raw = col.type === "text" ? run[col.key] : run.metrics?.[col.key];
                    const isBest = col.type !== "text" && bests[col.key] === origIdx;
                    return (
                      <td key={col.key} style={cellStyle(col.type === "text", isBest)}>
                        {col.format ? col.format(raw) : raw ?? "—"}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function headerStyle(active) {
  return {
    padding: "8px 12px",
    textAlign: "left",
    borderBottom: "1px solid var(--border)",
    color: active ? "var(--accent)" : "var(--text3)",
    fontWeight: 600,
    letterSpacing: "0.8px",
    fontSize: "10px",
    whiteSpace: "nowrap",
    cursor: "pointer",
    userSelect: "none",
    background: "var(--bg3)",
  };
}

function cellStyle(text, highlight = false) {
  return {
    padding: "9px 12px",
    textAlign: text ? "left" : "right",
    whiteSpace: "nowrap",
    color: highlight ? "var(--green)" : "var(--text2)",
    fontFamily: text ? "inherit" : "var(--font-mono)",
    fontWeight: highlight ? 700 : 500,
  };
}
