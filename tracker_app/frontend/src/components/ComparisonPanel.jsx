import React, { useState } from "react";
import { BarChart2, Trash2, ChevronUp, ChevronDown } from "lucide-react";

const COLS = [
  { key: "label", header: "Pipeline", mono: false },
  { key: "avg_fps", header: "Avg FPS", unit: "fps", higher: true },
  { key: "min_fps", header: "Min FPS", unit: "fps", higher: true },
  { key: "unique_tracks", header: "Unique IDs", higher: true },
  { key: "avg_tracks_per_frame", header: "Tracks/Frame", higher: true },
  { key: "max_simultaneous_tracks", header: "Max Tracks", higher: true },
  {
    key: "avg_track_lifetime_frames",
    header: "Avg Lifetime",
    unit: "fr",
    higher: true,
  },
  { key: "map_iou_50", header: "mAP@0.50", higher: true },
  { key: "map_small", header: "mAP Small", higher: true },
  { key: "total_wall_time_s", header: "Wall Time", unit: "s", higher: false },
];

function bestIdx(runs, key, higher) {
  if (runs.length < 2) return -1;
  const vals = runs.map((r) => r.metrics?.[key] ?? null);
  if (vals.every((v) => v === null)) return -1;
  return vals.reduce((best, v, i) => {
    if (v === null) return best;
    if (best === -1) return i;
    return higher ? (v > vals[best] ? i : best) : v < vals[best] ? i : best;
  }, -1);
}

export default function ComparisonPanel({ runs, onClear, onAddDemo }) {
  const [sortCol, setSortCol] = useState(null);
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
          <span
            style={{
              fontSize: 12,
              fontWeight: 700,
              letterSpacing: "1px",
              color: "var(--text2)",
            }}
          >
            PIPELINE COMPARISON
          </span>
        </div>
        <div
          style={{
            marginTop: 10,
            fontSize: 12,
            color: "var(--text3)",
            lineHeight: 1.6,
          }}
        >
          No runs yet. Start a realtime/file run, or add demo rows to validate
          table behavior.
        </div>
        {onAddDemo && (
          <button
            onClick={onAddDemo}
            style={{
              marginTop: 12,
              padding: "8px 12px",
              borderRadius: "8px",
              background: "var(--bg3)",
              border: "1px solid var(--border)",
              color: "var(--text2)",
              fontWeight: 600,
              fontSize: 12,
              cursor: "pointer",
            }}
          >
            + Add Demo Comparison Data
          </button>
        )}
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
    if (!sortCol) return 0;
    const av =
      sortCol === "label" ? a.label : (a.metrics?.[sortCol] ?? -Infinity);
    const bv =
      sortCol === "label" ? b.label : (b.metrics?.[sortCol] ?? -Infinity);
    if (av < bv) return sortAsc ? -1 : 1;
    if (av > bv) return sortAsc ? 1 : -1;
    return 0;
  });

  const bests = Object.fromEntries(
    COLS.filter((c) => c.key !== "label").map((c) => [
      c.key,
      bestIdx(runs, c.key, c.higher),
    ]),
  );

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
      {/* Header */}
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
        <span
          style={{
            fontSize: "12px",
            fontWeight: 700,
            letterSpacing: "1px",
            color: "var(--text2)",
          }}
        >
          PIPELINE COMPARISON
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
        <span
          style={{
            marginLeft: "auto",
            fontSize: "10px",
            color: "var(--text3)",
          }}
        >
          Click column headers to sort ·{" "}
          <span style={{ color: "var(--green)" }}>green</span> = best
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

      {/* Table */}
      <div style={{ maxHeight: 320, overflow: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "12px",
          }}
        >
          <thead>
            <tr>
              {COLS.map((col) => {
                const active = sortCol === col.key;
                return (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    style={{
                      padding: "8px 12px",
                      textAlign: col.key === "label" ? "left" : "right",
                      borderBottom: "1px solid var(--border)",
                      color: active ? "var(--accent)" : "var(--text3)",
                      fontWeight: 600,
                      letterSpacing: "0.8px",
                      fontSize: "10px",
                      whiteSpace: "nowrap",
                      cursor: "pointer",
                      userSelect: "none",
                      background: "var(--bg3)",
                    }}
                  >
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 3,
                      }}
                    >
                      {col.header}
                      {active ? (
                        sortAsc ? (
                          <ChevronUp size={10} />
                        ) : (
                          <ChevronDown size={10} />
                        )
                      ) : null}
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {sorted.map((run, rowIdx) => {
              // find original index for best highlighting
              const origIdx = runs.indexOf(run);
              return (
                <tr
                  key={run.id}
                  style={{
                    borderBottom:
                      rowIdx < sorted.length - 1
                        ? "1px solid var(--border)"
                        : "none",
                    background:
                      rowIdx % 2 === 0
                        ? "transparent"
                        : "rgba(255,255,255,0.015)",
                  }}
                >
                  {COLS.map((col) => {
                    const isBest =
                      col.key !== "label" && bests[col.key] === origIdx;
                    const raw =
                      col.key === "label"
                        ? run.label
                        : (run.metrics?.[col.key] ?? null);
                    const display =
                      raw === null
                        ? "—"
                        : typeof raw === "number"
                          ? raw.toFixed(raw % 1 === 0 ? 0 : 2)
                          : raw;
                    return (
                      <td
                        key={col.key}
                        style={{
                          padding: "9px 12px",
                          textAlign: col.key === "label" ? "left" : "right",
                          fontFamily:
                            col.key !== "label"
                              ? "var(--font-mono)"
                              : undefined,
                          color: isBest
                            ? "var(--green)"
                            : col.key === "label"
                              ? "var(--text)"
                              : "var(--text2)",
                          fontWeight: isBest
                            ? 700
                            : col.key === "label"
                              ? 600
                              : 400,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {display}
                        {raw !== null && col.unit && (
                          <span
                            style={{
                              fontSize: "10px",
                              color: "var(--text3)",
                              marginLeft: 3,
                            }}
                          >
                            {col.unit}
                          </span>
                        )}
                        {isBest && (
                          <span
                            style={{
                              fontSize: "9px",
                              marginLeft: 4,
                              color: "var(--green)",
                            }}
                          >
                            ▲
                          </span>
                        )}
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
