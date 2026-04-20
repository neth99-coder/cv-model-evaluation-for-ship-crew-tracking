import React, { useEffect, useRef } from "react";
import { ChevronDown, Info } from "lucide-react";

const CAPABILITIES = {
  boxmot: {
    trackers: ["bytetrack", "botsort", "ocsort", "deepocsort", "strongsort"],
    detectors: [
      "yolov8n",
      "yolov8s",
      "yolov8m",
      "yolov5n",
      "yolov5s",
      "fasterrcnn",
      "ssd_mobilenet",
    ],
    reidSupport: {
      bytetrack: false,
      botsort: true,
      ocsort: false,
      deepocsort: true,
      strongsort: true,
    },
    reidModels: ["osnet_x0_25", "osnet_x1_0", "resnet50", "mlfn"],
  },
  fairmot: {
    trackers: ["fairmot"],
    detectors: ["dla34", "hrnet", "resnet50"],
    reidSupport: { fairmot: true },
    reidModels: ["built-in (no selection needed)"],
  },
  deepsort: {
    trackers: ["deepsort"],
    detectors: [
      "yolov8n",
      "yolov8s",
      "yolov8m",
      "yolov5n",
      "yolov5s",
      "fasterrcnn",
      "ssd_mobilenet",
    ],
    reidSupport: { deepsort: true },
    reidModels: ["osnet_x0_25", "osnet_x1_0", "resnet50"],
  },
};

const FRAMEWORK_COLORS = {
  boxmot: "#f5a623",
  fairmot: "#4d9fff",
  deepsort: "#3ddc84",
};

const FRAMEWORK_DESC = {
  boxmot:
    "Unified tracker library supporting ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT",
  fairmot:
    "Joint detection & embedding model — built-in Re-ID with DLA34/HRNet/ResNet50 backbone",
  deepsort:
    "Classic DeepSORT with pluggable YOLOv8/v5 detector and configurable Re-ID model",
};

function Select({ label, value, options, onChange, disabled }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <label
        style={{
          fontSize: "11px",
          fontWeight: 600,
          letterSpacing: "1px",
          color: "var(--text3)",
        }}
      >
        {label}
      </label>
      <div style={{ position: "relative" }}>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          style={{
            width: "100%",
            appearance: "none",
            padding: "8px 32px 8px 12px",
            background: disabled ? "var(--bg)" : "var(--bg3)",
            border: "1px solid var(--border)",
            borderRadius: "6px",
            color: disabled ? "var(--text3)" : "var(--text)",
            fontSize: "13px",
            cursor: disabled ? "not-allowed" : "pointer",
          }}
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          style={{
            position: "absolute",
            right: 10,
            top: "50%",
            transform: "translateY(-50%)",
            color: "var(--text3)",
            pointerEvents: "none",
          }}
        />
      </div>
    </div>
  );
}

export default function ModelSelector({ config, onChange }) {
  const {
    framework,
    tracker,
    detector,
    reidModel,
    manualReid,
    confThreshold,
    colorEnabled,
    colorSegmenter,
  } = config;
  const cap = CAPABILITIES[framework];
  const reidEnabled = cap.reidSupport[tracker] && framework !== "fairmot";
  const manualReidEnabled = framework === "boxmot" || framework === "deepsort";
  const mountedRef = useRef(false);

  // Auto-reset when framework changes
  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true;
      return;
    }
    const newTracker = cap.trackers.includes(tracker) ? tracker : cap.trackers[0];
    const newDetector = cap.detectors.includes(detector)
      ? detector
      : cap.detectors[0];
    const newReid =
      !!cap.reidSupport[newTracker] &&
      cap.reidModels[0] !== "built-in (no selection needed)" &&
      reidModel &&
      cap.reidModels.includes(reidModel)
        ? reidModel
        : cap.reidModels[0] !== "built-in (no selection needed)"
        ? cap.reidModels[0]
        : null;
    onChange({
      framework,
      tracker: newTracker,
      detector: newDetector,
      reidModel: newReid,
      manualReid: manualReidEnabled ? !!manualReid : false,
      confThreshold,
      colorEnabled,
      colorSegmenter,
    });
  }, [framework]); // eslint-disable-line

  useEffect(() => {
    if (cap.trackers.includes(tracker)) return;
    onChange((c) => ({ ...c, tracker: cap.trackers[0] }));
  }, [cap, tracker]); // eslint-disable-line

  useEffect(() => {
    if (manualReidEnabled || !manualReid) return;
    onChange((c) => ({ ...c, manualReid: false }));
  }, [manualReidEnabled, manualReid]); // eslint-disable-line

  // Auto-reset reid when tracker changes
  useEffect(() => {
    const supportsReid = !!cap.reidSupport[tracker] && framework !== "fairmot";
    if (!supportsReid) {
      onChange((c) => ({ ...c, reidModel: null }));
      return;
    }

    // Critical: ensure state has a real selected value. The select control can
    // display a fallback value even when config.reidModel is null.
    const fallback = cap.reidModels[0] || null;
    if (!fallback) return;
    if (!reidModel || !cap.reidModels.includes(reidModel)) {
      onChange((c) => ({ ...c, reidModel: fallback }));
    }
  }, [framework, tracker, reidModel]); // eslint-disable-line

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Framework tabs */}
      <div>
        <div
          style={{
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "1px",
            color: "var(--text3)",
            marginBottom: "8px",
          }}
        >
          FRAMEWORK
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          {Object.keys(CAPABILITIES).map((fw) => {
            const active = fw === framework;
            return (
              <button
                key={fw}
                onClick={() => onChange((c) => ({ ...c, framework: fw }))}
                style={{
                  flex: 1,
                  padding: "8px 4px",
                  background: active ? "var(--accent-dim)" : "var(--bg3)",
                  border: `1px solid ${active ? FRAMEWORK_COLORS[fw] : "var(--border)"}`,
                  borderRadius: "6px",
                  color: active ? FRAMEWORK_COLORS[fw] : "var(--text2)",
                  fontWeight: active ? 700 : 500,
                  fontSize: "12px",
                  letterSpacing: "0.5px",
                  cursor: "pointer",
                  transition: "all 0.15s",
                  textTransform: "uppercase",
                }}
              >
                {fw}
              </button>
            );
          })}
        </div>
        <p
          style={{
            marginTop: "8px",
            padding: "8px 10px",
            background: "var(--bg3)",
            borderRadius: "6px",
            fontSize: "11px",
            color: "var(--text2)",
            lineHeight: 1.6,
            borderLeft: `2px solid ${FRAMEWORK_COLORS[framework]}`,
          }}
        >
          {FRAMEWORK_DESC[framework]}
        </p>
      </div>

      {/* Tracker */}
      <Select
        label="TRACKER"
        value={tracker}
        options={cap.trackers}
        onChange={(v) => onChange((c) => ({ ...c, tracker: v }))}
      />

      {/* Detector */}
      <Select
        label="DETECTOR"
        value={detector}
        options={cap.detectors}
        onChange={(v) => onChange((c) => ({ ...c, detector: v }))}
      />

      {/* RE-ID */}
      <div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            marginBottom: "6px",
          }}
        >
          <label
            style={{
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "1px",
              color: "var(--text3)",
            }}
          >
            RE-ID MODEL
          </label>
          {!reidEnabled && framework !== "fairmot" && (
            <span
              style={{
                fontSize: "10px",
                padding: "1px 6px",
                background: "var(--bg3)",
                borderRadius: "20px",
                color: "var(--text3)",
                border: "1px solid var(--border)",
              }}
            >
              not supported by {tracker}
            </span>
          )}
          {framework === "fairmot" && (
            <span
              style={{
                fontSize: "10px",
                padding: "1px 6px",
                background: "#4d9fff22",
                borderRadius: "20px",
                color: "var(--blue)",
                border: "1px solid var(--blue)",
              }}
            >
              built-in
            </span>
          )}
          {reidEnabled && (
            <span
              style={{
                fontSize: "10px",
                padding: "1px 6px",
                background: "var(--accent-dim)",
                borderRadius: "20px",
                color: "var(--accent)",
                border: "1px solid var(--accent)",
              }}
            >
              supported
            </span>
          )}
        </div>
        {reidEnabled ? (
          <Select
            label=""
            value={reidModel || cap.reidModels[0]}
            options={cap.reidModels}
            onChange={(v) => onChange((c) => ({ ...c, reidModel: v }))}
          />
        ) : (
          <div
            style={{
              padding: "8px 12px",
              background: "var(--bg)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              fontSize: "12px",
              color: "var(--text3)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {framework === "fairmot"
              ? "— built into model architecture —"
              : "— disabled for this tracker —"}
          </div>
        )}
      </div>

      <div
        style={{
          padding: "10px",
          borderRadius: "8px",
          border: "1px solid var(--border)",
          background: "var(--bg3)",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "1px",
            color: "var(--text3)",
          }}
        >
          MANUAL RE-ID
          <Info size={12} style={{ color: "var(--text3)" }} />
        </div>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            fontSize: "12px",
            color: manualReidEnabled ? "var(--text2)" : "var(--text3)",
          }}
        >
          <input
            type="checkbox"
            checked={!!manualReid}
            onChange={(e) =>
              onChange((c) => ({ ...c, manualReid: e.target.checked }))
            }
            disabled={!manualReidEnabled}
          />
          Enable manual Re-ID (ID stitching)
        </label>
        {!manualReidEnabled && (
          <div style={{ fontSize: "11px", color: "var(--text3)" }}>
            Available only for BoxMOT and DeepSORT.
          </div>
        )}
      </div>

      {/* Confidence slider */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: "6px",
          }}
        >
          <label
            style={{
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "1px",
              color: "var(--text3)",
            }}
          >
            CONFIDENCE THRESHOLD
          </label>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "12px",
              color: "var(--accent)",
            }}
          >
            {confThreshold.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min="0.1"
          max="0.95"
          step="0.05"
          value={confThreshold}
          onChange={(e) =>
            onChange((c) => ({
              ...c,
              confThreshold: parseFloat(e.target.value),
            }))
          }
          style={{
            width: "100%",
            accentColor: "var(--accent)",
            background: "transparent",
            cursor: "pointer",
          }}
        />
      </div>

      {/* Cloth color settings */}
      <div
        style={{
          padding: "10px",
          borderRadius: "8px",
          border: "1px solid var(--border)",
          background: "var(--bg3)",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}
      >
        <div
          style={{
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "1px",
            color: "var(--text3)",
          }}
        >
          CLOTH COLOR RECOGNITION
        </div>
        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            fontSize: "12px",
            color: "var(--text2)",
          }}
        >
          <input
            type="checkbox"
            checked={!!colorEnabled}
            onChange={(e) =>
              onChange((c) => ({ ...c, colorEnabled: e.target.checked }))
            }
          />
          Enable cloth color recognition
        </label>
        <Select
          label="COLOR SEGMENTER"
          value={colorSegmenter || "grabcut"}
          options={["grabcut", "yolov8n-seg", "yolov8s-seg"]}
          onChange={(v) => onChange((c) => ({ ...c, colorSegmenter: v }))}
          disabled={!colorEnabled}
        />
      </div>
    </div>
  );
}
