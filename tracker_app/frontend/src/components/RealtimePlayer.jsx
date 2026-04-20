import React, { useEffect, useRef, useState, useCallback } from "react";
import { Square, Play, AlertCircle } from "lucide-react";
import MetricsPanel from "./MetricsPanel.jsx";

export default function RealtimePlayer({
  videoId,
  modelConfig,
  onStop,
  onJobComplete,
}) {
  const sessionId = useRef(`sess_${Date.now()}`).current;
  const wsRef = useRef(null);
  const imgRef = useRef(null);
  const [status, setStatus] = useState("connecting"); // connecting | streaming | done | error
  const [liveMetrics, setLiveMetrics] = useState(null);
  const [finalMetrics, setFinalMetrics] = useState(null);
  const [frameInfo, setFrameInfo] = useState(null);
  const [error, setError] = useState(null);
  const [isStopping, setIsStopping] = useState(false);
  const [effectivePipeline, setEffectivePipeline] = useState(null);
  const uiUpdateAtRef = useRef(0);

  const connect = useCallback(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${proto}://${location.host}/ws/track/${sessionId}`,
    );
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("streaming");
      ws.send(
        JSON.stringify({
          video_id: videoId,
          framework: modelConfig.framework,
          tracker: modelConfig.tracker,
          detector: modelConfig.detector,
          reid_model: modelConfig.reidModel || null,
          color_enabled: modelConfig.colorEnabled ?? true,
          color_segmenter: modelConfig.colorSegmenter || "grabcut",
          conf_threshold: modelConfig.confThreshold,
        }),
      );
    };

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "pipeline") {
        setEffectivePipeline(msg.pipeline || null);
      } else if (msg.type === "frame") {
        if (imgRef.current) {
          imgRef.current.src = `data:image/jpeg;base64,${msg.frame}`;
        }
        const now = performance.now();
        if (now - uiUpdateAtRef.current >= 120) {
          setLiveMetrics({
            fps: msg.fps,
            tracker_fps: msg.tracker_fps,
            track_count: msg.track_count,
          });
          setFrameInfo({ current: msg.frame_idx, total: msg.total_frames });
          uiUpdateAtRef.current = now;
        }
      } else if (msg.type === "done") {
        const resolvedPipeline = msg.pipeline || effectivePipeline;
        if (resolvedPipeline) setEffectivePipeline(resolvedPipeline);
        setFinalMetrics(msg.metrics);
        setStatus("done");
        onJobComplete?.({
          config: resolvedPipeline || modelConfig,
          metrics: msg.metrics,
        });
      } else if (msg.error) {
        setError(msg.error);
        setStatus("error");
      }
    };

    ws.onerror = () => {
      setStatus("error");
      setError("WebSocket connection failed");
    };
    ws.onclose = () => {
      setIsStopping(false);
      setStatus((prev) => (prev === "streaming" ? "done" : prev));
    };
  }, [videoId, modelConfig, sessionId]); // eslint-disable-line

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      fetch(`/api/track/stop/${sessionId}`, { method: "POST" }).catch(() => {});
    };
  }, []); // eslint-disable-line

  const handleStop = async () => {
    try {
      setIsStopping(true);
      await fetch(`/api/track/stop/${sessionId}`, { method: "POST" });
      // Keep WebSocket open so backend can send final "done" metrics payload.
    } catch {
      setIsStopping(false);
    }
  };

  const shownPipeline = effectivePipeline || {
    framework: modelConfig.framework,
    tracker: modelConfig.tracker,
    detector: modelConfig.detector,
    reid_model: modelConfig.reidModel || "none",
  };
  const pipelineRows = [
    ["Framework", shownPipeline.framework],
    ["Tracker", shownPipeline.tracker],
    ["Detector", shownPipeline.detector],
    ["Re-ID", shownPipeline.reid_model || "none"],
    [
      "Color",
      shownPipeline.cloth_color?.enabled
        ? shownPipeline.cloth_color?.segmenter
        : "off",
    ],
    ["Embedder", shownPipeline.embedder || "n/a"],
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        height: "100%",
      }}
    >
      {/* Video canvas */}
      <div
        style={{
          flex: 1,
          position: "relative",
          background: "#000",
          borderRadius: "10px",
          overflow: "hidden",
          border: "1px solid var(--border)",
          minHeight: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {status === "connecting" && (
          <div style={{ textAlign: "center", color: "var(--text2)" }}>
            <div
              style={{
                width: 32,
                height: 32,
                border: "2px solid var(--accent)",
                borderTopColor: "transparent",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
                margin: "0 auto 12px",
              }}
            />
            Connecting to tracker…
          </div>
        )}
        {status === "error" && (
          <div
            style={{
              textAlign: "center",
              color: "var(--red)",
              padding: "24px",
            }}
          >
            <AlertCircle size={32} style={{ marginBottom: 12 }} />
            <div style={{ fontWeight: 700 }}>Connection Error</div>
            <div
              style={{ fontSize: "12px", color: "var(--text2)", marginTop: 4 }}
            >
              {error}
            </div>
          </div>
        )}
        <img
          ref={imgRef}
          alt="Tracking stream"
          style={{
            maxWidth: "100%",
            maxHeight: "100%",
            display:
              status === "streaming" || status === "done" ? "block" : "none",
            objectFit: "contain",
          }}
        />
        {/* Overlay badges */}
        {status === "streaming" && (
          <div
            style={{
              position: "absolute",
              top: 10,
              left: 10,
              display: "flex",
              flexDirection: "column",
              gap: "8px",
            }}
          >
            <div style={{ display: "flex", gap: "6px" }}>
              <span
                style={{
                  padding: "3px 8px",
                  borderRadius: "4px",
                  background: "rgba(255,77,109,0.85)",
                  fontSize: "10px",
                  fontWeight: 700,
                  letterSpacing: "1px",
                  color: "#fff",
                }}
              >
                ● LIVE
              </span>
              {isStopping && (
                <span
                  style={{
                    padding: "3px 8px",
                    borderRadius: "4px",
                    background: "rgba(245,166,35,0.9)",
                    fontSize: "10px",
                    fontWeight: 700,
                    letterSpacing: "1px",
                    color: "#111",
                  }}
                >
                  STOPPING...
                </span>
              )}
            </div>
            <div
              style={{
                padding: "7px 9px",
                borderRadius: "6px",
                background: "rgba(0,0,0,0.75)",
                border: "1px solid rgba(255,255,255,0.15)",
                fontSize: "10px",
                fontFamily: "var(--font-mono)",
                color: "var(--text2)",
                lineHeight: 1.5,
              }}
            >
              {pipelineRows.map(([k, v]) => (
                <div key={k}>
                  <span style={{ color: "var(--text3)" }}>{k}: </span>
                  <span style={{ color: "var(--accent)" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {status === "done" && (
          <div
            style={{
              position: "absolute",
              top: 10,
              left: 10,
              padding: "3px 8px",
              borderRadius: "4px",
              background: "rgba(61,220,132,0.85)",
              fontSize: "10px",
              fontWeight: 700,
              letterSpacing: "1px",
              color: "#000",
            }}
          >
            ✓ COMPLETE
          </div>
        )}
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: "8px" }}>
        {status === "streaming" && (
          <button
            onClick={handleStop}
            disabled={isStopping}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "10px 20px",
              borderRadius: "8px",
              background: isStopping ? "var(--bg3)" : "var(--red)",
              border: "none",
              color: isStopping ? "var(--text3)" : "#fff",
              fontWeight: 700,
              fontSize: "13px",
              cursor: isStopping ? "default" : "pointer",
            }}
          >
            <Square size={14} fill="#fff" />
            {isStopping ? "STOPPING..." : "STOP"}
          </button>
        )}
        <button
          onClick={onStop}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "10px 20px",
            borderRadius: "8px",
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            color: "var(--text2)",
            fontWeight: 600,
            fontSize: "13px",
            cursor: "pointer",
          }}
        >
          ← Back
        </button>
      </div>

      {/* Metrics */}
      {(liveMetrics || finalMetrics) && (
        <MetricsPanel
          metrics={finalMetrics}
          live={status === "streaming" ? liveMetrics : null}
          frameInfo={frameInfo}
        />
      )}
    </div>
  );
}
