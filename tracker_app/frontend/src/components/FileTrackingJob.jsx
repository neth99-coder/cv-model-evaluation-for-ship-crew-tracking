import React, { useState, useEffect, useRef } from "react";
import { Download, CheckCircle, AlertCircle, Loader } from "lucide-react";
import axios from "axios";
import MetricsPanel from "./MetricsPanel.jsx";

export default function FileTrackingJob({
  videoId,
  modelConfig,
  onBack,
  onJobComplete,
}) {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | submitting | running | done | error
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const submitJob = async () => {
    setStatus("submitting");
    setError(null);
    try {
      const params = {
        video_id: videoId,
        framework: modelConfig.framework,
        tracker: modelConfig.tracker,
        detector: modelConfig.detector,
        conf_threshold: modelConfig.confThreshold,
      };
      if (modelConfig.reidModel) params.reid_model = modelConfig.reidModel;
      const res = await axios.post("/api/track/file", null, { params });
      setJobId(res.data.job_id);
      setStatus("running");
    } catch (e) {
      setStatus("error");
      setError(e.response?.data?.detail || e.message);
    }
  };

  // Poll for job status
  useEffect(() => {
    if (!jobId || status !== "running") return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`/api/track/status/${jobId}`);
        const job = res.data;
        setProgress(job.progress || 0);
        if (job.status === "done") {
          clearInterval(pollRef.current);
          setStatus("done");
          setMetrics(job.metrics);
          onJobComplete?.({ config: job.pipeline || modelConfig, metrics: job.metrics });
        } else if (job.status === "error") {
          clearInterval(pollRef.current);
          setStatus("error");
          setError(job.error || "Unknown error");
        }
      } catch (e) {
        console.error(e);
      }
    }, 1000);
    return () => clearInterval(pollRef.current);
  }, [jobId, status]);

  const downloadVideo = () => {
    window.open(`/api/track/download/${jobId}`, "_blank");
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        animation: "slideIn 0.3s ease",
      }}
    >
      {/* Config summary */}
      <div
        style={{
          padding: "14px 16px",
          borderRadius: "10px",
          background: "var(--bg3)",
          border: "1px solid var(--border)",
        }}
      >
        <div
          style={{
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "1px",
            color: "var(--text3)",
            marginBottom: "8px",
          }}
        >
          JOB CONFIGURATION
        </div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          {[
            ["Framework", modelConfig.framework],
            ["Tracker", modelConfig.tracker],
            ["Detector", modelConfig.detector],
            ["Re-ID", modelConfig.reidModel || "none"],
            ["Confidence", modelConfig.confThreshold.toFixed(2)],
          ].map(([k, v]) => (
            <div
              key={k}
              style={{
                padding: "4px 10px",
                borderRadius: "20px",
                background: "var(--bg2)",
                border: "1px solid var(--border)",
                fontSize: "11px",
                color: "var(--text2)",
              }}
            >
              <span style={{ color: "var(--text3)" }}>{k}: </span>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  color: "var(--accent)",
                }}
              >
                {v}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Status area */}
      {status === "idle" && (
        <button
          onClick={submitJob}
          style={{
            padding: "14px",
            borderRadius: "10px",
            background: "var(--accent)",
            border: "none",
            color: "#000",
            fontWeight: 800,
            fontSize: "14px",
            letterSpacing: "1px",
            cursor: "pointer",
          }}
        >
          ▶ START TRACKING JOB
        </button>
      )}

      {status === "submitting" && (
        <div
          style={{
            textAlign: "center",
            padding: "20px",
            color: "var(--text2)",
          }}
        >
          <Loader
            size={24}
            style={{ animation: "spin 0.8s linear infinite", marginBottom: 8 }}
          />
          <div>Submitting job…</div>
        </div>
      )}

      {status === "running" && (
        <div
          style={{
            padding: "16px",
            borderRadius: "10px",
            background: "var(--bg3)",
            border: "1px solid var(--border)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              marginBottom: "12px",
            }}
          >
            <div
              style={{
                width: 14,
                height: 14,
                border: "2px solid var(--accent)",
                borderTopColor: "transparent",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
                flexShrink: 0,
              }}
            />
            <span style={{ fontWeight: 600, fontSize: "13px" }}>
              Processing video…
            </span>
            <span
              style={{
                marginLeft: "auto",
                fontFamily: "var(--font-mono)",
                color: "var(--accent)",
                fontSize: "14px",
              }}
            >
              {progress}%
            </span>
          </div>
          <div
            style={{
              height: "6px",
              background: "var(--bg2)",
              borderRadius: "3px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${progress}%`,
                background:
                  "linear-gradient(90deg, var(--accent), var(--accent2))",
                borderRadius: "3px",
                transition: "width 0.5s ease",
              }}
            />
          </div>
          <div
            style={{
              fontSize: "11px",
              color: "var(--text3)",
              marginTop: "8px",
              fontFamily: "var(--font-mono)",
            }}
          >
            Job ID: {jobId}
          </div>
        </div>
      )}

      {status === "done" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              padding: "14px 16px",
              borderRadius: "10px",
              background: "rgba(61,220,132,0.08)",
              border: "1px solid var(--green)",
            }}
          >
            <CheckCircle size={18} color="var(--green)" />
            <span style={{ fontWeight: 700, color: "var(--green)" }}>
              Tracking Complete!
            </span>
          </div>
          <button
            onClick={downloadVideo}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
              padding: "12px",
              borderRadius: "10px",
              background: "var(--accent)",
              border: "none",
              color: "#000",
              fontWeight: 800,
              fontSize: "13px",
              cursor: "pointer",
            }}
          >
            <Download size={16} />
            DOWNLOAD TRACKED VIDEO
          </button>
        </div>
      )}

      {status === "error" && (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "14px 16px",
            borderRadius: "10px",
            background: "rgba(255,77,109,0.08)",
            border: "1px solid var(--red)",
          }}
        >
          <AlertCircle
            size={18}
            color="var(--red)"
            style={{ flexShrink: 0, marginTop: 2 }}
          />
          <div>
            <div style={{ fontWeight: 700, color: "var(--red)" }}>
              Job Failed
            </div>
            <div
              style={{ fontSize: "12px", color: "var(--text2)", marginTop: 4 }}
            >
              {error}
            </div>
          </div>
        </div>
      )}

      {/* Back button */}
      <button
        onClick={onBack}
        style={{
          padding: "10px",
          borderRadius: "8px",
          background: "var(--bg3)",
          border: "1px solid var(--border)",
          color: "var(--text2)",
          fontWeight: 600,
          fontSize: "12px",
          cursor: "pointer",
        }}
      >
        ← Back to Config
      </button>

      {/* Metrics */}
      {metrics && <MetricsPanel metrics={metrics} />}
    </div>
  );
}
