import React, { useState } from "react";
import { Crosshair, Radio, FileVideo, ChevronRight } from "lucide-react";
import ModelSelector from "../components/ModelSelector.jsx";
import VideoUploader from "../components/VideoUploader.jsx";
import RealtimePlayer from "../components/RealtimePlayer.jsx";
import FileTrackingJob from "../components/FileTrackingJob.jsx";
import ComparisonPanel from "../components/ComparisonPanel.jsx";

const CONFIG_STORAGE_KEY = "tracker_app.object_tracking.config";

const DEFAULT_CONFIG = {
  framework: "boxmot",
  tracker: "bytetrack",
  detector: "yolov8n",
  reidModel: null,
  manualReid: false,
  confThreshold: 0.4,
  colorEnabled: false,
  colorSegmenter: "grabcut",
};

const MODE_REALTIME = "realtime";
const MODE_FILE = "file";

function loadStoredConfig() {
  if (typeof window === "undefined") return DEFAULT_CONFIG;
  try {
    const raw = window.localStorage.getItem(CONFIG_STORAGE_KEY);
    if (!raw) return DEFAULT_CONFIG;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export default function ObjectTracking() {
  const [config, setConfig] = useState(loadStoredConfig);
  const [video, setVideo] = useState(null); // { video_id, filename }
  const [mode, setMode] = useState(null); // null | 'realtime' | 'file'
  const [activeView, setActiveView] = useState("config"); // 'config' | 'realtime' | 'file'
  const [runHistory, setRunHistory] = useState([]); // completed run records

  React.useEffect(() => {
    window.localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(config));
  }, [config]);

  const formatPipelineLabel = (cfg) => {
    if (!cfg) return "unknown pipeline";
    const base = `${cfg.framework} / ${cfg.tracker} / ${cfg.detector}`;
    const reid = cfg.reid_model || cfg.reidModel;
    const embedder = cfg.embedder;
    const reidPart = reid ? ` + ${reid}` : "";
    const embedderPart = embedder ? ` (${embedder})` : "";
    return `${base}${reidPart}${embedderPart}`;
  };

  const addDemoRuns = () => {
    const now = Date.now();
    const demo = [
      {
        id: now + 1,
        config: {
          framework: "deepsort",
          tracker: "deepsort",
          detector: "yolov8n",
          reid_model: "osnet_x0_25",
          embedder: "torchreid",
        },
        metrics: {
          avg_fps: 24.6,
          min_fps: 18.9,
          unique_tracks: 13,
          avg_tracks_per_frame: 2.7,
          max_simultaneous_tracks: 5,
          avg_track_lifetime_frames: 41.2,
          map_iou_50: 0.612,
          map_small: 0.438,
          total_wall_time_s: 18.4,
        },
      },
      {
        id: now + 2,
        config: {
          framework: "boxmot",
          tracker: "botsort",
          detector: "yolov8s",
          reid_model: "resnet50",
        },
        metrics: {
          avg_fps: 19.1,
          min_fps: 14.2,
          unique_tracks: 12,
          avg_tracks_per_frame: 2.9,
          max_simultaneous_tracks: 6,
          avg_track_lifetime_frames: 47.5,
          map_iou_50: 0.584,
          map_small: 0.401,
          total_wall_time_s: 24.0,
        },
      },
    ].map((r) => ({ ...r, label: formatPipelineLabel(r.config) }));
    setRunHistory((h) => [...h, ...demo]);
  };

  const handleJobComplete = ({ config: cfg, metrics }) => {
    if (!metrics) return;
    const label = formatPipelineLabel(cfg);
    setRunHistory((h) => [
      ...h,
      { id: Date.now(), label, config: cfg, metrics },
    ]);
  };

  const handleStart = (selectedMode) => {
    if (!video) {
      alert("Please upload or select a video first.");
      return;
    }
    setMode(selectedMode);
    setActiveView(selectedMode);
  };

  const handleBack = () => {
    setActiveView("config");
    setMode(null);
  };

  // ── Realtime / File views ───────────────────────────────────────────────
  if (activeView === MODE_REALTIME) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          flexDirection: "column",
          padding: "16px",
          gap: "12px",
          overflow: "hidden",
        }}
      >
        <Header
          title="REAL-TIME TRACKING"
          sub={`${config.framework} / ${config.tracker} / ${config.detector}`}
        />
        <div style={{ flex: 1, minHeight: 0 }}>
          <RealtimePlayer
            videoId={video.video_id}
            modelConfig={config}
            onStop={handleBack}
            onJobComplete={handleJobComplete}
          />
        </div>
      </div>
    );
  }

  if (activeView === MODE_FILE) {
    return (
      <div style={{ height: "100%", padding: "16px", overflow: "auto" }}>
        <Header
          title="FILE TRACKING JOB"
          sub={`${config.framework} / ${config.tracker} / ${config.detector}`}
        />
        <div style={{ maxWidth: 760, marginTop: 16 }}>
          <FileTrackingJob
            videoId={video.video_id}
            modelConfig={config}
            onBack={handleBack}
            onJobComplete={handleJobComplete}
          />
          <ComparisonPanel
            runs={runHistory}
            onClear={() => setRunHistory([])}
            onAddDemo={addDemoRuns}
          />
        </div>
      </div>
    );
  }

  // ── Config view ─────────────────────────────────────────────────────────
  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        overflow: "hidden",
      }}
    >
      {/* Left panel — config */}
      <div
        style={{
          width: 320,
          flexShrink: 0,
          borderRight: "1px solid var(--border)",
          overflow: "auto",
          padding: "20px 16px",
          display: "flex",
          flexDirection: "column",
          gap: "24px",
        }}
      >
        <SectionTitle icon={Crosshair} title="MODEL CONFIGURATION" />
        <ModelSelector
          config={config}
          onChange={(c) =>
            setConfig((prev) => (typeof c === "function" ? c(prev) : c))
          }
        />
      </div>

      {/* Right panel — video + launch */}
      <div
        style={{
          flex: 1,
          overflow: "auto",
          padding: "20px 24px",
          display: "flex",
          flexDirection: "column",
          gap: "24px",
        }}
      >
        <SectionTitle icon={FileVideo} title="VIDEO INPUT" />
        <VideoUploader onVideoReady={setVideo} />

        {video && (
          <div style={{ animation: "slideIn 0.25s ease" }}>
            {/* Video confirmed banner */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                padding: "10px 14px",
                borderRadius: "8px",
                marginBottom: 20,
                background: "rgba(61,220,132,0.08)",
                border: "1px solid rgba(61,220,132,0.3)",
              }}
            >
              <span style={{ fontSize: "11px", color: "var(--green)" }}>●</span>
              <span
                style={{
                  fontSize: "12px",
                  color: "var(--green)",
                  fontWeight: 600,
                }}
              >
                {video.filename}
              </span>
              <span
                style={{
                  fontSize: "11px",
                  color: "var(--text3)",
                  marginLeft: "auto",
                }}
              >
                ID: {video.video_id?.slice(0, 8)}…
              </span>
            </div>

            <SectionTitle icon={Radio} title="LAUNCH MODE" />
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "12px",
                marginTop: 12,
              }}
            >
              <ModeCard
                icon={Radio}
                title="Real-Time"
                desc="Stream frames via WebSocket. Watch tracking happen live with metrics overlay."
                accentColor="var(--red)"
                onClick={() => handleStart(MODE_REALTIME)}
              />
              <ModeCard
                icon={FileVideo}
                title="Save to File"
                desc="Run tracker on full video, save annotated output. Download when complete."
                accentColor="var(--accent)"
                onClick={() => handleStart(MODE_FILE)}
              />
            </div>
          </div>
        )}

        <ComparisonPanel
          runs={runHistory}
          onClear={() => setRunHistory([])}
          onAddDemo={addDemoRuns}
        />
      </div>
    </div>
  );
}

function Header({ title, sub }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: "12px",
        flexShrink: 0,
      }}
    >
      <h2 style={{ fontWeight: 800, fontSize: "16px", letterSpacing: "1px" }}>
        {title}
      </h2>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "11px",
          color: "var(--text3)",
        }}
      >
        {sub}
      </span>
    </div>
  );
}

function SectionTitle({ icon: Icon, title }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <Icon size={14} color="var(--accent)" />
      <span
        style={{
          fontSize: "11px",
          fontWeight: 700,
          letterSpacing: "2px",
          color: "var(--text2)",
        }}
      >
        {title}
      </span>
    </div>
  );
}

function ModeCard({ icon: Icon, title, desc, accentColor, onClick }) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        gap: "10px",
        padding: "20px 16px",
        borderRadius: "12px",
        textAlign: "left",
        background: hovered ? "var(--bg3)" : "var(--bg2)",
        border: `1px solid ${hovered ? accentColor : "var(--border)"}`,
        cursor: "pointer",
        transition: "all 0.15s",
      }}
    >
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: "8px",
          background: hovered ? accentColor : "var(--bg3)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "all 0.15s",
        }}
      >
        <Icon size={18} color={hovered ? "#000" : accentColor} />
      </div>
      <div>
        <div
          style={{
            fontWeight: 800,
            fontSize: "14px",
            color: "var(--text)",
            marginBottom: 4,
          }}
        >
          {title}
        </div>
        <div
          style={{ fontSize: "12px", color: "var(--text2)", lineHeight: 1.5 }}
        >
          {desc}
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "4px",
          color: accentColor,
          fontSize: "12px",
          fontWeight: 700,
          marginTop: "auto",
        }}
      >
        Launch <ChevronRight size={14} />
      </div>
    </button>
  );
}
