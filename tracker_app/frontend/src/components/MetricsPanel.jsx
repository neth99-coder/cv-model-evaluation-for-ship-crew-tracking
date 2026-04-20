import React from "react";
import {
  Activity,
  Users,
  Zap,
  Clock,
  TrendingUp,
  BarChart2,
} from "lucide-react";

function Stat({ icon: Icon, label, value, unit, color }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        padding: "12px",
        background: "var(--bg3)",
        borderRadius: "8px",
        border: "1px solid var(--border)",
        minWidth: 0,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
        <Icon size={12} color={color || "var(--text3)"} />
        <span
          style={{
            fontSize: "10px",
            letterSpacing: "1px",
            color: "var(--text3)",
            fontWeight: 600,
          }}
        >
          {label}
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: "4px" }}>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "20px",
            fontWeight: 700,
            color: color || "var(--text)",
          }}
        >
          {value ?? "—"}
        </span>
        {unit && (
          <span style={{ fontSize: "11px", color: "var(--text3)" }}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

export default function MetricsPanel({ metrics, live, frameInfo }) {
  if (!metrics && !live) return null;

  const fps = live?.fps ?? metrics?.avg_fps;
  const trackerFps = live?.tracker_fps;
  const tracks = live?.track_count ?? metrics?.avg_tracks_per_frame;
  const uniqueTracks = metrics?.unique_tracks;
  const totalFrames = live?.frame_idx ?? metrics?.total_frames_processed;
  const maxTracks = metrics?.max_simultaneous_tracks;
  const avgLifetime = metrics?.avg_track_lifetime_frames;
  const mapIou50 = metrics?.map_iou_50;
  const mapSmall = metrics?.map_small;

  return (
    <div
      style={{
        padding: "16px",
        background: "var(--bg2)",
        borderRadius: "10px",
        border: "1px solid var(--border)",
        animation: "slideIn 0.3s ease",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "12px",
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
          {live ? "LIVE METRICS" : "SESSION METRICS"}
        </span>
        {live && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: "10px",
              fontFamily: "var(--font-mono)",
              color: "var(--green)",
            }}
          >
            ● LIVE
          </span>
        )}
      </div>

      <div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}
      >
        <Stat
          icon={Zap}
          label={live ? "PIPELINE FPS" : "FPS"}
          value={fps?.toFixed(1)}
          color="var(--accent)"
        />
        {live && trackerFps !== undefined && (
          <Stat
            icon={Zap}
            label="TRACKER FPS"
            value={trackerFps?.toFixed(1)}
            color="var(--text2)"
          />
        )}
        <Stat
          icon={Users}
          label="TRACKS/FRAME"
          value={typeof tracks === "number" ? tracks.toFixed(1) : tracks}
          color="var(--blue)"
        />
        {uniqueTracks !== undefined && (
          <Stat
            icon={Activity}
            label="UNIQUE IDs"
            value={uniqueTracks}
            color="var(--green)"
          />
        )}
        {totalFrames !== undefined && (
          <Stat icon={Clock} label="FRAMES" value={totalFrames} />
        )}
        {maxTracks !== undefined && (
          <Stat icon={TrendingUp} label="MAX TRACKS" value={maxTracks} />
        )}
        {avgLifetime !== undefined && (
          <Stat
            icon={Activity}
            label="AVG LIFETIME"
            value={avgLifetime?.toFixed(1)}
            unit="fr"
          />
        )}
        {mapIou50 !== undefined && (
          <Stat
            icon={TrendingUp}
            label="mAP@IoU=0.50"
            value={
              typeof mapIou50 === "number" ? mapIou50.toFixed(3) : mapIou50
            }
          />
        )}
        {mapSmall !== undefined && (
          <Stat
            icon={TrendingUp}
            label="mAP SMALL"
            value={
              typeof mapSmall === "number" ? mapSmall.toFixed(3) : mapSmall
            }
            color="var(--green)"
          />
        )}
      </div>

      {metrics?.map_note && (
        <div
          style={{ marginTop: "8px", fontSize: "11px", color: "var(--text3)" }}
        >
          {metrics.map_note}
        </div>
      )}

      {frameInfo && (
        <div style={{ marginTop: "10px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: "4px",
            }}
          >
            <span style={{ fontSize: "10px", color: "var(--text3)" }}>
              PROGRESS
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "11px",
                color: "var(--text2)",
              }}
            >
              {frameInfo.current} / {frameInfo.total}
            </span>
          </div>
          <div
            style={{
              height: "4px",
              background: "var(--bg3)",
              borderRadius: "2px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${Math.min(100, (frameInfo.current / frameInfo.total) * 100)}%`,
                background: "var(--accent)",
                borderRadius: "2px",
                transition: "width 0.3s",
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
