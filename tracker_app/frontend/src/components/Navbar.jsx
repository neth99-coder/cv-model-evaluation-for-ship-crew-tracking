import React from "react";
import { Crosshair, ScanFace } from "lucide-react";

export default function Navbar({ page, setPage }) {
  return (
    <nav
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0",
        height: "48px",
        flexShrink: 0,
        borderBottom: "1px solid var(--border)",
        background: "var(--bg2)",
        padding: "0 16px",
      }}
    >
      {/* Brand */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginRight: "24px",
        }}
      >
        <div
          style={{
            width: 28,
            height: 28,
            borderRadius: "6px",
            background: "var(--accent-dim)",
            border: "1px solid var(--accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Crosshair size={14} color="var(--accent)" />
        </div>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "13px",
            fontWeight: 700,
            color: "var(--text)",
            letterSpacing: "1px",
          }}
        >
          TRACKER<span style={{ color: "var(--accent)" }}>APP</span>
        </span>
      </div>

      {/* Nav tabs */}
      <div style={{ display: "flex", gap: "4px" }}>
        <NavTab
          icon={Crosshair}
          label="Object Tracking"
          active={page === "tracking"}
          onClick={() => setPage("tracking")}
        />
        <NavTab
          icon={ScanFace}
          label="Face Recognition"
          active={page === "face"}
          onClick={() => setPage("face")}
        />
      </div>
    </nav>
  );
}

function NavTab({ icon: Icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: "6px",
        padding: "6px 14px",
        borderRadius: "6px",
        background: active ? "var(--accent-dim)" : "transparent",
        border: active ? "1px solid var(--accent)" : "1px solid transparent",
        color: active ? "var(--accent)" : "var(--text2)",
        fontWeight: active ? 700 : 500,
        fontSize: "12px",
        letterSpacing: "0.5px",
        cursor: "pointer",
        transition: "all 0.15s",
      }}
    >
      <Icon size={13} />
      {label}
    </button>
  );
}
