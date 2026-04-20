import React, { useEffect, useRef, useState } from "react";
import { Upload, Film, TestTube } from "lucide-react";
import axios from "axios";

export default function VideoUploader({ onVideoReady }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [testVideos, setTestVideos] = useState([]);
  const [selectedTestVideoId, setSelectedTestVideoId] = useState("");

  useEffect(() => {
    const loadTestVideos = async () => {
      try {
        const res = await axios.get("/api/test-videos");
        const videos = res.data?.videos || [];
        setTestVideos(videos);
        if (videos.length > 0) {
          setSelectedTestVideoId(videos[0].video_id);
        }
      } catch (e) {
        console.warn("Failed to load test video list", e);
      }
    };
    loadTestVideos();
  }, []);

  const handleFile = async (file) => {
    if (!file || !file.type.startsWith("video/")) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post("/api/upload", form);
      setUploadedFile({ name: file.name, ...res.data });
      onVideoReady({ ...res.data, filename: file.name });
    } catch (e) {
      console.error(e);
      alert("Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleTestVideo = async () => {
    if (!selectedTestVideoId) {
      alert("No test videos available on server");
      return;
    }
    try {
      const selected = testVideos.find(
        (v) => v.video_id === selectedTestVideoId,
      );
      const filename = selected?.filename;
      const res = await axios.get("/api/test-video", {
        params: filename ? { name: filename } : undefined,
      });
      setUploadedFile({ name: res.data.filename, ...res.data });
      onVideoReady({ ...res.data, filename: res.data.filename });
    } catch (e) {
      alert("Selected test video not found on server");
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files[0]);
        }}
        style={{
          border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
          borderRadius: "10px",
          padding: "28px 16px",
          textAlign: "center",
          cursor: "pointer",
          background: dragging ? "var(--accent-dim)" : "var(--bg3)",
          transition: "all 0.15s",
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {uploading ? (
          <div style={{ color: "var(--accent)" }}>
            <div
              style={{
                width: 24,
                height: 24,
                border: "2px solid var(--accent)",
                borderTopColor: "transparent",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
                margin: "0 auto 8px",
              }}
            />
            Uploading…
          </div>
        ) : uploadedFile ? (
          <div style={{ color: "var(--green)" }}>
            <Film
              size={24}
              style={{ margin: "0 auto 8px", display: "block" }}
            />
            <div style={{ fontWeight: 700, fontSize: "13px" }}>
              {uploadedFile.name}
            </div>
            <div
              style={{
                fontSize: "11px",
                color: "var(--text3)",
                marginTop: "2px",
              }}
            >
              Click to replace
            </div>
          </div>
        ) : (
          <div style={{ color: "var(--text2)" }}>
            <Upload
              size={24}
              style={{
                margin: "0 auto 8px",
                display: "block",
                color: "var(--text3)",
              }}
            />
            <div style={{ fontWeight: 600, fontSize: "13px" }}>
              Drop video here or click to browse
            </div>
            <div
              style={{
                fontSize: "11px",
                color: "var(--text3)",
                marginTop: "4px",
              }}
            >
              MP4, AVI, MOV, MKV
            </div>
          </div>
        )}
      </div>

      {/* Use test video */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto",
          gap: "8px",
        }}
      >
        <select
          value={selectedTestVideoId}
          onChange={(e) => setSelectedTestVideoId(e.target.value)}
          style={{
            padding: "9px 10px",
            borderRadius: "8px",
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            color: "var(--text2)",
            fontSize: "12px",
          }}
          disabled={testVideos.length === 0}
        >
          {testVideos.length === 0 ? (
            <option value="">No test videos found in backend/test</option>
          ) : (
            testVideos.map((v) => (
              <option key={v.video_id} value={v.video_id}>
                {v.filename}
              </option>
            ))
          )}
        </select>
        <button
          onClick={handleTestVideo}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
            padding: "9px 12px",
            borderRadius: "8px",
            background: "var(--bg3)",
            border: "1px solid var(--border)",
            color: "var(--text2)",
            fontSize: "12px",
            fontWeight: 600,
            cursor: testVideos.length === 0 ? "not-allowed" : "pointer",
            transition: "all 0.15s",
            opacity: testVideos.length === 0 ? 0.6 : 1,
          }}
          disabled={testVideos.length === 0}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "var(--accent)";
            e.currentTarget.style.color = "var(--accent)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "var(--border)";
            e.currentTarget.style.color = "var(--text2)";
          }}
        >
          <TestTube size={14} />
          USE TEST VIDEO
        </button>
      </div>
    </div>
  );
}
