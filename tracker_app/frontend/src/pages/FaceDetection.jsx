import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  ScanFace,
  Upload,
  UserPlus,
  TestTube,
  Loader,
  CheckCircle,
  AlertCircle,
  FileVideo,
} from "lucide-react";
import FaceRunComparisonPanel from "../components/FaceRunComparisonPanel.jsx";

export default function FaceDetection() {
  const mediaInputRef = useRef(null);
  const sampleInputRef = useRef(null);
  const pollRef = useRef(null);
  const samplePreviewUrlsRef = useRef([]);

  const [models, setModels] = useState([]);
  const [assets, setAssets] = useState({ media: [], references: [] });
  const [runs, setRuns] = useState([]);
  const [modelName, setModelName] = useState("auraface");
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [selectedReferenceIds, setSelectedReferenceIds] = useState([]);
  const [uploadedMedia, setUploadedMedia] = useState(null);
  const [uploadedSamples, setUploadedSamples] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [modelsRes, assetsRes, runsRes] = await Promise.all([
          axios.get("/api/face/models"),
          axios.get("/api/face/test-assets"),
          axios.get("/api/face/runs"),
        ]);
        const nextModels = modelsRes.data?.models || [];
        const nextAssets = assetsRes.data || { media: [], references: [] };
        setModels(nextModels);
        setAssets(nextAssets);
        setRuns((runsRes.data?.runs || []).map(normalizeRun));
        if (nextModels.length > 0) setModelName((prev) => prev || nextModels[0].id);
        if (nextAssets.media?.length > 0) setSelectedAssetId((prev) => prev || nextAssets.media[0].asset_id);
      } catch (e) {
        setError(e.response?.data?.detail || e.message);
      }
    };
    load();
  }, []);

  useEffect(() => {
    return () => {
      clearInterval(pollRef.current);
      samplePreviewUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      if (uploadedMedia?.previewUrl) URL.revokeObjectURL(uploadedMedia.previewUrl);
    };
  }, [uploadedMedia]);

  const selectedAsset = useMemo(
    () => assets.media.find((asset) => asset.asset_id === selectedAssetId) || null,
    [assets.media, selectedAssetId],
  );

  const activeMedia = uploadedMedia || selectedAsset;

  const toggleReference = (assetId) => {
    setSelectedReferenceIds((current) =>
      current.includes(assetId) ? current.filter((id) => id !== assetId) : [...current, assetId],
    );
  };

  const handleMediaUpload = async (file) => {
    if (!file) return;
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post("/api/face/upload", form);
      if (uploadedMedia?.previewUrl) URL.revokeObjectURL(uploadedMedia.previewUrl);
      setUploadedMedia({
        ...res.data,
        kind: file.type?.startsWith("video/") ? "video" : "image",
        previewUrl: URL.createObjectURL(file),
      });
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    }
  };

  const handleSampleFiles = (files) => {
    const next = Array.from(files || []).map((file) => {
      const inferred = file.name.replace(/\.[^.]+$/, "");
      const previewUrl = URL.createObjectURL(file);
      samplePreviewUrlsRef.current.push(previewUrl);
      return {
        id: `${file.name}_${Date.now()}_${Math.random()}`,
        file,
        name: inferred,
        previewUrl,
      };
    });
    setUploadedSamples((current) => [...current, ...next]);
  };

  const updateSampleName = (id, name) => {
    setUploadedSamples((current) =>
      current.map((sample) => (sample.id === id ? { ...sample, name } : sample)),
    );
  };

  const removeSample = (id) => {
    setUploadedSamples((current) => {
      const sample = current.find((entry) => entry.id === id);
      if (sample?.previewUrl) {
        URL.revokeObjectURL(sample.previewUrl);
        samplePreviewUrlsRef.current = samplePreviewUrlsRef.current.filter((url) => url !== sample.previewUrl);
      }
      return current.filter((entry) => entry.id !== id);
    });
  };

  const submitJob = async () => {
    const mediaId = uploadedMedia?.media_id || selectedAssetId;
    if (!mediaId) {
      setError("Upload or select media before starting a face run.");
      return;
    }

    const form = new FormData();
    form.append("model_name", modelName);
    form.append("media_id", mediaId);
    form.append("reference_asset_ids", JSON.stringify(selectedReferenceIds));

    uploadedSamples.forEach((sample) => {
      if (!sample.name?.trim() || !sample.file) return;
      form.append("sample_names", sample.name.trim());
      form.append("sample_files", sample.file);
    });

    setStatus("submitting");
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      const res = await axios.post("/api/face/run/file", form);
      setJobId(res.data.job_id);
      setStatus("running");
    } catch (e) {
      setStatus("error");
      setError(e.response?.data?.detail || e.message);
    }
  };

  useEffect(() => {
    if (!jobId || status !== "running") return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`/api/face/run/status/${jobId}`);
        const job = res.data;
        setProgress(job.progress || 0);
        if (job.status === "done") {
          clearInterval(pollRef.current);
          setStatus("done");
          setResult(job);
          setRuns((current) => mergeRuns(current, [normalizeRun({ job_id: jobId, ...job })]));
        } else if (job.status === "error") {
          clearInterval(pollRef.current);
          setStatus("error");
          setError(job.error || "Face processing failed.");
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setStatus("error");
        setError(e.response?.data?.detail || e.message);
      }
    }, 1000);
    return () => clearInterval(pollRef.current);
  }, [jobId, status]);

  return (
    <div style={{ height: "100%", display: "flex", overflow: "hidden", minHeight: 0, minWidth: 0 }}>
      <div
        style={{
          width: 360,
          flexShrink: 0,
          borderRight: "1px solid var(--border)",
          overflow: "auto",
          padding: "20px 16px",
          display: "flex",
          flexDirection: "column",
          gap: "18px",
        }}
      >
        <SectionTitle icon={ScanFace} title="FACE RECOGNITION CONFIG" />

        <Panel title="MODEL">
          <select value={modelName} onChange={(e) => setModelName(e.target.value)} style={selectStyle}>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </select>
        </Panel>

        <Panel title="TARGET MEDIA">
          <input
            ref={mediaInputRef}
            type="file"
            accept="image/*,video/*"
            style={{ display: "none" }}
            onChange={(e) => handleMediaUpload(e.target.files?.[0])}
          />
          <button onClick={() => mediaInputRef.current?.click()} style={actionButtonStyle}>
            <Upload size={14} />
            Upload Image / Video
          </button>
          <select
            value={selectedAssetId}
            onChange={(e) => {
              setSelectedAssetId(e.target.value);
              if (uploadedMedia?.previewUrl) URL.revokeObjectURL(uploadedMedia.previewUrl);
              setUploadedMedia(null);
            }}
            style={{ ...selectStyle, marginTop: 10 }}
            disabled={assets.media.length === 0}
          >
            {assets.media.length === 0 ? (
              <option value="">No default test assets</option>
            ) : (
              assets.media.map((asset) => (
                <option key={asset.asset_id} value={asset.asset_id}>
                  {asset.filename}
                </option>
              ))
            )}
          </select>
          <div style={hintStyle}>
            {uploadedMedia
              ? `Uploaded: ${uploadedMedia.filename}`
              : selectedAsset
              ? `Selected server asset: ${selectedAsset.filename}`
              : "Choose media to process."}
          </div>
        </Panel>

        <Panel title="REFERENCE FACES">
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: 170, overflow: "auto" }}>
            {assets.references.length === 0 && <div style={hintStyle}>No default reference faces found.</div>}
            {assets.references.map((ref) => (
              <label
                key={ref.asset_id}
                style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "12px", color: "var(--text2)" }}
              >
                <input
                  type="checkbox"
                  checked={selectedReferenceIds.includes(ref.asset_id)}
                  onChange={() => toggleReference(ref.asset_id)}
                />
                <span>{ref.name}</span>
                <span style={{ color: "var(--text3)", marginLeft: "auto" }}>{ref.filename}</span>
              </label>
            ))}
          </div>
        </Panel>

        <Panel title="UPLOAD NAMED SAMPLE FACES">
          <input
            ref={sampleInputRef}
            type="file"
            accept="image/*"
            multiple
            style={{ display: "none" }}
            onChange={(e) => handleSampleFiles(e.target.files)}
          />
          <button onClick={() => sampleInputRef.current?.click()} style={actionButtonStyle}>
            <UserPlus size={14} />
            Add Sample Faces
          </button>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px", marginTop: 10 }}>
            {uploadedSamples.map((sample) => (
              <div
                key={sample.id}
                style={{ display: "grid", gridTemplateColumns: "42px 1fr auto", gap: "8px", alignItems: "center" }}
              >
                <img
                  src={sample.previewUrl}
                  alt={sample.name}
                  style={{ width: 42, height: 42, objectFit: "cover", borderRadius: 6, border: "1px solid var(--border)" }}
                />
                <input
                  value={sample.name}
                  onChange={(e) => updateSampleName(sample.id, e.target.value)}
                  placeholder="Person name"
                  style={inputStyle}
                />
                <button onClick={() => removeSample(sample.id)} style={miniButtonStyle}>
                  Remove
                </button>
              </div>
            ))}
            {uploadedSamples.length === 0 && <div style={hintStyle}>Upload headshots with names to extend the gallery for this run.</div>}
          </div>
        </Panel>

        <button onClick={submitJob} style={primaryButtonStyle}>
          Process Face Run
        </button>

        {status === "running" && (
          <Panel title="PROCESSING">
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <Loader size={18} style={{ animation: "spin 0.8s linear infinite", color: "var(--accent)" }} />
              <div style={{ fontSize: "12px", color: "var(--text2)" }}>Running face recognition... {progress}%</div>
            </div>
          </Panel>
        )}

        {status === "error" && error && (
          <Panel title="ERROR">
            <div style={{ display: "flex", gap: "8px", color: "var(--red)", fontSize: "12px" }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          </Panel>
        )}
      </div>

      <div
        style={{
          flex: 1,
          minWidth: 0,
          minHeight: 0,
          padding: "20px 24px",
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "18px",
        }}
      >
        <SectionTitle icon={FileVideo} title="RUN PREVIEW & RESULTS" />

        <div
          style={{
            minHeight: 320,
            maxHeight: "75vh",
            border: "1px solid var(--border)",
            borderRadius: "12px",
            background: "#000",
            display: "flex",
            alignItems: "stretch",
            justifyContent: "center",
            overflow: "auto",
            padding: "16px",
          }}
        >
          {result?.output_url ? (
            isVideoRun(result?.target_kind) ? (
              <video key={result.output_url} controls src={result.output_url} style={previewMediaStyle} />
            ) : (
              <img src={result.output_url} alt="Face recognition output" style={previewMediaStyle} />
            )
          ) : uploadedMedia?.previewUrl ? (
            isVideoRun(uploadedMedia.kind) ? (
              <video controls src={uploadedMedia.previewUrl} style={previewMediaStyle} />
            ) : (
              <img src={uploadedMedia.previewUrl} alt="Uploaded preview" style={previewMediaStyle} />
            )
          ) : (
            <div style={{ textAlign: "center", color: "var(--text3)" }}>
              <ScanFace size={52} style={{ marginBottom: 14, opacity: 0.35 }} />
              <div>Upload media or choose a default test asset to begin.</div>
            </div>
          )}
        </div>

        {result && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <Panel title="RUN SUMMARY">
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {Object.entries(result.stats || {}).map(([key, value]) => (
                  <div key={key} style={statRowStyle}>
                    <span style={{ color: "var(--text3)" }}>{humanize(key)}</span>
                    <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)" }}>
                      {Array.isArray(value) ? value.join(", ") : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="DETECTIONS">
              <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: 260, overflow: "auto" }}>
                {flattenDetections(result.detections).length === 0 ? (
                  <div style={hintStyle}>No faces were detected or identified.</div>
                ) : (
                  flattenDetections(result.detections).map((detection, idx) => (
                    <div key={`${detection.name}_${idx}`} style={statRowStyle}>
                      <span style={{ color: detection.name === "unknown" ? "var(--red)" : "var(--text)" }}>
                        {detection.frame_idx !== undefined ? `Frame ${detection.frame_idx}: ` : ""}
                        {detection.name}
                      </span>
                      <span style={{ color: "var(--accent)", fontFamily: "var(--font-mono)" }}>
                        {(detection.confidence || 0).toFixed(2)}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </Panel>
          </div>
        )}

        {status === "done" && result?.output_url && (
          <a href={result.output_url} target="_blank" rel="noreferrer" style={downloadLinkStyle}>
            <CheckCircle size={16} />
            Open processed output
          </a>
        )}

        <FaceRunComparisonPanel runs={runs} onClear={() => setRuns([])} />
      </div>
    </div>
  );
}

function mergeRuns(currentRuns, nextRuns) {
  const map = new Map(currentRuns.map((run) => [run.id, run]));
  nextRuns.forEach((run) => {
    map.set(run.id, run);
  });
  return Array.from(map.values()).sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
}

function normalizeRun(run) {
  const stats = run.stats || {};
  return {
    id: run.job_id,
    status: run.status,
    model: run.model_name,
    media: run.target_filename,
    kind: run.target_kind,
    createdAt: run.created_at || 0,
    completedAt: run.completed_at || 0,
    outputUrl: run.output_url,
    metrics: {
      faces_detected: stats.total_faces_detected ?? stats.faces_detected ?? 0,
      identified_faces: stats.identified_faces ?? 0,
      unknown_faces: stats.unknown_faces ?? 0,
      frames_processed: stats.frames_processed,
      avg_match_confidence: stats.avg_match_confidence ?? 0,
      inference_time_ms: stats.avg_inference_time_ms ?? stats.inference_time_ms,
      enrolled_count: stats.enrolled_count ?? 0,
    },
  };
}

function flattenDetections(detections) {
  if (!Array.isArray(detections)) return [];
  return detections.flatMap((entry) => {
    if (entry?.frame_idx !== undefined && Array.isArray(entry.detections)) {
      return entry.detections.map((detection) => ({ ...detection, frame_idx: entry.frame_idx }));
    }
    return [entry];
  });
}

function humanize(value) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function isVideoRun(kind) {
  return kind === "video";
}

function SectionTitle({ icon: Icon, title }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <Icon size={15} color="var(--accent)" />
      <span style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "2px", color: "var(--text2)" }}>
        {title}
      </span>
    </div>
  );
}

function Panel({ title, children }) {
  return (
    <div
      style={{
        padding: "14px",
        borderRadius: "10px",
        background: "var(--bg3)",
        border: "1px solid var(--border)",
      }}
    >
      <div style={{ fontSize: "11px", fontWeight: 600, letterSpacing: "1px", color: "var(--text3)", marginBottom: 10 }}>
        {title}
      </div>
      {children}
    </div>
  );
}

const selectStyle = {
  width: "100%",
  padding: "9px 10px",
  borderRadius: "8px",
  background: "var(--bg2)",
  border: "1px solid var(--border)",
  color: "var(--text)",
  fontSize: "12px",
};

const inputStyle = {
  width: "100%",
  padding: "9px 10px",
  borderRadius: "8px",
  background: "var(--bg2)",
  border: "1px solid var(--border)",
  color: "var(--text)",
  fontSize: "12px",
};

const actionButtonStyle = {
  width: "100%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "8px",
  padding: "10px 12px",
  borderRadius: "8px",
  background: "var(--bg2)",
  border: "1px solid var(--border)",
  color: "var(--text2)",
  fontSize: "12px",
};

const primaryButtonStyle = {
  width: "100%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "8px",
  padding: "12px 14px",
  borderRadius: "10px",
  background: "var(--accent)",
  border: "none",
  color: "#000",
  fontSize: "13px",
  fontWeight: 800,
  letterSpacing: "0.8px",
  cursor: "pointer",
};

const miniButtonStyle = {
  padding: "8px 10px",
  borderRadius: "8px",
  background: "var(--bg2)",
  border: "1px solid var(--border)",
  color: "var(--text2)",
  fontSize: "12px",
};

const hintStyle = {
  fontSize: "12px",
  color: "var(--text3)",
  lineHeight: 1.5,
  marginTop: "8px",
};

const statRowStyle = {
  display: "flex",
  justifyContent: "space-between",
  gap: "12px",
  fontSize: "12px",
};

const downloadLinkStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: "8px",
  width: "fit-content",
  padding: "10px 12px",
  borderRadius: "8px",
  textDecoration: "none",
  background: "rgba(61,220,132,0.08)",
  border: "1px solid rgba(61,220,132,0.3)",
  color: "var(--green)",
  fontSize: "12px",
  fontWeight: 600,
};

const previewMediaStyle = {
  display: "block",
  margin: "auto",
  maxWidth: "100%",
  maxHeight: "calc(75vh - 32px)",
  width: "auto",
  height: "auto",
  objectFit: "contain",
  objectPosition: "center top",
};
