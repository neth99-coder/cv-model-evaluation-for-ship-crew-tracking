import { useEffect, useMemo, useState } from "react";

type SourceMode = "realtime" | "uploaded_file";
type TrackingBackend = "auto" | "boxmot" | "custom";

type DetectionModelOption = {
  id: string;
  label: string;
};

type TrackerOption = {
  id: string;
  label: string;
  supportsBoxmot: boolean;
  supportsCustom: boolean;
  requiresReid: boolean;
};

type ReidOption = {
  id: string;
  label: string;
  provider: "boxmot-osnet" | "boxmot-fastreid" | "custom" | "none";
};

type VideoMeta = {
  name: string;
  sizeBytes: number;
  durationSec: number;
  width: number;
  height: number;
};

type TrackingInfo = {
  app: string;
  generatedAt: string;
  runWindow?: {
    startedAt: string;
    endedAt: string;
  };
  sourceMode: string;
  selections: {
    detectionModel: string;
    trackingBackend: string;
    trackingModel: string;
    reidModel: string;
    conf: number;
    imgsz: number;
  };
  trackingOutputFile: {
    fileName: string;
    absolutePath: string;
    relativePath: string;
    sizeBytes: number;
    downloadUrl: string;
  };
  execution?: {
    python: string;
    command: string;
    stdout: string;
    stderr: string;
  };
};

const DETECTION_MODELS: DetectionModelOption[] = [
  { id: "yolov8n.pt", label: "YOLOv8 Nano" },
  { id: "yolov8s.pt", label: "YOLOv8 Small" },
  { id: "yolov8m.pt", label: "YOLOv8 Medium" },
  { id: "yolov8l.pt", label: "YOLOv8 Large" },
  { id: "yolov8x.pt", label: "YOLOv8 XL" },
  { id: "yolov5s.pt", label: "YOLOv5 Small" },
  { id: "yolov5m.pt", label: "YOLOv5 Medium" },
  { id: "yolov5l.pt", label: "YOLOv5 Large" },
  { id: "yolov5x.pt", label: "YOLOv5 XL" },
  { id: "faster_rcnn", label: "Faster R-CNN" },
  { id: "ssd_mobilenet", label: "SSD MobileNet" },
];

const TRACKING_MODELS: TrackerOption[] = [
  {
    id: "deepsort",
    label: "DeepSORT",
    supportsBoxmot: true,
    supportsCustom: true,
    requiresReid: true,
  },
  {
    id: "bytetrack",
    label: "ByteTrack",
    supportsBoxmot: true,
    supportsCustom: true,
    requiresReid: false,
  },
  {
    id: "botsort",
    label: "BoT-SORT",
    supportsBoxmot: true,
    supportsCustom: true,
    requiresReid: true,
  },
  {
    id: "strongsort",
    label: "StrongSORT",
    supportsBoxmot: true,
    supportsCustom: false,
    requiresReid: true,
  },
  {
    id: "deepocsort",
    label: "DeepOCSORT",
    supportsBoxmot: true,
    supportsCustom: false,
    requiresReid: true,
  },
  {
    id: "ocsort",
    label: "OCSORT",
    supportsBoxmot: true,
    supportsCustom: false,
    requiresReid: false,
  },
  {
    id: "hybridsort",
    label: "HybridSORT",
    supportsBoxmot: true,
    supportsCustom: false,
    requiresReid: true,
  },
  {
    id: "boosttrack",
    label: "BoostTrack",
    supportsBoxmot: true,
    supportsCustom: false,
    requiresReid: true,
  },
  {
    id: "sfsort",
    label: "SFSORT",
    supportsBoxmot: true,
    supportsCustom: false,
    requiresReid: false,
  },
];

const REID_MODELS_BOXMOT: ReidOption[] = [
  {
    id: "osnet_x0_25_msmt17",
    label: "OSNet x0.25 (MSMT17)",
    provider: "boxmot-osnet",
  },
  {
    id: "osnet_x0_5_msmt17",
    label: "OSNet x0.5 (MSMT17)",
    provider: "boxmot-osnet",
  },
  {
    id: "osnet_x1_0_msmt17",
    label: "OSNet x1.0 (MSMT17)",
    provider: "boxmot-osnet",
  },
  {
    id: "fastreid_sbs_s50",
    label: "FastReID SBS-S50",
    provider: "boxmot-fastreid",
  },
  {
    id: "fastreid_sbs_r50",
    label: "FastReID SBS-R50",
    provider: "boxmot-fastreid",
  },
];

const REID_MODELS_CUSTOM: ReidOption[] = [
  {
    id: "custom_internal",
    label: "Custom tracker internal Re-ID",
    provider: "custom",
  },
];

const REID_NONE: ReidOption[] = [
  { id: "none", label: "No Re-ID", provider: "none" },
];

function triggerDownload(url: string, filenameHint?: string) {
  const anchor = document.createElement("a");
  anchor.href = url;
  if (filenameHint) {
    anchor.download = filenameHint;
  }
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

async function readApiJson<T>(resp: Response): Promise<T> {
  const contentType = resp.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return (await resp.json()) as T;
  }

  const text = await resp.text();
  if (text.startsWith("<!DOCTYPE") || text.startsWith("<html")) {
    throw new Error(
      "The GUI received HTML instead of API JSON. Start the backend with `npm run api` in person-tracking-eval-gui and make sure the Vite `/api` proxy is active.",
    );
  }

  throw new Error(text || "Unexpected non-JSON response from tracking API.");
}

function App() {
  const [detectionModel, setDetectionModel] = useState<string>("yolov8n.pt");
  const [trackingBackend, setTrackingBackend] =
    useState<TrackingBackend>("auto");
  const [trackingModel, setTrackingModel] = useState<string>("deepsort");
  const [reidModel, setReidModel] = useState<string>("osnet_x0_25_msmt17");
  const [sourceMode, setSourceMode] = useState<SourceMode>("uploaded_file");

  const [conf, setConf] = useState<number>(0.25);
  const [imgsz, setImgsz] = useState<number>(640);
  const [maxFrames, setMaxFrames] = useState<number>(0);
  const [timeLimit, setTimeLimit] = useState<number>(0);

  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);

  const [trackingInfo, setTrackingInfo] = useState<TrackingInfo | null>(null);
  const [videoDownloadUrl, setVideoDownloadUrl] = useState<string>("");
  const [infoDownloadUrl, setInfoDownloadUrl] = useState<string>("");
  const [realtimeStatus, setRealtimeStatus] = useState<string>("");
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const availableTrackingModels = useMemo(() => {
    if (trackingBackend === "custom") {
      return TRACKING_MODELS.filter((model) => model.supportsCustom);
    }
    if (trackingBackend === "boxmot") {
      return TRACKING_MODELS.filter((model) => model.supportsBoxmot);
    }
    return TRACKING_MODELS;
  }, [trackingBackend]);

  const selectedTrackingModel = useMemo(
    () =>
      availableTrackingModels.find((model) => model.id === trackingModel) ??
      null,
    [availableTrackingModels, trackingModel],
  );

  const availableReidModels = useMemo(() => {
    if (!selectedTrackingModel) {
      return REID_NONE;
    }
    if (!selectedTrackingModel.requiresReid) {
      return REID_NONE;
    }
    if (trackingBackend === "custom") {
      return REID_MODELS_CUSTOM;
    }
    return REID_MODELS_BOXMOT;
  }, [selectedTrackingModel, trackingBackend]);

  const infoPreview = useMemo(
    () => (trackingInfo ? JSON.stringify(trackingInfo, null, 2) : ""),
    [trackingInfo],
  );

  useEffect(() => {
    if (!availableTrackingModels.some((model) => model.id === trackingModel)) {
      setTrackingModel(availableTrackingModels[0]?.id ?? "");
    }
  }, [availableTrackingModels, trackingModel]);

  useEffect(() => {
    if (!availableReidModels.some((model) => model.id === reidModel)) {
      setReidModel(availableReidModels[0]?.id ?? "none");
    }
  }, [availableReidModels, reidModel]);

  useEffect(() => {
    if (sourceMode !== "realtime") {
      setRealtimeStatus("");
    }
  }, [sourceMode]);

  async function handleVideoUpload(file: File | null) {
    setError("");
    setVideoFile(file);
    setVideoMeta(null);
    setTrackingInfo(null);
    setVideoDownloadUrl("");
    setInfoDownloadUrl("");

    if (!file) {
      return;
    }

    const objectUrl = URL.createObjectURL(file);

    try {
      const meta = await new Promise<VideoMeta>((resolve, reject) => {
        const video = document.createElement("video");
        video.preload = "metadata";
        video.src = objectUrl;

        video.onloadedmetadata = () => {
          resolve({
            name: file.name,
            sizeBytes: file.size,
            durationSec: Number.isFinite(video.duration) ? video.duration : 0,
            width: video.videoWidth || 0,
            height: video.videoHeight || 0,
          });
        };

        video.onerror = () =>
          reject(new Error("Unable to read video metadata"));
      });

      setVideoMeta(meta);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  }

  async function startRealtimeTracking() {
    setError("");
    setTrackingInfo(null);
    setVideoDownloadUrl("");
    setInfoDownloadUrl("");
    setRealtimeStatus("");
    setIsExecuting(true);

    if (!videoFile) {
      setError("Upload a video before starting realtime tracking.");
      setIsExecuting(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("video", videoFile);
      formData.append("detectionModel", detectionModel);
      formData.append("trackingBackend", trackingBackend);
      formData.append("trackingModel", trackingModel);
      formData.append("reidModel", reidModel);
      formData.append("conf", String(conf));
      formData.append("imgsz", String(imgsz));
      formData.append("maxFrames", String(maxFrames > 0 ? maxFrames : 0));
      formData.append("timeLimit", String(timeLimit > 0 ? timeLimit : 0));

      const resp = await fetch("/api/track/realtime/start", {
        method: "POST",
        body: formData,
      });

      const payload = await readApiJson<{
        ok?: boolean;
        error?: string;
        message?: string;
      }>(resp);
      if (!resp.ok || !payload.ok) {
        throw new Error(
          payload.error || "Failed to start realtime tracking window.",
        );
      }

      setRealtimeStatus(payload.message || "Realtime tracking window started.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setIsExecuting(false);
    }
  }

  async function stopRealtimeTracking() {
    setError("");
    try {
      const resp = await fetch("/api/track/realtime/stop", { method: "POST" });
      const payload = await readApiJson<{
        ok?: boolean;
        error?: string;
        message?: string;
      }>(resp);
      if (!resp.ok || !payload.ok) {
        throw new Error(
          payload.error || "Failed to stop realtime tracking window.",
        );
      }
      setRealtimeStatus(payload.message || "Realtime tracking window stopped.");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function runFileTracking() {
    setError("");
    setRealtimeStatus("");

    if (!videoFile) {
      setError("Upload a video before executing tracking.");
      return;
    }

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("detectionModel", detectionModel);
    formData.append("trackingBackend", trackingBackend);
    formData.append("trackingModel", trackingModel);
    formData.append("reidModel", reidModel);
    formData.append("conf", String(conf));
    formData.append("imgsz", String(imgsz));
    formData.append("maxFrames", String(maxFrames > 0 ? maxFrames : 0));
    formData.append("timeLimit", String(timeLimit > 0 ? timeLimit : 0));

    setIsExecuting(true);
    try {
      const resp = await fetch("/api/track/file", {
        method: "POST",
        body: formData,
      });

      const payload = await readApiJson<{
        ok?: boolean;
        error?: string;
        info?: TrackingInfo;
        infoDownloadUrl?: string;
        videoDownloadUrl?: string;
      }>(resp);

      if (!resp.ok || !payload.ok || !payload.info) {
        throw new Error(payload.error || "Tracking execution failed.");
      }

      setTrackingInfo(payload.info);
      setInfoDownloadUrl(payload.infoDownloadUrl || "");
      setVideoDownloadUrl(
        payload.videoDownloadUrl ||
          payload.info.trackingOutputFile.downloadUrl ||
          "",
      );

      if (payload.videoDownloadUrl) {
        triggerDownload(
          payload.videoDownloadUrl,
          payload.info.trackingOutputFile.fileName,
        );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setIsExecuting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">Person Tracking Evaluation GUI</p>
        <h1>Execute Tracking by Selected Options</h1>
        <p className="hero-subtitle">
          Realtime mode uses the uploaded video and opens the native Python
          tracking window with person tracking. Downloaded-file mode executes
          tracking and downloads the annotated output file.
        </p>
      </header>

      <main className="grid">
        <section className="card">
          <h2>1. Object Detection Model</h2>
          <select
            value={detectionModel}
            onChange={(e) => setDetectionModel(e.target.value)}
          >
            {DETECTION_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </section>

        <section className="card">
          <h2>2. Tracking Backend</h2>
          <select
            value={trackingBackend}
            onChange={(e) =>
              setTrackingBackend(e.target.value as TrackingBackend)
            }
          >
            <option value="auto">auto (prefer BoxMOT, fallback custom)</option>
            <option value="boxmot">boxmot (BoxMOT only)</option>
            <option value="custom">custom (custom adapters only)</option>
          </select>
        </section>

        <section className="card">
          <h2>3. Tracking Model</h2>
          <select
            value={trackingModel}
            onChange={(e) => setTrackingModel(e.target.value)}
          >
            {availableTrackingModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </section>

        <section className="card">
          <h2>4. Pluggable Re-ID Model</h2>
          <select
            value={reidModel}
            onChange={(e) => setReidModel(e.target.value)}
          >
            {availableReidModels.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <p className="muted">
            Includes FastReID options and applies compatibility by backend and
            tracker.
            {selectedTrackingModel && !selectedTrackingModel.requiresReid
              ? " The selected tracker does not use Re-ID, so only `No Re-ID` is available."
              : ""}
          </p>
        </section>

        <section className="card span-2">
          <h2>5. Source Type</h2>
          <div className="inline-options">
            <label>
              <input
                type="radio"
                name="source-mode"
                checked={sourceMode === "realtime"}
                onChange={() => setSourceMode("realtime")}
              />
              Realtime
            </label>
            <label>
              <input
                type="radio"
                name="source-mode"
                checked={sourceMode === "uploaded_file"}
                onChange={() => setSourceMode("uploaded_file")}
              />
              Downloaded output file
            </label>
          </div>

          {sourceMode === "realtime" ? (
            <p className="muted">
              Realtime mode uses the uploaded video and launches the Python
              tracking window. There is no browser preview.
            </p>
          ) : (
            <p className="muted">
              Uploaded-file mode executes tracking and downloads the annotated
              output file.
            </p>
          )}
        </section>

        <section className="card">
          <h2>6. Detection Runtime</h2>
          <label>
            Confidence threshold
            <input
              type="number"
              step="0.01"
              min={0}
              max={1}
              value={conf}
              onChange={(e) => setConf(Number(e.target.value) || 0.25)}
            />
          </label>
          <label>
            Image size
            <input
              type="number"
              min={320}
              value={imgsz}
              onChange={(e) => setImgsz(Number(e.target.value) || 640)}
            />
          </label>
        </section>

        <section className="card">
          <h2>7. Limits</h2>
          <label>
            Max frames (0 = unlimited)
            <input
              type="number"
              min={0}
              value={maxFrames}
              onChange={(e) => setMaxFrames(Number(e.target.value) || 0)}
            />
          </label>
          <label>
            Time limit seconds (0 = unlimited)
            <input
              type="number"
              min={0}
              value={timeLimit}
              onChange={(e) => setTimeLimit(Number(e.target.value) || 0)}
            />
          </label>
        </section>

        <section className="card span-2">
          <h2>8. Upload Test Video</h2>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              void handleVideoUpload(file);
            }}
          />
          {videoMeta ? (
            <div className="meta-grid">
              <div>
                <span className="meta-label">File</span>
                <strong>{videoMeta.name}</strong>
              </div>
              <div>
                <span className="meta-label">Size</span>
                <strong>
                  {(videoMeta.sizeBytes / (1024 * 1024)).toFixed(2)} MB
                </strong>
              </div>
              <div>
                <span className="meta-label">Duration</span>
                <strong>{videoMeta.durationSec.toFixed(2)} s</strong>
              </div>
              <div>
                <span className="meta-label">Resolution</span>
                <strong>
                  {videoMeta.width} x {videoMeta.height}
                </strong>
              </div>
            </div>
          ) : (
            <p className="muted">No video uploaded yet.</p>
          )}
        </section>

        <section className="card span-2 actions">
          {sourceMode === "realtime" ? (
            <>
              <button
                className="primary"
                onClick={startRealtimeTracking}
                disabled={isExecuting}
              >
                {isExecuting
                  ? "Starting realtime..."
                  : "Start Realtime Tracking"}
              </button>
              <button className="secondary" onClick={stopRealtimeTracking}>
                Stop Realtime Tracking
              </button>
              {realtimeStatus && <p className="muted">{realtimeStatus}</p>}
            </>
          ) : (
            <button
              className="primary"
              onClick={runFileTracking}
              disabled={isExecuting}
            >
              {isExecuting
                ? "Running tracking..."
                : "Run Tracking and Download Output"}
            </button>
          )}

          {videoDownloadUrl && (
            <button
              className="secondary"
              onClick={() => triggerDownload(videoDownloadUrl)}
            >
              Download Annotated Video
            </button>
          )}

          {infoDownloadUrl && (
            <button
              className="secondary"
              onClick={() => triggerDownload(infoDownloadUrl)}
            >
              Download Info JSON
            </button>
          )}

          {error && <p className="error">{error}</p>}
        </section>

        <section className="card span-2">
          <h2>Info JSON Preview</h2>
          <pre>
            {infoPreview ||
              "Run file tracking to generate info JSON with tracking output file details."}
          </pre>
        </section>
      </main>
    </div>
  );
}

export default App;
