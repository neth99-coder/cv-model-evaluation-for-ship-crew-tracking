import express from "express";
import multer from "multer";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import crypto from "node:crypto";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const TRACKING_DIR = path.join(REPO_ROOT, "person_tracking");
const PYTHON_BIN =
  process.env.TRACKING_PYTHON || path.join(REPO_ROOT, ".venv", "bin", "python");

const RUNTIME_DIR = path.join(__dirname, "runtime");
const UPLOADS_DIR = path.join(RUNTIME_DIR, "uploads");
const OUTPUTS_DIR = path.join(RUNTIME_DIR, "outputs");

for (const dir of [RUNTIME_DIR, UPLOADS_DIR, OUTPUTS_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

const app = express();
const upload = multer({ dest: UPLOADS_DIR });
let realtimeProcess = null;
let realtimeUploadPath = null;

app.use(express.json());

function runProcess(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: { ...process.env, ...(options.env || {}) },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => {
      reject(err);
    });

    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stdout, stderr, code });
      } else {
        reject(
          new Error(`Process exited with code ${code}\n${stderr || stdout}`),
        );
      }
    });
  });
}

function safeName(input) {
  return String(input || "").replace(/[^a-zA-Z0-9_.-]/g, "_");
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, python: PYTHON_BIN });
});

app.post(
  "/api/track/realtime/start",
  upload.single("video"),
  async (req, res) => {
    try {
      if (
        realtimeProcess &&
        realtimeProcess.exitCode === null &&
        !realtimeProcess.killed
      ) {
        res
          .status(409)
          .json({ error: "A realtime tracking window is already running." });
        return;
      }

      if (!req.file) {
        res
          .status(400)
          .json({
            error: "Missing uploaded video file for realtime tracking.",
          });
        return;
      }

      const detectionModel = String(
        req.body.detectionModel || "yolov8n.pt",
      ).trim();
      const trackingBackend = String(req.body.trackingBackend || "auto").trim();
      const trackingModel = String(
        req.body.trackingModel || "bytetrack",
      ).trim();
      const reidModel = String(req.body.reidModel || "none").trim();
      const conf = String(req.body.conf || "0.25").trim();
      const imgsz = String(req.body.imgsz || "640").trim();
      const maxFrames = String(req.body.maxFrames || "0").trim();
      const timeLimit = String(req.body.timeLimit || "0").trim();

      const args = [
        path.join(TRACKING_DIR, "live_tracking.py"),
        "--model",
        trackingModel,
        "--backend",
        trackingBackend,
        "--source",
        req.file.path,
        "--detector",
        detectionModel,
        "--conf",
        conf,
        "--imgsz",
        imgsz,
      ];

      if (
        reidModel &&
        reidModel !== "none" &&
        reidModel !== "custom_internal"
      ) {
        args.push("--reid", reidModel);
      }
      if (Number(maxFrames) > 0) {
        args.push("--max-frames", maxFrames);
      }
      if (Number(timeLimit) > 0) {
        args.push("--time-limit", timeLimit);
      }

      realtimeProcess = spawn(PYTHON_BIN, args, {
        cwd: TRACKING_DIR,
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
      });
      realtimeUploadPath = req.file.path;

      realtimeProcess.stdout.on("data", (chunk) => {
        process.stdout.write(`[realtime-window] ${chunk.toString()}`);
      });
      realtimeProcess.stderr.on("data", (chunk) => {
        process.stderr.write(`[realtime-window] ${chunk.toString()}`);
      });
      realtimeProcess.on("close", () => {
        if (realtimeUploadPath) {
          fs.rm(realtimeUploadPath, { force: true }, () => {});
          realtimeUploadPath = null;
        }
        realtimeProcess = null;
      });

      res.json({
        ok: true,
        message: "Realtime tracking window started for the uploaded video.",
        source: req.file.originalname,
      });
    } catch (err) {
      if (req.file?.path) {
        fs.rm(req.file.path, { force: true }, () => {});
      }
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : "Unexpected realtime start error",
      });
    }
  },
);

app.post("/api/track/realtime/stop", (_req, res) => {
  if (
    !realtimeProcess ||
    realtimeProcess.exitCode !== null ||
    realtimeProcess.killed
  ) {
    res.json({
      ok: true,
      message: "No realtime tracking window is currently running.",
    });
    return;
  }

  realtimeProcess.kill("SIGTERM");
  realtimeProcess = null;
  res.json({ ok: true, message: "Realtime tracking window stopped." });
});

app.post("/api/track/file", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "Missing uploaded video file" });
      return;
    }

    const detectionModel = String(
      req.body.detectionModel || "yolov8n.pt",
    ).trim();
    const trackingBackend = String(req.body.trackingBackend || "auto").trim();
    const trackingModel = String(req.body.trackingModel || "bytetrack").trim();
    const reidModel = String(req.body.reidModel || "none").trim();
    const conf = String(req.body.conf || "0.25").trim();
    const imgsz = String(req.body.imgsz || "640").trim();
    const maxFrames = String(req.body.maxFrames || "0").trim();
    const timeLimit = String(req.body.timeLimit || "0").trim();

    const runId = `${Date.now()}_${crypto.randomBytes(4).toString("hex")}`;
    const outputVideoName = `tracked_${safeName(trackingModel)}_${runId}.mp4`;
    const outputInfoName = `tracking_info_${runId}.json`;
    const outputVideoPath = path.join(OUTPUTS_DIR, outputVideoName);
    const outputInfoPath = path.join(OUTPUTS_DIR, outputInfoName);

    const args = [
      path.join(TRACKING_DIR, "live_tracking.py"),
      "--model",
      trackingModel,
      "--backend",
      trackingBackend,
      "--source",
      req.file.path,
      "--detector",
      detectionModel,
      "--conf",
      conf,
      "--imgsz",
      imgsz,
      "--no-show",
      "--save-out",
      outputVideoPath,
    ];

    if (reidModel && reidModel !== "none" && reidModel !== "custom_internal") {
      args.push("--reid", reidModel);
    }
    if (Number(maxFrames) > 0) {
      args.push("--max-frames", maxFrames);
    }
    if (Number(timeLimit) > 0) {
      args.push("--time-limit", timeLimit);
    }

    const startedAt = new Date().toISOString();
    const { stdout, stderr } = await runProcess(PYTHON_BIN, args, {
      cwd: TRACKING_DIR,
    });
    const endedAt = new Date().toISOString();

    if (!fs.existsSync(outputVideoPath)) {
      throw new Error("Tracking completed but no output video was generated.");
    }

    const st = fs.statSync(outputVideoPath);
    const info = {
      app: "person-tracking-eval-gui",
      generatedAt: endedAt,
      runWindow: { startedAt, endedAt },
      sourceMode: "uploaded_file",
      selections: {
        detectionModel,
        trackingBackend,
        trackingModel,
        reidModel,
        conf: Number(conf),
        imgsz: Number(imgsz),
      },
      trackingOutputFile: {
        fileName: outputVideoName,
        absolutePath: outputVideoPath,
        relativePath: path.relative(REPO_ROOT, outputVideoPath),
        sizeBytes: st.size,
        downloadUrl: `/api/download/video/${outputVideoName}`,
      },
      execution: {
        python: PYTHON_BIN,
        command: [PYTHON_BIN, ...args].join(" "),
        stdout,
        stderr,
      },
    };

    fs.writeFileSync(outputInfoPath, JSON.stringify(info, null, 2), "utf-8");

    fs.rm(req.file.path, { force: true }, () => {});

    res.json({
      ok: true,
      message: "Tracking execution completed",
      info,
      infoDownloadUrl: `/api/download/info/${outputInfoName}`,
      videoDownloadUrl: `/api/download/video/${outputVideoName}`,
    });
  } catch (err) {
    res.status(500).json({
      error:
        err instanceof Error
          ? err.message
          : "Unexpected tracking execution error",
    });
  }
});

app.get("/api/download/video/:name", (req, res) => {
  const fileName = safeName(req.params.name);
  const filePath = path.join(OUTPUTS_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: "Video file not found" });
    return;
  }
  res.download(filePath, fileName);
});

app.get("/api/download/info/:name", (req, res) => {
  const fileName = safeName(req.params.name);
  const filePath = path.join(OUTPUTS_DIR, fileName);
  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: "Info file not found" });
    return;
  }
  res.download(filePath, fileName);
});

const port = Number(process.env.PORT || 8787);
app.listen(port, () => {
  console.log(`Tracking API listening on http://localhost:${port}`);
  console.log(`Using python: ${PYTHON_BIN}`);
});
