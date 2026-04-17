# Person Tracking Evaluation GUI (React)

React + TypeScript + Vite GUI for executing person tracking with selected options.

The app supports:

1. Object detection model selection
2. Tracking backend selection (`auto`, `boxmot`, `custom`)
3. Tracking model selection
4. Re-ID model selection (including FastReID options)
5. Source mode selection (`realtime` or `downloaded file`)
6. Test video upload
7. Realtime execution in the native Python tracking window using the uploaded video
8. File-mode execution with downloadable annotated output video
9. Downloadable info JSON that includes tracking output file details

## Tech stack

- React 19
- TypeScript
- Vite

## Project structure

```text
person-tracking-eval-gui/
├── src/
│   ├── App.tsx        # Main GUI workflow
│   ├── index.css      # Application styles
│   └── main.tsx       # App bootstrap
├── public/
├── package.json
├── tsconfig*.json
└── vite.config.ts
```

## Run locally

From the `person-tracking-eval-gui` folder:

```bash
npm install
npm run api
npm run dev
```

Keep both commands running in separate terminals.

Open the URL shown in terminal (usually `http://localhost:5173`).

## Build and preview

```bash
npm run build
npm run preview
```

## How to use the GUI

1. Select `Object Detection Model`
2. Select `Tracking Backend`
3. Select `Tracking Model`
4. Select `Re-ID Model`
5. Select source mode:
   - `Realtime`
   - `Downloaded output file`
6. Upload test video
7. For `Realtime`: click `Start Realtime Tracking` to open the native Python tracking window for the uploaded video
8. For `Downloaded output file`: click `Run Tracking and Download Output`
9. Download `Annotated Video` and `Info JSON`

## Included model catalogs

Detection models (from `person_detection` and `person_tracking` detector pipeline):

- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`
- `yolov5s.pt`, `yolov5m.pt`, `yolov5l.pt`, `yolov5x.pt`
- `faster_rcnn`
- `ssd_mobilenet`

Tracking models (from `person_tracking`):

- `deepsort`, `bytetrack`, `botsort`, `strongsort`, `deepocsort`, `ocsort`, `hybridsort`, `boosttrack`, `sfsort`

Re-ID models:

- BoxMOT OSNet: `osnet_x0_25_msmt17`, `osnet_x0_5_msmt17`, `osnet_x1_0_msmt17`
- BoxMOT FastReID: `fastreid_sbs_s50`, `fastreid_sbs_r50`
- Custom backend: `custom_internal`
- ReID-free trackers: `none`

The GUI automatically filters incompatible tracker/Re-ID combinations based on selected backend.

## JSON output

The downloaded info JSON includes:

- Selected execution options (detector/backend/tracker/re-id/conf/imgsz)
- Run timing window
- Tracking output file metadata:
  - output file name
  - absolute path
  - repository-relative path
  - file size in bytes
  - download URL
- Raw execution command and stdout/stderr logs

## Runtime notes

This GUI now executes real tracking through `server.mjs`.

- File mode calls `person_tracking/live_tracking.py` with selected options.
- Re-ID is pluggable through `--reid` for BoxMOT runs.
- Realtime mode starts `person_tracking/live_tracking.py` with the native OpenCV display window.
- Realtime uses the uploaded video file as the source.
- The GUI does not expose webcam/stream/source detail selectors.

Recommended backend payload fields:

- `objectDetectionModel`
- `trackingBackend`
- `trackingModel`
- `reidModel`
- `sourceMode`
- `videoFile`

## Scripts

- `npm run dev`: start development server
- `npm run api`: start tracking API server (`http://localhost:8787`)
- `npm run build`: production build
- `npm run preview`: preview production build
- `npm run lint`: lint source code
