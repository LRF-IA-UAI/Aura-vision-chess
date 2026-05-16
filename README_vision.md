# AURA — Vision Module
**Chess Robot · CAETI, Universidad Abierta Interamericana**

---

## What this is

AURA is a robot that plays physical chess against a human. This repository is the vision module: it reads the board state from a live camera feed, generates a FEN string, sends it to Stockfish, and returns the best move. The move is then executed by a robotic arm (separate module).

The core problem: you have a physical board in front of a camera that's not perfectly overhead, not perfectly still, and lit inconsistently. You need to know, reliably and in real time, which squares are occupied and by which team — without touching the board. That's what this solves.

---

## Physical setup required

You need all three of these, no substitutes:

**1. A camera, fixed.** Developed and tested with a Logitech C920e mounted above the board. It needs to stay still — the calibration is one-shot and assumes the camera doesn't move. Any USB webcam that OpenCV can open will technically work, but the C920e at 1080p gives enough resolution to distinguish piece colors reliably.

**2. A chess board with a red border.** The tracking system uses the red border as a geometric anchor to recompute the perspective transform on every frame. If the board shifts slightly or the camera vibrates, the system corrects automatically as long as the red border is visible. The color detection is HSV-based (H ∈ [0°,10°] ∪ [170°,180°]), so the red needs to be reasonably saturated — a painted wooden frame or red tape works fine.

**3. Colored pieces, not standard black/white.** The occupancy classifier distinguishes **red team vs green team** using HSV color masks. Standard black-and-white chess pieces will not classify correctly into teams (they'll show as UNKNOWN). This is intentional — the physical robot setup uses custom-colored pieces.

---

## Software dependencies

```bash
pip install opencv-python numpy ultralytics python-chess
```

You also need:

**Stockfish binary** — download the Windows build from [stockfishchess.org](https://stockfishchess.org/download/) and note the path to the `.exe`.

**chess_model.pt** — a YOLOv8 model trained on chess pieces. The easiest way to get one is from Roboflow Universe (search "chess pieces detection"). Download in YOLOv8 format, rename to `chess_model.pt`, drop it in this folder. If the file isn't present, `piece_classifier.py` will try to download it automatically via the Roboflow API (needs `ROBOFLOW_API_KEY` set) or fall back to a generic YOLOv8n that won't detect pieces.

**ChessEngine module** — `pipeline.py` imports `ChessEngine` from a sibling directory. The path is hardcoded at the top of `pipeline.py`:

```python
_ENGINE_DIR = r"C:\Users\Botmaker\Desktop\robot-ajedrecista-integracion-stockfish-y-python"
```

Update that and the Stockfish path to match your machine before running.

---

## Running it

```bash
python camera_pipeline.py
```

On startup it scans available cameras. If there's only one, it picks it automatically. If there are multiple, it asks for the index. Then it loads YOLO and connects to Stockfish — this takes a few seconds the first time.

### Controls

| Key | What it does |
|-----|-------------|
| `R` | **Calibrate.** Point the camera at the empty board with the red border fully visible. Press R — it detects the red border, runs `findChessboardCorners` on the grid for sub-pixel accuracy, computes the perspective transform, and saves the empty board reference for occupancy detection. From this point on, the transform updates automatically every frame as long as the red border is visible. |
| `B` | **Occupancy mode.** Opens a second window showing each square classified as empty / red team / green team / unknown. Uses pixel-level change detection against the empty board reference — no YOLO, runs at full frame rate. `+`/`-` adjusts the detection threshold (default 15%, meaning 15% of pixels in a cell must change vs the reference). |
| `SPACE` | **Full analysis.** Captures the current frame, runs YOLO to identify pieces, builds a FEN, queries Stockfish, and displays the suggested move overlaid on the warped board. Runs in a background thread so the feed stays live. |
| `P` | **Photo analysis.** Same as SPACE but also saves 8 artifacts to `capturas/captura_<timestamp>/`: the original frame, warped board, grid overlay, all 64 cell crops, YOLO result, FEN, Stockfish output, and a summary. Useful for debugging. |
| `D` | **Debug overlay.** Shows Canny contours, the detected board polygon, calibrated corners (orange), and the red border detected this frame (cyan). Orange = what the system is using; cyan = what the red detector sees right now. If they diverge, the border isn't being tracked cleanly. |
| `Q` | Quit. |

---

## How the tracking works

On every frame, before anything else, `_update_red_border_tracking()` runs:

1. Converts the frame to HSV, builds a red mask, morphologically closes gaps.
2. Finds contours with `RETR_CCOMP` — specifically looks for the *inner* contour of the red frame (the hole inside the border), not the outer edge.
3. Approximates to 4 vertices, validates aspect ratio.
4. Applies the homography computed at calibration time (`red_to_board_homography`) to map those 4 corners to the playing area corners.
5. Recomputes the perspective matrix.

This means if the board slides a centimeter during a game, the warp corrects on the next frame. If the red border goes out of frame entirely, the system holds the last valid matrix (status shows STALE in yellow on the HUD).

The calibration step (R) uses `findChessboardCorners` on the 7×7 inner grid to get sub-pixel accurate playing-area corners, then expands them 2.5% outward from the centroid to compensate for the slight undershoot typical of the extrapolation. It also computes the homography from the red border corners to those precise corners — that's what gets reused every frame.

---

## Current state and known issues

**What works reliably:**
- Red border tracking under moderate lighting changes
- Occupancy detection (the two-stage pixel-diff method is much more robust than mean diff for small objects like chess pieces)
- Team color classification when pieces are red/green

**What's fragile:**

- **Lighting.** If there's a strong directional light casting shadows on the red border, detection drops out and the system goes STALE. Diffuse overhead lighting works best.
- **The red-vs-board color calibration is fixed.** The HSV thresholds for red detection (`H ∈ [0,10] ∪ [170,180]`, `S > 80`, `V > 50`) are hardcoded. If your red border is a different shade, edit `_detect_red_border_corners()` in `camera_pipeline.py`.
- **YOLO piece identification.** Quality depends entirely on `chess_model.pt`. The model we use was trained on overhead views of standard pieces, so it works reasonably well on our setup but will degrade on unusual angles or piece designs.
- **Paths are hardcoded.** Both the Stockfish binary path and the `ChessEngine` module directory are hardcoded in `pipeline.py`. Fix those for your machine before running.
- **Single camera assumed.** The code scans indices 0–5 and uses the first available camera, or asks you to pick. There's no persistent camera config — it always asks on startup if multiple cameras are connected.

---

## File overview

```
camera_pipeline.py          Main entry point — live feed, all keyboard controls
pipeline.py                 Orchestrates YOLO + FEN builder + Stockfish
board_detector.py           Perspective correction, corner detection strategies
piece_classifier.py         YOLOv8 wrapper, loads chess_model.pt
chess_model.pt              YOLOv8 weights for piece detection (not in repo — get from Roboflow)
generate_aruco_markers.py   Utility to generate ArUco marker PDFs (previous tracking method)
_aruco_fallback.py          ArUco-based tracking code, kept as reference
```

The `pipeline.py` module is designed to be imported, not just run directly — `camera_pipeline.py` imports `detector`, `classifier`, and `engine` from it at startup.

---

## Context

This is a university robotics project at CAETI (Centro de Altos Estudios en Tecnología Informática), UAI. The full system includes a robotic arm, a communication layer, and this vision module. The vision module is developed independently and exposes its output (FEN + best move) over a simple interface so the other modules don't need to know anything about OpenCV or cameras.
