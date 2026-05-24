# EdgeDetect — CPU Object Detection with Smart Alerts

Real-time object detection that runs entirely on CPU. Uses **SSDLite320\_MobileNetV3\_Large** — a lightweight architecture from Google designed for edge devices.

---

## Features

- **~34 FPS on CPU** (12th gen i5) — MobileNetV3 backbone with depthwise separable convolutions
- **Motion gating** — skips model inference when nothing moves; ~1ms idle frame cost
- **Object tracking** — IoU-based identity persistence across frames (not just bounding boxes)
- **Alert zones** — polygonal regions of interest; alerts only fire inside defined zones
- **Temporal filtering** — requires N consecutive detections before alerting (default: 3)
- **Per-object cooldown** — no repeat alerts for the same tracked object within 5s
- **Multi-channel alerts** — console, sound (system beep), CSV log, snapshot capture
- **Activity heatmap** — color overlay showing where objects spend time (toggle with `h`)
- **Headless mode** — runs on servers without a display (`--no-display`)
- **Interactive controls** — draw zones, toggle overlays, save frames at runtime

---

## Quick Start

```bash
git clone https://github.com/YOUR_USER/EdgeDetect.git
cd EdgeDetect

pip install -r requirements.txt

python detect.py
```

On first run, the model weights (~13MB) download automatically from PyTorch Hub. An alert sound file is also generated.

---

## Usage

```bash
# Live webcam (default)
python detect.py

# Video file
python detect.py -s path/to/video.mp4

# Single image (no video loop)
python detect.py -i image.jpg

# Headless mode with 60s timeout
python detect.py --no-display -t 60

# Custom config
python detect.py -c my_config.yaml

# Override camera index
python detect.py -s 2
```

### Controls (live display mode)

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `h` | Toggle heatmap overlay |
| `z` | Toggle alert zone visibility |
| `d` | Enter zone drawing mode (click to add points) |
| `s` | Save current frame as snapshot |
| `p` | Pause / resume |

---

## Configuration

All settings in `config.yaml`:

```yaml
model:
  confidence: 0.55       # minimum detection confidence
  device: cpu

alerts:
  persistence: 3         # frames before alert triggers
  cooldown: 5            # seconds between repeat alerts
  classes: []            # empty = all; or ["person", "car"]
  snapshot: true
  sound: true
  log: true

heatmap:
  enabled: true
  decay: 0.97            # per-frame fade rate
  colormap: JET

zones:
  draw_on_start: false   # prompts for zone drawing on launch
  list: []               # pre-defined zones
```

---

## Architecture

```
Camera ─→ Motion Gate ─→ SSDLite Model ─→ Tracker ─→ Zone Check ─→ Alerts
                              ↓
                         Heatmap overlay → Display
```

| Component | File | Line | Role |
|---|---|---|---|
| `VideoSource` | `detect.py` | `VideoSource` | Frame provider (camera / file) |
| `ModelManager` | `detect.py` | `ModelManager` | Model load & inference |
| `ObjectTracker` | `detect.py` | `ObjectTracker` | IoU-based identity tracking |
| `AlertZone` | `detect.py` | `AlertZone` | Polygon region-of-interest |
| `AlertManager` | `detect.py` | `AlertManager` | Persistence + cooldown + alert actions |
| `Heatmap` | `detect.py` | `Heatmap` | Exponential-decay activity overlay |
| `DetectionApp` | `detect.py` | `DetectionApp` | Main loop & orchestration |


---

## Model Details

**SSDLite320\_MobileNetV3\_Large** is a one-stage detector that uses:
- **MobileNetV3 backbone** — depthwise separable convolutions (8× fewer params than standard conv)
- **SSD head** — multi-scale feature maps for detecting objects at different sizes
- **Hard-swish activation** and **squeeze-and-excite** blocks
- **320×320 input** — balances speed vs accuracy for CPU
- **COCO training** — 80 common object classes (person, car, dog, etc.)

---

## Performance

| Scene | CPU Usage | FPS |
|---|---|---|
| Idle (no motion) | ~3% (motion gated) | 60+ |
| Active (detections) | ~30% | 30-34 |

Tested on 12th Gen Intel Core i5-12450H (12 cores). No GPU required.

---

