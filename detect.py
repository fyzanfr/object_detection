#!/usr/bin/env python3
"""
Smart Object Detection with Alert Zones
Uses: TorchVision SSDLite320_MobileNetV3_Large (lightweight, CPU-friendly)
"""

import argparse
import csv
import os
import subprocess
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml

warnings.filterwarnings("ignore")

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

ALERT_SOUND_PATH = Path(__file__).parent / "alert.wav"


def generate_alert_sound(path=ALERT_SOUND_PATH, freq=880, duration=0.25):
    import wave
    sample_rate = 22050
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    envelope = np.exp(-3 * t / duration)
    wave_data = np.sin(2 * np.pi * freq * t) * envelope * 0.5
    fade_len = int(sample_rate * 0.01)
    wave_data[:fade_len] *= np.arange(fade_len) / fade_len
    wave_data = (wave_data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(wave_data.tobytes())
    return path


class VideoSource:
    def __init__(self, source, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    @property
    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def fps(self):
        if self.is_file:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 30

    def release(self):
        if self.cap:
            self.cap.release()


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg["model"]["device"])
        self.model = None
        self.input_size = cfg["camera"]["resize_input"]
        self.load()

    def load(self):
        name = self.cfg["model"]["name"]
        weights = "DEFAULT"
        self.model = torchvision.models.detection.__dict__[name](
            weights=weights, weights_backbone="DEFAULT"
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def infer(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        img = cv2.resize(rgb, (self.input_size, self.input_size))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        outputs = self.model(tensor)[0]
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy()

        conf = self.cfg["model"]["confidence"]
        keep = scores >= conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        scale = np.array([w, h, w, h], dtype=float) / self.input_size
        boxes = boxes * scale
        return boxes, scores, labels

    def warmup(self):
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.infer(dummy)


class ObjectTracker:
    def __init__(self, iou_threshold=0.3, max_disappeared=5):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        current_ids = set()
        matched = [False] * len(detections)

        for tid, track in list(self.tracks.items()):
            best_iou = 0
            best_idx = -1
            for j, det in enumerate(detections):
                if matched[j]:
                    continue
                iou = self._iou(track["box"], det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= self.iou_threshold:
                self.tracks[tid] = {
                    "box": detections[best_idx]["box"],
                    "label": detections[best_idx]["label"],
                    "score": detections[best_idx]["score"],
                    "disappeared": 0,
                    "life": track.get("life", 0) + 1,
                }
                matched[best_idx] = True
                current_ids.add(tid)
            else:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        for j, m in enumerate(matched):
            if not m:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box": detections[j]["box"],
                    "label": detections[j]["label"],
                    "score": detections[j]["score"],
                    "disappeared": 0,
                    "life": 1,
                }
                current_ids.add(tid)

        return {tid: self.tracks[tid] for tid in current_ids if self.tracks[tid]["life"] >= 1}

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0
        aa = (ax2 - ax1) * (ay2 - ay1)
        bb = (bx2 - bx1) * (by2 - by1)
        return inter / (aa + bb - inter)


class AlertZone:
    def __init__(self, name, polygon):
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.color = tuple(np.random.randint(50, 255, 3).tolist())

    def contains(self, box):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return cv2.pointPolygonTest(self.polygon, (cx, cy), False) >= 0

    def draw(self, frame):
        cv2.polylines(frame, [self.polygon], True, self.color, 2)
        cx, cy = self.polygon.mean(axis=0).astype(int)
        cv2.putText(frame, self.name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, self.color, 2)


class AlertManager:
    def __init__(self, cfg):
        self.cfg = cfg["alerts"]
        self.zones = []
        for z in cfg["zones"]["list"]:
            self.zones.append(AlertZone(z["name"], z["polygon"]))
        self.log_dir = Path(cfg["alerts"]["snapshot_dir"])
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "alerts.csv"
        self._init_log()
        self._cooldowns = {}
        self._persistence = {}
        self._persistence_count = cfg["alerts"]["persistence"]
        self._cooldown_time = cfg["alerts"]["cooldown"]
        self._alert_classes = cfg["alerts"]["classes"]

        if cfg["alerts"]["sound"] and not ALERT_SOUND_PATH.exists():
            generate_alert_sound()

    def _init_log(self):
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "class", "confidence", "zone",
                            "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])

    def update_zones(self, zones):
        self.zones = zones

    def check(self, tracks, frame):
        now = time.time()
        triggered = []
        for tid, trk in tracks.items():
            label = trk["label"]
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
            if self._alert_classes and class_name not in self._alert_classes:
                continue
            if class_name not in ("__background__", "N/A"):
                pass
            zone_name = None
            for zone in self.zones:
                if zone.contains(trk["box"]):
                    zone_name = zone.name
                    break
            if self.zones and zone_name is None:
                continue
            key = (tid, label)
            if key not in self._persistence:
                self._persistence[key] = 0
            self._persistence[key] += 1
            if self._persistence[key] < self._persistence_count:
                continue
            if key in self._cooldowns and now - self._cooldowns[key] < self._cooldown_time:
                continue
            self._cooldowns[key] = now
            triggered.append({
                "track_id": tid, "class": class_name, "label": label,
                "confidence": float(trk["score"]),
                "zone": zone_name, "box": trk["box"].tolist(),
            })
            if self.cfg["snapshot"]:
                self._save_snapshot(frame, triggered[-1])
            if self.cfg["console"]:
                self._console_alert(triggered[-1])
            if self.cfg["sound"]:
                self._sound_alert()
            if self.cfg["log"]:
                self._log_alert(triggered[-1])
        return triggered

    def _save_snapshot(self, frame, alert):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = f"{ts}_{alert['class']}_id{alert['track_id']}.jpg"
        path = self.log_dir / fname
        x1, y1, x2, y2 = map(int, alert["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{alert['class']} {alert['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
        cv2.imwrite(str(path), frame)

    def _console_alert(self, alert):
        zone = f" [{alert['zone']}]" if alert["zone"] else ""
        print(f"  ALERT: {alert['class']} ({alert['confidence']:.2f}){zone}")

    def _sound_alert(self):
        try:
            if ALERT_SOUND_PATH.exists():
                subprocess.Popen(["paplay", str(ALERT_SOUND_PATH)],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _log_alert(self, alert):
        with open(self.log_file, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(), alert["class"],
                f"{alert['confidence']:.4f}", alert.get("zone", ""),
                alert["track_id"], *[f"{x:.1f}" for x in alert["box"]],
            ])


class Heatmap:
    def __init__(self, cfg, shape):
        self.enabled = cfg["heatmap"]["enabled"]
        self.decay = cfg["heatmap"]["decay"]
        cmap_name = cfg["heatmap"]["colormap"]
        self.cmap = getattr(cv2, f"COLORMAP_{cmap_name}", cv2.COLORMAP_JET)
        self.accum = np.zeros(shape[:2], dtype=np.float32)
        self.mask = np.zeros(shape[:2], dtype=np.uint8)

    def update(self, tracks, frame_shape):
        if not self.enabled:
            return
        self.accum *= self.decay
        self.mask.fill(0)
        for tid, trk in tracks.items():
            x1, y1, x2, y2 = map(int, trk["box"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max(5, min(x2 - x1, y2 - y1) // 2)
            cv2.circle(self.mask, (cx, cy), max(2, r // 2), 1, -1)
        self.accum[self.mask > 0] += 0.3

    def overlay(self, frame, alpha=0.4):
        if not self.enabled or self.accum.max() < 0.01:
            return frame
        norm = (self.accum / max(self.accum.max(), 1e-6) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, self.cmap)
        return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)


class DetectionApp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.video = VideoSource(cfg["camera"]["source"],
                                 cfg["camera"]["width"],
                                 cfg["camera"]["height"])
        if not self.video.is_opened:
            raise RuntimeError(f"Cannot open video source: {cfg['camera']['source']}")
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")
        self.frame_shape = frame.shape
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.motion_thresh = 2.0

        print("Loading model...")
        self.model = ModelManager(cfg)
        print("Warming up...")
        self.model.warmup()
        print("Ready.")

        self.tracker = ObjectTracker()
        self.heatmap = Heatmap(cfg, self.frame_shape)
        self.alert_mgr = AlertManager(cfg)

        self.zones_enabled = cfg["zones"]["enabled"]
        self.zones = []
        if cfg["zones"]["draw_on_start"]:
            self._interactive_zones(frame)
        else:
            for z in cfg["zones"]["list"]:
                self.zones.append(AlertZone(z["name"], z["polygon"]))
        self.alert_mgr.update_zones(self.zones)

        self.show_heatmap = cfg["heatmap"]["enabled"]

    def _has_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(self.prev_gray, gray)
        mean_diff = diff.mean()
        self.prev_gray = gray
        return mean_diff > self.motion_thresh

    def _interactive_zones(self, frame):
        print("Click points to define zones. Press 'c' to close polygon, Enter to confirm, Esc to skip.")
        clone = frame.copy()
        points = []
        while True:
            display = clone.copy()
            for p in points:
                cv2.circle(display, p, 4, (0, 255, 0), -1)
            if len(points) > 1:
                pts = np.array(points, np.int32)
                cv2.polylines(display, [pts], False, (0, 255, 0), 2)

            def mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append((x, y))

            cv2.imshow("Draw Zones", display)
            cv2.setMouseCallback("Draw Zones", mouse)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("c") and len(points) > 2:
                break
            elif key == 13 and len(points) > 2:
                name = f"zone_{len(self.zones) + 1}"
                self.zones.append(AlertZone(name, points))
                print(f"  Added {name} with {len(points)} points")
                points = []
                clone = frame.copy()
                for z in self.zones:
                    z.draw(clone)
            elif key == 27:
                break
        cv2.destroyWindow("Draw Zones")

    def run(self, timeout=None):
        start_time = time.time()
        frame_count = 0
        inf_count = 0
        fps_display = 0
        fps_timer = time.time()
        skip_counter = 0

        print("\nControls:")
        print("  q / Esc : Quit")
        print("  h       : Toggle heatmap")
        print("  z       : Toggle zones")
        print("  d       : Enter zone drawing mode")
        print("  s       : Save current frame")
        print("  p       : Pause/resume\n")

        while self.video.is_opened:
            if timeout and time.time() - start_time > timeout:
                break
            ret, frame = self.video.read()
            if not ret:
                break

            display = frame.copy()
            frame_count += 1
            skip_counter += 1

            if skip_counter >= 3 or self._has_motion(frame):
                skip_counter = 0
                boxes, scores, labels = self.model.infer(frame)
                inf_count += 1

                detections = []
                for i in range(len(boxes)):
                    detections.append({
                        "box": boxes[i],
                        "score": scores[i],
                        "label": labels[i],
                    })

                tracks = self.tracker.update(detections)
            else:
                tracks = self.tracker.update([])

            if self.show_heatmap:
                self.heatmap.update(tracks, self.frame_shape)
                display = self.heatmap.overlay(display)

            for tid, trk in tracks.items():
                x1, y1, x2, y2 = map(int, trk["box"])
                label = trk["label"]
                class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"id_{label}"
                label_text = f"{class_name} {trk['score']:.2f} #{tid}"
                color = (0, 255, 0) if trk["life"] < 5 else (0, 200, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                if self.cfg["display"]["show_all_labels"]:
                    (tw, th), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX,
                        self.cfg["display"]["font_scale"], 1)
                    cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(display, label_text, (x1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.cfg["display"]["font_scale"], (0, 0, 0), 1)

            if self.zones_enabled:
                for z in self.zones:
                    z.draw(display)

            triggered = self.alert_mgr.check(tracks, frame)

            if self.cfg["display"]["fps"]:
                now = time.time()
                if now - fps_timer >= 0.5:
                    fps_display = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer = now
                cv2.putText(display, f"FPS: {fps_display:.1f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.cfg["display"]["show_video"]:
                cv2.imshow("Object Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("h"):
                    self.show_heatmap = not self.show_heatmap
                    print(f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}")
                elif key == ord("z"):
                    self.zones_enabled = not self.zones_enabled
                    print(f"Zones: {'ON' if self.zones_enabled else 'OFF'}")
                elif key == ord("d"):
                    ret2, freeze = self.video.read()
                    self._interactive_zones(freeze)
                    self.alert_mgr.update_zones(self.zones)
                elif key == ord("s"):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"snapshot_{ts}.jpg", display)
                    print(f"Saved snapshot_{ts}.jpg")
                elif key == ord("p"):
                    paused = True
                    while paused:
                        k2 = cv2.waitKey(100) & 0xFF
                        if k2 == ord("p"):
                            paused = False
            else:
                cv2.waitKey(1)

        self.video.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Smart Object Detection with Alerts")
    parser.add_argument("-c", "--config", default="config.yaml", help="Config file path")
    parser.add_argument("-s", "--source", help="Override video source (file path or camera index)")
    parser.add_argument("-i", "--image", help="Run on a single image (no video)")
    parser.add_argument("-t", "--timeout", type=int, default=0,
                        help="Run for N seconds then exit (0 = unlimited)")
    parser.add_argument("--no-display", action="store_true", help="Run headless")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: cannot read image {args.image}")
            return 1
        cfg["camera"]["source"] = args.image
        app = DetectionApp(cfg)
        boxes, scores, labels = app.model.infer(frame)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            lbl = COCO_CLASSES[labels[i]] if labels[i] < len(COCO_CLASSES) else f"cls_{labels[i]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{lbl} {scores[i]:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out = f"detected_{Path(args.image).name}"
        cv2.imwrite(out, frame)
        print(f"Saved {out} ({len(boxes)} detections)")
        return 0

    if args.source is not None:
        try:
            cfg["camera"]["source"] = int(args.source)
        except ValueError:
            cfg["camera"]["source"] = args.source

    if args.no_display or "DISPLAY" not in os.environ:
        cfg["display"]["show_video"] = False

    app = DetectionApp(cfg)
    app.run(timeout=args.timeout if args.timeout > 0 else None)


if __name__ == "__main__":
    main()
