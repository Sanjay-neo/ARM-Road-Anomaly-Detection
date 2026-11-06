"""
╔══════════════════════════════════════════════════════════╗
║   ARM Edge AI — Real-Time Driver Alert System            ║
║   Detects road hazards AHEAD and warns the driver        ║
║   Classes : Crack | Pothole | Obstacle                   ║
║   Model   : YOLO26n on Raspberry Pi (NCNN)               ║
╚══════════════════════════════════════════════════════════╝

PURPOSE:
  Camera mounted on vehicle detects road hazards in real-time
  and alerts the driver BEFORE they hit them.

Run: python live_demo.py
Keys: Q=Quit  S=Screenshot  P=Pause  +/-=Confidence
"""

import cv2
import time
import sys
import os
import threading
import winsound                      # Windows built-in beep (no install needed)
from collections import deque

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
MODEL_PATH    = "D:/ARM_hackathon/runs/detect/road_anomaly_yolo26/weights/best.pt"
CAMERA_INDEX  = 0
CONF          = 0.35
IOU           = 0.45
IMGSZ         = 640
DEVICE        = 0        # GPU for laptop demo; change to 'cpu' for Raspberry Pi

# ── Alert thresholds (bbox area as % of frame) ─
# Larger bbox = hazard is CLOSER to the vehicle
DANGER_THRESH = 0.06     # >6% of frame area  → DANGER (red, loud beep)
WARN_THRESH   = 0.02     # >2% of frame area  → WARNING (orange, soft beep)

# ── Per-class colors (BGR) ────────────────────
COLORS = {
    0: (0,  180, 255),   # crack    → amber
    1: (30,  30, 255),   # pothole  → red
    2: (40, 210,  40),   # obstacle → green
}
NAMES  = {0: "CRACK", 1: "POTHOLE", 2: "OBSTACLE"}
ALERTS = {0: "⚠ CRACK AHEAD",  1: "🚨 POTHOLE AHEAD", 2: "⛔ OBSTACLE AHEAD"}

# ══════════════════════════════════════════════
#  ALERT SOUND (non-blocking)
# ══════════════════════════════════════════════
_beep_busy = False

def beep_alert(danger=False):
    """Play a non-blocking beep. danger=True → urgent beep."""
    global _beep_busy
    if _beep_busy:
        return
    def _play():
        global _beep_busy
        _beep_busy = True
        if danger:
            winsound.Beep(1200, 180)
            time.sleep(0.08)
            winsound.Beep(1200, 180)
        else:
            winsound.Beep(800, 120)
        _beep_busy = False
    threading.Thread(target=_play, daemon=True).start()


# ══════════════════════════════════════════════
#  PROXIMITY LEVEL
# ══════════════════════════════════════════════
def get_proximity(box_area_ratio):
    """Return proximity level based on bbox size relative to frame."""
    if box_area_ratio >= DANGER_THRESH:
        return "DANGER"
    elif box_area_ratio >= WARN_THRESH:
        return "WARNING"
    return "DETECTED"


# ══════════════════════════════════════════════
#  DRAW DETECTIONS
# ══════════════════════════════════════════════
def draw_detections(frame, results):
    """Draw boxes + return list of active alerts."""
    h, w     = frame.shape[:2]
    frame_area = h * w
    alerts   = []     # (proximity_level, class_name, conf, area_ratio)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color  = COLORS.get(cls_id, (255, 255, 255))
        name   = NAMES.get(cls_id, str(cls_id))

        box_area  = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area
        proximity  = get_proximity(area_ratio)

        alerts.append((proximity, cls_id, conf, area_ratio))

        # Box thickness varies by proximity
        thick = 4 if proximity == "DANGER" else 3 if proximity == "WARNING" else 2

        # Filled transparent interior
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        fill_alpha = 0.18 if proximity == "DANGER" else 0.10
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # Box border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)

        # Corner accents for DANGER
        if proximity == "DANGER":
            clen = 20
            for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame, (cx, cy), (cx+dx*clen, cy), (0,0,255), 3)
                cv2.line(frame, (cx, cy), (cx, cy+dy*clen), (0,0,255), 3)

        # Label
        prox_tag  = f"[{proximity}]" if proximity != "DETECTED" else ""
        label     = f"{name} {conf:.0%}  {prox_tag}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        ly = max(y1 - 2, lh + 10)
        cv2.rectangle(frame, (x1, ly - lh - 8), (x1 + lw + 10, ly), color, -1)
        cv2.putText(frame, label, (x1+5, ly-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, cv2.LINE_AA)

    return alerts


# ══════════════════════════════════════════════
#  DRAW DRIVER ALERT BANNER
# ══════════════════════════════════════════════
def draw_alert_banner(frame, alerts):
    """Draw a full-width alert banner at bottom when hazard detected."""
    h, w = frame.shape[:2]

    if not alerts:
        # All clear banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-50), (w, h), (20, 100, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "  ✓  ROAD CLEAR — SAFE TO PROCEED",
                    (w//2 - 220, h-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (100, 255, 100), 2, cv2.LINE_AA)
        return

    # Determine highest severity
    has_danger  = any(a[0] == "DANGER"  for a in alerts)
    has_warning = any(a[0] == "WARNING" for a in alerts)

    if has_danger:
        bg_color  = (0, 0, 200)
        txt_color = (255, 255, 255)
        prefix    = "🚨  DANGER — "
        beep_alert(danger=True)
    elif has_warning:
        bg_color  = (0, 120, 220)
        txt_color = (255, 255, 255)
        prefix    = "⚠   WARNING — "
        beep_alert(danger=False)
    else:
        bg_color  = (30, 130, 30)
        txt_color = (220, 255, 220)
        prefix    = "DETECTED — "

    # Build alert message from all detected classes
    unique_classes = list({a[1] for a in alerts})
    msg = prefix + " & ".join(ALERTS[c] for c in unique_classes[:2])

    # Pulsing border effect for danger
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-60), (w, h), bg_color, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    if has_danger:
        t = time.time()
        border_color = (0,0,255) if int(t*4) % 2 == 0 else (255,100,100)
        cv2.rectangle(frame, (0, h-60), (w, h), border_color, 3)

    cv2.putText(frame, msg, (w//2 - min(len(msg)*9, w//2-20), h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, txt_color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  DRAW HUD
# ══════════════════════════════════════════════
def draw_hud(frame, fps, counts, paused):
    h, w = frame.shape[:2]

    # Panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (250, 155), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    # Title
    cv2.putText(frame, "ARM EDGE AI  |  DRIVER ALERT", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (8, 23), (242, 23), (80, 80, 120), 1)

    # FPS
    fps_color = (80,255,80) if fps > 15 else (0,165,255) if fps > 8 else (0,60,255)
    cv2.putText(frame, f"FPS : {fps:5.1f}", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)

    # Per class counts
    y = 68
    for cls_id, name in NAMES.items():
        color = COLORS[cls_id]
        cnt   = counts.get(cls_id, 0)
        cv2.circle(frame, (16, y-5), 7, color, -1)
        label_color = (255,255,255) if cnt > 0 else (100,100,100)
        cv2.putText(frame, f"{name}: {cnt}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, label_color, 1, cv2.LINE_AA)
        y += 24

    # Model info
    cv2.putText(frame, "YOLO26n  |  Epoch 80/150", (8, y+6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1, cv2.LINE_AA)

    # Paused
    if paused:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,180), 5)
        cv2.putText(frame, "PAUSED", (w//2-60, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,60,255), 4, cv2.LINE_AA)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║   ARM Edge AI — Real-Time Driver Alert System            ║
║   Detecting: Crack | Pothole | Obstacle                  ║
╚══════════════════════════════════════════════════════════╝
""")

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("  ⏳ Loading YOLO26n model...")
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    print("  ✅ Model loaded!\n")
    print("  ⚠  Alert levels:")
    print("     DETECTED → hazard visible but far")
    print("     WARNING  → hazard getting closer (soft beep)")
    print("     DANGER   → hazard very close! (double beep)")
    print()
    print("  Controls:  Q=Quit  S=Screenshot  P=Pause  +/-=Confidence\n")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"  ❌ Cannot open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1")
        sys.exit(1)

    WIN = "ARM Edge AI — Real-Time Driver Alert  [Q=Quit | S=Save | P=Pause | +/-=Conf]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    fps_buf     = deque(maxlen=20)
    paused      = False
    frame       = None
    shot_n      = 0
    conf_thresh = CONF

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠️  Camera feed lost.")
                break

            t0      = time.perf_counter()
            results = model(frame, conf=conf_thresh, iou=IOU,
                            imgsz=IMGSZ, device=DEVICE, verbose=False)
            elapsed = time.perf_counter() - t0

            fps_buf.append(1.0 / elapsed if elapsed > 0 else 0)
            avg_fps = sum(fps_buf) / len(fps_buf)

            counts  = {0:0, 1:0, 2:0}
            alerts  = draw_detections(frame, results)
            for _, cls_id, _, _ in alerts:
                counts[cls_id] = counts.get(cls_id, 0) + 1

            draw_alert_banner(frame, alerts)
            draw_hud(frame, avg_fps, counts, paused)

            # Confidence display bottom-right
            cf_txt = f"Conf threshold: {conf_thresh:.2f}  [+ / -]"
            cv2.putText(frame, cf_txt,
                        (frame.shape[1]-280, frame.shape[0]-68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1, cv2.LINE_AA)

        cv2.imshow(WIN, frame if frame is not None else
                   __import__('numpy').zeros((720,1280,3), dtype='uint8'))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('p'), ord('P')):
            paused = not paused
            print(f"  {'⏸  Paused' if paused else '▶  Resumed'}")
        elif key in (ord('s'), ord('S')):
            shot_n += 1
            path = f"D:/ARM_hackathon/alert_demo_{shot_n:03d}.jpg"
            cv2.imwrite(path, frame)
            print(f"  💾 Screenshot → {path}")
        elif key in (ord('+'), ord('=')):
            conf_thresh = min(0.95, round(conf_thresh + 0.05, 2))
            print(f"  Confidence → {conf_thresh:.2f}")
        elif key == ord('-'):
            conf_thresh = max(0.05, round(conf_thresh - 0.05, 2))
            print(f"  Confidence → {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  ✅ Alert system closed.")


if __name__ == "__main__":
    main()
