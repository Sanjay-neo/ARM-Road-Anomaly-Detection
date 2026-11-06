"""
╔══════════════════════════════════════════════════════════╗
║   YOLO26 — Road Anomaly Demo                             ║
║   Classes : crack | pothole | obstacle                   ║
║   Model   : last.pt (epoch 80)                           ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python demo.py                    # Webcam (default)
  python demo.py --source 0         # Webcam (explicit)
  python demo.py --source image.jpg # Single image
  python demo.py --source video.mp4 # Video file
  python demo.py --source D:/ARM_hackathon/road_anomaly/test/images  # Folder

Keys during webcam/video:
  Q or ESC  → Quit
  S         → Save current frame as screenshot
  P         → Pause / Unpause
"""

import argparse
import os
import sys
import time
import cv2

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
MODEL_PATH  = "D:/ARM_hackathon/runs/detect/road_anomaly_yolo26/weights/best.pt"
CONF        = 0.35     # Confidence threshold (lower = more detections)
IOU         = 0.45     # NMS IoU threshold
IMGSZ       = 640      # Inference size
DEVICE      = 0        # 0 = GPU, 'cpu' = CPU

# ── Per-class colors (BGR) ─────────────────────
COLORS = {
    0: (0,   200, 255),   # crack    → orange
    1: (0,   60,  255),   # pothole  → red
    2: (50,  220, 50 ),   # obstacle → green
}
CLASS_NAMES = {0: "crack", 1: "pothole", 2: "obstacle"}

# ══════════════════════════════════════════════
#  DRAW RESULTS
# ══════════════════════════════════════════════
def draw_detections(frame, results, fps=None):
    """Draw bounding boxes, labels, confidence on frame."""
    h, w = frame.shape[:2]

    # Count per class
    counts = {0: 0, 1: 0, 2: 0}

    for box in results[0].boxes:
        cls_id  = int(box.cls[0])
        conf    = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color   = COLORS.get(cls_id, (255, 255, 255))
        label   = CLASS_NAMES.get(cls_id, str(cls_id))
        counts[cls_id] += 1

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text  = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── HUD overlay ──────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 22
    cv2.putText(frame, "YOLO26 Road Anomaly", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 22

    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)
        y += 22

    for cls_id, name in CLASS_NAMES.items():
        color = COLORS[cls_id]
        count = counts[cls_id]
        cv2.putText(frame, f"  {name}: {count}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        y += 22

    return frame


# ══════════════════════════════════════════════
#  IMAGE MODE
# ══════════════════════════════════════════════
def run_image(model, source):
    """Run on a single image and display result."""
    print(f"  Running on image: {source}")
    results = model(source, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE, verbose=False)

    frame = cv2.imread(source)
    if frame is None:
        print(f"  ❌ Could not read image: {source}")
        return

    frame = draw_detections(frame, results)

    # Print detections to console
    n = len(results[0].boxes)
    print(f"  ✅ Detected {n} object(s):")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        print(f"     → {CLASS_NAMES[cls_id]:10s}  conf={conf:.3f}")

    # Save result
    out_path = source.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(out_path, frame)
    print(f"  💾 Saved → {out_path}")

    # Show
    win = "Road Anomaly Detection — press any key to close"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════
#  FOLDER MODE
# ══════════════════════════════════════════════
def run_folder(model, folder):
    """Run on all images in a folder."""
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    print(f"  Found {len(files)} images in {folder}")

    out_dir = os.path.join(folder, "detections")
    os.makedirs(out_dir, exist_ok=True)

    for i, fname in enumerate(files):
        path   = os.path.join(folder, fname)
        results = model(path, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE, verbose=False)
        frame  = cv2.imread(path)
        frame  = draw_detections(frame, results)
        out    = os.path.join(out_dir, fname)
        cv2.imwrite(out, frame)
        n = len(results[0].boxes)
        print(f"  [{i+1}/{len(files)}] {fname} → {n} detections  saved to {out}")

    print(f"\n  ✅ All done! Results saved to: {out_dir}")


# ══════════════════════════════════════════════
#  WEBCAM / VIDEO MODE
# ══════════════════════════════════════════════
def run_stream(model, source):
    """Run real-time on webcam or video."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"  ❌ Cannot open source: {source}")
        sys.exit(1)

    is_video  = not str(source).isdigit()
    total_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    win_name  = "Road Anomaly Detection  |  Q=Quit  S=Save  P=Pause"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    paused     = False
    frame_idx  = 0
    fps_vals   = []
    screenshot_n = 0

    print(f"\n  🎥 Stream started. Press Q to quit, S to save frame, P to pause.\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if is_video:
                    print("  Video ended.")
                break

            t0      = time.perf_counter()
            results = model(frame, conf=CONF, iou=IOU, imgsz=IMGSZ,
                            device=DEVICE, verbose=False, stream=False)
            elapsed = time.perf_counter() - t0

            fps_vals.append(1.0 / elapsed if elapsed > 0 else 0)
            if len(fps_vals) > 20:
                fps_vals.pop(0)
            avg_fps = sum(fps_vals) / len(fps_vals)

            frame = draw_detections(frame, results, fps=avg_fps)
            frame_idx += 1

        # Pause indicator
        if paused:
            cv2.putText(frame, "  PAUSED  ", (frame.shape[1]//2 - 80, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 255), 3)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):    # Q or ESC
            break
        elif key in (ord('p'), ord('P')):       # Pause
            paused = not paused
            print(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
        elif key in (ord('s'), ord('S')):       # Screenshot
            screenshot_n += 1
            fname = f"D:/ARM_hackathon/screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  💾 Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  ✅ Demo ended after {frame_idx} frames.")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="YOLO26 Road Anomaly Demo")
    parser.add_argument("--source",  type=str, default="0",
                        help="Source: 0=webcam, image.jpg, video.mp4, or folder path")
    parser.add_argument("--model",   type=str, default=MODEL_PATH,
                        help="Path to .pt model weights")
    parser.add_argument("--conf",    type=float, default=CONF,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--device",  default=DEVICE,
                        help="Device: 0=GPU, cpu=CPU")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   YOLO26 — Road Anomaly Detection DEMO                   ║
║   crack (orange) | pothole (red) | obstacle (green)      ║
╚══════════════════════════════════════════════════════════╝

  Model  : {args.model}
  Source : {args.source}
  Conf   : {args.conf}
  Device : {args.device}
""")

    if not os.path.exists(args.model):
        print(f"  ❌ Model not found: {args.model}")
        print(f"     Make sure training completed and best.pt exists.")
        sys.exit(1)

    print("  ⏳ Loading model...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    print("  ✅ Model loaded!\n")

    src = args.source

    # Route to correct mode
    if os.path.isdir(src):
        run_folder(model, src)
    elif os.path.isfile(src) and src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        run_image(model, src)
    else:
        # Webcam or video
        run_stream(model, src)


if __name__ == "__main__":
    main()
