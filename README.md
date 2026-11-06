# 🚗 ARM Edge AI — Real-Time Road Anomaly Detection

> **ARM Hackathon Project** — Detecting road hazards in real-time using YOLO26n, trained on a laptop GPU and deployed on Raspberry Pi edge devices.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLO-v2.6n-orange?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)
![Platform](https://img.shields.io/badge/Edge-Raspberry%20Pi%20%7C%20NCNN-red?logo=raspberry-pi)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🎯 What It Does

A vehicle-mounted camera detects road hazards **ahead of the driver** in real time and raises alerts before the vehicle hits them.

| Class | Color | Alert |
|-------|-------|-------|
| 🟠 Crack | Orange | `⚠ CRACK AHEAD` |
| 🔴 Pothole | Red | `🚨 POTHOLE AHEAD` |
| 🟢 Obstacle | Green | `⛔ OBSTACLE AHEAD` |

### Proximity Alert System
| Level | Trigger (bbox area) | Visual | Audio |
|-------|---------------------|--------|-------|
| `DETECTED` | Any size | Bounding box | — |
| `WARNING` | ≥ 2% of frame | Orange banner | Soft beep |
| `DANGER` | ≥ 6% of frame | 🔴 Pulsing red banner + corner accents | Double urgent beep |

---

## 🏗️ Architecture

```
Dataset (Roboflow/Kaggle)
        ↓
   allign.py          ← Copies & organises dataset
        ↓
 train_yolo26.py      ← Trains YOLO26n on RTX 4060 (FP16, batch=32)
        ↓
   best.pt saved      ← runs/detect/road_anomaly_yolo26/weights/
        ↓
 ┌──────────────────────────────────┐
 │  demo.py       → Flexible demo  │
 │  live_demo.py  → Driver alerts  │
 │  webcam_stream.py → Flask MJPEG │
 └──────────────────────────────────┘
        ↓
  NCNN Export → Raspberry Pi ARM64
```

---

## 📦 Setup

```bash
git clone https://github.com/Sanjay-neo/ARM-Road-Anomaly-Detection.git
cd ARM-Road-Anomaly-Detection
pip install -r requirements.txt
```

> **PyTorch with CUDA (RTX 4060 / CUDA 12.1):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## ⬇️ Download Model Weights

Download `best.pt` from the [Releases](https://github.com/Sanjay-neo/ARM-Road-Anomaly-Detection/releases) page and place it at:
```
runs/detect/road_anomaly_yolo26/weights/best.pt
```

---

## 🚀 Usage

### 🔴 Real-Time Driver Alert System (Main Demo)
```bash
python live_demo.py
```
Controls: `Q` = Quit &nbsp;|&nbsp; `S` = Screenshot &nbsp;|&nbsp; `P` = Pause &nbsp;|&nbsp; `+/-` = Confidence

---

### 🎥 Flexible Demo (Image / Video / Webcam / Folder)
```bash
python demo.py                                    # Webcam (default)
python demo.py --source 0                         # Webcam (explicit)
python demo.py --source road.jpg                  # Single image
python demo.py --source road.mp4                  # Video file
python demo.py --source road_anomaly/test/images  # Entire folder
```

---

### 🏋️ Train from Scratch
```bash
python train_yolo26.py               # Fresh training
python train_yolo26.py --resume      # Resume from last checkpoint
python train_yolo26.py --verify-only # Dataset check only
python train_yolo26.py --export-only # Export best.pt to NCNN + ONNX
```

---

### 📡 Flask Webcam Stream (for Raspberry Pi)
```bash
python webcam_stream.py
# → Open http://<laptop-ip>:5000/video_feed on the Pi
```

---

## 🧠 Model Info

| Property | Value |
|----------|-------|
| Architecture | YOLO26n (Nano) |
| Input resolution | 640 × 640 |
| Classes | 3 — crack, pothole, obstacle |
| Training GPU | NVIDIA RTX 4060 Laptop 8 GB |
| Precision | FP16 / AMP (Ada Lovelace) |
| Batch size | 32 |
| Epochs | 150 (early stop @ 80) |
| Cache | Disk (RAM-safe for 12 GB systems) |
| Edge export | NCNN (Raspberry Pi ARM64) |

---

## 📂 Project Structure

```
ARM-Road-Anomaly-Detection/
│
├── train_yolo26.py          # Full 5-step training pipeline
├── demo.py                  # Flexible inference demo
├── live_demo.py             # Real-time driver alert system
├── webcam_stream.py         # Flask MJPEG webcam streamer
├── allign.py                # Dataset alignment utility
│
├── requirements.txt
├── .gitignore
│
└── runs/detect/road_anomaly_yolo26/
    ├── args.yaml            # Exact training config used
    ├── labels.jpg           # Class distribution chart
    └── train_batch0.jpg     # Sample training batch
```

---

## 📊 Dataset

| Dataset | Classes | Source |
|---------|---------|--------|
| Hazard in Road | 3 (crack, pothole, obstacle) | Roboflow |
| Pothole Detection | 1 (pothole) | Kaggle |

---

## 🍓 Raspberry Pi Deployment

```python
from ultralytics import YOLO

# Load NCNN model (exported after training)
model = YOLO("runs/detect/road_anomaly_yolo26/weights/best_ncnn_model")
results = model("road.jpg", conf=0.4)
results[0].show()
```

---

## 📄 License

MIT © 2026 [SANJAY S](https://github.com/Sanjay-neo)
