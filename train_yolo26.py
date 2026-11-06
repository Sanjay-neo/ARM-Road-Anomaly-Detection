"""
╔══════════════════════════════════════════════════════════╗
║   YOLO26 — Road Anomaly Detection Trainer                ║
║   GPU    : NVIDIA RTX 4060 Laptop (8 GB VRAM)           ║
║   Classes: crack | pothole | obstacle                    ║
║   Target : Raspberry Pi (exported to NCNN after train)   ║
╚══════════════════════════════════════════════════════════╝

Usage:
  python train_yolo26.py               # Fresh training
  python train_yolo26.py --resume      # Resume from last.pt
  python train_yolo26.py --resume --weights path/to/last.pt
  python train_yolo26.py --verify-only # Dataset check only
"""

import argparse
import os
import sys

# ══════════════════════════════════════════════
#  CONFIG — Optimized for RTX 4060 Laptop 8GB
# ══════════════════════════════════════════════
DATA_YAML     = "D:/ARM_hackathon/road_anomaly/data.yaml"
MODEL         = "yolo26n.pt"           # Nano = best speed on Raspberry Pi
PROJECT       = "D:/ARM_hackathon/runs/detect"
RUN_NAME      = "road_anomaly_yolo26"

# ── Core ──────────────────────────────────────
DEVICE        = 0                      # GPU 0 (RTX 4060)
EPOCHS        = 150
IMGSZ         = 640
BATCH         = 32                     # 4060 8GB handles 32 comfortably at 640px
PATIENCE      = 30                     # Early stop if no improvement for 30 epochs
WORKERS       = 4                      # ⚠️  12GB RAM: 4 workers to avoid memory pressure
                                       #    Each worker buffers batches in RAM

# ── Speed / Memory ────────────────────────────
AMP           = True                   # FP16 Mixed Precision (Ada Lovelace loves this)
CACHE         = "disk"                 # ⚠️  12GB RAM: use disk cache (preprocesses images
                                       #    once → saves .npy files, no RAM pressure)
                                       #    Set to "ram" only if you have 32GB+ RAM

# ── Learning Rate ─────────────────────────────
LR0           = 0.01                   # Initial LR (SGD)
LRF           = 0.01                   # Final LR factor (cosine decay → LR0 * LRF)
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3.0
WARMUP_BIAS_LR= 0.1

# ── Augmentation (road-scene optimized) ───────
DEGREES       = 5.0                    # Slight rotation for road angles
TRANSLATE     = 0.1
SCALE         = 0.5
SHEAR         = 2.0
FLIPUD        = 0.1                    # Potholes look similar upside down
FLIPLR        = 0.5
MOSAIC        = 1.0                    # Keep full mosaic during training
MIXUP         = 0.15                   # Slightly more mixup for generalization
COPY_PASTE    = 0.1
CLOSE_MOSAIC  = 20                     # Disable mosaic last 20 epochs (fine-tuning)
HSV_H         = 0.015
HSV_S         = 0.7
HSV_V         = 0.4

# ── Save / Log ────────────────────────────────
SAVE_PERIOD   = 10                     # Checkpoint every 10 epochs
PLOTS         = True                   # Save result plots

# ══════════════════════════════════════════════
#  COLORS
# ══════════════════════════════════════════════
G = "\033[92m";  Y = "\033[93m";  R = "\033[91m"
C = "\033[96m";  B = "\033[1m";   X = "\033[0m"

def ok(m):     print(f"{G}  ✅ {m}{X}")
def warn(m):   print(f"{Y}  ⚠️  {m}{X}")
def fail(m):   print(f"{R}  ❌ {m}{X}")
def info(m):   print(f"{C}  ℹ️  {m}{X}")
def sep(title):print(f"\n{B}{C}{'━'*56}\n  {title}\n{'━'*56}{X}")


# ══════════════════════════════════════════════
#  STEP 1 — GPU Check
# ══════════════════════════════════════════════
def check_gpu():
    sep("STEP 1 — GPU & Environment Check")
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")

        if not torch.cuda.is_available():
            fail("CUDA not available! Check NVIDIA drivers and PyTorch CUDA build.")
            sys.exit(1)

        gpu_name  = torch.cuda.get_device_name(0)
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_ver  = torch.version.cuda

        ok(f"GPU     : {gpu_name}")
        ok(f"VRAM    : {vram_gb:.1f} GB")
        ok(f"CUDA    : {cuda_ver}")
        ok(f"AMP/FP16: {'Enabled ⚡' if AMP else 'Disabled'}")

        if vram_gb < 6:
            warn(f"VRAM {vram_gb:.1f}GB < 6GB — reducing batch to 16")
            global BATCH
            BATCH = 16

    except ImportError:
        fail("PyTorch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    try:
        import ultralytics
        ok(f"ultralytics {ultralytics.__version__}")
    except ImportError:
        fail("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)


# ══════════════════════════════════════════════
#  STEP 2 — Dataset Verification
# ══════════════════════════════════════════════
def verify_dataset():
    sep("STEP 2 — Dataset Verification")
    import yaml

    if not os.path.exists(DATA_YAML):
        fail(f"data.yaml not found: {DATA_YAML}"); sys.exit(1)
    ok(f"data.yaml : {DATA_YAML}")

    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    base  = cfg.get("path", "")
    nc    = cfg.get("nc", 0)
    names = cfg.get("names", {})

    info(f"Dataset   : {base}")
    info(f"Classes   : {nc} → {list(names.values()) if isinstance(names, dict) else names}")

    if nc == 3:
        ok("Class count = 3 (crack / pothole / obstacle) ✓")
    else:
        warn(f"Expected 3 classes, got {nc}")

    all_ok     = True
    total_imgs = 0

    split_map = {
        "train": cfg.get("train", "train/images"),
        "val"  : cfg.get("val",   "valid/images"),
        "test" : cfg.get("test",  "test/images"),
    }

    for split, rel in split_map.items():
        img_dir = os.path.join(base, rel)
        lbl_dir = img_dir.replace("images", "labels")

        for d, label in [(img_dir, "images"), (lbl_dir, "labels")]:
            if not os.path.isdir(d):
                fail(f"{split}/{label} missing: {d}"); all_ok = False

        if not all_ok:
            continue

        imgs     = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
        lbls     = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)}
        unpaired = imgs - lbls
        empty    = sum(
            1 for f in os.listdir(lbl_dir)
            if open(os.path.join(lbl_dir, f)).read().strip() == ""
        )

        ok(f"{split:5s} → {len(imgs):>6,} imgs | {len(lbls):>6,} lbls "
           f"| unpaired={len(unpaired)} | background={empty}")
        if unpaired:
            warn(f"  {len(unpaired)} images have no label in {split}")

        total_imgs += len(imgs)

    if not all_ok:
        fail("Fix the dataset errors above before training."); sys.exit(1)

    ok(f"Total     : {total_imgs:,} images across all splits")
    print(f"\n{G}{B}  ✅ Dataset OK — ready to train!{X}")
    return cfg


# ══════════════════════════════════════════════
#  STEP 3 — Train
# ══════════════════════════════════════════════
def train(resume=False, resume_weights=None):
    sep("STEP 3 — Training YOLO26 on RTX 4060")
    from ultralytics import YOLO

    if resume:
        ckpt = resume_weights or os.path.join(PROJECT, RUN_NAME, "weights", "last.pt")
        if not os.path.exists(ckpt):
            fail(f"Checkpoint not found: {ckpt}"); sys.exit(1)
        info(f"▶ Resuming from : {ckpt}")
        model = YOLO(ckpt)
        extra = dict(resume=True)
    else:
        info(f"▶ Loading base model : {MODEL}")
        model = YOLO(MODEL)
        extra = {}

    print(f"\n{B}  Training Config (RTX 4060 Optimized):{X}")
    config_rows = [
        ("device",        f"cuda:{DEVICE} (RTX 4060)"),
        ("epochs",        EPOCHS),
        ("imgsz",         IMGSZ),
        ("batch",         f"{BATCH}  ← tuned for 8GB VRAM"),
        ("AMP (FP16)",    f"{'ON ⚡' if AMP else 'OFF'}"),
        ("cache",         f"'{CACHE}'"),
        ("workers",       WORKERS),
        ("lr0 → lrf",     f"{LR0} → {LR0 * LRF:.5f}"),
        ("patience",      f"{PATIENCE} epochs early stop"),
        ("close_mosaic",  f"last {CLOSE_MOSAIC} epochs"),
        ("save_period",   f"every {SAVE_PERIOD} epochs"),
    ]
    for k, v in config_rows:
        print(f"  {C}{k:<18}{X}: {v}")

    print()
    results = model.train(
        # ── Dataset ──────────────────────────
        data         = DATA_YAML,

        # ── GPU ──────────────────────────────
        device       = DEVICE,
        amp          = AMP,            # FP16 — 2x speed on Ada Lovelace
        cache        = CACHE,          # Images cached in RAM
        workers      = WORKERS,

        # ── Run ──────────────────────────────
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH,
        patience     = PATIENCE,
        project      = PROJECT,
        name         = RUN_NAME,
        exist_ok     = True,           # Needed for resume

        # ── LR ───────────────────────────────
        lr0          = LR0,
        lrf          = LRF,
        momentum     = MOMENTUM,
        weight_decay = WEIGHT_DECAY,
        warmup_epochs= WARMUP_EPOCHS,
        warmup_bias_lr= WARMUP_BIAS_LR,

        # ── Augmentation ─────────────────────
        degrees      = DEGREES,
        translate    = TRANSLATE,
        scale        = SCALE,
        shear        = SHEAR,
        flipud       = FLIPUD,
        fliplr       = FLIPLR,
        mosaic       = MOSAIC,
        mixup        = MIXUP,
        copy_paste   = COPY_PASTE,
        close_mosaic = CLOSE_MOSAIC,
        hsv_h        = HSV_H,
        hsv_s        = HSV_S,
        hsv_v        = HSV_V,

        # ── Save / Log ───────────────────────
        val          = True,
        save         = True,
        save_period  = SAVE_PERIOD,
        plots        = PLOTS,

        **extra
    )
    return model, results


# ══════════════════════════════════════════════
#  STEP 4 — Export for Raspberry Pi
# ══════════════════════════════════════════════
def export_for_rpi():
    sep("STEP 4 — Exporting for Raspberry Pi")
    best_pt = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")

    if not os.path.exists(best_pt):
        warn("best.pt not found — run training first."); return

    from ultralytics import YOLO
    model = YOLO(best_pt)

    # NCNN — fastest on Raspberry Pi ARM64
    info("Exporting → NCNN (fastest on Raspberry Pi ARM64)...")
    try:
        model.export(format="ncnn", imgsz=IMGSZ)
        ok("NCNN export done ✓")
    except Exception as e:
        warn(f"NCNN failed: {e}")

    # ONNX — works with OpenCV DNN or ONNX Runtime
    info("Exporting → ONNX (alternative, OpenCV DNN compatible)...")
    try:
        model.export(format="onnx", imgsz=IMGSZ, simplify=True)
        ok("ONNX export done ✓")
    except Exception as e:
        warn(f"ONNX failed: {e}")


# ══════════════════════════════════════════════
#  STEP 5 — Summary
# ══════════════════════════════════════════════
def print_summary():
    sep("STEP 5 — Done!")
    out     = os.path.join(PROJECT, RUN_NAME)
    best_pt = os.path.join(out, "weights", "best.pt")
    ncnn    = os.path.join(out, "weights", "best_ncnn_model")

    print(f"""
  {G}{B}Training complete!{X}

  {C}Output folder{X}  →  {out}/
  {C}Best model   {X}  →  {best_pt}
  {C}NCNN (RPi)  {X}  →  {ncnn}/
  {C}Loss plots  {X}  →  {out}/results.png

  {Y}{B}Run on Raspberry Pi:{X}
      from ultralytics import YOLO
      model = YOLO("{ncnn}/best_ncnn_model")
      results = model("road.jpg", conf=0.4)
      results[0].show()
    """)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="YOLO26 Road Anomaly Trainer — RTX 4060 Optimized"
    )
    parser.add_argument("--resume",      action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--weights",     type=str, default=None,
                        help="Specific .pt checkpoint path to resume from")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run dataset verification (no training)")
    parser.add_argument("--export-only", action="store_true",
                        help="Only export best.pt (skip training)")
    args = parser.parse_args()

    print(f"\n{B}{C}{'═'*56}")
    print(f"  🚗  YOLO26 Road Anomaly Detector")
    print(f"  🖥️  GPU : RTX 4060 Laptop (Ada Lovelace, 8 GB)")
    print(f"  🍓  Out : Raspberry Pi (NCNN)")
    print(f"{'═'*56}{X}\n")

    if args.export_only:
        export_for_rpi()
        return

    check_gpu()
    verify_dataset()

    if args.verify_only:
        info("--verify-only complete."); return

    print(f"\n{Y}{B}  All checks passed. Ready to train on RTX 4060!{X}")
    if not args.resume:
        input(f"\n  {B}Press ENTER to start training  |  Ctrl+C to cancel >{X} ")

    model, results = train(resume=args.resume, resume_weights=args.weights)
    export_for_rpi()
    print_summary()


if __name__ == "__main__":
    main()
