# ======================================================================
# master_pipeline.py
# FLIR ADAS - Automated ML Pipeline for Self-Driving Car Object Detection
# Robotic Process Automation (RPA) using Python
#
# Usage:
#   python master_pipeline.py                  # full pipeline
#   python master_pipeline.py --skip-train     # skip training (use existing model)
#   python master_pipeline.py --skip-track     # skip tracking (use existing results)
#   python master_pipeline.py --report-only    # generate report from existing results
# ======================================================================

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import ctypes
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Argument parser ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FLIR ADAS Automated ML Pipeline")
parser.add_argument("--skip-train",  action="store_true", help="Skip training, use existing best.pt")
parser.add_argument("--skip-track",  action="store_true", help="Skip tracking, use existing results")
parser.add_argument("--report-only", action="store_true", help="Only generate report from existing results")
args = parser.parse_args()

# ── Paths ──────────────────────────────────────────────────────────────
ROOT         = Path(r"c:\Users\SHANTANU\flir_adas")
DATA_DIR     = ROOT / "data"
ANNOT_DIR    = DATA_DIR / "annotations"
RAW_TRAIN    = DATA_DIR / "training_data"
RAW_TEST     = DATA_DIR / "test_data"
IMG_TRAIN    = DATA_DIR / "images" / "train"
IMG_VAL      = DATA_DIR / "images" / "val"
LBL_TRAIN    = DATA_DIR / "labels" / "train"
LBL_VAL      = DATA_DIR / "labels" / "val"
TRAINING_DIR = ROOT / "training"
OUTPUT_RUNS  = Path(r"D:\flir_adas_runs\runs")
LOG_DIR      = Path(r"D:\flir_adas_runs\logs")
REPORT_DIR   = Path(r"D:\flir_adas_runs\reports")

ANNOT_TRAIN_COCO = ANNOT_DIR / "training_coco.json"
ANNOT_TEST_COCO  = ANNOT_DIR / "test_coco.json"
ANNOT_INDEX      = ANNOT_DIR / "training_index.json"

# ── Calibration constants ──────────────────────────────────────────────
DEFAULT_FPS         = 30.0
REAL_CAR_WIDTH_M    = 1.8    # standard car width in meters
MEDIAN_CAR_WIDTH_PX = 22.0   # measured from 71281 annotations
METERS_PER_PIXEL    = REAL_CAR_WIDTH_M / MEDIAN_CAR_WIDTH_PX

# ── Logging setup ──────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("flir_pipeline")
log.info(f"Log file: {log_file}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Check data availability
# ══════════════════════════════════════════════════════════════════════
def step1_check_data():
    log.info("=" * 60)
    log.info("STEP 1 — Checking data availability")
    log.info("=" * 60)

    required = {
        "training_data dir"   : RAW_TRAIN,
        "test_data dir"       : RAW_TEST,
        "training_coco.json"  : ANNOT_TRAIN_COCO,
        "test_coco.json"      : ANNOT_TEST_COCO,
        "training_index.json" : ANNOT_INDEX,
    }

    all_ok = True
    for name, path in required.items():
        ok = path.exists()
        log.info(f"  {'OK' if ok else 'MISSING'}  {name}: {path}")
        if not ok:
            all_ok = False

    if not all_ok:
        log.error("Missing required data files. Aborting pipeline.")
        sys.exit(1)

    train_imgs = list(RAW_TRAIN.glob("*.jpg")) + list(RAW_TRAIN.glob("*.png"))
    test_imgs  = list(RAW_TEST.glob("*.jpg"))  + list(RAW_TEST.glob("*.png"))
    log.info(f"  Training images : {len(train_imgs)}")
    log.info(f"  Test images     : {len(test_imgs)}")
    log.info("STEP 1 complete\n")
    return {"train_count": len(train_imgs), "test_count": len(test_imgs)}


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Convert COCO annotations to YOLO format + remap classes
# ══════════════════════════════════════════════════════════════════════
def step2_convert_annotations():
    log.info("=" * 60)
    log.info("STEP 2 — Converting COCO annotations to YOLO format")
    log.info("=" * 60)

    existing_train = list(LBL_TRAIN.glob("*.txt")) if LBL_TRAIN.exists() else []
    existing_val   = list(LBL_VAL.glob("*.txt"))   if LBL_VAL.exists()   else []

    if len(existing_train) > 5000 and len(existing_val) > 1000:
        log.info(f"  Labels already converted: {len(existing_train)} train, {len(existing_val)} val")
        log.info("STEP 2 skipped (already done)\n")
        return {"train_labels": len(existing_train), "val_labels": len(existing_val)}

    LBL_TRAIN.mkdir(parents=True, exist_ok=True)
    LBL_VAL.mkdir(parents=True, exist_ok=True)

    def convert_coco_to_yolo(coco_json_path, label_dir):
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images     = {im["id"]: im for im in coco["images"]}
        cat_ids    = sorted(c["id"] for c in coco["categories"])
        cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        ann_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_img[ann["image_id"]].append(ann)

        written = 0
        for img_id, img_info in images.items():
            w, h  = img_info["width"], img_info["height"]
            lines = []
            for ann in ann_by_img.get(img_id, []):
                cid = ann["category_id"]
                if cid not in cat_to_idx:
                    continue
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                lines.append(f"{cat_to_idx[cid]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stem = Path(img_info["file_name"]).stem
            (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            written += 1
        return written, coco["categories"]

    log.info("  Converting training annotations...")
    n_train, categories = convert_coco_to_yolo(ANNOT_TRAIN_COCO, LBL_TRAIN)
    log.info(f"    {n_train} label files written")

    log.info("  Converting test annotations...")
    n_val, _ = convert_coco_to_yolo(ANNOT_TEST_COCO, LBL_VAL)
    log.info(f"    {n_val} label files written")

    # Save full 80-class names
    all_names = [c["name"] for c in sorted(categories, key=lambda x: x["id"])]
    (DATA_DIR / "names.txt").write_text("\n".join(all_names), encoding="utf-8")

    # Remap to only used classes
    log.info("  Remapping to used classes only...")
    counter = Counter()
    for ldir in [LBL_TRAIN, LBL_VAL]:
        for lf in ldir.glob("*.txt"):
            for line in lf.read_text().strip().splitlines():
                if line.strip():
                    counter[int(line.split()[0])] += 1

    used_indices = sorted(counter.keys())
    new_names    = [all_names[i] for i in used_indices]
    old_to_new   = {old: new for new, old in enumerate(used_indices)}

    for ldir in [LBL_TRAIN, LBL_VAL]:
        for lf in ldir.glob("*.txt"):
            lines     = lf.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                if old_cls not in old_to_new:
                    continue
                parts[0] = str(old_to_new[old_cls])
                new_lines.append(" ".join(parts))
            lf.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    (DATA_DIR / "names.txt").write_text("\n".join(new_names), encoding="utf-8")
    log.info(f"  Final classes ({len(new_names)}): {new_names}")
    log.info("STEP 2 complete\n")
    return {"train_labels": n_train, "val_labels": n_val, "classes": new_names}


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Setup image folders with hardlinks
# ══════════════════════════════════════════════════════════════════════
def step3_setup_images():
    log.info("=" * 60)
    log.info("STEP 3 — Setting up image folders (hardlinks)")
    log.info("=" * 60)

    def is_junction(p):
        try:
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(p))
            return bool(attrs & 0x400)
        except Exception:
            return False

    for p in [IMG_TRAIN, IMG_VAL]:
        if p.exists() and is_junction(p):
            os.rmdir(p)
            log.info(f"  Removed junction: {p.name}")
        p.mkdir(parents=True, exist_ok=True)

    def hardlink_images(src, dst, label):
        imgs  = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        count = 0
        for img in imgs:
            dst_file = dst / img.name
            if not dst_file.exists():
                try:
                    os.link(img, dst_file)
                except OSError:
                    shutil.copy2(img, dst_file)
                count += 1
        log.info(f"  {label}: {count} new links -> {dst}")
        return count

    n_train = hardlink_images(RAW_TRAIN, IMG_TRAIN, "TRAIN")
    n_val   = hardlink_images(RAW_TEST,  IMG_VAL,   "VAL")

    # Verify YOLO path swap works
    train_resolved = str(IMG_TRAIN.resolve()).replace("\\", "/")
    lbl_via_swap   = train_resolved.replace("images", "labels")
    lbl_actual     = str(LBL_TRAIN.resolve()).replace("\\", "/")
    assert lbl_via_swap == lbl_actual, (
        f"YOLO will not find labels!\n  Expected: {lbl_via_swap}\n  Actual: {lbl_actual}"
    )
    log.info("  YOLO path swap check: PASSED")
    log.info("STEP 3 complete\n")
    return {"train_links": n_train, "val_links": n_val}


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Write data.yaml
# ══════════════════════════════════════════════════════════════════════
def step4_write_yaml():
    log.info("=" * 60)
    log.info("STEP 4 — Writing data.yaml")
    log.info("=" * 60)

    names     = (DATA_DIR / "names.txt").read_text().strip().splitlines()
    train_abs = str(IMG_TRAIN.resolve()).replace("\\", "/")
    val_abs   = str(IMG_VAL.resolve()).replace("\\", "/")

    for c in ROOT.rglob("*.cache"):
        c.unlink()
        log.info(f"  Deleted stale cache: {c.name}")

    name_lines = "\n".join(f"  - {n}" for n in names)
    yaml_text  = (
        f"train: {train_abs}\n"
        f"val: {val_abs}\n"
        f"nc: {len(names)}\n"
        f"names:\n{name_lines}\n"
    )

    yaml_path = TRAINING_DIR / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    assert "images/train" in yaml_text
    assert "images/val"   in yaml_text
    log.info(f"  Written: {yaml_path}")
    log.info(f"  nc={len(names)}, classes={names}")
    log.info("STEP 4 complete\n")
    return {"yaml_path": str(yaml_path), "nc": len(names)}


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Train YOLOv8x model
# ══════════════════════════════════════════════════════════════════════
def step5_train(skip=False):
    log.info("=" * 60)
    log.info("STEP 5 — Training YOLOv8x model")
    log.info("=" * 60)

    best_pt = OUTPUT_RUNS / "yolov8x_flir" / "weights" / "best.pt"

    if skip or best_pt.exists():
        reason = "--skip-train" if skip else "weights already exist"
        log.info(f"  Skipping training ({reason}): {best_pt}")
        if not best_pt.exists():
            log.error("No weights found. Run without --skip-train first.")
            sys.exit(1)
        log.info("STEP 5 skipped\n")
        return {"skipped": True, "weights": str(best_pt)}

    from ultralytics import YOLO
    import torch

    torch.cuda.empty_cache()
    log.info(f"  GPU  : {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    base_weights = next(
        (p for p in [TRAINING_DIR / "yolov8x.pt", ROOT / "yolov8x.pt"] if p.exists()),
        "yolov8x.pt"
    )
    log.info(f"  Base weights: {base_weights}")

    model = YOLO(str(base_weights))
    t0    = time.time()
    model.train(
        data         = str(TRAINING_DIR / "data.yaml"),
        epochs       = 50,
        imgsz        = 640,
        batch        = 2,
        device       = 0,
        project      = str(OUTPUT_RUNS),
        name         = "yolov8x_flir",
        exist_ok     = True,
        workers      = 0,
        amp          = True,
        cache        = False,
        close_mosaic = 10,
    )

    elapsed = (time.time() - t0) / 3600
    log.info(f"  Training complete in {elapsed:.1f} hours")
    log.info(f"  Best weights: {best_pt}")
    log.info("STEP 5 complete\n")
    return {"skipped": False, "weights": str(best_pt), "hours": round(elapsed, 1)}


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate model on validation set
# ══════════════════════════════════════════════════════════════════════
def step6_evaluate():
    log.info("=" * 60)
    log.info("STEP 6 — Evaluating model on validation set")
    log.info("=" * 60)

    from ultralytics import YOLO

    best_pt = OUTPUT_RUNS / "yolov8x_flir" / "weights" / "best.pt"
    model   = YOLO(str(best_pt))
    names   = (DATA_DIR / "names.txt").read_text().strip().splitlines()

    metrics = model.val(
        data   = str(TRAINING_DIR / "data.yaml"),
        imgsz  = 640,
        batch  = 4,
        device = 0,
        split  = "val",
        plots  = True,
    )

    results = {
        "mAP50"     : round(float(metrics.box.map50), 4),
        "mAP50_95"  : round(float(metrics.box.map),   4),
        "precision" : round(float(metrics.box.mp),     4),
        "recall"    : round(float(metrics.box.mr),     4),
        "per_class" : {},
    }
    for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
        if i < len(names):
            results["per_class"][names[i]] = {
                "AP50": round(float(ap50), 4), "AP50_95": round(float(ap), 4)
            }

    log.info(f"  mAP50={results['mAP50']}  mAP50-95={results['mAP50_95']}")
    log.info(f"  Precision={results['precision']}  Recall={results['recall']}")
    for cls, v in results["per_class"].items():
        log.info(f"    {cls:<20} AP50={v['AP50']:.4f}")
    log.info("STEP 6 complete\n")
    return results


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — Parse Labelbox frame metadata
# ══════════════════════════════════════════════════════════════════════
def step7_parse_metadata():
    log.info("=" * 60)
    log.info("STEP 7 — Parsing Labelbox frame metadata")
    log.info("=" * 60)

    with open(ANNOT_INDEX, "r", encoding="utf-8") as f:
        frames_json = json.load(f)

    disk_images   = list(RAW_TRAIN.glob("*.jpg")) + list(RAW_TRAIN.glob("*.png"))
    disk_name_map = {p.stem: p.name for p in disk_images}

    frame_meta = {}
    for fr in frames_json.get("frames", []):
        video_meta       = fr.get("videoMetadata", {})
        video_id         = video_meta.get("videoId")
        frame_idx        = video_meta.get("frameIndex")
        dataset_frame_id = fr.get("datasetFrameId")
        if not dataset_frame_id:
            continue
        for stem, fname in disk_name_map.items():
            if dataset_frame_id in stem:
                frame_meta[fname] = {"videoId": video_id, "frameIndex": frame_idx}
                break

    log.info(f"  Mapped {len(frame_meta)} / {len(disk_images)} frames")
    log.info("STEP 7 complete\n")
    return frame_meta


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — ByteTrack tracking + speed estimation
# ══════════════════════════════════════════════════════════════════════
def step8_track_and_speed(frame_meta, skip=False):
    log.info("=" * 60)
    log.info("STEP 8 — ByteTrack tracking + speed estimation")
    log.info("=" * 60)

    out_px  = Path(r"D:\flir_adas_runs\tracking_results.json")
    out_kmh = Path(r"D:\flir_adas_runs\tracking_results_kmh.json")

    if skip and out_kmh.exists():
        log.info("  Loading existing tracking results (--skip-track)")
        with open(out_kmh) as f:
            data = json.load(f)
        log.info(f"  Loaded {len(data)} tracks")
        log.info("STEP 8 skipped\n")
        return data

    from ultralytics import YOLO

    best_pt = OUTPUT_RUNS / "yolov8x_flir" / "weights" / "best.pt"
    model   = YOLO(str(best_pt))

    def center_xyxy(xyxy):
        x1, y1, x2, y2 = xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    results = model.track(
        source  = str(RAW_TRAIN),
        tracker = "bytetrack.yaml",
        persist = True,
        stream  = True,
    )

    tracks = defaultdict(list)
    for res in results:
        fname = Path(res.path).name
        meta  = frame_meta.get(fname)
        if not meta or res.boxes is None:
            continue
        t = meta["frameIndex"] / DEFAULT_FPS
        for box in res.boxes:
            if box.id is None:
                continue
            tid    = int(box.id.item())
            cx, cy = center_xyxy(box.xyxy[0].tolist())
            tracks[tid].append((t, cx, cy, fname))

    track_speeds = {}
    for tid, pts in tracks.items():
        pts    = sorted(pts, key=lambda x: x[0])
        speeds = []
        for (t1, x1, y1, _), (t2, x2, y2, _) in zip(pts, pts[1:]):
            dt = t2 - t1
            if dt <= 0:
                continue
            speed_px_s = math.hypot(x2 - x1, y2 - y1) / dt
            speeds.append((round(t2, 3), round(speed_px_s, 2)))
        track_speeds[tid] = speeds

    with open(out_px, "w") as f:
        json.dump({str(k): v for k, v in track_speeds.items()}, f, indent=2)

    track_speeds_kmh = {}
    all_speeds = []
    for tid, readings in track_speeds.items():
        kmh = [(t, round(px_s * METERS_PER_PIXEL * 3.6, 2)) for t, px_s in readings]
        track_speeds_kmh[str(tid)] = kmh
        all_speeds.extend([s for _, s in kmh])

    with open(out_kmh, "w") as f:
        json.dump(track_speeds_kmh, f, indent=2)

    valid = [s for s in all_speeds if 0 < s < 150]
    log.info(f"  Tracked objects   : {len(track_speeds_kmh)}")
    log.info(f"  Speed readings    : {len(all_speeds)}  ({len(valid)} valid)")
    if valid:
        log.info(f"  Average speed     : {sum(valid)/len(valid):.1f} km/h")
        log.info(f"  Max speed         : {max(valid):.1f} km/h")
    log.info(f"  Saved: {out_kmh}")
    log.info("STEP 8 complete\n")
    return track_speeds_kmh


# ══════════════════════════════════════════════════════════════════════
# STEP 9 — Generate report
# ══════════════════════════════════════════════════════════════════════
def step9_generate_report(data_stats, eval_results, track_speeds_kmh):
    log.info("=" * 60)
    log.info("STEP 9 — Generating pipeline report")
    log.info("=" * 60)

    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = REPORT_DIR / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    all_speeds = [s for readings in track_speeds_kmh.values() for _, s in readings]
    valid      = [s for s in all_speeds if 0 < s < 150]
    avg_speed  = sum(valid) / len(valid) if valid else 0
    max_speed  = max(valid) if valid else 0
    min_speed  = min(valid) if valid else 0

    names = (DATA_DIR / "names.txt").read_text().strip().splitlines() if (DATA_DIR / "names.txt").exists() else []

    per_class_lines = ""
    if eval_results and "per_class" in eval_results:
        per_class_lines = "\n".join(
            f"    {cls:<20} AP50={v['AP50']:.4f}  AP50-95={v['AP50_95']:.4f}"
            for cls, v in eval_results["per_class"].items()
        )

    report = f"""
================================================================================
  FLIR ADAS -- Automated ML Pipeline Report
  Generated : {timestamp}
================================================================================

PIPELINE OVERVIEW
-----------------
  Automated RPA pipeline for self-driving car object detection, multi-object
  tracking, and speed estimation using FLIR ADAS RGB dashcam data.

  Steps executed:
    Step 1  Check data availability
    Step 2  Convert COCO annotations to YOLO format + remap classes
    Step 3  Setup image folders with hardlinks
    Step 4  Write data.yaml configuration
    Step 5  Train YOLOv8x detection model (50 epochs)
    Step 6  Evaluate model on validation set
    Step 7  Parse Labelbox frame metadata
    Step 8  ByteTrack multi-object tracking + speed estimation
    Step 9  Generate this report

DATASET
-------
  Source          : FLIR ADAS DualCapture RGB (Santa Barbara, CA)
  Training images : {data_stats.get('train_count', 'N/A')}
  Test images     : {data_stats.get('test_count', 'N/A')}
  Classes         : {len(names)}
  Class list      : {', '.join(names)}
  Original format : COCO JSON
  Converted to    : YOLO TXT format

MODEL
-----
  Architecture    : YOLOv8x (68M parameters, 257.5 GFLOPs)
  Training epochs : 50
  Image size      : 640 x 640
  Batch size      : 2 (RTX 3050 6GB)
  Optimizer       : AdamW (lr=0.000526)
  AMP             : Enabled (mixed precision)
  Pretrained on   : COCO (RGB, 80 classes)
  Fine-tuned on   : FLIR ADAS dataset (15 classes)

EVALUATION RESULTS
------------------
  mAP50           : {eval_results.get('mAP50', 'N/A')}
  mAP50-95        : {eval_results.get('mAP50_95', 'N/A')}
  Precision       : {eval_results.get('precision', 'N/A')}
  Recall          : {eval_results.get('recall', 'N/A')}

  Per-class performance:
{per_class_lines}

TRACKING RESULTS
----------------
  Algorithm       : ByteTrack (multi-object tracking)
  Source          : {RAW_TRAIN}
  Tracked objects : {len(track_speeds_kmh)}
  Speed readings  : {len(all_speeds)}
  Valid readings  : {len(valid)} (filtered 0-150 km/h)

SPEED ESTIMATION
----------------
  Method          : Centroid displacement between frames / time delta
  Calibration     : Ground truth car width (empirical)
    Real car width    = {REAL_CAR_WIDTH_M} m (standard)
    Median px width   = {MEDIAN_CAR_WIDTH_PX} px (measured from 71,281 annotations)
    Meters per pixel  = {METERS_PER_PIXEL:.4f} m/px
    Frame rate        = {DEFAULT_FPS} fps
  Average speed   : {avg_speed:.1f} km/h
  Max speed       : {max_speed:.1f} km/h
  Min speed       : {min_speed:.2f} km/h

OUTPUT FILES
------------
  Best model      : D:\\flir_adas_runs\\runs\\yolov8x_flir\\weights\\best.pt
  Tracking px/s   : D:\\flir_adas_runs\\tracking_results.json
  Tracking km/h   : D:\\flir_adas_runs\\tracking_results_kmh.json
  Pipeline log    : {log_file}
  This report     : {report_path}

================================================================================
  END OF REPORT
================================================================================
"""

    report_path.write_text(report.strip(), encoding="utf-8")
    log.info(f"  Report saved: {report_path}")
    log.info("STEP 9 complete\n")
    return str(report_path)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    start_time = time.time()

    log.info("=" * 60)
    log.info("  FLIR ADAS -- Automated ML Pipeline")
    log.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    if args.report_only:
        log.info("--report-only mode")
        out_kmh = Path(r"D:\flir_adas_runs\tracking_results_kmh.json")
        if not out_kmh.exists():
            log.error(f"No tracking results at {out_kmh}. Run full pipeline first.")
            sys.exit(1)
        with open(out_kmh) as f:
            track_speeds_kmh = json.load(f)
        step9_generate_report({}, {}, track_speeds_kmh)
        return

    # Run full pipeline
    data_stats       = step1_check_data()
    step2_convert_annotations()
    step3_setup_images()
    step4_write_yaml()
    step5_train(skip=args.skip_train)
    eval_results     = step6_evaluate()
    frame_meta       = step7_parse_metadata()
    track_speeds_kmh = step8_track_and_speed(frame_meta, skip=args.skip_track)
    report_path      = step9_generate_report(data_stats, eval_results, track_speeds_kmh)

    elapsed = (time.time() - start_time) / 3600
    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE in {elapsed:.2f} hours")
    log.info(f"  mAP50  : {eval_results.get('mAP50', 'N/A')}")
    log.info(f"  Tracks : {len(track_speeds_kmh)}")
    log.info(f"  Report : {report_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
