"""
infer.py — Inference Engine for M2a Vision Pipeline
=====================================================
Module: M2a Vision DL | MicroPlastiNet Pipeline
Author: MicroPlastiNet Team

PURPOSE
-------
Run end-to-end microplastic detection + classification on an input image.
Returns a structured JSON payload compatible with the M3 Graph GNN input format
and the M4 Dashboard display schema.

PIPELINE FLOW
-------------
  Image
   ↓
  [TinyYOLO Detector]  → bounding boxes, objectness scores
   ↓
  [NMS]               → filtered non-overlapping detections
   ↓
  [MPClassifier]      → per-particle shape class + confidence
   ↓
  JSON output          → forwarded to M3 (source attribution)

PIXEL-TO-SIZE CALIBRATION
--------------------------
Real microplastics are 1μm – 5mm. Under 10× magnification:
  1px ≈ 2.5μm  (10× objective, typical camera: 2.5μm/px)
  1px ≈ 0.625μm (40× objective)
Default: 10× @ 2.5μm/px → size_mm = px_diagonal * 0.0025

For accurate sizing in field deployment, calibrate with a stage micrometer
and pass --pixel_size_um.

USAGE
-----
  # Command line:
  python infer.py --image path/to/image.jpg \\
                  --det_checkpoint checkpoints/best_detector.pt \\
                  --clf_checkpoint checkpoints/best_classifier.pt \\
                  --output results.json

  # Python API:
  from infer import MicroplasticInference
  engine = MicroplasticInference(det_ckpt="...", clf_ckpt="...")
  result = engine.infer("sample.jpg")
  print(result)

OUTPUT JSON SCHEMA
------------------
{
  "image_path": "...",
  "timestamp": "2025-01-15T14:30:00Z",
  "sensor_id": "station_01",          # from --sensor_id flag
  "pixel_size_um": 2.5,
  "total_count": 5,
  "particles": [
    {
      "particle_id": 1,
      "bbox": [x1, y1, x2, y2],       # pixel coordinates
      "bbox_normalized": [cx, cy, w, h],
      "size_mm": 0.12,                 # estimated physical size
      "area_px2": 1234,
      "shape": "fragment",
      "shape_confidence": 0.87,
      "detection_confidence": 0.91
    },
    ...
  ],
  "shape_distribution": {
    "fragment": 2, "fiber": 1, "film": 1, "bead": 1, "foam": 0
  },
  "mean_size_mm": 0.09,
  "size_range_mm": [0.04, 0.18],
  "processing_time_ms": 42.1,
  "model_versions": {"detector": "TinyYOLO", "classifier": "EfficientNet-B0"},
  "data_quality": "synthetic"          # "real" when using camera data
}
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    build_detector, build_classifier, load_checkpoint,
    NUM_CLASSES, IMG_SIZE, ANCHORS
)
from dataset import SHAPE_CLASSES


# ─────────────────────── NMS & Decoding ─────────────────────────────────────

def decode_yolo_predictions(
    raw_preds: List[torch.Tensor],
    conf_thresh: float = 0.35,
    img_size: int = IMG_SIZE,
) -> List[Dict]:
    """
    Decode raw YOLO multi-scale predictions into bounding box candidates.

    Each candidate: {bbox_normalized, confidence, cls_logits, scale}

    Parameters
    ----------
    raw_preds  : List of 3 scale tensors from TinyYOLO.forward().
    conf_thresh: Minimum objectness confidence threshold.
    img_size   : Input image dimension (used for anchor scaling).

    Returns
    -------
    List of detection dicts before NMS.
    """
    candidates = []

    for scale_i, pred in enumerate(raw_preds):
        # pred shape: (1, num_anchors, H, W, 5+C)
        pred = pred.squeeze(0)  # (3, H, W, 10)
        num_anch, H, W, _ = pred.shape

        anchors_wh = torch.tensor(
            ANCHORS[scale_i], dtype=torch.float32, device=pred.device)

        # Decode sigmoid for objectness + offsets
        tx = torch.sigmoid(pred[..., 0])   # x offset
        ty = torch.sigmoid(pred[..., 1])   # y offset
        tw = pred[..., 2]                  # log scale width
        th = pred[..., 3]                  # log scale height
        obj = torch.sigmoid(pred[..., 4])  # objectness
        cls_logits = pred[..., 5:]         # (3, H, W, C)

        # Grid offsets
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=pred.device, dtype=torch.float32),
            torch.arange(W, device=pred.device, dtype=torch.float32),
            indexing="ij",
        )

        # Absolute grid positions (normalized 0–1)
        bx = (tx + grid_x.unsqueeze(0)) / W
        by = (ty + grid_y.unsqueeze(0)) / H

        # Anchor-scaled widths & heights
        aw = anchors_wh[:, 0].view(num_anch, 1, 1)  # (3, 1, 1)
        ah = anchors_wh[:, 1].view(num_anch, 1, 1)
        bw = aw * torch.exp(tw.clamp(-4, 4))
        bh = ah * torch.exp(th.clamp(-4, 4))

        # Flatten confident detections
        mask = obj > conf_thresh
        if mask.sum() == 0:
            continue

        for a_i in range(num_anch):
            for h_i in range(H):
                for w_i in range(W):
                    if obj[a_i, h_i, w_i].item() < conf_thresh:
                        continue
                    cx_n = bx[a_i, h_i, w_i].item()
                    cy_n = by[a_i, h_i, w_i].item()
                    bw_n = bw[a_i, h_i, w_i].item()
                    bh_n = bh[a_i, h_i, w_i].item()

                    x1 = cx_n - bw_n / 2
                    y1 = cy_n - bh_n / 2
                    x2 = cx_n + bw_n / 2
                    y2 = cy_n + bh_n / 2

                    candidates.append({
                        "bbox_norm": [
                            float(np.clip(x1, 0, 1)),
                            float(np.clip(y1, 0, 1)),
                            float(np.clip(x2, 0, 1)),
                            float(np.clip(y2, 0, 1)),
                        ],
                        "confidence": float(obj[a_i, h_i, w_i].item()),
                        "cls_logits": cls_logits[a_i, h_i, w_i].cpu(),
                        "scale": scale_i,
                    })

    return candidates


def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] normalized boxes."""
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union


def nms(candidates: List[Dict], iou_thresh: float = 0.45) -> List[Dict]:
    """
    Non-Maximum Suppression: remove overlapping detections.

    Parameters
    ----------
    candidates : List of detection dicts from decode_yolo_predictions().
    iou_thresh : Overlap threshold for suppression.

    Returns
    -------
    Filtered list of non-overlapping detections.
    """
    if not candidates:
        return []
    candidates_sorted = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept = []
    while candidates_sorted:
        best = candidates_sorted.pop(0)
        kept.append(best)
        candidates_sorted = [
            c for c in candidates_sorted
            if bbox_iou(best["bbox_norm"], c["bbox_norm"]) < iou_thresh
        ]
    return kept


# ────────────────────────── Inference Engine ────────────────────────────────

class MicroplasticInference:
    """
    End-to-end microplastic detection + shape classification engine.

    Loads TinyYOLO (detection) and MPClassifier (EfficientNet-B0 shape),
    processes an image, and returns a structured JSON-compatible dict.

    Parameters
    ----------
    det_checkpoint  : Path to TinyYOLO .pt checkpoint (or None for untrained).
    clf_checkpoint  : Path to MPClassifier .pt checkpoint (or None for untrained).
    device          : Torch device.
    conf_thresh     : Detection objectness threshold (default 0.35).
    nms_thresh      : NMS IoU threshold (default 0.45).
    pixel_size_um   : Microns per pixel at current magnification (default 2.5).
    sensor_id       : Station/sensor identifier for JSON output.
    """

    def __init__(
        self,
        det_checkpoint: Optional[str] = None,
        clf_checkpoint: Optional[str] = None,
        device: Optional[torch.device] = None,
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
        pixel_size_um: float = 2.5,
        sensor_id: str = "station_01",
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.pixel_size_um = pixel_size_um
        self.sensor_id = sensor_id

        # ── Load detector ────────────────────────────────────────────────
        self.detector = build_detector().to(self.device)
        if det_checkpoint and Path(det_checkpoint).exists():
            self.detector, _ = load_checkpoint(
                self.detector, det_checkpoint, self.device)
        else:
            print("  [WARN] No detector checkpoint loaded — using random weights")
        self.detector.eval()

        # ── Load classifier ──────────────────────────────────────────────
        self.classifier = build_classifier(pretrained=False).to(self.device)
        if clf_checkpoint and Path(clf_checkpoint).exists():
            self.classifier, _ = load_checkpoint(
                self.classifier, clf_checkpoint, self.device)
        else:
            print("  [WARN] No classifier checkpoint — using random weights")
        self.classifier.eval()

        # Preprocessing
        self.det_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.clf_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _px_to_mm(self, px_length: float) -> float:
        """Convert pixel length to millimeters using calibration constant."""
        return px_length * self.pixel_size_um / 1000.0

    def _classify_crop(self, img_pil: Image.Image, bbox_norm: List[float]) -> Tuple[str, float]:
        """
        Crop particle region from image and run shape classifier.

        Parameters
        ----------
        img_pil    : Full PIL image.
        bbox_norm  : Normalized [x1, y1, x2, y2] bounding box.

        Returns
        -------
        (shape_name, confidence) tuple.
        """
        W, H = img_pil.size
        x1, y1, x2, y2 = bbox_norm
        px1, py1 = int(x1 * W), int(y1 * H)
        px2, py2 = int(x2 * W), int(y2 * H)

        # Add context padding
        pad = 0.15
        pw = (px2 - px1) * pad
        ph = (py2 - py1) * pad
        px1 = max(0, int(px1 - pw)); py1 = max(0, int(py1 - ph))
        px2 = min(W, int(px2 + pw)); py2 = min(H, int(py2 + ph))

        if px2 <= px1 or py2 <= py1:
            return "fragment", 0.5  # fallback

        crop = img_pil.crop((px1, py1, px2, py2))
        crop_t = self.clf_transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(crop_t)
            probs = F.softmax(logits, dim=1).squeeze(0)
            conf, cls_idx = probs.max(0)

        return SHAPE_CLASSES[cls_idx.item()], float(conf.item())

    def infer(self, image_path: str) -> Dict:
        """
        Run full detection + classification pipeline on one image.

        Parameters
        ----------
        image_path : Path to input image (JPEG/PNG).

        Returns
        -------
        result_dict : Structured dict matching the M2a→M3 JSON schema.
        """
        t_start = time.time()

        # ── Load image ───────────────────────────────────────────────────
        img_pil = Image.open(image_path).convert("RGB")
        W_orig, H_orig = img_pil.size
        img_tensor = self.det_transform(img_pil).unsqueeze(0).to(self.device)

        # ── Detection ────────────────────────────────────────────────────
        with torch.no_grad():
            raw_preds = self.detector(img_tensor)

        candidates = decode_yolo_predictions(
            raw_preds, conf_thresh=self.conf_thresh, img_size=IMG_SIZE)
        detections = nms(candidates, iou_thresh=self.nms_thresh)

        # ── Per-particle classification ───────────────────────────────────
        particles = []
        shape_counts = {s: 0 for s in SHAPE_CLASSES}

        for i, det in enumerate(detections):
            bbox_n = det["bbox_norm"]
            x1, y1, x2, y2 = bbox_n

            # Pixel coordinates in original image
            px1 = int(x1 * W_orig); py1 = int(y1 * H_orig)
            px2 = int(x2 * W_orig); py2 = int(y2 * H_orig)
            w_px = px2 - px1; h_px = py2 - py1

            if w_px <= 0 or h_px <= 0:
                continue

            # Physical size (diagonal)
            diag_px = float(np.sqrt(w_px ** 2 + h_px ** 2))
            size_mm = self._px_to_mm(diag_px)
            area_px2 = w_px * h_px

            shape, shape_conf = self._classify_crop(img_pil, bbox_n)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1

            particles.append({
                "particle_id": i + 1,
                "bbox": [px1, py1, px2, py2],
                "bbox_normalized": [
                    round((x1 + x2) / 2, 4),
                    round((y1 + y2) / 2, 4),
                    round(x2 - x1, 4),
                    round(y2 - y1, 4),
                ],
                "size_mm": round(size_mm, 4),
                "area_px2": area_px2,
                "shape": shape,
                "shape_confidence": round(shape_conf, 4),
                "detection_confidence": round(det["confidence"], 4),
            })

        t_elapsed = (time.time() - t_start) * 1000.0  # ms

        sizes = [p["size_mm"] for p in particles]
        result = {
            "image_path": str(image_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensor_id": self.sensor_id,
            "pixel_size_um": self.pixel_size_um,
            "image_dimensions": {"width": W_orig, "height": H_orig},
            "total_count": len(particles),
            "particles": particles,
            "shape_distribution": shape_counts,
            "mean_size_mm": round(float(np.mean(sizes)), 4) if sizes else 0.0,
            "size_range_mm": [
                round(min(sizes), 4), round(max(sizes), 4)
            ] if sizes else [0.0, 0.0],
            "processing_time_ms": round(t_elapsed, 2),
            "model_versions": {
                "detector": "TinyYOLO-v1",
                "classifier": "EfficientNet-B0-MPClassifier-v1",
            },
            "data_quality": "synthetic",
            "note": (
                "SYNTHETIC DATA. Field accuracy: 60-70% (camera); "
                "~85% with UV fluorescence (MP-Set dataset)."
            ),
        }
        return result

    def infer_and_annotate(
        self, image_path: str, output_image_path: str
    ) -> Tuple[Dict, np.ndarray]:
        """
        Run inference and draw annotated bounding boxes on the image.

        Parameters
        ----------
        image_path       : Input image path.
        output_image_path: Where to save the annotated image.

        Returns
        -------
        (result_dict, annotated_image_bgr)
        """
        result = self.infer(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            img_bgr = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()

        # Color palette per shape class
        COLORS = {
            "fragment": (50, 120, 255),   # blue
            "fiber":    (50, 220, 100),   # green
            "film":     (220, 180, 50),   # yellow
            "bead":     (220, 50, 220),   # magenta
            "foam":     (50, 220, 220),   # cyan
        }

        for p in result["particles"]:
            x1, y1, x2, y2 = p["bbox"]
            color = COLORS.get(p["shape"], (200, 200, 200))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            label = f"{p['shape']} {p['shape_confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            lx, ly = x1, max(y1 - 5, label_size[1] + 3)
            cv2.rectangle(img_bgr,
                          (lx, ly - label_size[1] - 3),
                          (lx + label_size[0] + 4, ly + 3),
                          color, -1)
            cv2.putText(img_bgr, label, (lx + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Summary overlay (top-left)
        summary = (f"N={result['total_count']} | "
                   f"mean={result['mean_size_mm']:.3f}mm | "
                   f"{result['processing_time_ms']:.0f}ms")
        cv2.putText(img_bgr, summary, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(img_bgr, summary, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                    cv2.LINE_AA)

        cv2.imwrite(output_image_path, img_bgr)
        return result, img_bgr


# ─────────────────────────────── CLI ────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2a Inference: detect and classify microplastics in image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--det_checkpoint",
                        default="checkpoints/best_detector.pt",
                        help="TinyYOLO checkpoint")
    parser.add_argument("--clf_checkpoint",
                        default="checkpoints/best_classifier.pt",
                        help="MPClassifier checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: print to stdout)")
    parser.add_argument("--annotated_image", default=None,
                        help="Save annotated image to this path")
    parser.add_argument("--conf_thresh", type=float, default=0.35)
    parser.add_argument("--nms_thresh", type=float, default=0.45)
    parser.add_argument("--pixel_size_um", type=float, default=2.5,
                        help="Microns per pixel (calibration)")
    parser.add_argument("--sensor_id", default="station_01")
    return parser.parse_args()


def main():
    args = parse_args()

    engine = MicroplasticInference(
        det_checkpoint=args.det_checkpoint,
        clf_checkpoint=args.clf_checkpoint,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
        pixel_size_um=args.pixel_size_um,
        sensor_id=args.sensor_id,
    )

    if args.annotated_image:
        result, _ = engine.infer_and_annotate(args.image, args.annotated_image)
        print(f"Annotated image saved to {args.annotated_image}")
    else:
        result = engine.infer(args.image)

    json_str = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"Results saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
