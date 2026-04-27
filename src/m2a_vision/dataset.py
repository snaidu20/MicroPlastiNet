"""
dataset.py — Synthetic Microplastic Microscopy Dataset Generator
================================================================
Module: M2a Vision DL | MicroPlastiNet Pipeline
Author: MicroPlastiNet Team

PURPOSE
-------
Generates a synthetic microscopy image dataset that mimics real microplastic
particle images for training and validation of the M2a detection/classification
pipeline.

REAL DATASETS (use these when available — drop-in replacements):
  - Kaggle Microplastic CV Dataset:
      https://www.kaggle.com/code/mathieuduverne/microplastic-detection-yolov8-map-50-76-2
      Format: YOLO annotation format (class x_center y_center w h, normalized)
  - MP-Set Fluorescence Dataset:
      https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset
      Format: COCO JSON with UV fluorescence channel

NOTE: Synthetic data is used here ONLY because Kaggle datasets cannot be
downloaded in this sandbox environment. The Dataset class is designed to be
a drop-in replacement — simply point `root_dir` at real dataset directories
organized in the same structure (YOLO format), and training will use real data.

PARTICLE MORPHOLOGY REFERENCE
------------------------------
Five shape classes modeled after the GESAMP (2015) and Rocha-Santos (2015)
microplastic classification scheme:
  0: fragment   — irregular angular shards (most common, ~40% of MPs)
  1: fiber      — elongated filaments (synthetic textiles, ~30%)
  2: film       — thin translucent sheets (packaging films, ~15%)
  3: bead       — spherical pellets (nurdles, microbeads, ~10%)
  4: foam       — irregular porous particles (EPS, ~5%)

Image appearance mimics:
  - 10x–40x stereo microscope on water filter paper (bright-field)
  - Particle colors: white/clear, blue, black, red (most common MPs)
  - Background: textured filter paper with slight vignette + noise
"""

import os
import random
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─────────────────────────── Constants ──────────────────────────────────────

SHAPE_CLASSES = ["fragment", "fiber", "film", "bead", "foam"]
IMG_SIZE = 416          # Standard YOLO input size
MAX_PARTICLES = 8       # Max particles per image
MIN_PARTICLES = 1

# Realistic MP colors observed under bright-field microscopy
MP_COLORS = [
    (220, 210, 195),  # white/clear (most common)
    (180, 200, 220),  # light blue
    (40,  40,  50),   # black
    (190, 70,  60),   # red
    (100, 160, 100),  # green
    (210, 190, 100),  # yellow
    (160, 120, 80),   # brown
]


# ─────────────────────── Synthetic Image Generation ─────────────────────────

class SyntheticParticleRenderer:
    """
    Generates realistic synthetic microplastic microscopy images using
    procedural geometry and texture techniques.

    Each image contains 1–8 particles on a textured water/filter-paper
    background, returned alongside YOLO-format bounding box annotations.
    """

    def __init__(self, img_size: int = IMG_SIZE, seed: Optional[int] = None):
        self.img_size = img_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── Background ──────────────────────────────────────────────────────────

    def _make_background(self) -> np.ndarray:
        """
        Creates a textured background simulating a filter membrane under
        bright-field microscopy: slight cream/gray tint, Gaussian noise,
        and subtle circular particle artifacts.
        """
        s = self.img_size
        # Base cream/off-white background (filter paper)
        base_color = np.array([235, 228, 215], dtype=np.float32)
        bg = np.ones((s, s, 3), dtype=np.float32) * base_color

        # Perlin-like texture: layered low-amplitude Gaussian blurs of noise
        for scale in [4, 8, 16, 32]:
            noise = np.random.normal(0, 6, (s // scale, s // scale, 3))
            noise_up = cv2.resize(noise, (s, s), interpolation=cv2.INTER_LINEAR)
            bg += noise_up

        # Vignette: darker edges (lens falloff)
        cx, cy = s / 2, s / 2
        Y, X = np.ogrid[:s, :s]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / (s * 0.6)
        vignette = 1.0 - 0.25 * np.clip(dist, 0, 1)
        bg *= vignette[:, :, None]

        # Occasional dust/debris specks
        n_specks = random.randint(5, 25)
        for _ in range(n_specks):
            sx, sy = random.randint(0, s - 1), random.randint(0, s - 1)
            r = random.randint(1, 3)
            intensity = random.uniform(0.7, 1.1)
            cv2.circle(bg, (sx, sy), r, (200 * intensity,) * 3, -1)

        return np.clip(bg, 0, 255).astype(np.uint8)

    # ── Particle Drawers ─────────────────────────────────────────────────────

    def _draw_fragment(self, canvas: np.ndarray, color: Tuple) -> Tuple[int, int, int, int]:
        """Irregular angular shard — most common MP morphology."""
        s = self.img_size
        size = random.randint(20, 70)
        cx = random.randint(size, s - size)
        cy = random.randint(size, s - size)

        n_pts = random.randint(5, 9)
        angles = sorted(np.random.uniform(0, 2 * np.pi, n_pts))
        radii = np.random.uniform(size * 0.4, size * 0.9, n_pts)
        pts = np.array([
            [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
            for r, a in zip(radii, angles)
        ], dtype=np.int32)

        # Draw filled shape with slight translucency blend
        mask = np.zeros((s, s), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        alpha = random.uniform(0.55, 0.85)
        canvas[mask > 0] = (
            canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)

        # Edge highlight
        cv2.polylines(canvas, [pts], True,
                      tuple(int(c * 0.6) for c in color), 1, cv2.LINE_AA)

        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return x1, y1, x2, y2

    def _draw_fiber(self, canvas: np.ndarray, color: Tuple) -> Tuple[int, int, int, int]:
        """Elongated filament — synthetic textile fiber morphology."""
        s = self.img_size
        length = random.randint(50, 150)
        width = random.randint(2, 6)
        angle = random.uniform(0, np.pi)
        cx = random.randint(length // 2, s - length // 2)
        cy = random.randint(20, s - 20)

        # Slightly curved fiber via polyline
        n_segs = random.randint(4, 8)
        pts = []
        for i in range(n_segs + 1):
            t = i / n_segs
            x = cx + (t - 0.5) * length * np.cos(angle)
            y = cy + (t - 0.5) * length * np.sin(angle)
            # Add gentle curvature
            x += np.sin(t * np.pi) * random.uniform(-10, 10)
            y += np.sin(t * np.pi) * random.uniform(-10, 10)
            pts.append((int(x), int(y)))

        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], False, color, width, cv2.LINE_AA)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = width * 2
        return (max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(s, max(xs) + pad), min(s, max(ys) + pad))

    def _draw_film(self, canvas: np.ndarray, color: Tuple) -> Tuple[int, int, int, int]:
        """Thin translucent sheet — packaging film fragment."""
        s = self.img_size
        size = random.randint(30, 90)
        cx = random.randint(size, s - size)
        cy = random.randint(size, s - size)

        # Irregular quadrilateral with slight transparency
        pts = np.array([
            [cx + random.randint(-size, size), cy + random.randint(-size, size)]
            for _ in range(4)
        ], dtype=np.int32)

        mask = np.zeros((s, s), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
        alpha = random.uniform(0.25, 0.50)  # films are thin, more transparent
        canvas[mask > 0] = (
            canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
        cv2.polylines(canvas, [pts], True, color, 1, cv2.LINE_AA)

        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return x1, y1, x2, y2

    def _draw_bead(self, canvas: np.ndarray, color: Tuple) -> Tuple[int, int, int, int]:
        """Spherical pellet — nurdle or microbead morphology."""
        s = self.img_size
        r = random.randint(10, 35)
        cx = random.randint(r + 5, s - r - 5)
        cy = random.randint(r + 5, s - r - 5)

        # Main circle with gradient shading (specular highlight)
        alpha = random.uniform(0.70, 0.92)
        mask = np.zeros((s, s), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        canvas[mask > 0] = (
            canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)

        # Specular highlight (upper-left)
        hi_r = max(2, r // 4)
        hi_x = cx - r // 3
        hi_y = cy - r // 3
        cv2.circle(canvas, (hi_x, hi_y), hi_r,
                   (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60)),
                   -1, cv2.LINE_AA)

        # Rim
        cv2.circle(canvas, (cx, cy), r,
                   tuple(int(c * 0.7) for c in color), 1, cv2.LINE_AA)

        return cx - r, cy - r, cx + r, cy + r

    def _draw_foam(self, canvas: np.ndarray, color: Tuple) -> Tuple[int, int, int, int]:
        """Irregular porous particle — EPS foam fragment."""
        s = self.img_size
        size = random.randint(25, 65)
        cx = random.randint(size, s - size)
        cy = random.randint(size, s - size)

        # Outer blob
        n_pts = random.randint(8, 14)
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        radii = np.random.uniform(size * 0.5, size * 0.95, n_pts)
        pts = np.array([
            [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
            for r, a in zip(radii, angles)
        ], dtype=np.int32)

        mask = np.zeros((s, s), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        # Porous: carve out small holes
        n_holes = random.randint(3, 10)
        for _ in range(n_holes):
            hx = cx + random.randint(-size // 2, size // 2)
            hy = cy + random.randint(-size // 2, size // 2)
            hr = random.randint(3, 10)
            cv2.circle(mask, (hx, hy), hr, 0, -1)

        alpha = 0.75
        canvas[mask > 0] = (
            canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
        cv2.polylines(canvas, [pts], True,
                      tuple(int(c * 0.65) for c in color), 1, cv2.LINE_AA)

        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return x1, y1, x2, y2

    # ── Full Image ───────────────────────────────────────────────────────────

    DRAWERS = [
        "_draw_fragment", "_draw_fiber", "_draw_film",
        "_draw_bead", "_draw_foam"
    ]

    def generate_image(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate a single synthetic microscopy image with particle annotations.

        Returns
        -------
        img : np.ndarray  shape (H, W, 3) uint8, BGR
        annotations : List[Dict]
            Each dict: {class_id, class_name, bbox_xyxy, bbox_yolo}
            bbox_yolo is [cx, cy, w, h] normalized to [0,1]
        """
        canvas = self._make_background()
        n_particles = random.randint(MIN_PARTICLES, MAX_PARTICLES)
        annotations = []

        for _ in range(n_particles):
            cls_id = random.randint(0, 4)
            color = random.choice(MP_COLORS)
            draw_fn = getattr(self, self.DRAWERS[cls_id])
            try:
                x1, y1, x2, y2 = draw_fn(canvas, color)
            except Exception:
                continue

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self.img_size - 1, x2); y2 = min(self.img_size - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1; h = y2 - y1
            cx = (x1 + x2) / 2 / self.img_size
            cy = (y1 + y2) / 2 / self.img_size
            nw = w / self.img_size
            nh = h / self.img_size

            annotations.append({
                "class_id": cls_id,
                "class_name": SHAPE_CLASSES[cls_id],
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_yolo": [cx, cy, nw, nh],
            })

        # Slight final blur to simulate microscope defocus
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
        return canvas, annotations


# ────────────────────────── Dataset Generation ──────────────────────────────

def generate_dataset(
    out_dir: str,
    n_train: int = 2000,
    n_val: int = 500,
    img_size: int = IMG_SIZE,
    seed: int = 42,
) -> None:
    """
    Generate the full synthetic dataset on disk in YOLO directory format:

        out_dir/
          train/images/   *.jpg
          train/labels/   *.txt  (YOLO format: class cx cy w h per line)
          val/images/
          val/labels/
          dataset.json    (summary statistics)

    This layout is identical to what Kaggle's Microplastic CV dataset uses,
    making this a drop-in swap when real data becomes available.

    Parameters
    ----------
    out_dir  : Root directory for the dataset.
    n_train  : Number of training images (default 2000).
    n_val    : Number of validation images (default 500).
    img_size : Pixel size of generated images (square, default 416).
    seed     : Random seed for reproducibility.
    """
    renderer = SyntheticParticleRenderer(img_size=img_size, seed=seed)

    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = Path(out_dir) / split / "images"
        lbl_dir = Path(out_dir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        class_counts = {c: 0 for c in SHAPE_CLASSES}
        print(f"Generating {n} {split} images …")

        for i in range(n):
            img_bgr, anns = renderer.generate_image()
            img_path = img_dir / f"mp_{split}_{i:05d}.jpg"
            lbl_path = lbl_dir / f"mp_{split}_{i:05d}.txt"

            cv2.imwrite(str(img_path), img_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            with open(lbl_path, "w") as f:
                for ann in anns:
                    cx, cy, nw, nh = ann["bbox_yolo"]
                    f.write(f"{ann['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    class_counts[ann["class_name"]] += 1

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n}")

        print(f"  Class distribution ({split}): {class_counts}")

    # Write dataset summary JSON
    meta = {
        "name": "MicroPlastiNet-Synthetic",
        "note": "SYNTHETIC DATA — replace with real Kaggle/MP-Set data for production",
        "real_datasets": {
            "Kaggle Microplastic CV": "https://www.kaggle.com/code/mathieuduverne/microplastic-detection-yolov8-map-50-76-2",
            "MP-Set Fluorescence": "https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset",
        },
        "n_train": n_train,
        "n_val": n_val,
        "img_size": img_size,
        "classes": SHAPE_CLASSES,
        "annotation_format": "YOLO (class cx cy w h, normalized)",
    }
    with open(Path(out_dir) / "dataset.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to {out_dir}")


# ─────────────────────── PyTorch Dataset Classes ────────────────────────────

class MicroplasticDetectionDataset(Dataset):
    """
    PyTorch Dataset for microplastic particle detection (YOLO format).

    Compatible with both the synthetic generator above and real datasets:
      - Kaggle Microplastic CV Dataset (YOLO format)
      - MP-Set Fluorescence Dataset (requires conversion from COCO JSON)

    Parameters
    ----------
    root_dir  : Path to split directory (e.g. data/train/).
                Expected layout: images/ and labels/ subdirectories.
    img_size  : Resize target (square). Default 416.
    transform : Optional torchvision transforms. Default: ToTensor + normalize.
    augment   : Whether to apply training augmentations.
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = IMG_SIZE,
        transform=None,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.augment = augment

        self.img_dir = self.root_dir / "images"
        self.lbl_dir = self.root_dir / "labels"

        self.img_files = sorted(self.img_dir.glob("*.jpg")) + \
                         sorted(self.img_dir.glob("*.png"))

        # Default transform: normalize to ImageNet stats (standard for
        # EfficientNet/YOLO fine-tuning)
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Augmentations for training robustness
        self.aug_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=30),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.img_files[idx]
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.augment:
            img_tensor = self.aug_transform(img)
        else:
            img_tensor = self.transform(img)

        # Load YOLO labels: each row → [class_id, cx, cy, w, h]
        boxes = []
        labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        bbox = [float(p) for p in parts[1:]]
                        labels.append(cls_id)
                        boxes.append(bbox)

        return {
            "image": img_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_path": str(img_path),
        }


class MicroplasticClassificationDataset(Dataset):
    """
    PyTorch Dataset for per-particle shape classification with EfficientNet-B0.

    Crops individual particles from detection output (or ground-truth boxes)
    for training the shape classifier.

    Parameters
    ----------
    root_dir  : Path to split directory with images/ and labels/ subdirs.
    img_size  : Crop resize target (square). Default 128.
    augment   : Apply training augmentations.
    """

    CROP_SIZE = 128  # EfficientNet-B0 accepts 224; we resize to 224 inside

    def __init__(
        self,
        root_dir: str,
        img_size: int = 224,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.augment = augment

        self.img_dir = self.root_dir / "images"
        self.lbl_dir = self.root_dir / "labels"

        # Build flat list of (image_path, class_id, bbox_yolo)
        self.samples: List[Tuple] = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        bbox = [float(p) for p in parts[1:]]
                        self.samples.append((img_path, cls_id, bbox))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.aug_transform = T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            T.RandomRotation(45),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, cls_id, (cx, cy, nw, nh) = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Convert YOLO bbox to pixel crop with padding
        pad = 0.15  # 15% context around particle
        x1 = max(0, int((cx - nw / 2 - pad * nw) * W))
        y1 = max(0, int((cy - nh / 2 - pad * nh) * H))
        x2 = min(W, int((cx + nw / 2 + pad * nw) * W))
        y2 = min(H, int((cy + nh / 2 + pad * nh) * H))
        crop = img.crop((x1, y1, x2, y2))

        tf = self.aug_transform if self.augment else self.transform
        return tf(crop), cls_id


# ──────────────────────────── DataLoader Factories ──────────────────────────

def get_detection_loaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 2,
    img_size: int = IMG_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for the detection task.

    Parameters
    ----------
    data_dir   : Root of the dataset (contains train/ and val/ subdirs).
    batch_size : Batch size for training.
    num_workers: DataLoader workers.
    img_size   : Image resize dimension.
    """
    train_ds = MicroplasticDetectionDataset(
        os.path.join(data_dir, "train"), img_size=img_size, augment=True)
    val_ds = MicroplasticDetectionDataset(
        os.path.join(data_dir, "val"), img_size=img_size, augment=False)

    def collate_fn(batch):
        """Custom collate: variable number of boxes per image."""
        images = torch.stack([b["image"] for b in batch])
        boxes = [b["boxes"] for b in batch]
        labels = [b["labels"] for b in batch]
        paths = [b["image_path"] for b in batch]
        return {"image": images, "boxes": boxes, "labels": labels, "paths": paths}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=False)

    return train_loader, val_loader


def get_classification_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for the shape classification task.

    Parameters
    ----------
    data_dir   : Root of the dataset.
    batch_size : Batch size.
    num_workers: DataLoader workers.
    img_size   : Crop resize size (224 for EfficientNet-B0).
    """
    train_ds = MicroplasticClassificationDataset(
        os.path.join(data_dir, "train"), img_size=img_size, augment=True)
    val_ds = MicroplasticClassificationDataset(
        os.path.join(data_dir, "val"), img_size=img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader


# ─────────────────────────────── CLI ────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic microplastic dataset")
    parser.add_argument("--out_dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        out_dir=args.out_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        img_size=args.img_size,
        seed=args.seed,
    )
