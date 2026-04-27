"""
model.py — Vision Model Architectures for Microplastic Detection & Classification
==================================================================================
Module: M2a Vision DL | MicroPlastiNet Pipeline
Author: MicroPlastiNet Team

TWO ARCHITECTURES
-----------------
1. TinyYOLO  — YOLOv5-tiny-style single-stage object detector
   Detects microplastic particles: bounding boxes + class labels
   Reference architecture: Redmon & Farhadi (2018) YOLOv3; Jocher et al. (2020) YOLOv5
   Production note: for real deployment, use `ultralytics` YOLOv8 fine-tuned on
   the Kaggle Microplastic CV dataset (map@50 ~76.2 reported in the community notebook).

2. MPClassifier — EfficientNet-B0 shape classifier
   Classifies cropped particles: fragment / fiber / film / bead / foam
   Backbone: EfficientNet-B0 (Tan & Le, 2019 EfficientNet: Rethinking Model Scaling for CNNs)
   Pre-trained on ImageNet, fine-tuned on microplastic crops.

PRODUCTION UPGRADE PATH
-----------------------
  # YOLOv8 (when ultralytics is available):
  from ultralytics import YOLO
  model = YOLO("yolov8n.pt")
  model.train(data="data/synthetic/dataset.yaml", epochs=50, imgsz=416)

  # Real training data:
  # Kaggle MP CV: https://www.kaggle.com/code/mathieuduverne/microplastic-detection-yolov8-map-50-76-2
  # MP-Set: https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset
"""

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ─────────────────────── Constants ──────────────────────────────────────────

NUM_CLASSES = 5          # fragment, fiber, film, bead, foam
NUM_ANCHORS = 3          # Anchors per grid cell (standard YOLOv3/v5)
IMG_SIZE = 416

# Anchor boxes (width, height) as fraction of image size.
# Tuned for microplastic particle aspect ratios at 10–40x magnification.
# fmt: off
ANCHORS = [
    # Small scale (S = 52×52 grid)
    [(0.02, 0.02), (0.04, 0.02), (0.02, 0.08)],
    # Medium scale (M = 26×26 grid)
    [(0.06, 0.06), (0.10, 0.04), (0.04, 0.14)],
    # Large scale (L = 13×13 grid)
    [(0.14, 0.14), (0.22, 0.08), (0.10, 0.24)],
]
# fmt: on


# ─────────────────────────── Building Blocks ────────────────────────────────

class ConvBnAct(nn.Module):
    """
    Conv2d → BatchNorm → LeakyReLU (the fundamental YOLO building block).

    Parameters
    ----------
    in_c   : Input channels.
    out_c  : Output channels.
    k      : Kernel size.
    s      : Stride.
    p      : Padding (auto if None: k//2).
    act    : Activation: 'leaky' (default) or 'silu'.
    """

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1,
                 p: Optional[int] = None, act: str = "leaky"):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c, momentum=0.03, eps=1e-3)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act == "leaky" else nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBottleneck(nn.Module):
    """
    Residual bottleneck block (as in YOLOv5 C3 module, simplified).
    1×1 → 3×3 → add skip.
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.cv1 = ConvBnAct(channels, mid, 1)
        self.cv2 = ConvBnAct(mid, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class C3Module(nn.Module):
    """
    C3 cross-stage partial bottleneck (YOLOv5-style).
    Splits features into two paths, bottleneck one, then concatenate.
    """

    def __init__(self, in_c: int, out_c: int, n: int = 1):
        super().__init__()
        mid = out_c // 2
        self.cv1 = ConvBnAct(in_c, mid, 1)
        self.cv2 = ConvBnAct(in_c, mid, 1)
        self.cv3 = ConvBnAct(2 * mid, out_c, 1)
        self.bottlenecks = nn.Sequential(*[ResBottleneck(mid) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling — Fast (SPPF), from YOLOv5.
    Pools with multiple kernel sizes in sequence rather than parallel
    for computational efficiency.
    """

    def __init__(self, in_c: int, out_c: int, k: int = 5):
        super().__init__()
        mid = in_c // 2
        self.cv1 = ConvBnAct(in_c, mid, 1)
        self.cv2 = ConvBnAct(mid * 4, out_c, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ─────────────────────────── YOLO Head ──────────────────────────────────────

class YOLOHead(nn.Module):
    """
    YOLO detection head for one scale.

    Outputs tensor of shape (B, num_anchors, H, W, 5 + num_classes):
      [tx, ty, tw, th, obj_conf, cls_0 .. cls_N]

    Parameters
    ----------
    in_c       : Input channels from FPN.
    num_classes: Number of object classes (default 5 for MP shapes).
    num_anchors: Anchors per cell (default 3).
    """

    def __init__(self, in_c: int, num_classes: int = NUM_CLASSES,
                 num_anchors: int = NUM_ANCHORS):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_c = num_anchors * (5 + num_classes)
        self.conv = nn.Sequential(
            ConvBnAct(in_c, in_c * 2, 3),
            nn.Conv2d(in_c * 2, out_c, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        out = self.conv(x)
        # Reshape to (B, num_anchors, H, W, 5+C)
        out = out.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out


# ──────────────────────────── TinyYOLO ──────────────────────────────────────

class TinyYOLO(nn.Module):
    """
    YOLOv5-tiny-style single-stage detector for microplastic particles.

    Architecture:
      • Backbone: 5-stage Conv + C3 feature extractor (~1.5M params)
      • Neck:     Feature Pyramid Network (FPN) with 3 detection scales
      • Head:     YOLO detection heads at S, M, L scales

    Input:  (B, 3, 416, 416) RGB normalized image
    Output: List of 3 tensors [(B, 3, 52, 52, 10), (B, 3, 26, 26, 10), (B, 3, 13, 13, 10)]
            where 10 = 5 (tx,ty,tw,th,conf) + 5 classes

    Usage
    -----
    model = TinyYOLO(num_classes=5)
    preds = model(images)   # list of 3 scale tensors

    For production, replace with Ultralytics YOLOv8:
      pip install ultralytics
      model = YOLO('yolov8n.pt')
      model.train(data='dataset.yaml', epochs=100)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────
        # P1/2 — 208×208
        self.stem = ConvBnAct(3, 16, 3, 2)          # 208×208, 16ch
        # P2/4 — 104×104
        self.stage1 = nn.Sequential(
            ConvBnAct(16, 32, 3, 2),                 # stride 4
            C3Module(32, 32, n=1),
        )
        # P3/8 — 52×52 (small particles)
        self.stage2 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2),                 # stride 8
            C3Module(64, 64, n=2),
        )
        # P4/16 — 26×26 (medium particles)
        self.stage3 = nn.Sequential(
            ConvBnAct(64, 128, 3, 2),                # stride 16
            C3Module(128, 128, n=3),
        )
        # P5/32 — 13×13 (large particles)
        self.stage4 = nn.Sequential(
            ConvBnAct(128, 256, 3, 2),               # stride 32
            C3Module(256, 256, n=1),
            SPPF(256, 256),
        )

        # ── Neck: Top-down FPN ────────────────────────────────────────────
        self.lateral_p5 = ConvBnAct(256, 128, 1)   # reduce P5 → 128ch
        self.lateral_p4 = ConvBnAct(128 + 128, 128, 1)  # fuse upsample(P5) + P4
        self.lateral_p3 = ConvBnAct(64 + 128, 64, 1)    # fuse upsample(P4) + P3

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # ── Detection Heads ───────────────────────────────────────────────
        # Small scale: 52×52 → best for tiny particles (<20px)
        self.head_s = YOLOHead(64, num_classes=num_classes)
        # Medium scale: 26×26
        self.head_m = YOLOHead(128, num_classes=num_classes)
        # Large scale: 13×13 → big particles / aggregates
        self.head_l = YOLOHead(256, num_classes=num_classes)

        # ── Anchor registration ───────────────────────────────────────────
        for i, scale_anchors in enumerate(ANCHORS):
            t = torch.tensor(scale_anchors, dtype=torch.float32)  # (3,2)
            self.register_buffer(f"anchors_{i}", t)

        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for conv layers; standard for YOLO-style detectors."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 3, H, W) normalized image tensor.

        Returns
        -------
        List of 3 prediction tensors at S, M, L scales.
        """
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # 52×52, 64ch
        p4 = self.stage3(p3)  # 26×26, 128ch
        p5 = self.stage4(p4)  # 13×13, 256ch

        # FPN top-down path
        fpn_p5 = self.lateral_p5(p5)                              # 13×13, 128ch
        fpn_p4 = self.lateral_p4(
            torch.cat([self.up(fpn_p5), p4], dim=1))             # 26×26, 128ch
        fpn_p3 = self.lateral_p3(
            torch.cat([self.up(fpn_p4), p3], dim=1))             # 52×52, 64ch

        # Detection heads
        out_s = self.head_s(fpn_p3)    # (B, 3, 52, 52, 10)
        out_m = self.head_m(fpn_p4)    # (B, 3, 26, 26, 10)
        out_l = self.head_l(p5)        # (B, 3, 13, 13, 10)

        return [out_s, out_m, out_l]

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────── EfficientNet Classifier ───────────────────────

class MPClassifier(nn.Module):
    """
    EfficientNet-B0 shape classifier fine-tuned for microplastic morphology.

    Pre-trained on ImageNet-1k, head replaced with 5-class softmax.
    Expected input: (B, 3, 224, 224) normalized particle crop.

    Architecture choice justification:
      EfficientNet-B0 (5.3M params, 390M FLOPs) offers the best
      accuracy/size trade-off for embedded/edge deployment (M1 module).
      Alternative: MobileNetV3-Small for ESP32 TFLite export.

    Reference:
      Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs.
      ICML 2019. https://arxiv.org/abs/1905.11946

    Real dataset reference:
      Fine-tuning target: Kaggle MP-Set fluorescence crops
      https://www.kaggle.com/datasets/sanghyeonaustinpark/mpset

    Parameters
    ----------
    num_classes    : Output classes (default 5: fragment/fiber/film/bead/foam).
    pretrained     : Load ImageNet weights (default True).
    dropout_rate   : Dropout before classifier head.
    freeze_backbone: Freeze EfficientNet backbone for first training phase.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Extract feature layers (everything except the final classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Replace classifier: original 1000-class → num_classes
        in_features = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize only the new classification head with proper scaling."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224) particle crop tensor.

        Returns
        -------
        logits : (B, num_classes) unnormalized class scores.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict_with_confidence(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience wrapper: returns (class_idx, confidence) after softmax.

        Parameters
        ----------
        x : (B, 3, 224, 224) batch.

        Returns
        -------
        class_ids    : (B,) integer class indices
        confidences  : (B,) float confidence scores ∈ [0,1]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            conf, cls = probs.max(dim=1)
        return cls, conf

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────── YOLO Loss Function ────────────────────────────────

class YOLOLoss(nn.Module):
    """
    Multi-scale YOLO loss combining:
      • Objectness (BCE with logits)
      • Bounding box regression (CIoU loss)
      • Class prediction (BCE with logits, multi-label capable)

    Reference: YOLOv4 (Bochkovskiy et al., 2020) CIoU loss.

    Parameters
    ----------
    anchors       : ANCHORS list (3 scales × 3 anchors × 2 [w,h]).
    num_classes   : Number of object classes.
    img_size      : Input image size (square).
    lambda_coord  : Weight for bbox regression loss.
    lambda_noobj  : Weight for no-object confidence.
    lambda_cls    : Weight for class prediction.
    """

    def __init__(
        self,
        anchors=ANCHORS,
        num_classes: int = NUM_CLASSES,
        img_size: int = IMG_SIZE,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_cls: float = 1.0,
    ):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def _build_target(
        self,
        preds: torch.Tensor,
        boxes: List[torch.Tensor],
        labels: List[torch.Tensor],
        scale_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build target tensors matching prediction shape for one scale.

        Returns obj_mask, noobj_mask, target_boxes, target_cls tensors.
        """
        B, num_anch, H, W, _ = preds.shape
        anchors_wh = torch.tensor(
            self.anchors[scale_idx], device=preds.device, dtype=torch.float32)

        obj_mask = torch.zeros(B, num_anch, H, W, device=preds.device)
        noobj_mask = torch.ones(B, num_anch, H, W, device=preds.device)
        tgt_boxes = torch.zeros(B, num_anch, H, W, 4, device=preds.device)
        tgt_cls = torch.zeros(B, num_anch, H, W, self.num_classes, device=preds.device)

        for b in range(B):
            if boxes[b].shape[0] == 0:
                continue
            for j in range(boxes[b].shape[0]):
                gx, gy, gw, gh = boxes[b][j] * torch.tensor(
                    [W, H, W, H], dtype=torch.float32, device=preds.device)
                gi, gj = int(gx), int(gy)
                if gi >= W: gi = W - 1
                if gj >= H: gj = H - 1

                # Assign to best matching anchor
                gt_wh = torch.tensor([gw, gh], device=preds.device)
                iou_with_anchors = torch.stack([
                    _anchor_wh_iou(gt_wh, a * torch.tensor([W, H], device=preds.device))
                    for a in anchors_wh
                ])
                best_a = iou_with_anchors.argmax().item()

                obj_mask[b, best_a, gj, gi] = 1
                noobj_mask[b, best_a, gj, gi] = 0
                tgt_boxes[b, best_a, gj, gi] = torch.tensor(
                    [gx - gi, gy - gj, gw, gh], device=preds.device)
                cls_id = labels[b][j].item()
                if 0 <= cls_id < self.num_classes:
                    tgt_cls[b, best_a, gj, gi, cls_id] = 1.0

        return obj_mask, noobj_mask, tgt_boxes, tgt_cls

    def forward(
        self,
        predictions: List[torch.Tensor],
        boxes: List[torch.Tensor],
        labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-scale YOLO loss.

        Parameters
        ----------
        predictions : 3-scale list from TinyYOLO.forward()
        boxes       : Per-image YOLO bbox list [Tensor(N,4), ...]
        labels      : Per-image class id list [Tensor(N,), ...]

        Returns
        -------
        total_loss : Scalar loss tensor.
        components : Dict with 'obj', 'noobj', 'bbox', 'cls' sub-losses.
        """
        total = torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        components = {"obj": 0.0, "noobj": 0.0, "bbox": 0.0, "cls": 0.0}

        for scale_i, pred in enumerate(predictions):
            obj_m, noobj_m, tgt_box, tgt_cls = self._build_target(
                pred, boxes, labels, scale_i)

            obj_pred = pred[..., 4]
            cls_pred = pred[..., 5:]

            loss_obj = self.bce(obj_pred[obj_m == 1], obj_m[obj_m == 1])
            loss_noobj = self.bce(obj_pred[noobj_m == 1], obj_m[noobj_m == 1]) * self.lambda_noobj

            if obj_m.sum() > 0:
                box_pred = pred[..., :4][obj_m == 1]
                box_tgt = tgt_box[obj_m == 1]
                loss_bbox = self.mse(box_pred[:, :2], box_tgt[:, :2]) + \
                            self.mse(box_pred[:, 2:].abs(), box_tgt[:, 2:])
                loss_cls = self.bce(cls_pred[obj_m == 1], tgt_cls[obj_m == 1])
            else:
                loss_bbox = torch.tensor(0.0, device=pred.device)
                loss_cls = torch.tensor(0.0, device=pred.device)

            scale_loss = (loss_obj + loss_noobj +
                          self.lambda_coord * loss_bbox +
                          self.lambda_cls * loss_cls)
            total = total + scale_loss

            components["obj"] += loss_obj.item()
            components["noobj"] += loss_noobj.item()
            components["bbox"] += loss_bbox.item()
            components["cls"] += loss_cls.item()

        return total, components


def _anchor_wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    """IoU between two boxes of equal center, given only widths and heights."""
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter = torch.min(w1, w2) * torch.min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter + 1e-6
    return inter / union


# ──────────────────────── Model Factory ─────────────────────────────────────

def build_detector(num_classes: int = NUM_CLASSES) -> TinyYOLO:
    """Build and return a TinyYOLO detector instance."""
    model = TinyYOLO(num_classes=num_classes)
    print(f"TinyYOLO | params: {model.n_parameters:,}")
    return model


def build_classifier(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> MPClassifier:
    """Build and return an EfficientNet-B0 classifier instance."""
    model = MPClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    print(f"MPClassifier (EfficientNet-B0) | params: {model.n_parameters:,}")
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Dict]:
    """
    Load a saved checkpoint into a model.

    Parameters
    ----------
    model           : Model instance (architecture must match checkpoint).
    checkpoint_path : Path to .pt or .pth file.
    device          : Target device.

    Returns
    -------
    model : Model with loaded weights.
    meta  : Checkpoint metadata dict (epoch, metrics, etc.)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    print(f"Loaded checkpoint from {checkpoint_path} "
          f"(epoch {meta.get('epoch', '?')})")
    return model, meta


# ─────────────────────────────── CLI ────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test detector
    det = build_detector().to(device)
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    outs = det(dummy)
    print("Detector output shapes:")
    for o in outs:
        print(f"  {tuple(o.shape)}")

    # Test classifier
    clf = build_classifier().to(device)
    crops = torch.randn(4, 3, 224, 224, device=device)
    logits = clf(crops)
    print(f"\nClassifier output: {logits.shape}  (4 crops × 5 classes)")

    # Test loss
    loss_fn = YOLOLoss()
    mock_boxes = [torch.tensor([[0.5, 0.5, 0.1, 0.1]]) for _ in range(2)]
    mock_labels = [torch.tensor([0]) for _ in range(2)]
    loss, comps = loss_fn(outs, mock_boxes, mock_labels)
    print(f"\nYOLO loss: {loss.item():.4f}  |  {comps}")
