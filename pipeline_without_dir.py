"""
pipeline_without_dir.py

Same pipeline as run_full_pipeline.py -- identical logic, identical
preprocessing, identical model architectures -- but with NO
dependency on the Plate_Recognition_Project folder structure. Every
model file is expected to sit in the SAME folder as this script
itself, using simple renamed filenames instead of the nested
models/<stage>/<run_name>/weights/best.pt layout. Built specifically
to avoid directory-structure conflicts when moving this pipeline
somewhere else (a separate deployment folder, a demo, etc.).

Full end-to-end inference: raw image -> {VC, PC, PN}.

    raw image
        |
        v
    YOLO#1 -- VC detector          -> vehicle category + box
        |
        v  (crop vehicle, +10% padding)
    YOLO#2 -- Plate OBB detector   -> plate quad (4 corners)
        |
        v  (perspective-correct, order_points -> warpPerspective)
    canonical plate image (400x149)
        |
        +-----------------------------+
        v                             v
    YOLO#3 -- PN locator          (whole corrected plate)
        |                             |
        v  (crop + enhance_crop)      |
    PN OCR (CRNN)                     |  (mask PN region white,
        |                             |   using the SAME box from
        v                             |   YOLO#3 above)
    "2C-5289"                         v
                                  PC classifier (CNN)
                                       |
                                       v
                                  "PC_PhnomPenh" or "Unknown"

Every preprocessing step matches what its model actually saw during
training -- same reasoning as run_full_pipeline.py, see
README_pipeline_integration.md for the full explanation of each
handoff. Changing any of these without retraining will silently
degrade results.

Reads (all expected in the SAME folder as this script):
    vc.pt
    plate.pt
    pn_bbx.pt
    pn_crnn_best.pt + pn_ocr.json
    pc_classifier_best.pt + pc_class_list.json

Usage:
    Option A (edit this file): set IMAGE_PATH below, then just run:
        python pipeline_without_dir.py
    Option B (command line): pass the path as an argument:
        python pipeline_without_dir.py path/to/image.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# EDIT THIS to test a specific image directly, without needing to type
# a command-line argument every time. Only used if you run the script
# with no argument -- passing a path on the command line still
# overrides this.
IMAGE_PATH = r"C:\Users\DIN REAKSA\Desktop\smartcity\Plate_Recognition_Project\data\YOLO1_Data\images\VanMiniBus\p4__2D-7008_Siem_Reap.jpg"

MODELS_DIR = Path(__file__).resolve().parent  # flat -- every model file sits
                                                # right next to this script,
                                                # no project folder structure
                                                # needed at all

VC_WEIGHTS = MODELS_DIR / "vc2.pt"
PLATE_WEIGHTS = MODELS_DIR / "plate.pt"
PN_LOCATOR_WEIGHTS = MODELS_DIR / "pn_bbx.pt"
PN_OCR_WEIGHTS = MODELS_DIR / "pn_crnn_best.pt"
PN_OCR_CHARSET = MODELS_DIR / "pn_ocr.json"
PC_WEIGHTS = MODELS_DIR / "pc_classifier_best.pt"
PC_CLASS_LIST = MODELS_DIR / "pc_class_list.json"

# --- Handoff constants -- see README_pipeline_integration.md for why ---
VC_CROP_PADDING_FRAC = 0.10      # step 1: fixed 10% padding at inference
                                   # (train used random 0-25%, val/test 5%;
                                   # 10% is a reasonable middle ground within
                                   # that trained-for tolerance range)
PLATE_CANONICAL_SIZE = (400, 149) # step 2: EXACT size PN locator, PN OCR,
                                   # and PC classifier were all trained
                                   # against -- do not change without
                                   # retraining everything downstream
PN_CROP_PADDING_PX = 4            # step 3: matches evaluate_pn_ocr_crnn.py
PN_INPUT_SIZE = (160, 32)         # (width, height) -- CRNN input
PC_INPUT_SIZE = (400, 149)        # (width, height) -- same as canonical plate
PC_CONFIDENCE_THRESHOLD = 0.5     # step 5: below this, report "Unknown"
                                   # rather than a forced, possibly-wrong guess

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# PN OCR: CRNN architecture, copied verbatim from
# 18.train_pn_ocr_crnn.py / 19.evaluate_pn_ocr_crnn.py -- must match
# exactly or the saved weights won't load/behave correctly.
# =========================================================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, (2, 1)), nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)
        rnn_out, _ = self.lstm(features)
        rnn_out = self.dropout(rnn_out)
        logits = self.fc(rnn_out)
        return logits.permute(1, 0, 2)


def greedy_decode(logits, idx_to_char):
    preds = logits.argmax(dim=2).permute(1, 0)
    texts = []
    for seq in preds:
        chars = []
        prev = -1
        for idx in seq.tolist():
            if idx != 0 and idx != prev:
                chars.append(idx_to_char.get(idx, ""))
            prev = idx
        texts.append("".join(chars))
    return texts


def enhance_crop(crop):
    """Verbatim from 19.evaluate_pn_ocr_crnn.py -- the CRNN has never
    seen a PN crop that wasn't processed this exact way."""
    upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
    return sharpened


def preprocess_pn_crop(crop):
    """Matches 19.evaluate_pn_ocr_crnn.py's preprocess_for_model."""
    image = cv2.resize(crop, PN_INPUT_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


# =========================================================
# PC classifier: architecture matches train_pc_classifier.py exactly.
# =========================================================

class PCClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.flatten(1)
        features = self.dropout(features)
        return self.fc(features)


def preprocess_pc_plate(plate_image):
    """Matches train_pc_classifier.py's PCDataset normalization exactly."""
    image = cv2.resize(plate_image, PC_INPUT_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


# =========================================================
# Geometry: order_points, copied verbatim from
# prepare_pc_classification_data.py -- TL/TR/BR/BL ordering
# regardless of the original annotation/detection order.
# =========================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_correct(image, quad, size=PLATE_CANONICAL_SIZE):
    w, h = size
    dst_rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    ordered = order_points(np.array(quad, dtype=np.float32))
    M = cv2.getPerspectiveTransform(ordered, dst_rect)
    corrected = cv2.warpPerspective(image, M, (w, h))
    return corrected, M


# =========================================================
# Pipeline
# =========================================================

def _draw_label(image, box_pts, text, color):
    """Draws a filled label box with text anchored above the given
    point set (a quad or an axis-aligned box's 4 corners)."""
    pts = np.array(box_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=3)

    text_x = int(box_pts[:, 0].min())
    text_y = max(24, int(box_pts[:, 1].min()) - 8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(image, (text_x, text_y - th - 8), (text_x + tw + 8, text_y + 4),
                  color, thickness=-1)
    cv2.putText(image, text, (text_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_annotations(raw_image, vc_box, vc_label, quad, pc_label, pn_box_reprojected,
                      pn_text, crop_offset):
    """
    Draws THREE separate labeled boxes directly on a copy of the
    ORIGINAL raw image -- not the vehicle crop. Drawing on the crop
    made the "VC box" meaningless (it always touched all four edges,
    since that panel already IS the crop). Here:
      - VC box: the actual tight detected box, already in raw-image
        coordinates -- no offset needed.
      - PC box (plate quad) and PN box: both computed relative to the
        vehicle crop, so both need crop_offset (cx1, cy1) added to
        shift them into raw-image coordinates.
    """
    annotated = raw_image.copy()
    cx1, cy1 = crop_offset

    vc_x1, vc_y1, vc_x2, vc_y2 = vc_box
    vc_pts = np.array([[vc_x1, vc_y1], [vc_x2, vc_y1], [vc_x2, vc_y2], [vc_x1, vc_y2]],
                       dtype=np.float32)
    _draw_label(annotated, vc_pts, vc_label, (0, 255, 0))

    quad_raw = quad + np.array([cx1, cy1], dtype=np.float32)
    pc_label_display = pc_label[3:] if pc_label.startswith("PC_") else pc_label
    _draw_label(annotated, quad_raw, pc_label_display, (255, 200, 0))

    pn_raw = pn_box_reprojected + np.array([cx1, cy1], dtype=np.float32)
    _draw_label(annotated, pn_raw, pn_text, (0, 100, 255))

    return annotated


class PlateRecognitionPipeline:
    def __init__(self):
        print("Loading models...")
        self.vc_model = YOLO(str(VC_WEIGHTS))
        self.plate_model = YOLO(str(PLATE_WEIGHTS))
        self.pn_locator_model = YOLO(str(PN_LOCATOR_WEIGHTS))

        with open(PN_OCR_CHARSET, "r", encoding="utf-8") as f:
            self.pn_char_to_idx = json.load(f)
        self.pn_idx_to_char = {idx: ch for ch, idx in self.pn_char_to_idx.items()}
        pn_num_classes = len(self.pn_char_to_idx) + 1  # +1 for CTC blank
        self.pn_model = CRNN(pn_num_classes).to(DEVICE)
        self.pn_model.load_state_dict(torch.load(PN_OCR_WEIGHTS, map_location=DEVICE))
        self.pn_model.eval()

        with open(PC_CLASS_LIST, "r", encoding="utf-8") as f:
            self.pc_class_list = json.load(f)
        self.pc_model = PCClassifier(len(self.pc_class_list)).to(DEVICE)
        self.pc_model.load_state_dict(torch.load(PC_WEIGHTS, map_location=DEVICE))
        self.pc_model.eval()
        print("All models loaded.")

    def run(self, image_or_path, return_debug=False):
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_or_path}")
        else:
            image = image_or_path  # already a decoded BGR numpy array
        img_h, img_w = image.shape[:2]
        debug = {"raw_image": image}

        # ---- Step 1: VC detector ----
        vc_result = self.vc_model.predict(image, verbose=False)[0]
        if len(vc_result.boxes) == 0:
            return {"error": "No vehicle detected", **(debug if return_debug else {})}
        best_idx = vc_result.boxes.conf.argmax().item()
        vc_box = vc_result.boxes.xyxy[best_idx].cpu().numpy()
        vc_cls_idx = int(vc_result.boxes.cls[best_idx].item())
        vc_label = vc_result.names[vc_cls_idx]

        x1, y1, x2, y2 = vc_box
        box_w, box_h = x2 - x1, y2 - y1
        pad_x, pad_y = box_w * VC_CROP_PADDING_FRAC, box_h * VC_CROP_PADDING_FRAC
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))
        vehicle_crop = image[cy1:cy2, cx1:cx2]
        debug["vehicle_crop"] = vehicle_crop

        # ---- Step 2: Plate OBB detector -> perspective correction ----
        plate_result = self.plate_model.predict(vehicle_crop, verbose=False)[0]
        if plate_result.obb is None or len(plate_result.obb) == 0:
            return {"VC": vc_label, "error": "No plate detected",
                    **(debug if return_debug else {})}
        best_plate_idx = plate_result.obb.conf.argmax().item()
        quad = plate_result.obb.xyxyxyxy[best_plate_idx].cpu().numpy().reshape(4, 2)
        corrected_plate, perspective_M = perspective_correct(vehicle_crop, quad)
        debug["plate_crop"] = corrected_plate
        debug["plate_quad"] = quad

        # ---- Step 3a: PN locator on the corrected plate ----
        pn_loc_result = self.pn_locator_model.predict(corrected_plate, verbose=False)[0]
        if len(pn_loc_result.boxes) == 0:
            return {"VC": vc_label, "error": "No PN region detected",
                    **(debug if return_debug else {})}
        best_pn_idx = pn_loc_result.boxes.conf.argmax().item()
        px1, py1, px2, py2 = pn_loc_result.boxes.xyxy[best_pn_idx].cpu().numpy()

        # ---- Step 3b: PN crop -> enhance -> CRNN -> text ----
        ppx1 = max(0, int(px1) - PN_CROP_PADDING_PX)
        ppy1 = max(0, int(py1) - PN_CROP_PADDING_PX)
        ppx2 = min(corrected_plate.shape[1], int(px2) + PN_CROP_PADDING_PX)
        ppy2 = min(corrected_plate.shape[0], int(py2) + PN_CROP_PADDING_PX)
        pn_crop = corrected_plate[ppy1:ppy2, ppx1:ppx2]
        pn_crop = enhance_crop(pn_crop)
        pn_tensor = preprocess_pn_crop(pn_crop).to(DEVICE)
        with torch.no_grad():
            pn_logits = self.pn_model(pn_tensor)
            pn_text = greedy_decode(pn_logits, self.pn_idx_to_char)[0]

        # ---- Step 4: mask the SAME PN box white, then PC classifier ----
        masked_plate = corrected_plate.copy()
        cv2.rectangle(masked_plate, (int(px1), int(py1)), (int(px2), int(py2)),
                      (255, 255, 255), thickness=-1)
        pc_tensor = preprocess_pc_plate(masked_plate).to(DEVICE)
        with torch.no_grad():
            pc_logits = self.pc_model(pc_tensor)
            pc_probs = torch.softmax(pc_logits, dim=1)
            top_prob, pred_idx = pc_probs.max(1)
            top_prob = top_prob.item()
            pc_label = self.pc_class_list[pred_idx.item()]

        # ---- Step 5: confidence-threshold Unknown fallback ----
        if top_prob < PC_CONFIDENCE_THRESHOLD:
            pc_label = "Unknown"

        result = {
            "VC": vc_label,
            "PC": pc_label,
            "PC_confidence": round(top_prob, 3),
            "PN": pn_text,
        }
        if return_debug:
            px1, py1, px2, py2 = pn_loc_result.boxes.xyxy[best_pn_idx].cpu().numpy()
            pn_corners = np.array([[px1, py1], [px2, py1], [px2, py2], [px1, py2]],
                                   dtype=np.float32)
            M_inv = np.linalg.inv(perspective_M)
            pn_box_reprojected = cv2.perspectiveTransform(
                pn_corners.reshape(-1, 1, 2), M_inv
            ).reshape(-1, 2)
            debug["annotated_image"] = draw_annotations(
                image, (x1, y1, x2, y2), vc_label, quad, pc_label,
                pn_box_reprojected, pn_text, crop_offset=(cx1, cy1)
            )
            result.update(debug)
        return result


def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = IMAGE_PATH
        print(f"No command-line argument given -- using IMAGE_PATH from the "
              f"top of the script:\n  {image_path}\n")

    pipeline = PlateRecognitionPipeline()
    result = pipeline.run(image_path)

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
