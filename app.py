# to start:
# sudo systemctl restart empty-shelves
# sudo systemctl status empty-shelves --no-pager

import os

# Keep CPU worker behaviour sane when using multiple gunicorn workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO

from rack_analytics import (
    analyze_rack_fleet,
    render_rackvision_overlay,
    RackStatus,
    ReplenishmentUrgency,
)

# Optional AVIF support if pillow-avif-plugin is installed.
try:
    import pillow_avif  # noqa: F401
except ImportError:
    pillow_avif = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "empty-shelves-yolo26m.pt"
STATIC_DIR = BASE_DIR / "static"
RESULT_DIR = BASE_DIR / "output" / "results"
INPUT_DIR = BASE_DIR / "input"

DEFAULT_CONF = 0.20
DEFAULT_IMGSZ = 640

# YOLO NMS setting. Lower = more aggressive duplicate removal.
DEFAULT_IOU = 0.35
MAX_DETECTIONS = 150

# App-level sanity filters.
# These remove tiny noise boxes and very large "empty area" boxes.
MIN_BOX_AREA_RATIO = 0.0003
MAX_BOX_AREA_RATIO = 0.35

# App-level overlap suppression.
# If two boxes strongly overlap, keep only the better/tighter one.
OVERLAP_IOU_THRESHOLD = 0.30
CONTAINMENT_THRESHOLD = 0.70

# If a smaller contained box has at least this fraction of the larger box score,
# prefer the smaller/tighter box.
TIGHTER_BOX_SCORE_RATIO = 0.85

RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Empty Shelf Gap Detector",
    description="YOLO26m gap detection service",
    version="1.2.0",
)

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    """
    Reads uploaded image bytes, applies EXIF orientation correction,
    converts to OpenCV BGR image.
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def get_class_name(names, cls_id: int) -> str:
    if hasattr(names, "get"):
        return names.get(cls_id, str(cls_id))

    try:
        return names[cls_id]
    except Exception:
        return str(cls_id)


def box_area_xyxy(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_intersection_xyxy(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def box_iou_xyxy(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    inter = box_intersection_xyxy(a, b)
    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def box_overlap_over_smaller(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """
    Useful when one box is mostly inside another.

    IoU can be low when a small box sits inside a large box,
    but for visual output we still want to suppress duplicates.
    """
    inter = box_intersection_xyxy(a, b)
    area_a = box_area_xyxy(a)
    area_b = box_area_xyxy(b)
    smaller = min(area_a, area_b)

    if smaller <= 0:
        return 0.0

    return inter / smaller


def extract_candidate_boxes(result, image_shape) -> List[Dict]:
    """
    Extract boxes from YOLO result, clamp to image bounds, and apply
    tiny/huge area filters.
    """
    h, w = image_shape[:2]
    image_area = h * w

    candidates: List[Dict] = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        bbox = (x1, y1, x2, y2)
        area = box_area_xyxy(bbox)
        area_ratio = area / image_area if image_area else 0.0

        # Remove very tiny noisy detections.
        if area_ratio < MIN_BOX_AREA_RATIO:
            continue

        # Remove very large "empty region" detections.
        # Increase MAX_BOX_AREA_RATIO if you genuinely expect very large shelf gaps.
        if area_ratio > MAX_BOX_AREA_RATIO:
            continue

        candidates.append(
            {
                "cls_id": cls_id,
                "score": score,
                "box": bbox,
                "area": area,
                "area_ratio": area_ratio,
            }
        )

    return candidates


def suppress_duplicate_boxes(result, image_shape) -> List[Dict]:
    """
    Suppresses duplicate/overlapping/contained boxes.

    Logic:
    - Start with higher-confidence boxes.
    - If a new box overlaps an existing kept box, suppress it.
    - If a smaller tighter box is mostly inside a larger kept box and has
      similar confidence, replace the larger box with the smaller one.
    """
    candidates = extract_candidate_boxes(result, image_shape)

    # Higher confidence first.
    candidates.sort(key=lambda item: item["score"], reverse=True)

    kept: List[Dict] = []

    for candidate in candidates:
        candidate_box = candidate["box"]
        candidate_area = candidate["area"]
        candidate_score = candidate["score"]

        should_add = True

        for idx, accepted in enumerate(kept):
            accepted_box = accepted["box"]
            accepted_area = accepted["area"]
            accepted_score = accepted["score"]

            iou = box_iou_xyxy(candidate_box, accepted_box)
            containment = box_overlap_over_smaller(candidate_box, accepted_box)

            is_duplicate = (
                iou >= OVERLAP_IOU_THRESHOLD
                or containment >= CONTAINMENT_THRESHOLD
            )

            if not is_duplicate:
                continue

            # If the new candidate is a tighter box inside/overlapping a larger box,
            # and the confidence is close enough, keep the tighter one.
            candidate_is_tighter = candidate_area < accepted_area
            candidate_score_close = candidate_score >= accepted_score * TIGHTER_BOX_SCORE_RATIO

            if candidate_is_tighter and candidate_score_close:
                kept[idx] = candidate

            should_add = False
            break

        if should_add:
            kept.append(candidate)

    # Draw low confidence first so higher confidence boxes/labels stay visible.
    kept.sort(key=lambda item: item["score"])

    return kept


def draw_small_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> None:
    """
    Draws a compact but clearly readable label.
    This only draws a small label chip, not an inner detection box.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]

    pad_x = 5
    pad_y = 4

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tw, th = text_size

    # Default label position: just above the bbox.
    chip_x1 = max(0, x)
    chip_y2 = y - 2
    chip_y1 = chip_y2 - th - baseline - (pad_y * 2)

    # If label goes above image, place it just inside the bbox top.
    if chip_y1 < 0:
        chip_y1 = min(h - 1, y + 2)
        chip_y2 = chip_y1 + th + baseline + (pad_y * 2)

    chip_x2 = chip_x1 + tw + (pad_x * 2)

    # If label goes beyond right edge, shift left.
    if chip_x2 >= w:
        shift = chip_x2 - w + 2
        chip_x1 = max(0, chip_x1 - shift)
        chip_x2 = min(w - 1, chip_x2 - shift)

    chip_y1 = max(0, min(h - 1, chip_y1))
    chip_y2 = max(0, min(h - 1, chip_y2))

    # Solid dark label chip.
    cv2.rectangle(
        img,
        (chip_x1, chip_y1),
        (chip_x2, chip_y2),
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )

    # Thin cyan border around label chip.
    cv2.rectangle(
        img,
        (chip_x1, chip_y1),
        (chip_x2, chip_y2),
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    text_x = chip_x1 + pad_x
    text_y = chip_y2 - baseline - pad_y

    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(
    image_bgr: np.ndarray,
    result,
    show_labels: bool = True,
    show_conf: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """
    Draw a single high-visibility outline around each final accepted gap.
    Duplicate, overlapping, contained, tiny, and huge boxes are filtered.
    """
    output = image_bgr.copy()
    h, w = output.shape[:2]

    raw_count = len(result.boxes)
    boxes = suppress_duplicate_boxes(result, output.shape)

    # Single visible outline only.
    # BGR color: cyan.
    box_color = (255, 255, 0)

    # Make outline visible without becoming bulky.
    line_width = 2 if w < 1200 else 3

    # Small readable font.
    font_scale = max(0.40, min(0.58, w / 1900.0))
    font_thickness = 1

    for item in boxes:
        cls_id = item["cls_id"]
        score = item["score"]
        x1, y1, x2, y2 = item["box"]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            line_width,
            cv2.LINE_AA,
        )

        if show_labels:
            class_name = get_class_name(result.names, cls_id)
            label = f"{class_name} {score:.2f}" if show_conf else class_name

            draw_small_label(
                output,
                label,
                x1,
                y1,
                font_scale=font_scale,
                thickness=font_thickness,
            )

    return output, len(boxes), raw_count


def encode_jpeg(image_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode output image")

    return encoded.tobytes()


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return Response(
        content="ok\n",
        media_type="text/plain",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    show_labels: bool = Form(True),
    show_conf: bool = Form(True),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    if conf < 0.01 or conf > 0.95:
        raise HTTPException(status_code=400, detail="conf must be between 0.01 and 0.95")

    file_bytes = await image.read()

    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 15 MB allowed.")

    original_bgr = read_image_from_upload(file_bytes)

    start = time.perf_counter()

    results = model.predict(
        source=original_bgr,
        imgsz=DEFAULT_IMGSZ,
        conf=conf,
        iou=DEFAULT_IOU,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    annotated_bgr, detection_count, raw_count = draw_detections(
        original_bgr,
        result,
        show_labels=show_labels,
        show_conf=show_conf,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    output_bytes = encode_jpeg(annotated_bgr)

    result_name = f"{uuid.uuid4().hex}.jpg"
    result_path = RESULT_DIR / result_name
    result_path.write_bytes(output_bytes)

    headers = {
        "X-Detections": str(detection_count),
        "X-Raw-Detections": str(raw_count),
        "X-Suppressed-Detections": str(raw_count - detection_count),
        "X-Inference-Ms": str(elapsed_ms),
        "X-Result-File": result_name,
        "Cache-Control": "no-store",
    }

    return Response(
        content=output_bytes,
        media_type="image/jpeg",
        headers=headers,
    )


@app.post("/api/detect-meta")
async def detect_meta(
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    if conf < 0.01 or conf > 0.95:
        raise HTTPException(status_code=400, detail="conf must be between 0.01 and 0.95")

    file_bytes = await image.read()

    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 15 MB allowed.")

    original_bgr = read_image_from_upload(file_bytes)

    start = time.perf_counter()

    results = model.predict(
        source=original_bgr,
        imgsz=DEFAULT_IMGSZ,
        conf=conf,
        iou=DEFAULT_IOU,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )

    result = results[0]
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    filtered_boxes = suppress_duplicate_boxes(result, original_bgr.shape)
    raw_count = len(result.boxes)

    detections = []

    for item in filtered_boxes:
        cls_id = item["cls_id"]
        score = item["score"]
        x1, y1, x2, y2 = item["box"]

        detections.append(
            {
                "class_id": cls_id,
                "class_name": get_class_name(result.names, cls_id),
                "confidence": round(score, 4),
                "box_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "area_ratio": round(item["area_ratio"], 6),
            }
        )

    return JSONResponse(
        {
            "detections": detections,
            "count": len(detections),
            "raw_count": raw_count,
            "suppressed_count": raw_count - len(detections),
            "conf": conf,
            "iou": DEFAULT_IOU,
            "inference_ms": elapsed_ms,
            "filters": {
                "min_box_area_ratio": MIN_BOX_AREA_RATIO,
                "max_box_area_ratio": MAX_BOX_AREA_RATIO,
                "overlap_iou_threshold": OVERLAP_IOU_THRESHOLD,
                "containment_threshold": CONTAINMENT_THRESHOLD,
                "tighter_box_score_ratio": TIGHTER_BOX_SCORE_RATIO,
            },
        }
    )


@app.get("/api/presets")
def get_presets():
    presets = []
    if INPUT_DIR.exists():
        for file in sorted(INPUT_DIR.glob("*.jpg")) + sorted(INPUT_DIR.glob("*.png")):
            presets.append(
                {
                    "id": file.name,
                    "filename": file.name,
                    "label": f"Warehouse Feed - {file.stem.upper()}",
                    "url": f"/api/preset-image/{file.name}",
                }
            )
    return JSONResponse({"presets": presets})


@app.get("/api/preset-image/{filename}")
def get_preset_image(filename: str):
    file_path = INPUT_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Preset image not found")
    return FileResponse(file_path)


@app.post("/api/rack-analyze")
async def rack_analyze(
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    mode: str = Form("grid"),
    num_tiers: int = Form(4),
    num_bays: int = Form(1),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    file_bytes = await image.read()
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 15 MB allowed.")

    original_bgr = read_image_from_upload(file_bytes)
    start = time.perf_counter()

    results = model.predict(
        source=original_bgr,
        imgsz=DEFAULT_IMGSZ,
        conf=conf,
        iou=DEFAULT_IOU,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )
    result = results[0]
    filtered_boxes = suppress_duplicate_boxes(result, original_bgr.shape)

    telemetry = analyze_rack_fleet(
        image_shape=original_bgr.shape,
        gap_boxes=filtered_boxes,
        mode=mode,
        num_tiers=num_tiers,
        num_bays=num_bays,
    )

    total_elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    telemetry["summary"]["total_pipeline_ms"] = total_elapsed_ms
    telemetry["image_metadata"] = {
        "width": original_bgr.shape[1],
        "height": original_bgr.shape[0],
        "raw_detections": len(result.boxes),
        "filtered_gaps": len(filtered_boxes),
    }

    return JSONResponse(telemetry)


@app.post("/api/rack-visualize")
async def rack_visualize(
    image: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONF),
    mode: str = Form("grid"),
    num_tiers: int = Form(4),
    num_bays: int = Form(1),
    show_racks: bool = Form(True),
    show_gaps: bool = Form(True),
    show_progress_bars: bool = Form(True),
    show_labels: bool = Form(True),
    show_heatmap: bool = Form(False),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    file_bytes = await image.read()
    if len(file_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 15 MB allowed.")

    original_bgr = read_image_from_upload(file_bytes)
    start = time.perf_counter()

    results = model.predict(
        source=original_bgr,
        imgsz=DEFAULT_IMGSZ,
        conf=conf,
        iou=DEFAULT_IOU,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )
    result = results[0]
    filtered_boxes = suppress_duplicate_boxes(result, original_bgr.shape)

    telemetry = analyze_rack_fleet(
        image_shape=original_bgr.shape,
        gap_boxes=filtered_boxes,
        mode=mode,
        num_tiers=num_tiers,
        num_bays=num_bays,
    )

    annotated_bgr = render_rackvision_overlay(
        image_bgr=original_bgr,
        analysis_data=telemetry,
        show_racks=show_racks,
        show_gaps=show_gaps,
        show_progress_bars=show_progress_bars,
        show_labels=show_labels,
        show_heatmap=show_heatmap,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    output_bytes = encode_jpeg(annotated_bgr)

    headers = {
        "X-Fleet-Occupancy": str(telemetry["summary"]["fleet_occupancy_pct"]),
        "X-Critical-Alerts": str(telemetry["summary"]["critical_replenishment_alerts"]),
        "X-Total-Orders": str(telemetry["summary"]["total_replenishment_orders"]),
        "X-Inference-Ms": str(elapsed_ms),
        "Cache-Control": "no-store",
    }

    return Response(
        content=output_bytes,
        media_type="image/jpeg",
        headers=headers,
    )


@app.post("/api/replenishment/dispatch")
async def dispatch_replenishment(payload: dict):
    order_id = payload.get("order_id", f"ORD-{uuid.uuid4().hex[:6].upper()}")
    rack_id = payload.get("rack_id", "UNKNOWN")
    agv_id = f"AGV-BOT-{np.random.randint(101, 199)}"

    dispatch_event = {
        "success": True,
        "order_id": order_id,
        "rack_id": rack_id,
        "assigned_unit": agv_id,
        "dispatch_status": "DISPATCHED",
        "eta_minutes": int(np.random.randint(4, 12)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"Autonomous Restock Unit {agv_id} dispatched to {rack_id}.",
    }
    return JSONResponse(dispatch_event)