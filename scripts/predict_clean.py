"""
RackVision AI - CLI Inference and Operations Center
Digital Solutions BU - Operational Warehouse Intelligence
"""

from pathlib import Path
import argparse
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to sys.path to access rack_analytics
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rack_analytics import analyze_rack_fleet, render_rackvision_overlay


def run_rackvision_cli(
    model_path: str,
    source_path: str,
    output_dir: str,
    conf: float = 0.20,
    imgsz: int = 640,
    num_tiers: int = 4,
    num_bays: int = 1,
    mode: str = "grid",
    show_racks: bool = True,
    show_gaps: bool = True,
    show_bars: bool = True,
    show_labels: bool = True,
    show_heatmap: bool = False,
):
    print("=" * 80)
    print("  RACKVISION AI: COMMAND & CONTROL CENTER (Digital Solutions BU)")
    print("=" * 80)
    print(f"[*] Loading YOLO Gap Model: {model_path}")
    model = YOLO(model_path)

    source = Path(source_path)
    if not source.exists():
        print(f"[!] Error: Source file not found: {source_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Processing Image: {source_path}")
    start_time = time.perf_counter()

    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        device="cpu",
        verbose=False,
    )

    for result in results:
        img_bgr = result.orig_img.copy()
        h, w = img_bgr.shape[:2]

        # Extract gap candidates
        gap_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            gap_boxes.append({
                "cls_id": cls_id,
                "score": score,
                "box": (x1, y1, x2, y2),
            })

        # Run Rack-level analysis
        telemetry = analyze_rack_fleet(
            image_shape=img_bgr.shape,
            gap_boxes=gap_boxes,
            mode=mode,
            num_tiers=num_tiers,
            num_bays=num_bays,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Print Executive C2 ASCII Table
        summary = telemetry["summary"]
        print("\n" + "-" * 80)
        print(f"  OPERATIONAL RACK TELEMETRY REPORT | Total Racks: {summary['total_racks']} | Latency: {elapsed_ms:.1f}ms")
        print("-" * 80)
        print(f"{'Rack ID':<15} | {'Tier':<8} | {'Bay':<8} | {'Status':<12} | {'Occupancy':<10} | {'Vacancy':<9} | {'Urgency':<10}")
        print("-" * 80)

        for r in telemetry["racks"]:
            occ_str = f"{r['occupancy_pct']:.1f}%"
            vac_str = f"{r['vacancy_pct']:.1f}%"
            print(f"{r['rack_id']:<15} | {r['tier_id']:<8} | {r['bay_id']:<8} | {r['status']:<12} | {occ_str:<10} | {vac_str:<9} | {r['urgency']:<10}")

        print("-" * 80)
        print(f"[*] Fleet Fill Rate: {summary['fleet_occupancy_pct']}%  (Empty Space: {summary['fleet_vacancy_pct']}%)")
        print(f"[*] Stock Status Breakdown: {summary['stocked_racks']} STOCKED | {summary['partial_racks']} PARTIAL | {summary['low_stock_racks']} LOW STOCK | {summary['empty_racks']} EMPTY")

        if telemetry["replenishment_queue"]:
            print("\n" + "!" * 80)
            print("  [CRITICAL ALERT] REPLENISHMENT DISPATCH QUEUE:")
            print("!" * 80)
            for idx, ord_item in enumerate(telemetry["replenishment_queue"], 1):
                print(f"  {idx}. [{ord_item['urgency']}] {ord_item['rack_id']} ({ord_item['bay']} - {ord_item['tier']}) -> {ord_item['action']} (Est. Units: {ord_item['estimated_units']})")
        else:
            print("\n[+] No immediate replenishment required. All racks healthy.")

        # Render Annotated Visualization
        annotated_img = render_rackvision_overlay(
            image_bgr=img_bgr,
            analysis_data=telemetry,
            show_racks=show_racks,
            show_gaps=show_gaps,
            show_progress_bars=show_bars,
            show_labels=show_labels,
            show_heatmap=show_heatmap,
        )

        output_path = output_dir / Path(result.path).name
        cv2.imwrite(str(output_path), annotated_img)
        print(f"\n[+] Saved Annotated RackVision HUD image: {output_path}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RackVision AI Command & Control CLI")

    BASE_DIR = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--model",
        default=str(BASE_DIR / "models" / "empty-shelves-yolo26m.pt"),
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--source",
        default=str(BASE_DIR / "input" / "test.jpg"),
        help="Path to input image file or directory",
    )
    parser.add_argument(
        "--output",
        default=str(BASE_DIR / "output" / "test"),
        help="Output directory to save annotated image",
    )
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--tiers", type=int, default=4, help="Number of shelf tiers/levels")
    parser.add_argument("--bays", type=int, default=1, help="Number of vertical rack bays")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "auto"], help="Partitioning mode")
    parser.add_argument("--no-racks", action="store_true", help="Disable rack shelf outlines")
    parser.add_argument("--no-gaps", action="store_true", help="Disable gap bounding boxes")
    parser.add_argument("--no-bars", action="store_true", help="Disable shelf progress bars")
    parser.add_argument("--no-labels", action="store_true", help="Disable HUD labels")
    parser.add_argument("--heatmap", action="store_true", help="Enable gap density heatmap")

    args = parser.parse_args()

    run_rackvision_cli(
        model_path=args.model,
        source_path=args.source,
        output_dir=args.output,
        conf=args.conf,
        imgsz=args.imgsz,
        num_tiers=args.tiers,
        num_bays=args.bays,
        mode=args.mode,
        show_racks=not args.no_racks,
        show_gaps=not args.no_gaps,
        show_bars=not args.no_bars,
        show_labels=not args.no_labels,
        show_heatmap=args.heatmap,
    )