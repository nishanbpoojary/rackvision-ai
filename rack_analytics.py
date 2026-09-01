"""
RackVision AI - Rack Analytics & Command and Control Engine
Digital Solutions BU - Operational Warehouse Intelligence

Features:
- RackRegionBuilder: AI-driven shelf tier & bay discovery from gap alignments
- Strict Rack ROI Bounding: Excludes non-rack background (floors, carpets, walls, ceilings)
- Exact Occupancy & Vacancy area calculation via polygon/pixel mask union
- Status Classification: STOCKED, PARTIAL, LOW_STOCK, EMPTY
- Automated Replenishment Dispatch Queue generation
- High-definition Command & Control OpenCV visualizer
"""

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import time
import uuid

import cv2
import numpy as np


class RackStatus(str, Enum):
    STOCKED = "STOCKED"          # >= 75% occupied
    PARTIAL = "PARTIAL"          # 40% - 74% occupied
    LOW_STOCK = "LOW_STOCK"      # 15% - 39% occupied
    EMPTY = "EMPTY"              # < 15% occupied


class ReplenishmentUrgency(str, Enum):
    CRITICAL = "CRITICAL"        # Empty racks (Immediate Action)
    HIGH = "HIGH"                # Low stock racks
    MEDIUM = "MEDIUM"            # Partial stock racks
    NONE = "NONE"                # Stocked racks


# Color schemes for HUD (BGR format for OpenCV)
STATUS_COLORS = {
    RackStatus.STOCKED: {
        "border": (70, 215, 85),      # Neon Green BGR
        "fill": (25, 70, 30),
        "text": (255, 255, 255),
        "badge_bg": (45, 140, 55),
        "hex": "#22f58b"
    },
    RackStatus.PARTIAL: {
        "border": (240, 200, 0),      # Bright Cyan-Blue BGR
        "fill": (60, 50, 0),
        "text": (255, 255, 255),
        "badge_bg": (180, 140, 0),
        "hex": "#00e5ff"
    },
    RackStatus.LOW_STOCK: {
        "border": (0, 165, 255),      # Bright Amber/Orange BGR
        "fill": (0, 45, 75),
        "text": (255, 255, 255),
        "badge_bg": (0, 130, 220),
        "hex": "#ff9f1c"
    },
    RackStatus.EMPTY: {
        "border": (60, 60, 255),      # Neon Red/Crimson BGR
        "fill": (15, 15, 70),
        "text": (255, 255, 255),
        "badge_bg": (40, 40, 210),
        "hex": "#ff385c"
    },
}


@dataclass
class RackRegion:
    rack_id: str
    bay_id: str
    tier_id: str
    tier_index: int
    bay_index: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class RackAnalysisResult:
    rack_id: str
    bay_id: str
    tier_id: str
    tier_index: int
    bay_index: int
    bbox: Tuple[int, int, int, int]
    total_area_px: int
    gap_area_px: int
    occupied_area_px: int
    occupancy_pct: float
    vacancy_pct: float
    status: str
    urgency: str
    gap_count: int
    status_color_hex: str
    recommended_action: str
    estimated_units_needed: int


class RackRegionBuilder:
    """
    Intelligent Rack Region Builder.
    Identifies the exact physical rack fixture in the image and discovers shelf tiers
    from gap alignments, filtering out floor/carpet, ceilings, and background walls.
    """

    @staticmethod
    def build_regions(
        image_shape: Tuple[int, int, ...],
        gap_boxes: List[Dict],
        mode: str = "auto",
        default_tiers: int = 4,
        default_bays: int = 1,
    ) -> Tuple[List[RackRegion], Dict]:
        h, w = image_shape[:2]

        valid_boxes = []
        for b in gap_boxes:
            box = b.get("box", b.get("bbox", None))
            if not box:
                continue
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                valid_boxes.append(b)

        # Fallback if no gap boxes are present in image
        if not valid_boxes:
            # Conservative center ROI to avoid floor and ceiling
            roi_y1 = int(h * 0.15)
            roi_y2 = int(h * 0.85)
            roi_x1 = int(w * 0.05)
            roi_x2 = int(w * 0.95)
            tier_count = default_tiers
            bay_count = default_bays

            regions: List[RackRegion] = []
            tier_h = (roi_y2 - roi_y1) / max(1, tier_count)
            bay_w = (roi_x2 - roi_x1) / max(1, bay_count)

            for b_idx in range(bay_count):
                bay_letter = chr(65 + b_idx)
                bx1 = int(roi_x1 + b_idx * bay_w)
                bx2 = int(roi_x1 + (b_idx + 1) * bay_w)
                for t_idx in range(tier_count):
                    by1 = int(roi_y1 + t_idx * tier_h)
                    by2 = int(roi_y1 + (t_idx + 1) * tier_h)
                    regions.append(
                        RackRegion(
                            rack_id=f"RACK-{bay_letter}-T{t_idx + 1}",
                            bay_id=f"Bay {bay_letter}",
                            tier_id=f"Tier {t_idx + 1}",
                            tier_index=t_idx,
                            bay_index=b_idx,
                            bbox=(bx1, by1, bx2, by2),
                        )
                    )

            meta = {
                "rack_roi": (roi_x1, roi_y1, roi_x2, roi_y2),
                "detected_tiers": tier_count,
                "detected_bays": bay_count,
                "floor_excluded_px": h - roi_y2,
                "ceiling_excluded_px": roi_y1,
            }
            return regions, meta

        # 1. Determine Rack Extents from Gaps
        all_x1 = [int(b["box"][0]) for b in valid_boxes]
        all_y1 = [int(b["box"][1]) for b in valid_boxes]
        all_x2 = [int(b["box"][2]) for b in valid_boxes]
        all_y2 = [int(b["box"][3]) for b in valid_boxes]

        gaps_min_x = min(all_x1)
        gaps_max_x = max(all_x2)
        gaps_min_y = min(all_y1)
        gaps_max_y = max(all_y2)

        gap_heights = [y2 - y1 for y1, y2 in zip(all_y1, all_y2)]
        med_height = float(np.median(gap_heights)) if gap_heights else (h * 0.2)

        # Pad bounding box to cover full width/height of rack shelf structure,
        # while strictly cutting off floor below and ceiling above.
        pad_x = int(w * 0.05)
        pad_y = int(med_height * 0.12)

        rack_x1 = max(0, gaps_min_x - pad_x)
        rack_x2 = min(w, gaps_max_x + pad_x)
        rack_y1 = max(0, gaps_min_y - pad_y)
        rack_y2 = min(h, gaps_max_y + pad_y)
        rack_h = rack_y2 - rack_y1

        # 2. 1D Shelf Tier Discovery from Gap Centers and Baselines
        gap_centers = np.array([(y1 + y2) / 2.0 for y1, y2 in zip(all_y1, all_y2)])
        sorted_order = np.argsort(gap_centers)
        sorted_centers = gap_centers[sorted_order]

        # Group gaps that lie along the same horizontal shelf level
        cluster_dist_thresh = max(16.0, med_height * 0.60)
        clusters = []
        curr_cluster = [sorted_order[0]]

        for i in range(1, len(sorted_centers)):
            curr_val = sorted_centers[i]
            prev_mean = np.mean([gap_centers[idx] for idx in curr_cluster])

            if (curr_val - prev_mean) <= cluster_dist_thresh:
                curr_cluster.append(sorted_order[i])
            else:
                clusters.append(curr_cluster)
                curr_cluster = [sorted_order[i]]
        if curr_cluster:
            clusters.append(curr_cluster)

        detected_tier_count = len(clusters)
        active_tiers = detected_tier_count if mode == "auto" else default_tiers
        active_bays = default_bays

        # Build clean tier boundaries within the rack ROI
        regions = []
        bay_w = (rack_x2 - rack_x1) / max(1, active_bays)

        if mode == "auto" and detected_tier_count > 1:
            # Cutoffs between cluster centers
            cluster_means = [np.mean([gap_centers[idx] for idx in c]) for c in clusters]
            tier_bounds = [rack_y1]
            for i in range(len(cluster_means) - 1):
                mid_y = int((cluster_means[i] + cluster_means[i + 1]) / 2.0)
                tier_bounds.append(mid_y)
            tier_bounds.append(rack_y2)

            for b_idx in range(active_bays):
                bay_letter = chr(65 + b_idx)
                bx1 = int(rack_x1 + b_idx * bay_w)
                bx2 = int(rack_x1 + (b_idx + 1) * bay_w) if b_idx < active_bays - 1 else rack_x2

                for t_idx in range(active_tiers):
                    y1 = tier_bounds[t_idx]
                    y2 = tier_bounds[t_idx + 1]
                    regions.append(
                        RackRegion(
                            rack_id=f"RACK-{bay_letter}-T{t_idx + 1}",
                            bay_id=f"Bay {bay_letter}",
                            tier_id=f"Tier {t_idx + 1}",
                            tier_index=t_idx,
                            bay_index=b_idx,
                            bbox=(bx1, y1, bx2, y2),
                        )
                    )
        else:
            tier_h = rack_h / max(1, active_tiers)
            for b_idx in range(active_bays):
                bay_letter = chr(65 + b_idx)
                bx1 = int(rack_x1 + b_idx * bay_w)
                bx2 = int(rack_x1 + (b_idx + 1) * bay_w) if b_idx < active_bays - 1 else rack_x2

                for t_idx in range(active_tiers):
                    y1 = int(rack_y1 + t_idx * tier_h)
                    y2 = int(rack_y1 + (t_idx + 1) * tier_h) if t_idx < active_tiers - 1 else rack_y2
                    regions.append(
                        RackRegion(
                            rack_id=f"RACK-{bay_letter}-T{t_idx + 1}",
                            bay_id=f"Bay {bay_letter}",
                            tier_id=f"Tier {t_idx + 1}",
                            tier_index=t_idx,
                            bay_index=b_idx,
                            bbox=(bx1, y1, bx2, y2),
                        )
                    )

        meta = {
            "rack_roi": (rack_x1, rack_y1, rack_x2, rack_y2),
            "detected_tiers": detected_tier_count,
            "detected_bays": active_bays,
            "floor_excluded_px": h - rack_y2,
            "ceiling_excluded_px": rack_y1,
        }
        return regions, meta


def compute_gap_union_area_in_region(
    rack_bbox: Tuple[int, int, int, int],
    gap_boxes: List[Dict],
    image_shape: Tuple[int, int, ...]
) -> Tuple[int, int]:
    """
    Calculates the exact union pixel area of all gap boxes intersecting the rack region.
    Prevents double-counting overlapping gap detections.
    """
    rx1, ry1, rx2, ry2 = rack_bbox
    rack_w = max(1, rx2 - rx1)
    rack_h = max(1, ry2 - ry1)

    mask = np.zeros((rack_h, rack_w), dtype=np.uint8)
    intersecting_count = 0

    for item in gap_boxes:
        box = item.get("box", item.get("bbox", None))
        if not box:
            continue
        gx1, gy1, gx2, gy2 = map(int, box)

        # Compute intersection with rack region
        ix1 = max(rx1, gx1)
        iy1 = max(ry1, gy1)
        ix2 = min(rx2, gx2)
        iy2 = min(ry2, gy2)

        if ix2 > ix1 and iy2 > iy1:
            lx1 = ix1 - rx1
            ly1 = iy1 - ry1
            lx2 = ix2 - rx1
            ly2 = iy2 - ry1
            mask[ly1:ly2, lx1:lx2] = 1
            intersecting_count += 1

    gap_area_px = int(np.count_nonzero(mask))
    return gap_area_px, intersecting_count


def classify_rack_status(occupancy_pct: float) -> Tuple[RackStatus, ReplenishmentUrgency, str, int]:
    """
    Maps occupancy percentage to operational rack status, urgency, and recommended actions.
    """
    if occupancy_pct >= 75.0:
        status = RackStatus.STOCKED
        urgency = ReplenishmentUrgency.NONE
        action = "Fully Stocked - No Action Required"
        units_needed = 0
    elif occupancy_pct >= 40.0:
        status = RackStatus.PARTIAL
        urgency = ReplenishmentUrgency.MEDIUM
        action = "Moderate Stock - Monitor in Next Cycle"
        units_needed = int((100.0 - occupancy_pct) * 0.5)
    elif occupancy_pct >= 15.0:
        status = RackStatus.LOW_STOCK
        urgency = ReplenishmentUrgency.HIGH
        action = "Low Inventory Warning - Schedule Replenishment"
        units_needed = int((100.0 - occupancy_pct) * 0.8)
    else:
        status = RackStatus.EMPTY
        urgency = ReplenishmentUrgency.CRITICAL
        action = "CRITICAL REPLENISHMENT - Dispatch Immediate Restock"
        units_needed = int(100.0 - occupancy_pct)

    return status, urgency, action, max(1 if status in [RackStatus.EMPTY, RackStatus.LOW_STOCK] else 0, units_needed)


def analyze_rack_fleet(
    image_shape: Tuple[int, int, ...],
    gap_boxes: List[Dict],
    mode: str = "auto",
    num_tiers: int = 4,
    num_bays: int = 1,
) -> Dict:
    """
    Performs warehouse rack occupancy and status analysis strictly bounded to the physical rack.
    Guarantees that each detected gap is assigned uniquely to its corresponding rack tier/bay.
    """
    start_time = time.perf_counter()

    regions, rack_meta = RackRegionBuilder.build_regions(
        image_shape=image_shape,
        gap_boxes=gap_boxes,
        mode=mode,
        default_tiers=num_tiers,
        default_bays=num_bays,
    )

    # 1. Assign each detected gap box uniquely to its primary rack
    rack_assigned_gaps: Dict[str, List[Dict]] = {r.rack_id: [] for r in regions}
    
    for item in gap_boxes:
        box = item.get("box", item.get("bbox", None))
        if not box:
            continue
        gx1, gy1, gx2, gy2 = map(int, box)
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0

        best_rack_id = None
        best_score = -1.0

        for r in regions:
            rx1, ry1, rx2, ry2 = r.bbox
            ix1 = max(rx1, gx1)
            iy1 = max(ry1, gy1)
            ix2 = min(rx2, gx2)
            iy2 = min(ry2, gy2)

            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            is_center_inside = (rx1 <= gcx <= rx2 and ry1 <= gcy <= ry2)
            # Prioritize rack containing the gap center, else maximum overlap area
            score = inter_area + (1e7 if is_center_inside else 0.0)

            if score > best_score and (inter_area > 0 or is_center_inside):
                best_score = score
                best_rack_id = r.rack_id

        # Fallback: if gap is outside all rack boundaries, assign to nearest rack
        if not best_rack_id and regions:
            min_dist = float("inf")
            for r in regions:
                rx1, ry1, rx2, ry2 = r.bbox
                rcx = (rx1 + rx2) / 2.0
                rcy = (ry1 + ry2) / 2.0
                dist = (gcx - rcx) ** 2 + (gcy - rcy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_rack_id = r.rack_id

        if best_rack_id:
            rack_assigned_gaps[best_rack_id].append(item)

    # 2. Compute individual rack analytics & pixel occupancy
    rack_results: List[RackAnalysisResult] = []
    total_warehouse_area = 0
    total_warehouse_gap_area = 0

    stocked_count = 0
    partial_count = 0
    low_stock_count = 0
    empty_count = 0

    replenishment_queue = []

    for region in regions:
        rx1, ry1, rx2, ry2 = region.bbox
        rack_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        total_warehouse_area += rack_area

        gap_area, _ = compute_gap_union_area_in_region(region.bbox, gap_boxes, image_shape)
        total_warehouse_gap_area += gap_area

        occupied_area = max(0, rack_area - gap_area)
        occupancy_pct = round((occupied_area / rack_area) * 100.0, 1)
        vacancy_pct = round((gap_area / rack_area) * 100.0, 1)

        # Exact unique count of gaps situated in this rack
        gap_cnt = len(rack_assigned_gaps.get(region.rack_id, []))

        status, urgency, action, units = classify_rack_status(occupancy_pct)

        if status == RackStatus.STOCKED:
            stocked_count += 1
        elif status == RackStatus.PARTIAL:
            partial_count += 1
        elif status == RackStatus.LOW_STOCK:
            low_stock_count += 1
        elif status == RackStatus.EMPTY:
            empty_count += 1

        color_hex = STATUS_COLORS[status]["hex"]

        item_result = RackAnalysisResult(
            rack_id=region.rack_id,
            bay_id=region.bay_id,
            tier_id=region.tier_id,
            tier_index=region.tier_index,
            bay_index=region.bay_index,
            bbox=region.bbox,
            total_area_px=rack_area,
            gap_area_px=gap_area,
            occupied_area_px=occupied_area,
            occupancy_pct=occupancy_pct,
            vacancy_pct=vacancy_pct,
            status=status.value,
            urgency=urgency.value,
            gap_count=gap_cnt,
            status_color_hex=color_hex,
            recommended_action=action,
            estimated_units_needed=units,
        )
        rack_results.append(item_result)

        if urgency in [ReplenishmentUrgency.CRITICAL, ReplenishmentUrgency.HIGH, ReplenishmentUrgency.MEDIUM]:
            replenishment_queue.append(
                {
                    "order_id": f"ORD-{uuid.uuid4().hex[:6].upper()}",
                    "rack_id": region.rack_id,
                    "bay": region.bay_id,
                    "tier": region.tier_id,
                    "status": status.value,
                    "urgency": urgency.value,
                    "occupancy_pct": occupancy_pct,
                    "vacancy_pct": vacancy_pct,
                    "action": action,
                    "estimated_units": units,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "dispatch_status": "PENDING",
                }
            )

    urgency_weights = {
        ReplenishmentUrgency.CRITICAL.value: 3,
        ReplenishmentUrgency.HIGH.value: 2,
        ReplenishmentUrgency.MEDIUM.value: 1,
    }
    replenishment_queue.sort(key=lambda x: urgency_weights.get(x["urgency"], 0), reverse=True)

    fleet_occupancy_pct = round(
        ((total_warehouse_area - total_warehouse_gap_area) / max(1, total_warehouse_area)) * 100.0, 1
    )
    fleet_vacancy_pct = round(100.0 - fleet_occupancy_pct, 1)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "summary": {
            "total_racks": len(rack_results),
            "detected_shelf_tiers": rack_meta["detected_tiers"],
            "detected_bay_count": rack_meta["detected_bays"],
            "total_detected_gaps": len(gap_boxes),
            "fleet_occupancy_pct": fleet_occupancy_pct,
            "fleet_vacancy_pct": fleet_vacancy_pct,
            "stocked_racks": stocked_count,
            "partial_racks": partial_count,
            "low_stock_racks": low_stock_count,
            "empty_racks": empty_count,
            "critical_replenishment_alerts": empty_count,
            "total_replenishment_orders": len(replenishment_queue),
            "analysis_latency_ms": elapsed_ms,
            "rack_roi": rack_meta["rack_roi"],
            "floor_excluded_px": rack_meta["floor_excluded_px"],
            "ceiling_excluded_px": rack_meta["ceiling_excluded_px"],
        },
        "racks": [asdict(r) for r in rack_results],
        "replenishment_queue": replenishment_queue,
        "gaps": gap_boxes,
        "rack_metadata": rack_meta,
    }


def render_rackvision_overlay(
    image_bgr: np.ndarray,
    analysis_data: Dict,
    show_racks: bool = True,
    show_gaps: bool = True,
    show_progress_bars: bool = True,
    show_labels: bool = True,
    show_heatmap: bool = False,
) -> np.ndarray:
    """
    Renders high-definition Digital Solutions BU Command & Control HUD graphics over the image.
    Only annotates within the physical rack ROI, keeping floor and background clean.
    """
    output = image_bgr.copy()
    h, w = output.shape[:2]

    # Optional Heatmap overlay for empty gap spaces
    if show_heatmap and analysis_data.get("gaps"):
        heatmap_mask = np.zeros((h, w), dtype=np.uint8)
        for g in analysis_data["gaps"]:
            box = g.get("box", g.get("bbox", None))
            if box:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(heatmap_mask, (x1, y1), (x2, y2), 255, -1)

        colored_heat = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
        alpha = 0.35
        cv2.addWeighted(colored_heat, alpha, output, 1 - alpha, 0, output)

    # 1. Render Rack Regions & Status Frames
    if show_racks and analysis_data.get("racks"):
        line_thick = max(2, min(4, int(w / 400)))
        font_scale = max(0.40, min(0.65, w / 1600.0))

        # Overall Rack Structure ROI Outer Bracket
        rack_roi = analysis_data.get("summary", {}).get("rack_roi")
        if rack_roi:
            rx1, ry1, rx2, ry2 = rack_roi
            # Subtle corner brackets for the detected rack fixture
            c_len = min(20, (rx2 - rx1) // 10)
            cv2.line(output, (rx1, ry1), (rx1 + c_len, ry1), (0, 229, 255), 2)
            cv2.line(output, (rx1, ry1), (rx1, ry1 + c_len), (0, 229, 255), 2)
            cv2.line(output, (rx2, ry2), (rx2 - c_len, ry2), (0, 229, 255), 2)
            cv2.line(output, (rx2, ry2), (rx2, ry2 - c_len), (0, 229, 255), 2)

        for rack in analysis_data["racks"]:
            x1, y1, x2, y2 = rack["bbox"]
            status_enum = RackStatus(rack["status"])
            colors = STATUS_COLORS[status_enum]
            border_bgr = colors["border"]
            fill_bgr = colors["fill"]
            badge_bg = colors["badge_bg"]

            # Subtle transparent shelf fill
            overlay = output.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_bgr, -1)
            cv2.addWeighted(overlay, 0.18, output, 0.82, 0, output)

            # Status border
            cv2.rectangle(output, (x1, y1), (x2, y2), border_bgr, line_thick, cv2.LINE_AA)

            # Rack Header Badge
            if show_labels:
                badge_text = f"{rack['rack_id']} | {rack['status']} ({rack['occupancy_pct']}%)"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                badge_h = th + 10
                badge_w = tw + 14

                by1 = max(0, y1)
                by2 = min(h, by1 + badge_h)
                bx1 = max(0, x1)
                bx2 = min(w, bx1 + badge_w)

                cv2.rectangle(output, (bx1, by1), (bx2, by2), badge_bg, -1)
                cv2.rectangle(output, (bx1, by1), (bx2, by2), (255, 255, 255), 1, cv2.LINE_AA)

                text_x = bx1 + 6
                text_y = by1 + th + 5
                cv2.putText(
                    output,
                    badge_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Mini Occupancy Progress Bar along bottom of each shelf
            if show_progress_bars:
                bar_h = max(5, int(h / 140))
                bar_y1 = y2 - bar_h - 2
                bar_y2 = y2 - 2
                bar_w = (x2 - x1) - 10
                bar_x1 = x1 + 5
                bar_x2 = bar_x1 + bar_w

                if bar_y1 > y1 and bar_w > 20:
                    cv2.rectangle(output, (bar_x1, bar_y1), (bar_x2, bar_y2), (20, 20, 25), -1)
                    fill_w = int(bar_w * (rack["occupancy_pct"] / 100.0))
                    if fill_w > 0:
                        cv2.rectangle(output, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), border_bgr, -1)
                    cv2.rectangle(output, (bar_x1, bar_y1), (bar_x2, bar_y2), (180, 180, 180), 1)

    # 2. Render Gap Detections
    if show_gaps and analysis_data.get("gaps"):
        gap_line_thick = max(1, min(3, int(w / 700)))
        gap_font_scale = max(0.35, min(0.55, w / 2000.0))
        gap_color = (0, 235, 255)

        for gap in analysis_data["gaps"]:
            box = gap.get("box", gap.get("bbox", None))
            if not box:
                continue
            gx1, gy1, gx2, gy2 = map(int, box)
            score = gap.get("score", gap.get("confidence", 0.0))

            cv2.rectangle(output, (gx1, gy1), (gx2, gy2), gap_color, gap_line_thick, cv2.LINE_AA)

            if show_labels:
                label_text = f"GAP {score:.2f}" if score > 0 else "GAP"
                (gtw, gth), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, gap_font_scale, 1)

                lx1 = gx1
                ly1 = max(0, gy1 - gth - 5)
                lx2 = gx1 + gtw + 6
                ly2 = gy1

                if ly1 >= 0 and lx2 <= w:
                    cv2.rectangle(output, (lx1, ly1), (lx2, ly2), (0, 0, 0), -1)
                    cv2.rectangle(output, (lx1, ly1), (lx2, ly2), gap_color, 1)
                    cv2.putText(
                        output,
                        label_text,
                        (lx1 + 3, ly2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        gap_font_scale,
                        gap_color,
                        1,
                        cv2.LINE_AA,
                    )

    return output
