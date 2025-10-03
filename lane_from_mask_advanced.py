import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Optional, Dict

# -----------------------
# Config
# -----------------------
YOLO_WEIGHTS = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\best.pt"
SAM_CHECKPOINT = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\sam_vit_h_4b8939.pth"
IMAGE_PATH = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\road_lane.jpg"
ROAD_CLASS_ID = 0  # assumed
CONF_THRESHOLD = 0.25

# Visualization switches
SHOW_PLOTS = True
SAVE_OUTPUTS = False
SKIP_FILLED_POLYGON = False  # If True, bypass filling and use original road mask for hole extraction

# -----------------------
# Utilities
# -----------------------
def draw_line(img: np.ndarray, line: Tuple[int, int, int, int], color=(0, 0, 255), thickness=6):
    x1, y1, x2, y2 = map(int, line)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def fit_line_x_as_fy(points: np.ndarray) -> Optional[Tuple[float, float]]:
    # Fit x = m*y + b (works better for near-vertical lanes)
    if len(points) < 2:
        return None
    y = points[:, 1]
    x = points[:, 0]
    A = np.vstack([y, np.ones(len(y))]).T
    try:
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return float(m), float(b)
    except Exception:
        return None


def line_endpoints_from_x_fy(model: Tuple[float, float], y_bottom: int, y_top: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    m, b = model
    if abs(m) < 1e-6:
        # nearly vertical? actually m~0 is nearly horizontal in x= m*y + b
        x_bottom = int(b)
        x_top = int(b)
    else:
        x_bottom = int(m * y_bottom + b)
        x_top = int(m * y_top + b)
    x_bottom = max(0, min(w - 1, x_bottom))
    x_top = max(0, min(w - 1, x_top))
    return (x_bottom, y_bottom, x_top, y_top)


# -----------------------
# Step 0: Road segmentation using YOLO + SAM (as in yolo_sam.py)
# -----------------------

def get_road_mask_with_yolo_sam(image_bgr: np.ndarray) -> np.ndarray:
    yolo_model = YOLO(YOLO_WEIGHTS)
    results = yolo_model.predict(image_bgr, conf=CONF_THRESHOLD)
    if len(results) == 0 or results[0].boxes is None or results[0].boxes.xyxy is None:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    road_boxes = xyxy[class_ids == ROAD_CLASS_ID]

    if len(road_boxes) == 0:
        return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    # SAM
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    predictor.set_image(image_bgr)

    h, w = image_bgr.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for box in road_boxes:
        box = box.astype(np.int32)
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(np.uint8)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask  # uint8 {0,1}


# -----------------------
# Step 1: Get a filled road polygon (so holes become explicit)
# -----------------------

def fill_road_polygon(road_mask01: np.ndarray) -> np.ndarray:
    # road_mask01 is 0/1 uint8
    mask = (road_mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled  # 0/255


# -----------------------
# Step 2: Clean holes and extract representative points
# -----------------------

def extract_lane_holes_and_centroids(road_mask01: np.ndarray, filled_mask255: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    # holes are where filled - original
    original255 = (road_mask01 * 255).astype(np.uint8)
    holes = cv2.subtract(filled_mask255, original255)

    # clean holes
    h, w = holes.shape
    # focus on lower region
    roi = np.zeros_like(holes)
    cv2.fillPoly(roi, [np.array([[0, h-1], [w//6, int(h*0.55)], [5*w//6, int(h*0.55)], [w-1, h-1]], np.int32)], 255)
    holes = cv2.bitwise_and(holes, roi)

    # morphological filtering: keep narrow vertical dashes, remove blobs
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # connected components to get centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes, connectivity=8)
    points = []
    filtered = holes

    for i in range(1, num_labels):  # skip background
        x, y, w_i, h_i, area = stats[i]
        cx, cy = centroids[i]
        # filter: small width, decent height, reasonable area
        points.append((int(cx), int(cy)))
        cv2.circle(filtered, (int(cx), int(cy)), 1, 255, -1)

    return filtered, points


# -----------------------
# Step 3: Build sparse point image (already done as 'filtered')
# -----------------------

def build_point_image(points: List[Tuple[int, int]], shape: Tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    for (x, y) in points:
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            img[y, x] = 255
    # Dilate to help Hough connect
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)), iterations=1)
    return img


# Removed Hough and RANSAC helpers for a leaner file


# -----------------------
# Boundary estimation from mask edges
# -----------------------

def estimate_boundaries_from_mask(mask01: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    # Estimate left/right boundaries from leftmost/rightmost road pixels per row
    h, w = mask01.shape
    ys = []
    xs_left = []
    xs_right = []
    for y in range(int(h * 0.6), h, 3):
        row = mask01[y, :]
        nz = np.flatnonzero(row)
        if len(nz) > 2:
            xs_left.append(nz[0])
            xs_right.append(nz[-1])
            ys.append(y)
    if len(ys) < 5:
        return None, None
    pts_left = np.stack([np.array(xs_left), np.array(ys)], axis=1)
    pts_right = np.stack([np.array(xs_right), np.array(ys)], axis=1)
    left_model = fit_line_x_as_fy(pts_left)
    right_model = fit_line_x_as_fy(pts_right)
    return left_model, right_model  # x = m*y + b for each


    # IPM helpers removed for cleanup
    pass


# -----------------------
# New: Vertical centroid-tracking for middle lanes
# -----------------------

def get_road_vertical_extent(mask01: np.ndarray) -> Tuple[int, int]:
    # Compute top and bottom row indices where mask is present
    rows = np.any(mask01 > 0, axis=1)
    ys = np.where(rows)[0]
    if ys.size == 0:
        return 0, mask01.shape[0] - 1
    return int(ys.min()), int(ys.max())


def fit_model_x_of_y(points: List[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    if len(points) < 2:
        return None
    pts = np.array(points, dtype=np.float32)
    y = pts[:, 1]
    x = pts[:, 0]
    A = np.vstack([y, np.ones_like(y)]).T
    try:
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return float(m), float(b)
    except Exception:
        return None


def track_vertical_lanes_from_centroids(
    centroids: List[Tuple[int, int]],
    mask01: np.ndarray,
    h: int,
    w: int,
    dy: int = 6,
    base_window: int = 6,
    window_growth: float = 0.03,
    min_track_points: int = 6,
    max_missing_steps: int = 2,
    min_lane_separation: int = 25,
    max_abs_slope: float = 0.35,
) -> List[Tuple[int, int, int, int]]:
    """
    Build near-vertical lanes by scanning downward from seed centroids. The lateral window grows with depth.
    Unused points seed additional lanes. Resulting lines are extended to the vertical extent of the road mask.
    """
    if not centroids:
        return []
    y_top, y_bottom = get_road_vertical_extent(mask01)
    # Bin points by y to accelerate neighborhood search
    N = len(centroids)
    pts = np.array(centroids, dtype=np.int32)
    visited = np.zeros(N, dtype=bool)
    bins: Dict[int, List[int]] = {}
    for idx, (x, y) in enumerate(pts):
        b = int(y // dy)
        bins.setdefault(b, []).append(idx)

    # Helper to get candidate indices near a target y
    def candidates_at_y(y_target: int) -> List[int]:
        b = int(y_target // dy)
        return bins.get(b, []) + bins.get(b - 1, []) + bins.get(b + 1, [])

    models: List[Tuple[float, float]] = []  # x = m*y + b

    # Seeds sorted top-to-bottom
    seed_indices = list(range(N))
    seed_indices.sort(key=lambda i: pts[i, 1])

    def too_close_to_existing_models(x_seed: int, y_seed: int) -> bool:
        for (m, b) in models:
            x_pred = m * y_seed + b
            if abs(x_seed - x_pred) < min_lane_separation:
                return True
        return False

    for si in seed_indices:
        if visited[si]:
            continue
        x0, y0 = int(pts[si, 0]), int(pts[si, 1])
        if y0 < y_top or y0 > y_bottom:
            continue
        if too_close_to_existing_models(x0, y0):
            continue

        track: List[Tuple[int, int]] = [(x0, y0)]
        visited[si] = True
        cx = x0
        missing = 0

        for y in range(y0 + dy, y_bottom + 1, dy):
            win = int(base_window + window_growth * (y - y0))
            cand_idx = candidates_at_y(y)
            # Choose unvisited candidate closest in x within window
            best = None
            best_dx = None
            for ci in cand_idx:
                if visited[ci]:
                    continue
                px, py = int(pts[ci, 0]), int(pts[ci, 1])
                if abs(py - y) > dy:  # keep close in y
                    continue
                dx = abs(px - cx)
                if dx <= win:
                    if best is None or dx < best_dx:
                        best = ci
                        best_dx = dx
            if best is not None:
                bx, by = int(pts[best, 0]), int(pts[best, 1])
                track.append((bx, by))
                visited[best] = True
                cx = bx
                missing = 0
            else:
                missing += 1
                if missing > max_missing_steps:
                    break

        if len(track) < min_track_points:
            continue

        model = fit_model_x_of_y(track)
        if model is None:
            continue
        m, b = model
        # keep vertical-ish lines only
        if abs(m) > max_abs_slope:
            continue
        # Extend to road vertical extent
        x_bot = int(np.clip(m * y_bottom + b, 0, w - 1))
        x_top = int(np.clip(m * y_top + b, 0, w - 1))
        models.append((m, b))

    # Convert models to pixel lines
    lines_pix: List[Tuple[int, int, int, int]] = []
    for (m, b) in models:
        x_bot = int(np.clip(m * y_bottom + b, 0, w - 1))
        x_top = int(np.clip(m * y_top + b, 0, w - 1))
        lines_pix.append((x_bot, y_bottom, x_top, y_top))
    return lines_pix


def set_border(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    # mask[:thickness, :] = 1        # top border
    mask[-thickness:, :] = 1       # bottom border
    mask[:, :thickness] = 1        # left border
    mask[:, -thickness:] = 1       # right border
    return mask

# -----------------------
# Main pipeline
# -----------------------

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")
    original = image.copy()
    h, w = image.shape[:2]

    # Step 0: Road segmentation via YOLO + SAM
    road_mask00 = get_road_mask_with_yolo_sam(image)  # 0/1
    road_mask01 = set_border(road_mask00)
    print("road mask",road_mask01)

    # Outer boundary lanes from road mask edges (left/right)
    left_model, right_model = estimate_boundaries_from_mask(road_mask00)
    boundary_lines_pix = []
    if left_model is not None:
        lpx = line_endpoints_from_x_fy(left_model, h - 1, int(h * 0.6), w)
        if lpx is not None:
            boundary_lines_pix.append(lpx)
    if right_model is not None:
        rpx = line_endpoints_from_x_fy(right_model, h - 1, int(h * 0.6), w)
        if rpx is not None:
            boundary_lines_pix.append(rpx)

    # Step 1: Filled road polygon (no holes)
    if SKIP_FILLED_POLYGON:
        # Skip polygon filling: directly use original mask as "filled"
        filled255 = (road_mask01 * 255).astype(np.uint8)
    else:
        filled255 = fill_road_polygon(road_mask01)

    # Step 2: Holes and centroids
    holes_filtered, centroids = extract_lane_holes_and_centroids(road_mask01, filled255)
    print(f"Extracted {len(centroids)} lane points from holes.")

    # Step 3: Sparse point image (from centroids) for optional debug
    point_img = build_point_image(centroids, (h, w))

    # Step 4: Vertical centroid-tracking for middle lanes (preferred)
    middle_lines_pix = track_vertical_lanes_from_centroids(
        centroids,
        (filled255 > 0).astype(np.uint8),
        h,
        w,
        dy=6,
        base_window=6,
        window_growth=0.03,
        min_track_points=6,
        max_missing_steps=2,
        min_lane_separation=25,
        max_abs_slope=0.35,
    )

    # No fallback or IPM; keep only tracker output for a clean pipeline
    used_ipm = False

    # Compose output images
    overlay = original.copy()
    # Draw boundaries (far left/right lanes)
    for bl in boundary_lines_pix:
        draw_line(overlay, bl, color=(255, 0, 0), thickness=8)  # blue-ish
    # Draw middle lanes (from holes)
    for ml in middle_lines_pix:
        draw_line(overlay, ml, color=(0, 0, 255), thickness=8)  # red

    # Create debug visuals
    mask_vis = (road_mask01 * 255).astype(np.uint8)
    holes_vis = cv2.cvtColor(holes_filtered.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    point_vis = cv2.cvtColor(point_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Plots
    if SHOW_PLOTS:
        filled_title = 'Filled Road Polygon' if not SKIP_FILLED_POLYGON else 'No Fill (Using Original Mask)'
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 3, 1); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
        plt.subplot(2, 3, 2); plt.imshow(mask_vis, cmap='gray'); plt.title('Road Mask (0/1)'); plt.axis('off')
        plt.subplot(2, 3, 3); plt.imshow(filled255, cmap='gray'); plt.title(filled_title); plt.axis('off')
        plt.subplot(2, 3, 4); plt.imshow(holes_filtered, cmap='gray'); plt.title('Hole centroids / filtered'); plt.axis('off')
        plt.subplot(2, 3, 5); plt.imshow(point_img, cmap='gray'); plt.title('Sparse points (dilated)'); plt.axis('off')
        plt.subplot(2, 3, 6); plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        title = 'Overlay (Blue=Edges, Red=Middle Lanes)'
        plt.title(title); plt.axis('off')
        plt.tight_layout(); plt.show()

    if SAVE_OUTPUTS:
        cv2.imwrite('lane_adv_mask.png', mask_vis)
        cv2.imwrite('lane_adv_filled.png', filled255)
        cv2.imwrite('lane_adv_holes.png', holes_filtered)
        cv2.imwrite('lane_adv_points.png', point_img)
        cv2.imwrite('lane_adv_overlay.png', overlay)


if __name__ == '__main__':
    main()
