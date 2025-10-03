import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Optional, Dict

YOLO_WEIGHTS = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\best.pt"
SAM_CHECKPOINT = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\sam_vit_h_4b8939.pth"
ROAD_CLASS_ID = 0  # assumed
CONF_THRESHOLD = 0.25

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

def set_border(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    m = mask.copy()
    # mask[:thickness, :] = 1        # top border
    m[-thickness:, :] = 1       # bottom border
    m[:, :thickness] = 1        # left border
    m[:, -thickness:] = 1       # right border
    return m


def fill_road_polygon(road_mask01: np.ndarray) -> np.ndarray:
    # road_mask01 is 0/1 uint8
    mask = (road_mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled  # 0/255

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

def build_point_image(points: List[Tuple[int, int]], shape: Tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    for (x, y) in points:
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            img[y, x] = 255
    # Dilate to help Hough connect
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)), iterations=1)
    return img

def draw_line(img: np.ndarray, line: Tuple[int, int, int, int], color=(0, 0, 255), thickness=6):
    x1, y1, x2, y2 = map(int, line)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def extract_outer_lanes(mask, thickness=10, cx=None, min_segment_len=30, smooth=15):
    """
    Extract left/right outer road edges while avoiding drawing lines that connect
    disjoint contour segments.

    Args:
      mask: 2D binary (0/255) road mask
      thickness: line thickness for output masks
      cx: optional center x to split; if None uses image center
      min_segment_len: minimum number of contour points to accept a segment
      smooth: smoothing kernel size (odd int). If <=1, skip smoothing.

    Returns:
      left_mask, right_mask
    """
    H, W = mask.shape
    road = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(road), np.zeros_like(road)

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2:
        return np.zeros_like(road), np.zeros_like(road)

    if cx is None:
        cx = W // 2

    # helper: get contiguous segments of indices where cond is true
    def contiguous_segments(condition):
        # condition: boolean array length N
        idx = np.where(condition)[0]
        if idx.size == 0:
            return []
        # find breaks in consecutive indices
        diffs = np.diff(idx)
        breaks = np.where(diffs > 1)[0]
        segments = []
        start = 0
        for b in breaks:
            seg = idx[start:b+1]
            segments.append(seg)
            start = b+1
        segments.append(idx[start:])  # last segment
        return segments

    xs = contour[:, 0]
    cond_left = xs <= cx
    cond_right = xs >= cx

    left_segs = contiguous_segments(cond_left)
    right_segs = contiguous_segments(cond_right)

    def pick_largest_segment(segments):
        if not segments:
            return None
        # pick the segment with largest length (could also prefer mid-located)
        best = max(segments, key=lambda s: s.size)
        return best if best.size >= min_segment_len else None

    left_idx = pick_largest_segment(left_segs)
    right_idx = pick_largest_segment(right_segs)

    def build_mask_from_idx(idx_array):
        mask_out = np.zeros_like(road)
        if idx_array is None:
            return mask_out
        pts = contour[idx_array]
        # optionally smooth by convolving x coords along the run
        if smooth and smooth > 1:
            k = smooth if smooth % 2 == 1 else smooth+1
            pad = k//2
            xs_run = pts[:, 0].astype(float)
            xs_pad = np.pad(xs_run, (pad, pad), mode='edge')
            xs_s = np.convolve(xs_pad, np.ones(k)/k, mode='valid')
            pts_s = np.column_stack((np.round(xs_s).astype(int), pts[:,1]))
            pts_draw = np.clip(pts_s, [[0,0]], [[W-1,H-1]])
        else:
            pts_draw = pts
        if pts_draw.shape[0] > 1:
            cv2.polylines(mask_out, [pts_draw.reshape(-1,1,2)], False, 255, thickness)
        return mask_out

    left_mask = build_mask_from_idx(left_idx)
    right_mask = build_mask_from_idx(right_idx)
    return left_mask, right_mask
  
def extract_outer_lanes2(road_mask: np.ndarray,
                         thickness: int = 6,
                         smooth: int = 15,
                         min_points: int = 30,
                         max_width_ratio: float = 0.98):
    """
    Robust extraction of left/right border lane masks from a binary road mask.
    Avoids bottom-artifact by selecting the largest contiguous vertical region
    where the road width is < max_width_ratio * image_width.

    Args:
        road_mask: binary mask (H,W) with values 0/1 or 0/255
        thickness: polyline thickness (pixels)
        smooth: smoothing kernel size (odd int). <=1 disables smoothing.
        min_points: minimum vertical samples required to produce a lane.
        max_width_ratio: rows with width >= max_width_ratio*W are ignored.

    Returns:
        left_mask, right_mask (uint8 0/255)
    """
    H, W = road_mask.shape[:2]
    mask = (road_mask > 0).astype(np.uint8) * 255

    # compute per-row left/right x and width
    left_x = np.full(H, -1, dtype=int)
    right_x = np.full(H, -1, dtype=int)
    widths = np.zeros(H, dtype=int)
    for y in range(H):
        xs = np.where(mask[y] > 0)[0]
        if xs.size:
            left_x[y] = int(xs[0])
            right_x[y] = int(xs[-1])
            widths[y] = xs[-1] - xs[0] + 1

    # valid rows: have road pixels and not spanning too wide
    valid = (widths > 0) & (widths < (W * max_width_ratio))

    # helper: contiguous segments of True in valid
    def contiguous_segments(cond_bool):
        idx = np.where(cond_bool)[0]
        if idx.size == 0:
            return []
        diffs = np.diff(idx)
        breaks = np.where(diffs > 1)[0]
        segs = []
        start = 0
        for b in breaks:
            segs.append(idx[start:b+1])
            start = b+1
        segs.append(idx[start:])
        return segs

    segs = contiguous_segments(valid)
    if not segs:
        return np.zeros_like(mask), np.zeros_like(mask)

    # pick the largest vertical contiguous segment
    best_seg = max(segs, key=lambda s: s.size)
    if best_seg.size < min_points:
        return np.zeros_like(mask), np.zeros_like(mask)

    ys = best_seg  # array of y indices we will use

    # build point arrays
    left_pts = np.array([[left_x[y], y] for y in ys], dtype=int)
    right_pts = np.array([[right_x[y], y] for y in ys], dtype=int)

    # small safety: if any -1 present (shouldn't), drop those rows
    def drop_invalid(pts):
        good = pts[:,0] >= 0
        return pts[good]

    left_pts = drop_invalid(left_pts)
    right_pts = drop_invalid(right_pts)

    # smoothing function (moving average) with jump-clamp
    def smooth_run(pts):
        if pts.shape[0] < min_points:
            return None
        xs = pts[:,0].astype(float)
        n = xs.size
        if smooth and smooth > 1:
            k = smooth if (smooth % 2 == 1) else smooth + 1
            pad = k//2
            xs_pad = np.pad(xs, (pad, pad), mode='edge')
            xs_s = np.convolve(xs_pad, np.ones(k)/k, mode='valid')
        else:
            xs_s = xs
        # clamp large single-step jumps to avoid corner artifacts
        jump_thresh = max(30, int(W * 0.05))
        for i in range(1, xs_s.size):
            d = xs_s[i] - xs_s[i-1]
            if abs(d) > jump_thresh:
                xs_s[i] = xs_s[i-1] + np.sign(d) * jump_thresh
        pts_s = np.column_stack((np.round(xs_s).astype(int), pts[:,1]))
        pts_s = np.clip(pts_s, [0,0], [W-1,H-1])
        return pts_s

    left_pts_s = smooth_run(left_pts)
    right_pts_s = smooth_run(right_pts)

    # draw polylines
    left_mask = np.zeros_like(mask)
    right_mask = np.zeros_like(mask)
    if left_pts_s is not None and left_pts_s.shape[0] > 1:
        cv2.polylines(left_mask, [left_pts_s.reshape(-1,1,2)], False, 255, thickness, lineType=cv2.LINE_AA)
    if right_pts_s is not None and right_pts_s.shape[0] > 1:
        cv2.polylines(right_mask, [right_pts_s.reshape(-1,1,2)], False, 255, thickness, lineType=cv2.LINE_AA)

    return left_mask, right_mask


def extract_outer_lanes3(road_mask, thickness=10, smooth=15, min_points=30, top_cut_ratio=0.05):
    """
    Extract left and right lane boundary masks from a road mask by scanning each row
    for the leftmost and rightmost points, but cut off the top portion of the road contour.

    Args:
        road_mask: binary (0/255 or bool) road mask
        thickness: polyline thickness
        smooth: moving average kernel size for x-coords
        min_points: minimum required points per side
        top_cut_ratio: fraction of the *road contour height* to ignore at the top

    Returns:
        left_mask, right_mask
    """
    H, W = road_mask.shape
    mask = (road_mask > 0).astype(np.uint8) * 255

    # --- Find road contour bounds ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros_like(mask), np.zeros_like(mask)
    contour = max(contours, key=cv2.contourArea)

    y_min = contour[:,:,1].min()
    y_max = contour[:,:,1].max()
    road_height = y_max - y_min
    cut_y = int(y_min + road_height * top_cut_ratio)

    # --- Collect per-row left/right x values ---
    ys, left_xs, right_xs = [], [], []
    for y in range(cut_y, H):
        xs = np.where(mask[y] > 0)[0]
        if xs.size > 0:
            ys.append(y)
            left_xs.append(xs.min())
            right_xs.append(xs.max())

    def smooth_line(xs, ys):
        if len(xs) < min_points:
            return np.zeros_like(mask)
        xs = np.array(xs, dtype=np.float32)
        if smooth and smooth > 1 and len(xs) >= smooth:
            k = smooth if smooth % 2 == 1 else smooth + 1
            pad = k // 2
            xs_pad = np.pad(xs, (pad, pad), mode="edge")
            xs_s = np.convolve(xs_pad, np.ones(k)/k, mode="valid")
            xs = xs_s
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        pts = np.clip(pts, [0,0], [W-1,H-1])
        lane_mask = np.zeros_like(mask)
        cv2.polylines(lane_mask, [pts.reshape(-1,1,2)], False, 255, thickness)
        return lane_mask

    left_mask = smooth_line(left_xs, ys)
    right_mask = smooth_line(right_xs, ys)

    return left_mask, right_mask


def overlay_edge_lanes(image, left_mask, right_mask, color=(0,255,0), alpha=1):
    """
    Overlay left/right lane masks on the original image.

    Args:
      image (ndarray): Original image (H,W,3).
      left_mask (ndarray): Binary mask for left lane (H,W), dtype uint8.
      right_mask (ndarray): Binary mask for right lane (H,W), dtype uint8.
      left_color (tuple): BGR color for left lane.
      right_color (tuple): BGR color for right lane.
      alpha (float): transparency factor (0=only image, 1=only mask).
    """

    # make color overlays
    overlay = image.copy()

    # left lane: color where mask is >0
    overlay[left_mask > 0] = color

    # right lane: color where mask is >0
    overlay[right_mask > 0] = color

    # blend
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from typing import List, Tuple

def extract_middle_lane_masks(
    centroids: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    eps: int = 800,
    min_samples: int = 3,
    thickness: int = 10,
    fit_poly: bool = True,
    poly_order: int = 2
) -> List[np.ndarray]:
    """
    Group centroid points into lane lines and return masks for each lane.

    Args:
        centroids: list of (x,y) coordinates for middle lane markings.
        image_shape: (H, W) of the image/mask.
        eps: DBSCAN neighborhood radius (controls horizontal grouping).
        min_samples: minimum samples per cluster for DBSCAN.
        thickness: line thickness for drawing lanes.
        fit_poly: if True, fit polynomial curve for smoothing.
        poly_order: polynomial order (1=linear, 2=quadratic).

    Returns:
        lane_masks: list of binary masks (uint8, 0/255) for each lane line.
    """
    H, W = image_shape
    if len(centroids) == 0:
        return []

    points = np.array(centroids)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    lane_masks = []
    for lbl in set(labels):
        if lbl == -1:
            continue  # skip noise
        pts = points[labels == lbl]
        pts = pts[np.argsort(pts[:,1])]  # sort by y

        lane_mask = np.zeros((H, W), dtype=np.uint8)

        if fit_poly and len(pts) > poly_order:
            # Fit polynomial x=f(y)
            y = pts[:,1]
            x = pts[:,0]
            coeffs = np.polyfit(y, x, poly_order)
            y_new = np.arange(y.min(), y.max()+1)
            x_new = np.polyval(coeffs, y_new)
            smoothed_pts = np.stack([x_new, y_new], axis=1).astype(np.int32)
        else:
            smoothed_pts = pts.astype(np.int32)

        # Draw lane line
        if len(smoothed_pts) > 1:
            smoothed_pts = np.clip(smoothed_pts, [0,0], [W-1,H-1])
            cv2.polylines(lane_mask, [smoothed_pts.reshape(-1,1,2)], False, 255, thickness)
        print("!", lane_mask)
        lane_masks.append(lane_mask)

    return lane_masks

def overlay_middle_lanes(image, lane_masks):
    overlay = image.copy()
    for mask in lane_masks:
        print("middle mask: ", mask)
        overlay[mask > 0] = (0,0,255)
    return overlay

