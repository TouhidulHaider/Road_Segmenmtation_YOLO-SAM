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

def set_border(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
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
    # holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

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
from sklearn.cluster import DBSCAN

def extract_middle_lane_masks(
    centroids: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    eps: int = 200,
    min_samples: int = 3,
    thickness: int = 10,
    fit_poly: bool = True,
    poly_order: int = 1
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
        lane_masks.append(lane_mask)

    return lane_masks

def extract_middle_lane_masks_dbscan_scaled(
    centroids: List[Tuple[int,int]],
    image_shape: Tuple[int,int],
    beta: float = 0.12,          # vertical scaling factor: <1 to allow long vertical elongation
    eps: float = 40.0,          # DBSCAN neighborhood radius (in scaled units)
    min_samples: int = 2,
    thickness: int = 10,
    fit_poly: bool = True,
    poly_order: int = 1
):
    H, W = image_shape
    if len(centroids) == 0:
        return []

    pts = np.array(centroids, dtype=float)  # (N,2) : (x,y)
    scaled = pts.copy()
    scaled[:,1] *= beta   # compress vertical axis

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled)

    masks = []
    for lbl in sorted(set(labels)):
        if lbl == -1:  # noise
            continue
        cluster_pts = pts[labels == lbl]
        if cluster_pts.shape[0] < 2:
            continue
        cluster_pts = cluster_pts[np.argsort(cluster_pts[:,1])]  # top->bottom

        # smoothing
        if fit_poly and cluster_pts.shape[0] > poly_order:
            y_arr = cluster_pts[:,1]
            x_arr = cluster_pts[:,0]
            try:
                coeffs = np.polyfit(y_arr, x_arr, poly_order)
                y_new = np.arange(int(y_arr.min()), int(y_arr.max()) + 1)
                x_new = np.polyval(coeffs, y_new)
                smoothed = np.stack([x_new, y_new], axis=1).astype(np.int32)
            except np.linalg.LinAlgError:
                smoothed = cluster_pts.astype(np.int32)
        else:
            smoothed = cluster_pts.astype(np.int32)

        smoothed = np.clip(smoothed, [0,0], [W-1, H-1])
        mask = np.zeros((H,W), dtype=np.uint8)
        if len(smoothed) > 1:
            cv2.polylines(mask, [smoothed.reshape(-1,1,2)], isClosed=False, color=255, thickness=thickness)
            masks.append(mask)
    return masks


from sklearn.cluster import OPTICS

def extract_middle_lane_masks_optics(
    centroids: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    # OPTICS params
    min_samples: int = 4,
    min_cluster_size: int = 3,
    xi: float = 0.1,
    # common params
    thickness: int = 10,
    fit_poly: bool = True,
    poly_order: int = 1,
    # perspective scaling to address foreshortening
    use_perspective_scaling: bool = False,
    perspective_beta: float = 0.6
) -> List[np.ndarray]:
    """
    Cluster centroid points into lane lines using OPTICS and return masks for each lane.

    Args:
        centroids: list of (x,y) coordinates for middle lane markings.
        image_shape: (H, W) of the image/mask.
        min_samples, min_cluster_size, xi: OPTICS clustering parameters.
        thickness: line thickness for drawing lanes.
        fit_poly: if True, fit polynomial curve for smoothing.
        poly_order: polynomial order (1=linear, 2=quadratic).
        use_perspective_scaling: compress x based on y to reduce foreshortening effects.
        perspective_beta: amount of compression near bottom (0..1).

    Returns:
        lane_masks: list of binary masks (uint8, 0/255) for each lane line.
    """
    H, W = image_shape
    if len(centroids) == 0:
        return []

    points = np.array(centroids, dtype=np.float32)  # (N,2) = (x,y)

    # Optional: perspective-aware scaling of x to reduce foreshortening
    if use_perspective_scaling:
        y = points[:, 1]
        scale = 1.0 - perspective_beta * (y / float(max(1, H - 1)))
        scaled_points = points.copy()
        scaled_points[:, 0] *= scale
        clustering_input = scaled_points
    else:
        clustering_input = points

    # OPTICS clustering
    opt = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    labels = opt.fit_predict(clustering_input)  # -1 = noise

    lane_masks: List[np.ndarray] = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue  # skip noise

        pts = points[labels == lbl]  # use original coords for drawing
        if pts.shape[0] == 0:
            continue
        pts = pts[np.argsort(pts[:, 1])]  # sort by y (top -> bottom)

        lane_mask = np.zeros((H, W), dtype=np.uint8)

        if fit_poly and len(pts) > poly_order:
            y_arr = pts[:, 1]
            x_arr = pts[:, 0]
            try:
                coeffs = np.polyfit(y_arr, x_arr, poly_order)
                y_new = np.arange(int(y_arr.min()), int(y_arr.max()) + 1)
                x_new = np.polyval(coeffs, y_new)
                smoothed_pts = np.stack([x_new, y_new], axis=1).astype(np.int32)
            except np.linalg.LinAlgError:
                smoothed_pts = pts.astype(np.int32)
        else:
            smoothed_pts = pts.astype(np.int32)

        if len(smoothed_pts) > 1:
            smoothed_pts = np.clip(smoothed_pts, [0, 0], [W - 1, H - 1])
            cv2.polylines(lane_mask, [smoothed_pts.reshape(-1, 1, 2)],
                          isClosed=False, color=255, thickness=thickness)

        lane_masks.append(lane_mask)

    return lane_masks


from sklearn.cluster import AgglomerativeClustering

def extract_middle_lane_masks_hierarchical(
    centroids: List[Tuple[int, int]],
    image_shape: Tuple[int, int],
    # Hierarchical clustering params
    n_clusters: int = None,  # set to None to auto decide based on distance
    distance_threshold: float = 400.0,  # max distance to merge clusters
    linkage: str = "ward",  # "ward", "complete", "average", "single"
    # common params
    thickness: int = 10,
    fit_poly: bool = True,
    poly_order: int = 1,
    # perspective scaling
    use_perspective_scaling: bool = False,
    perspective_beta: float = 0.6
) -> List[np.ndarray]:
    """
    Cluster centroid points into lane lines using Hierarchical Clustering 
    and return masks for each lane.

    Args:
        centroids: list of (x,y) coordinates for middle lane markings.
        image_shape: (H, W) of the image/mask.
        n_clusters: number of clusters to form (if None, use distance_threshold).
        distance_threshold: threshold for forming clusters.
        linkage: linkage criterion for hierarchical clustering.
        thickness: line thickness for drawing lanes.
        fit_poly: if True, fit polynomial curve for smoothing.
        poly_order: polynomial order (1=linear, 2=quadratic).
        use_perspective_scaling: compress x based on y to reduce foreshortening.
        perspective_beta: amount of compression near bottom (0..1).

    Returns:
        lane_masks: list of binary masks (uint8, 0/255) for each lane line.
    """
    H, W = image_shape
    if len(centroids) == 0:
        return []

    points = np.array(centroids, dtype=np.float32)

    # Optional perspective-aware scaling
    if use_perspective_scaling:
        y = points[:, 1]
        scale = 1.0 - perspective_beta * (y / float(max(1, H - 1)))
        scaled_points = points.copy()
        scaled_points[:, 0] *= scale
        clustering_input = scaled_points
    else:
        clustering_input = points

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=None if n_clusters is not None else distance_threshold,
        linkage=linkage
    )
    labels = clustering.fit_predict(clustering_input)

    lane_masks: List[np.ndarray] = []
    for lbl in sorted(set(labels)):
        pts = points[labels == lbl]  # use original coordinates for drawing
        if pts.shape[0] == 0:
            continue
        pts = pts[np.argsort(pts[:, 1])]  # sort by y

        lane_mask = np.zeros((H, W), dtype=np.uint8)

        # Polynomial smoothing
        if fit_poly and len(pts) > poly_order:
            y_arr = pts[:, 1]
            x_arr = pts[:, 0]
            try:
                coeffs = np.polyfit(y_arr, x_arr, poly_order)
                y_new = np.arange(int(y_arr.min()), int(y_arr.max()) + 1)
                x_new = np.polyval(coeffs, y_new)
                smoothed_pts = np.stack([x_new, y_new], axis=1).astype(np.int32)
            except np.linalg.LinAlgError:
                smoothed_pts = pts.astype(np.int32)
        else:
            smoothed_pts = pts.astype(np.int32)

        if len(smoothed_pts) > 1:
            smoothed_pts = np.clip(smoothed_pts, [0, 0], [W - 1, H - 1])
            cv2.polylines(lane_mask, [smoothed_pts.reshape(-1, 1, 2)],
                          isClosed=False, color=255, thickness=thickness)

        lane_masks.append(lane_mask)

    return lane_masks


from sklearn.cluster import DBSCAN
import math

def extract_middle_lane_masks_dbscan_scaled_with_merge(
    centroids: List[Tuple[int,int]],
    image_shape: Tuple[int,int],
    beta: float = 0.12,          # vertical scaling factor used for clustering
    eps: float = 40.0,           # DBSCAN eps (in scaled coords)
    min_samples: int = 2,
    thickness: int = 5,
    fit_poly: bool = True,
    poly_order: int = 1,
    # merge params
    angle_thresh_deg: float = 10.0,   # max angle difference to consider similar (degrees)
    merge_dist_px: float = 50.0,      # max horizontal distance (pixels) to consider close
    min_cluster_size: int = 2         # discard clusters smaller than this earlier
) -> List[np.ndarray]:
    """
    DBSCAN on scaled coords, then merge clusters whose fitted lines are similar in angle
    and close horizontally. Returns list of binary masks (0/255).
    """
    H, W = image_shape
    if len(centroids) == 0:
        return []

    pts = np.array(centroids, dtype=float)  # (N,2) = (x,y)
    scaled = pts.copy()
    scaled[:,1] *= beta  # compress vertical axis

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled)

    # Collect clusters (use original coords for fitting/drawing)
    clusters = []
    for lbl in sorted(set(labels)):
        if lbl == -1:
            continue
        cluster_pts = pts[labels == lbl]
        if cluster_pts.shape[0] < min_cluster_size:
            continue
        cluster_pts = cluster_pts[np.argsort(cluster_pts[:,1])]
        clusters.append({
            "pts": cluster_pts  # np array (M,2)
        })

    if len(clusters) == 0:
        return []

    # helper: fit poly x = p(y) and compute slope (dx/dy) at y0 and a function to eval x(y)
    def fit_poly_for_cluster(cluster_pts, order):
        y = cluster_pts[:,1]
        x = cluster_pts[:,0]
        # if not enough points for poly, fallback to linear (order=1) or direct
        ord_use = min(order, len(y)-1) if len(y) > 1 else 0
        if ord_use <= 0:
            # constant x (vertical column of points) - represent by median x
            coeffs = np.array([np.median(x)])  # constant polynomial
            # polyval with coeffs of degree 0: np.polyval([c], y) yields c
        else:
            try:
                coeffs = np.polyfit(y, x, ord_use)
            except np.linalg.LinAlgError:
                coeffs = np.polyfit(y, x, ord_use, rcond=None)
        # derivative polynomial (coeffs of dx/dy)
        if coeffs.size > 1:
            dcoeffs = np.polyder(coeffs)
        else:
            dcoeffs = np.array([0.0])  # derivative of constant is zero
        # create lambdas
        def x_of_y(yv):
            return np.polyval(coeffs, yv)
        def slope_at(y0):
            return np.polyval(dcoeffs, y0)
        return coeffs, dcoeffs, x_of_y, slope_at

    # prepare cluster models
    for c in clusters:
        coeffs, dcoeffs, x_of_y, slope_at = fit_poly_for_cluster(c["pts"], poly_order)
        c["coeffs"] = coeffs
        c["dcoeffs"] = dcoeffs
        c["x_of_y"] = x_of_y
        c["slope_at"] = slope_at
        y_med = float(np.median(c["pts"][:,1]))
        slope = slope_at(y_med)
        angle_deg = math.degrees(math.atan(slope))
        c["angle_deg"] = angle_deg
        # store y-range and endpoints for distance fallback
        c["ymin"] = float(c["pts"][:,1].min())
        c["ymax"] = float(c["pts"][:,1].max())
        c["x_at_ymin"] = float(x_of_y(c["ymin"]))
        c["x_at_ymax"] = float(x_of_y(c["ymax"]))

    # union-find for merging
    n = len(clusters)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # compute pairwise and merge iteratively
    for i in range(n):
        for j in range(i+1, n):
            ci = clusters[i]
            cj = clusters[j]
            # angle difference (absolute)
            ang_diff = abs(ci["angle_deg"] - cj["angle_deg"])
            # normalize ang_diff to [0,90]
            if ang_diff > 180:
                ang_diff = abs(ang_diff - 360)
            if ang_diff > 90:
                ang_diff = 180 - ang_diff

            if ang_diff > angle_thresh_deg:
                continue  # not similar angle

            # compute horizontal closeness over y-overlap
            y_overlap_min = max(ci["ymin"], cj["ymin"])
            y_overlap_max = min(ci["ymax"], cj["ymax"])
            mean_horiz_dist = None
            if y_overlap_max >= y_overlap_min:
                # sample a few y in overlap
                ys = np.linspace(y_overlap_min, y_overlap_max, num=20)
                xi_vals = ci["x_of_y"](ys)
                xj_vals = cj["x_of_y"](ys)
                mean_horiz_dist = float(np.mean(np.abs(xi_vals - xj_vals)))
            else:
                
                mean_horiz_dist = line_distance_with_extrapolation(
                    ci, cj,
                    sample_count=20,
                    max_extrapolation_px=120.0,
                    use_perpendicular=True,   # recommended
                    beta=1.0
                )
            if mean_horiz_dist <= merge_dist_px:
                union(i, j)

    # build merged groups
    groups = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    merged_clusters = []
    for root, idx_list in groups.items():
        if len(idx_list) == 1:
            merged_pts = clusters[idx_list[0]]["pts"]
        else:
            # concatenate points of all clusters in this group
            all_pts = np.vstack([clusters[k]["pts"] for k in idx_list])
            # optional: sort by y
            merged_pts = all_pts[np.argsort(all_pts[:,1])]
        merged_clusters.append(merged_pts)

    # Optionally, you could re-run merging after re-fitting merged clusters (not done here),
    # but for most cases one pass suffices.

    # create masks from merged clusters
    masks = []
    for pts_cluster in merged_clusters:
        if pts_cluster.shape[0] < 2:
            continue
        pts_cluster = pts_cluster[np.argsort(pts_cluster[:,1])]
        if fit_poly and pts_cluster.shape[0] > poly_order:
            y_arr = pts_cluster[:,1]
            x_arr = pts_cluster[:,0]
            try:
                coeffs = np.polyfit(y_arr, x_arr, poly_order)
                y_new = np.arange(int(y_arr.min()), int(y_arr.max()) + 1)
                x_new = np.polyval(coeffs, y_new)
                smoothed = np.stack([x_new, y_new], axis=1).astype(np.int32)
            except np.linalg.LinAlgError:
                smoothed = pts_cluster.astype(np.int32)
        else:
            smoothed = pts_cluster.astype(np.int32)

        smoothed = np.clip(smoothed, [0,0], [W-1, H-1])
        mask = np.zeros((H, W), dtype=np.uint8)
        if len(smoothed) > 1:
            cv2.polylines(mask, [smoothed.reshape(-1,1,2)], isClosed=False, color=255, thickness=thickness)
            masks.append(mask)
    return masks



def overlay_middle_lanes(image, lane_masks):
    overlay = image.copy()
    for mask in lane_masks:
        print("middle mask: ", mask)
        overlay[mask > 0] = (0,0,255)
    return overlay

import numpy as np
import math

def line_distance_with_extrapolation(
    ci, cj,
    sample_count: int = 20,
    max_extrapolation_px: float = 120.0,
    use_perpendicular: bool = True,
    beta: float = 1.0
) -> float:
    """
    Compute distance between two fitted polynomials ci and cj.
    ci and cj must provide:
      - 'ymin', 'ymax' (floats)
      - 'x_of_y' callable: x_of_y(y_array) -> np.array
      - 'slope_at' callable: slope_at(y_array) -> np.array or float
    If they overlap in y, sample over overlap and compute mean distance (perp or horizontal).
    If they do NOT overlap:
      - Extrapolate polynomials across the vertical gap by sampling y between the nearer endpoints
      - If the gap exceeds max_extrapolation_px, return a large distance (avoid dangerous extrapolation)
    beta: vertical weighting for Euclidean distance if you want vertical contribution
    use_perpendicular: True -> use perpendicular distance (recommended), else horizontal abs diff.
    Returns: scalar distance (lower means closer)
    """

    # Helper safe-eval that accepts scalar or array
    def eval_x(f, ys):
        ys = np.array(ys, dtype=float)
        return np.asarray(f(ys), dtype=float)

    # overlap range
    overlap_min = max(ci["ymin"], cj["ymin"])
    overlap_max = min(ci["ymax"], cj["ymax"])

    # function to compute mean distance given two x arrays and slopes for perp metric
    def mean_distance_from_samples(xs_i, xs_j, slopes_i=None):
        dx = xs_i - xs_j  # horizontal differences
        if use_perpendicular:
            # need slopes_i (slope of reference curve i) to compute normal
            if slopes_i is None:
                # approximate slope zero if none provided (fallback to horizontal)
                norms = np.tile(np.array([0.0, 1.0]), (len(dx),1))
            else:
                mi = np.asarray(slopes_i, dtype=float)
                # normal vector for slope m is (-m, 1) normalized
                norms = np.vstack((-mi, np.ones_like(mi))).T
                norms = norms / np.linalg.norm(norms, axis=1, keepdims=True)
            # vector from Pi to Pj at same y is (dx, 0). Perpendicular distance is dot with normal:
            # d = | [dx, 0] . n | = | dx * n_x |
            n_x = norms[:, 0]
            d_perp = np.abs(dx * n_x)
            # But if slopes vary a lot between the two, you could average normals; using i's normals is fine.
            return float(np.mean(d_perp))
        else:
            # horizontal metric (optionally include vertical weight beta externally if needed)
            return float(np.mean(np.abs(dx)))

    # Case 1: overlapping -> sample in overlap
    if overlap_max >= overlap_min:
        ys = np.linspace(overlap_min, overlap_max, num=sample_count)
        xi = eval_x(ci["x_of_y"], ys)
        xj = eval_x(cj["x_of_y"], ys)
        if use_perpendicular:
            mi = np.atleast_1d(ci["slope_at"](ys))
            # we could also sample slopes of cj and average, but using ci's normals is fine
            return mean_distance_from_samples(xi, xj, slopes_i=mi)
        else:
            return mean_distance_from_samples(xi, xj, slopes_i=None)

    # Case 2: no overlap -> identify gap and extrapolate across it
    # Determine vertical gap distance (in pixels)
    # Identify which is above: if ci is above cj then ci.ymax < cj.ymin
    if ci["ymax"] < cj["ymin"]:
        top = ci
        bottom = cj
    elif cj["ymax"] < ci["ymin"]:
        top = cj
        bottom = ci
    else:
        # unlikely: numerical issues; fallback to endpoint min horizontal distance
        d_candidates = [
            abs(ci["x_at_ymin"] - cj["x_at_ymax"]),
            abs(ci["x_at_ymax"] - cj["x_at_ymin"]),
            abs(ci["x_at_ymin"] - cj["x_at_ymin"]),
            abs(ci["x_at_ymax"] - cj["x_at_ymax"])
        ]
        return float(min(d_candidates))

    gap = bottom["ymin"] - top["ymax"]  # positive because top.ymax < bottom.ymin
    if gap > max_extrapolation_px:
        # gap too big to trust extrapolation; return a large distance so they won't merge
        return float(1e6)

    # sample ys across the vertical corridor between top.ymax and bottom.ymin (inclusive)
    ys = np.linspace(top["ymax"], bottom["ymin"], num=sample_count)
    # Evaluate both polynomials at the same ys (this extrapolates each polynomial outside its fit range if needed)
    xi = eval_x(ci["x_of_y"], ys)
    xj = eval_x(cj["x_of_y"], ys)

    if use_perpendicular:
        # use slope from the curve that is being extrapolated at these ys
        # to be symmetric, compute slopes for both and average normals:
        mi = np.atleast_1d(ci["slope_at"](ys))
        mj = np.atleast_1d(cj["slope_at"](ys))
        # normals for i and j
        ni = np.vstack((-mi, np.ones_like(mi))).T
        ni = ni / np.linalg.norm(ni, axis=1, keepdims=True)
        nj = np.vstack((-mj, np.ones_like(mj))).T
        nj = nj / np.linalg.norm(nj, axis=1, keepdims=True)
        # use the average normal direction (but re-normalize)
        n_avg = ni + nj
        # handle cases where normals oppose (norm very small) — fallback to using i's normals
        norms_len = np.linalg.norm(n_avg, axis=1, keepdims=True)
        small = (norms_len.squeeze() < 1e-6)
        n_avg[small] = ni[small]  # fallback
        n_avg = n_avg / np.linalg.norm(n_avg, axis=1, keepdims=True)
        dx = xi - xj
        d_perp = np.abs(dx * n_avg[:, 0])  # since y is same for both samples, vector is (dx,0)
        return float(np.mean(d_perp))
    else:
        # horizontal mean (optionally you may include Beta-weighted vertical component)
        return float(np.mean(np.abs(xi - xj)))
