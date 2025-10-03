import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from typing import List, Tuple, Optional, Dict

from abir_utilities import build_point_image, extract_lane_holes_and_centroids, extract_middle_lane_masks, extract_outer_lanes, extract_outer_lanes2, extract_outer_lanes3, fill_road_polygon, get_road_mask_with_yolo_sam, overlay_edge_lanes, overlay_middle_lanes, set_border

# -----------------------
# Config
# -----------------------
IMAGE_PATH = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\road_lane.jpg"
output_path = r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\output"


# Visualization switches
SHOW_PLOTS = False
SAVE_OUTPUTS = True

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")
    original = image.copy()
    h, w = image.shape[:2]

    # Step 0: Road segmentation via YOLO + SAM
    # Try to load saved road mask first, otherwise generate and save it
    mask_save_path = f'{output_path}/road_mask.npy'
    try:
        road_mask00 = np.load(mask_save_path)
        print(f"Loaded road mask from {mask_save_path}")
    except FileNotFoundError:
        road_mask00 = get_road_mask_with_yolo_sam(image)  # 0/1
        np.save(mask_save_path, road_mask00)
        print(f"Generated and saved road mask to {mask_save_path}")
    road_mask01 = set_border(road_mask00)
    # print("road mask",road_mask00)

    # Outer boundary lanes from road mask edges (left/right)
    left_lane_mask, right_lane_mask = extract_outer_lanes3(road_mask00) # no. 3 is best so far
    overlay_lanes = overlay_edge_lanes(original, left_lane_mask, right_lane_mask)

    # print("Left lane points:", left_lane_mask)
    # print("Right lane points:", right_lane_mask)

    filled255 = fill_road_polygon(road_mask01)

    # Step 2: Holes and centroids
    holes_filtered, centroids = extract_lane_holes_and_centroids(road_mask01, filled255)
    print(f"Extracted {len(centroids)} lane points from holes.\n{centroids}")

    # Step 3: Sparse point image (from centroids) for optional debug
    point_img = build_point_image(centroids, (h, w))
    middle_lanes = extract_middle_lane_masks(centroids, overlay_lanes.shape[:2])
    overlay_lanes = overlay_middle_lanes(overlay_lanes, middle_lanes)

    # Create debug visuals
    mask_vis = (road_mask00 * 255).astype(np.uint8)

    # Plots
    if SHOW_PLOTS:
        filled_title = 'Filled Road Polygon'
        plt.figure(figsize=(18, 12))
        plt.subplot(2, 3, 1); plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
        plt.subplot(2, 3, 2); plt.imshow(mask_vis, cmap='gray'); plt.title('Road Mask (0/1)'); plt.axis('off')
        plt.subplot(2, 3, 3); plt.imshow(filled255, cmap='gray'); plt.title(filled_title); plt.axis('off')
        plt.subplot(2, 3, 4); plt.imshow(holes_filtered, cmap='gray'); plt.title('Hole centroids / filtered'); plt.axis('off')
        plt.subplot(2, 3, 5); plt.imshow(point_img, cmap='gray'); plt.title('Sparse points (dilated)'); plt.axis('off')
        plt.subplot(2, 3, 6); plt.imshow(cv2.cvtColor(overlay_lanes, cv2.COLOR_BGR2RGB))
        title = 'Overlay (Blue=Edges, Red=Middle Lanes)'
        plt.title(title); plt.axis('off')
        plt.tight_layout(); plt.show()

    if SAVE_OUTPUTS:
        cv2.imwrite(f'{output_path}/lane_adv_mask.png', mask_vis)
        cv2.imwrite(f'{output_path}/lane_adv_filled.png', filled255)
        cv2.imwrite(f'{output_path}/lane_adv_holes.png', holes_filtered)
        cv2.imwrite(f'{output_path}/lane_adv_points.png', point_img)
        cv2.imwrite(f'{output_path}/lane_adv_overlay.png', overlay_lanes)
        cv2.imwrite(f'{output_path}/left_lane_overlay.png', left_lane_mask)
        cv2.imwrite(f'{output_path}/right_lane_overlay.png', right_lane_mask)
        print(f"Saved output images to {output_path}")


if __name__ == '__main__':
    main()
