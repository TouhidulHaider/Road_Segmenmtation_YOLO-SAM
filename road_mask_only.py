from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your trained YOLO model
yolo_model = YOLO(r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\best.pt")

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Load image
image = cv2.imread(r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\road_lane.jpg")
original_image = image.copy()

# Run YOLO prediction to detect road bounding boxes
results = yolo_model.predict(image, conf=0.25)
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy()
road_boxes = boxes.xyxy.cpu().numpy()[class_ids == 0]  # Filter road class only

# Set image for SAM predictor
predictor.set_image(image)

# Initialize combined mask
combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)

# Generate segmentation masks for each detected road box
for box in road_boxes:
    box = box.astype(np.int32)
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]
    # Combine all road masks
    combined_mask = combined_mask | mask

# Create segmented road image (only road pixels)
segmented_road = cv2.bitwise_and(original_image, original_image, mask=combined_mask.astype(np.uint8)*255)

# Create binary mask visualization (white road, black background)
binary_mask = np.zeros_like(original_image)
binary_mask[combined_mask] = [255, 255, 255]

# Create colored overlay (green road on original image)
overlay_image = original_image.copy()
overlay_image[combined_mask] = [0, 255, 0]  # Green overlay for road

# === LANE DETECTION USING ROAD MASK ANALYSIS ===

height, width = combined_mask.shape

# 1. Find road mask edges (outer lane boundaries)
# Apply morphological operations to clean the mask
kernel = np.ones((5, 5), np.uint8)
cleaned_mask = cv2.morphologyEx(combined_mask.astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)

# Find contours of the road mask to get outer boundaries
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 2. Create inverted mask to find lane markings (black areas within road region)
# Dilate the road mask slightly to include nearby lane markings
dilate_kernel = np.ones((15, 15), np.uint8)
dilated_mask = cv2.dilate(cleaned_mask, dilate_kernel, iterations=1)

# Convert original image to grayscale
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply the dilated mask to focus only on road area
masked_gray = cv2.bitwise_and(gray_original, dilated_mask)

# 3. Detect lane markings as dark vertical lines within the road area
# Apply threshold to find dark areas (lane markings)
_, lane_markings = cv2.threshold(masked_gray, 100, 255, cv2.THRESH_BINARY_INV)

# Remove noise and connect dashed lines
# Use vertical morphological operations to connect dashed lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))  # Vertical kernel
lane_markings = cv2.morphologyEx(lane_markings, cv2.MORPH_CLOSE, vertical_kernel)

# Remove small noise
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
lane_markings = cv2.morphologyEx(lane_markings, cv2.MORPH_OPEN, horizontal_kernel)

# Apply the original road mask to keep only markings within the road
lane_markings = cv2.bitwise_and(lane_markings, dilated_mask)

# 4. Find edges of the road mask (outer lane boundaries)
mask_edges = cv2.Canny(cleaned_mask, 50, 150)

# Combine internal lane markings with road edges
combined_lane_features = cv2.bitwise_or(lane_markings, mask_edges)

# Define ROI for better lane detection
roi_vertices = np.array([
    [0, height],
    [width//6, height//2],
    [5*width//6, height//2],
    [width, height]
], np.int32)

roi_mask = np.zeros_like(combined_lane_features)
cv2.fillPoly(roi_mask, [roi_vertices], 255)
masked_edges = cv2.bitwise_and(combined_lane_features, roi_mask)

# Hough Line Transform to detect lane lines
lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,                    # Distance resolution in pixels
    theta=np.pi/180,          # Angle resolution in radians
    threshold=30,             # Lower threshold for better detection
    minLineLength=40,         # Minimum line length
    maxLineGap=50             # Larger gap to connect dashed lines
)

# Function to extract road boundary lanes from mask edges
def extract_boundary_lanes(mask, img_height, img_width):
    # Find the leftmost and rightmost points of the road mask at different heights
    left_points = []
    right_points = []
    
    # Sample points at different heights
    for y in range(int(img_height * 0.6), img_height, 5):  # Smaller step for more points
        if y < mask.shape[0]:
            row = mask[y, :]
            nonzero_x = np.nonzero(row)[0]
            if len(nonzero_x) > 0:
                left_points.append([nonzero_x[0], y])
                right_points.append([nonzero_x[-1], y])
    
    # Fit lines to the boundary points (treating y as function of x for lane lines)
    def fit_lane_line(points):
        if len(points) < 3:  # Need at least 3 points
            return None
        points = np.array(points)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Check if we have enough variation in y coordinates
        if np.std(y_coords) < 10:  # Not enough vertical variation
            return None
            
        try:
            # Fit y = slope * x + intercept (normal lane line equation)
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            slope, intercept = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            
            # For lane lines, we expect negative slopes for both sides in perspective view
            # but let's be more flexible and just check for reasonable slopes
            if abs(slope) > 5:  # Slope too steep (nearly vertical)
                return None
                
            # Calculate line endpoints
            y1 = img_height - 1  # Bottom of image
            y2 = int(img_height * 0.6)  # Middle-ish of image
            
            # Calculate corresponding x coordinates: x = (y - intercept) / slope
            if abs(slope) > 0.01:  # Avoid division by very small numbers
                x1 = int((y1 - intercept) / slope)
                x2 = int((y2 - intercept) / slope)
            else:
                # If slope is near zero (horizontal line), use average x
                avg_x = int(np.mean(x_coords))
                x1 = avg_x
                x2 = avg_x
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width - 1))
            x2 = max(0, min(x2, img_width - 1))
            
            # Ensure we have a reasonable line (not too short)
            if abs(x1 - x2) < 10 and abs(y1 - y2) < 50:
                return None
                
            return [x1, y1, x2, y2]
        except:
            return None
    
    left_boundary = fit_lane_line(left_points)
    right_boundary = fit_lane_line(right_points)
    
    return left_boundary, right_boundary

# Extract boundary lanes from road mask
left_boundary, right_boundary = extract_boundary_lanes(cleaned_mask, height, width)

# Function to separate left and right lanes
def separate_lanes(lines, img_width):
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip lines that are too horizontal or too short
            if abs(y2 - y1) < 20:  # Too horizontal
                continue
            if abs(x2 - x1) + abs(y2 - y1) < 30:  # Too short
                continue
                
            # Calculate slope
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter lines by slope and position
                # Be more restrictive about slopes to avoid horizontal lines
                if slope < -0.5 and x1 < img_width // 2:  # Left lane (steeper negative slope, left side)
                    left_lines.append(line[0])
                elif slope > 0.5 and x1 > img_width // 2:  # Right lane (steeper positive slope, right side)
                    right_lines.append(line[0])
            else:
                # Vertical line - check position and add to appropriate side
                if x1 < img_width // 2:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])
    
    return left_lines, right_lines

# Function to extrapolate lines
def extrapolate_lines(lines, img_height, img_width):
    if not lines:
        return None
    
    # Convert to numpy array for easier manipulation
    lines_array = np.array(lines)
    
    # Calculate average slope and intercept
    slopes = []
    intercepts = []
    
    for x1, y1, x2, y2 in lines_array:
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)
    
    if not slopes:
        return None
    
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)
    
    # Define y coordinates (from bottom to middle of image)
    y1 = img_height
    y2 = int(img_height * 0.6)
    
    # Calculate corresponding x coordinates
    x1 = int((y1 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
    x2 = int((y2 - avg_intercept) / avg_slope) if avg_slope != 0 else 0
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width))
    x2 = max(0, min(x2, img_width))
    
    return [x1, y1, x2, y2]

# Separate left and right lanes from Hough lines (internal lane markings)
left_lines, right_lines = separate_lanes(lines, width)

# Extrapolate internal lane lines
left_internal = extrapolate_lines(left_lines, height, width)
right_internal = extrapolate_lines(right_lines, height, width)

# Combine internal and boundary lanes
# Priority: use internal lanes if detected, otherwise use boundary lanes
final_left_lane = left_internal if left_internal is not None else left_boundary
final_right_lane = right_internal if right_internal is not None else right_boundary

# If we have both internal and boundary, we can detect multiple lanes
all_lanes = []
if left_boundary is not None:
    all_lanes.append(('Left Boundary', left_boundary))
if left_internal is not None:
    all_lanes.append(('Left Internal', left_internal))
if right_internal is not None:
    all_lanes.append(('Right Internal', right_internal))
if right_boundary is not None:
    all_lanes.append(('Right Boundary', right_boundary))

# Create visualization images
lane_image = original_image.copy()
boundary_image = original_image.copy()
internal_image = original_image.copy()
combined_lanes_image = original_image.copy()

# Draw boundary lanes (road edges)
if left_boundary is not None:
    x1, y1, x2, y2 = left_boundary
    cv2.line(boundary_image, (x1, y1), (x2, y2), (255, 0, 0), 6)  # Blue for boundary
    cv2.line(combined_lanes_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

if right_boundary is not None:
    x1, y1, x2, y2 = right_boundary
    cv2.line(boundary_image, (x1, y1), (x2, y2), (255, 0, 0), 6)  # Blue for boundary
    cv2.line(combined_lanes_image, (x1, y1), (x2, y2), (255, 0, 0), 4)

# Draw internal lane markings
if left_internal is not None:
    x1, y1, x2, y2 = left_internal
    cv2.line(internal_image, (x1, y1), (x2, y2), (0, 255, 0), 6)  # Green for internal
    cv2.line(combined_lanes_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

if right_internal is not None:
    x1, y1, x2, y2 = right_internal
    cv2.line(internal_image, (x1, y1), (x2, y2), (0, 255, 0), 6)  # Green for internal
    cv2.line(combined_lanes_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

# Draw final lane lines (main result)
if final_left_lane is not None:
    x1, y1, x2, y2 = final_left_lane
    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 8)  # Red for final lanes

if final_right_lane is not None:
    x1, y1, x2, y2 = final_right_lane
    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 8)  # Red for final lanes

# Draw all detected line segments (for debugging)
debug_image = original_image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow lines

# Create lane area visualization
lane_area_image = original_image.copy()
if final_left_lane is not None and final_right_lane is not None:
    # Create polygon for lane area
    lane_pts = np.array([
        [final_left_lane[0], final_left_lane[1]],   # Left bottom
        [final_left_lane[2], final_left_lane[3]],   # Left top
        [final_right_lane[2], final_right_lane[3]], # Right top
        [final_right_lane[0], final_right_lane[1]]  # Right bottom
    ], np.int32)
    
    # Fill lane area with semi-transparent color
    overlay = lane_area_image.copy()
    cv2.fillPoly(overlay, [lane_pts], (0, 255, 255))  # Yellow fill
    lane_area_image = cv2.addWeighted(lane_area_image, 0.7, overlay, 0.3, 0)
    
    # Draw lane boundaries
    if final_left_lane is not None:
        cv2.line(lane_area_image, (final_left_lane[0], final_left_lane[1]), (final_left_lane[2], final_left_lane[3]), (255, 0, 0), 8)
    if final_right_lane is not None:
        cv2.line(lane_area_image, (final_right_lane[0], final_right_lane[1]), (final_right_lane[2], final_right_lane[3]), (255, 0, 0), 8)

# Display results
plt.figure(figsize=(20, 15))

# Original image
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Binary mask
plt.subplot(3, 4, 2)
plt.imshow(cv2.cvtColor(binary_mask, cv2.COLOR_BGR2RGB))
plt.title("Road Segmentation Mask")
plt.axis('off')

# Lane markings detected
plt.subplot(3, 4, 3)
plt.imshow(lane_markings, cmap='gray')
plt.title("Detected Lane Markings")
plt.axis('off')

# Combined lane features (markings + edges)
plt.subplot(3, 4, 4)
plt.imshow(combined_lane_features, cmap='gray')
plt.title("Combined Lane Features")
plt.axis('off')

# Masked edges (ROI applied)
plt.subplot(3, 4, 5)
plt.imshow(masked_edges, cmap='gray')
plt.title("Edges with ROI")
plt.axis('off')

# Debug: All detected line segments
plt.subplot(3, 4, 6)
plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
plt.title("All Detected Line Segments")
plt.axis('off')

# Boundary lanes (road edges)
plt.subplot(3, 4, 7)
plt.imshow(cv2.cvtColor(boundary_image, cv2.COLOR_BGR2RGB))
plt.title("Road Boundary Lanes")
plt.axis('off')

# Internal lane markings
plt.subplot(3, 4, 8)
plt.imshow(cv2.cvtColor(internal_image, cv2.COLOR_BGR2RGB))
plt.title("Internal Lane Markings")
plt.axis('off')

# Combined lanes (boundary + internal)
plt.subplot(3, 4, 9)
plt.imshow(cv2.cvtColor(combined_lanes_image, cv2.COLOR_BGR2RGB))
plt.title("All Detected Lanes")
plt.axis('off')

# Final result
plt.subplot(3, 4, 10)
plt.imshow(cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
plt.title("Final Lane Lines")
plt.axis('off')

# Lane area visualization
plt.subplot(3, 4, 11)
plt.imshow(cv2.cvtColor(lane_area_image, cv2.COLOR_BGR2RGB))
plt.title("Lane Area Visualization")
plt.axis('off')

# Segmented road only
plt.subplot(3, 4, 12)
plt.imshow(cv2.cvtColor(segmented_road, cv2.COLOR_BGR2RGB))
plt.title("Segmented Road Only")
plt.axis('off')

plt.tight_layout()
plt.show()

# Print some statistics
road_pixels = np.sum(combined_mask)
total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
road_percentage = (road_pixels / total_pixels) * 100

print(f"Road pixels: {road_pixels}")
print(f"Total pixels: {total_pixels}")
print(f"Road coverage: {road_percentage:.2f}%")

# Print lane detection statistics
print("=== Lane Detection Results ===")
if final_left_lane is not None:
    print(f"Final left lane detected: {final_left_lane}")
else:
    print("Final left lane not detected")

if final_right_lane is not None:
    print(f"Final right lane detected: {final_right_lane}")
else:
    print("Final right lane not detected")

print("\n=== Boundary Lanes (Road Edges) ===")
if left_boundary is not None:
    print(f"Left boundary detected: {left_boundary}")
else:
    print("Left boundary not detected")

if right_boundary is not None:
    print(f"Right boundary detected: {right_boundary}")
else:
    print("Right boundary not detected")

print("\n=== Internal Lane Markings ===")
if left_internal is not None:
    print(f"Left internal lane detected: {left_internal}")
else:
    print("Left internal lane not detected")

if right_internal is not None:
    print(f"Right internal lane detected: {right_internal}")
else:
    print("Right internal lane not detected")

if lines is not None:
    print(f"Total line segments detected: {len(lines)}")
else:
    print("No line segments detected")

print(f"Total lanes detected: {len(all_lanes)}")
for lane_type, lane_coords in all_lanes:
    print(f"  - {lane_type}: {lane_coords}")

# Save the results
cv2.imwrite("road_mask_binary.png", binary_mask)
cv2.imwrite("road_segmented.png", segmented_road)
cv2.imwrite("lane_markings_detected.png", lane_markings)
cv2.imwrite("boundary_lanes.png", boundary_image)
cv2.imwrite("internal_lanes.png", internal_image)
cv2.imwrite("combined_lanes.png", combined_lanes_image)
cv2.imwrite("final_lane_lines.png", lane_image)
cv2.imwrite("lane_area_result.png", lane_area_image)
cv2.imwrite("debug_all_lines.png", debug_image)

print("Saved binary mask as 'road_mask_binary.png'")
print("Saved segmented road as 'road_segmented.png'")
print("Saved detected lane markings as 'lane_markings_detected.png'")
print("Saved boundary lanes as 'boundary_lanes.png'")
print("Saved internal lanes as 'internal_lanes.png'")
print("Saved combined lanes as 'combined_lanes.png'")
print("Saved final lane lines as 'final_lane_lines.png'")
print("Saved lane area visualization as 'lane_area_result.png'")
print("Saved debug image with all lines as 'debug_all_lines.png'")