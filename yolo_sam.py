from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2, numpy as np
import matplotlib.pyplot as plt

# Load your trained model
yolo_model = YOLO(r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\best.pt")

# Load SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)


image = cv2.imread(r"D:\University Project\LaneDetectionPlayground\Road_Segmenmtation_YOLO-SAM\road_lane.jpg")
img = image
results = yolo_model.predict(image, conf=0.25)
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy()
road_boxes = boxes.xyxy.cpu().numpy()[class_ids == 0]  # Filter road class only
predictor.set_image(image)

edges = cv2.Canny(image, threshold1=50, threshold2=150)

for box in road_boxes:
    box = box.astype(np.int32)
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]
    image[mask] = [0, 255, 0]  # Green overlay for road
    
    seg_road = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8)*255)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask.astype(np.uint8))
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)

    lane_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # Red lines
        
        
# matplotlib to show the image
plt.figure()
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Show only the segmented road
plt.figure()
plt.imshow(cv2.cvtColor(seg_road, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Segmented Road")
plt.show()
# segmentated road imag

plt.figure()
plt.imshow(cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Lane Lines Inside Road Mask")
plt.show()
# cap = cv2.VideoCapture(r"D:/cityscape/road.mp4")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLOv8 inference
#     results = yolo_model.predict(frame, conf=0.25)
#     boxes = results[0].boxes
#     class_ids = boxes.cls.cpu().numpy()
#     road_boxes = boxes.xyxy.cpu().numpy()[class_ids == 0]  # Filter road class only

#     # Set image for SAM
#     predictor.set_image(frame)

#     # Refine each road box with SAM
#     for box in road_boxes:
#         box = box.astype(np.int32)
#         masks, _, _ = predictor.predict(box=box, multimask_output=False)
#         mask = masks[0]

#         # Overlay mask on frame
#         frame[mask] = [0, 255, 0]  # Green overlay for road

#     cv2.imshow("Road Segmentation", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# For saving output video, uncomment below lines
# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
# Inside the loop: out.write(frame)
