from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load your trained model
model = YOLO(r"D:/cityscape/best.pt")

# results = model.predict(source= r"D:/cityscape/road_loane.jpg", save=True, conf=0.25)
results = model.predict(source= r"D:/cityscape/tusimple_sample1.jpg", save=False, conf=0.25)

# Load the image using OpenCV
image = cv2.imread(r"D:/cityscape/tusimple_sample1.jpg")

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract boxes and class IDs
boxes = results[0].boxes
class_ids = boxes.cls.cpu().numpy()
road_boxes = boxes.xyxy.cpu().numpy()[class_ids == 0]  # Assuming class 0 is 'road'

# Draw road bounding boxes
for box in road_boxes:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=5)  # Blue box

# Display the image with bounding boxes
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Road Bounding Boxes")
plt.show()

# cap = cv2.VideoCapture(r"D:/cityscape/road.mp4")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model.predict(frame, conf=0.25)
#     annotated_frame = results[0].plot()
#     cv2.imshow("YOLOv8 Inference", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


# the above part is good for video and image

