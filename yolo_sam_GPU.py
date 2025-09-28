from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2, numpy as np
import matplotlib.pyplot as plt
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO model on GPU
yolo_model = YOLO(r"D:/cityscape/best.pt")
yolo_model.to(device)

# Load SAM model on GPU
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device)
predictor = SamPredictor(sam)

# Load image
image = cv2.imread(r"D:/cityscape/tusimple_sample1.jpg")

# Run YOLO prediction
results = yolo_model.predict(image, conf=0.25)
boxes = results[0].boxes
class_ids = boxes.cls.to('cpu').numpy()
road_boxes = boxes.xyxy.to('cpu').numpy()[class_ids == 0]  # Filter road class only

# Set image for SAM
predictor.set_image(image)

# Refine each road box with SAM
for box in road_boxes:
    box = box.astype(np.int32)
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]
    image[mask] = [0, 255, 0]  # Green overlay for road

# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
