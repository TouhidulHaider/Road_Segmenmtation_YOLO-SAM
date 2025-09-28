from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
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

# '''
# Load SAM model on GPU
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device)
predictor = SamPredictor(sam)
# '''

'''
# Alternatively, use FastSAM for potentially better performance
# Create a FastSAM model
sam = FastSAM("FastSAM-s.pt")
sam.to(device)
# Create FastSAMPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-s.pt", save=False, imgsz=1024)
predictor = FastSAMPredictor(overrides=overrides)
'''


# Load image
image = cv2.imread(r"D:/cityscape/tusimple_sample1.jpg")
image = cv2.resize(image, (1024, 1024))

# Run YOLO prediction
results = yolo_model.predict(image, conf=0.25)
boxes = results[0].boxes
class_ids = boxes.cls.to('cpu').numpy()
road_boxes = boxes.xyxy.to('cpu').numpy()[class_ids == 0]  # Filter road class only

# '''
# Set image for SAM
predictor.set_image(image)
# '''

# FASTSAM does not require set_image, directly use the image in predict
# everything_segment = predictor(image) # fastsam predict

# ''' SAM 
# Refine each road box with SAM
for box in road_boxes:
    box = box.astype(np.int32)
    masks, _, _ = predictor.predict(box=box, multimask_output=False) # Original SAM
    mask = masks[0]
    image[mask] = [0, 255, 0]  # Green overlay for road
# '''

'''  FastSAM
#For FastSAM
masks = predictor.prompt(everything_segment, bboxes=road_boxes) # FastSAM
# Ensure mask is a 2D boolean array
mask_result = masks[0]
# Then access the mask tensor
mask_array = mask_result.masks.data[0].cpu().numpy()  # shape: [H, W]
# Convert to boolean mask
mask_bool = mask_array.astype(bool)
# Apply overlay
image[mask_bool] = [0, 255, 0]
'''


# If you want to overlay color without replacing the original image entirely:
# image = np.where(mask[..., None], [0, 255, 0], image) 



# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
