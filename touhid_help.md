pip install ultralytics opencv-python matplotlib numpy
pip install git+https://github.com/facebookresearch/segment-anything.git

https://github.com/Hyounjun-Oh/YOLOv8_cityscapes/tree/main/result_example/weights

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
or
https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth


OLOv8 gives fast bounding boxes for road regions.

SAM refines those boxes into pixel-perfect masks.

You get real-time, class-specific, and high-quality segmentation.

# Output (results)
image 1/1 D:\cityscape\road_loane.jpg: 448x640 1 road, 1 sidewalk, 1 building, 1 traffic sign, 50.0ms
Speed: 3.4ms preprocess, 50.0ms inference, 3.7ms postprocess per image at shape (1, 3, 448, 640)
[ultralytics.engine.results.Results object with attributes:

boxes: ultralytics.engine.results.Boxes object
keypoints: None
masks: ultralytics.engine.results.Masks object
names: {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'}
obb: None
orig_img: array([[[112, 205, 254],
        [121, 214, 255],
        [113, 205, 254],
        ...,
        [108, 199, 254],
        [109, 199, 255],
        [107, 197, 254]],

       [[ 95, 187, 236],
        [ 97, 189, 238],
        [ 94, 183, 233],
        ...,
        [ 86, 177, 232],
        [ 87, 176, 233],
        [ 78, 168, 225]],

       [[117, 201, 253],
        [107, 191, 243],
        [ 95, 180, 230],
        ...,
        [ 91, 181, 236],
        [ 83, 170, 227],
        [ 88, 177, 234]],

       ...,

       [[106, 108, 108],
        [163, 168, 166],
        [ 86,  93,  88],
        ...,
        [ 78,  82,  77],
        [ 88,  92,  87],
        [ 59,  63,  58]],

       [[ 70,  73,  71],
        [ 88,  93,  91],
        [113, 120, 115],
        ...,
        [ 90,  91,  87],
        [ 85,  89,  84],
        [101, 105, 100]],

       [[ 68,  71,  69],
        [ 65,  71,  66],
        [ 66,  73,  68],
        ...,
        [ 97,  98,  94],
        [100, 104,  99],
        [108, 112, 107]]], shape=(2848, 4272, 3), dtype=uint8)
orig_shape: (2848, 4272)
path: 'D:\\cityscape\\road_loane.jpg'
probs: None
save_dir: 'D:\\cityscape\\runs\\segment\\predict2'
speed: {'preprocess': 3.403100010473281, 'inference': 50.02909997710958, 'postprocess': 3.70850000763312}]