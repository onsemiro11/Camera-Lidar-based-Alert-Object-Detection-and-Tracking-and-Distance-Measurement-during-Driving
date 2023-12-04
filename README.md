
# Camera & Lidar based Alert Object Detection and Distance Measurement during Driving
A web interface for Sensor Fusion, Camera - Lidar Calibration and object detection & distance Measurement using streamlit.

It supports CPU and GPU inference, supports both images and videos and uploading your own custom models. but, up to now, you can't use your own lidar file.. I will setup it.

<img width="800" alt="image" src="https://github.com/onsemiro11/-Camera-and-Lidar-based-Alert-Object-Detection-and-Distance-Measurement-during-Driving/assets/49609175/18b32912-0a9d-47c9-9faa-df5be51dc18f">

## Index
1. Introduction
2. workflow
3. object detection (YOLO V7) Training
4. Implement our web interface (streamlit)


# 1. Introduction

When driving in real life, there are many accidents that occur with people who cannot be seen from the human perspective, kickboard users, or bicycle or motorcycle users. In this regard, we developed a service that detects objects requiring caution through camera and lidar sensors, calculates the distance, and notifies people to be careful when the object is detected within a certain distance. Detects an object through a camera and obtains the distance value through the lidar point cloud belonging to the detected object. Through this, the goal was to show the distance of the currently detected object and inform of the dangerous situation. The project was carried out with the sub-goal of incorporating camera-lidar calibration and tracking technology necessary for this process.

# 2. Workflow

<img width="600" alt="image" src="https://github.com/onsemiro11/-Camera-and-Lidar-based-Alert-Object-Detection-and-Distance-Measurement-during-Driving/assets/49609175/56493549-6720-4c33-b1f4-2a9343721c35">

# 3. object detection (YOLO V7)

https://github.com/WongKinYiu/yolov7
Please, refer to this github.


### git clone and install requirements file

``` shell
# git clone yolov7
git clone https://github.com/WongKinYiu/yolov7

# install required packages
cd /yolov7
pip install -r requirements.txt
```

### Setting your data set format. (data.yaml, folder)

### Training

``` shell
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```

Train and get own model weights file.

# 4. Implement our web interface (streamlit)

## How to run
After cloning the repo:
1. Install requirements
   - `pip install -r requirements.txt`
2. Add sample images to `data/sample_images`
3. Add sample video to `data/sample_videos` and call it `sample.mp4` or change name in the code.
4. Add the model file to `models/` and change `cfg_model_path` to its path.
```bash
git clone https://github.com/onsemiro11/Camera & Lidar based Alert Object Detection and Distance Measurement during Driving
cd Camera & Lidar based Alert Object Detection and Distance Measurement during Driving
streamlit run app.py
```

## References
https://discuss.streamlit.io/t/deploy-yolov5-object-detection-on-streamlit/27675

https://github.com/theos-ai/easy-yolov7

https://github.com/moaaztaha/Yolo-Interface-using-Streamlit

