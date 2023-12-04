from algorithm.object_detector import YOLOv7
from utils.detections import draw
from lidar2camera import LiDAR2Camera
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import open3d as o3d #opencv 와 같이 point data들을 다룰 수 있도록 도와주는 패키지
import numpy as np
import glob
import streamlit as st
import wget
from PIL import Image
import torch
import cv2
import os
import sys
import time

st.set_page_config(layout="wide")

cfg_model_path = ""
model = None
confidence = .25

def draw_3d_plot(frame,dataset_velo, points=0.2):

    f = plt.figure(figsize=(20, 10))
    axis = f.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])
    axis.set_facecolor('white')
    
    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]
    axis.scatter(*np.transpose(velo_frame[:, [1,0, 2]]), s=point_size, c='black', cmap='white')

    axis.set_xlim3d((-10,30))
    axis.set_ylim3d((-20,20))
    axis.set_zlim3d((-2,10))
    
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    width, height = f.canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    return image_array ,width, height


def webcam_input(data_src, yolov7):

    if data_src == 'Webcam data':
        st.sidebar.warning("Using Webcam Input")

    cap = cv2.VideoCapture(0) 

    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

    fps = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")

    st.markdown("---")
    output = st.empty()
    prev_time = 0
    curr_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("# Error reading from webcam")
            break

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = yolov7.detect(frame, track=True)
        detected_frame,box_range = draw(frame, detections)
        output.image(detected_frame, channels="RGB", use_column_width=True)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")

    cap.release()
    if stor_video is not None:
        stor_video.release()


def camera_input(data_src, yolov7,conf):
    vid_file = None
    stor_video = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    elif data_src == 'Upload your own data':
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())
    else:
        webcam_input(data_src, yolov7)

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0

        if st.sidebar.button('Save Video'):
            f = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            stor_video = cv2.VideoWriter(f'output_videos/{vid_file.split("/")[-1][:-4]}_output.mp4', fourcc, f, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("# End of video")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detections = yolov7.detect(frame,track=True)
            # print(detections)
            detected_frame,boxes_range = draw(frame, detections)
            output.image(detected_frame, channels="RGB", use_column_width=True)
            
            if stor_video:
                stor_video.write(cv2.cvtColor(detected_frame,  cv2.COLOR_BGR2RGB))

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        # cap.release()


def show_lidar():

    folder_path = '/Users/hyundolee/data_story/Auto Drive Project/yolov7_streamlit/data/velodyne_0000'
    
    fps = 0
    width = 0
    height = 0

    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")
    
    st.markdown("---")
    output_show = st.empty()
    prev_time = 0
    curr_time = 0

    point_files = sorted(glob.glob(folder_path+"/*.pcd"))

    dataset_velo = list()

    for index in range(len(point_files)):
        pcd_file = point_files[index]
        cloud = o3d.io.read_point_cloud(pcd_file)
        points= np.asarray(cloud.points)
        
        dataset_velo.append(points)


    frames = []
    n_frames = len(point_files)

    print('Preparing animation frames...')
    for i in range(n_frames):

        lidar_frame,width, height = draw_3d_plot(i,dataset_velo)
        # print(width, height)
        output_show.image(lidar_frame)#, use_column_width=True)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")




def Lidar2Camera(calib_file, video_images, video_points,yolov7,conf):

    l2c = LiDAR2Camera(calib_file)

    image = cv2.imread(video_images[0])
    h,w,c = image.shape
    fps = 0

    custom_size = st.sidebar.checkbox("Custom frame size")
    if custom_size:
        w = st.sidebar.number_input("Width", min_value=120, step=20, value=w)
        h = st.sidebar.number_input("Height", min_value=120, step=20, value=h)

    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{h}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{w}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")
    
    st.markdown("---")
    output_lidar = st.empty()
    bicycle_caution = st.empty()
    person_caution = st.empty()
    prev_time = 0
    curr_time = 0
    
    for idx, img in enumerate(video_images):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)

        #yolo detection
        detections = yolov7.detect(image, track=True)

        #lidar2camera
        img, center_list = l2c.show_lidar_on_image(point_cloud, image, detections, debug="True")
        detected_frame,box_range = draw(img, detections)

        box_class = {o[3]:o[2] for o in center_list}

        output_lidar.image(detected_frame, channels="RGB", use_column_width=True)
        
        if 'bicycle' in box_class.keys() and box_class['bicycle'] < 20:
            bicycle_caution.write("# 🚫 Be Detected Bicycle 🚲 Be careful!")
        if 'person' in box_class.keys() and box_class['person'] < 20:
            person_caution.write("# 🚫 Be Detected Person🧍 Be careful!")
        else:
            bicycle_caution.write("# 👷‍♂️ Now safety")
            person_caution.write("# 👷‍♂️ Now safety")

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st3_text.markdown(f"**{fps:.2f}**")




#===================   main   ===================

def main():
    video_images = sorted(glob.glob("data/image_0000/*.png"))
    video_points = sorted(glob.glob("data/velodyne_0000/*.pcd"))
    calib_files = sorted(glob.glob("data/calib/*.txt"))
    
    # global variables
    global model, confidence, cfg_model_path

    st.title("Perception for Autonomous Driving")
    

    st.sidebar.title("Settings")

    # upload model
    model_src = st.sidebar.radio("**Select yolov7 weight**", ["yolov7 tiny model", "yolov7 base model",'yolov7 kitti base',"yolov7 kitti tiny","Use your own model"])
    
    # URL, upload file (max 200 mb)
    if model_src == "Use your own model":
        user_model_path = get_user_model()
        if user_model_path:
            cfg_model_path = user_model_path

        st.sidebar.text(cfg_model_path.split("/")[-1])
        st.sidebar.markdown("---")
    elif model_src == "yolov7 tiny model":
        cfg_model_path = "weights/yolov7-tiny.pt"
    elif model_src == "yolov7 kitti base":
        cfg_model_path = "weights/yolov7_kitti_base.pt"
    elif model_src == "yolov7 kitti tiny":
        cfg_model_path = "weights/yolov7_kitti_tiny.pt"
    else:
        cfg_model_path = "weights/yolov7.pt"

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please added to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        # load model
        yolov7 = YOLOv7(conf_thres=confidence)
        yolov7.load(cfg_model_path, classes='data.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Select input type: ", ['camera', 'lidar', 'camera & lidar'])

        if input_option == 'camera':
            # input src option
            data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data','webcam'])

            st.subheader("object detection tracking by camera")
            camera_input(data_src, yolov7,confidence)
        elif input_option == 'lidar':
            # input src option
            data_src = st.sidebar.radio("Select input source: ", ['Sample data'])

            st.subheader("visualize lidar point cloud")
            show_lidar()
        elif input_option == 'camera & lidar':
            # input src option
            data_src = st.sidebar.radio("Select input source: ", ['Sample data'])
            
            st.subheader("camera & lidar calibration")
            Lidar2Camera(calib_files[0], video_images, video_points,yolov7,confidence)



if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
