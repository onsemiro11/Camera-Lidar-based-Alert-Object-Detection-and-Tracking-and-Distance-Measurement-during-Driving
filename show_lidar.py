from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import open3d as o3d
import numpy as np
from glob import glob
import cv2
import sys
import os

# Print iterations progress


def draw_3d_plot(frame,dataset_velo, points=0.2):
    """
    Saves a single frame for an animation: a 3D plot of the lidar data without ticks and all frame trackelts.
    Parameters
    ----------
    frame           : Absolute number of the frame.
    dataset_velo    : `raw` dataset list
    points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.

    Returns
    -------
    Saved frame filename.
    """

    f = plt.figure(figsize=(12, 8))
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
    
    plt.gca().set_aspect('equal', adjustable='box')  # 종횡비를 조절하여 정확한 각도 표현
    canvas = FigureCanvasAgg(f)
    canvas.draw()
    width, height = f.canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)

    # Display the NumPy array (optional)
    plt.imshow(image_array)
    plt.show()
    plt.close()

    return image_array

    # filename = 'frame_{0:0>4}.png'.format(frame)
    # plt.savefig(filename)
    # plt.close(f)
    # return filename



folder_path = '/Users/hyundolee/data_story/Auto Drive Project/yolov7_streamlit/data/velodyne'

point_files = sorted(glob(folder_path+"/*.pcd"))
print(point_files)
dataset_velo = list()

for index in range(len(point_files)):
  point_page = []
  pcd_file = point_files[index]
  cloud = o3d.io.read_point_cloud(pcd_file)
  points= np.asarray(cloud.points)
  
  dataset_velo.append(points)  


frames = []
n_frames = len(point_files)

print('Preparing animation frames...')
for i in range(n_frames):
    print(i)
    print_progress(i, n_frames - 1)

    draw_3d_plot(i,dataset_velo)
    # filename = draw_3d_plot(i,dataset_velo)
    # plt.show()
    # frames += [filename]
print('...Animation frames ready.')

# clip = ImageSequenceClip(frames, fps=5)

# clip.write_gif('pcl_data.gif', fps=5)
