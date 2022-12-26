import cv2
import numpy as np

vidcap = cv2.VideoCapture('bike.mp4')
success,image = vidcap.read()
count = 0
success = True
frames = []
while success:
    frames.append(image)
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file if you want
    success,image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1
print(count, " frames extracted")
frames = np.array(frames)
print("data shape =\t", frames.shape)

def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# downsample
from scipy import ndimage
ds_frames = ndimage.interpolation.zoom(frames,(1., 0.2, 0.2, 1.))
ds_frames = rgb2gray(ds_frames)
print("downsampled shape =\t", ds_frames.shape)
np.save("bike_frames_data.npy", ds_frames)
print("downsampled frames stored to frames_data.npy")