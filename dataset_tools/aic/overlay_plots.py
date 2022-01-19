import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import cv2
from scipy import ndimage
import os
import random


data_folder = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/' \
              'AIC21_Track1_Vehicle_Counting/Dataset_A'
frame_folder = 'cam_11_frames'
output_folder = 'cam_11_frames_plot_overlay'
Xs = []
Ys = []

input_frames = os.listdir(os.path.join(data_folder, frame_folder))

for id in range(len(input_frames)):
    if id % 20 == 0:  # maybe you don't need this, you want to show every frame
        image_name = 'img{:04d}.png'.format(id + 1)
        # data = image.imread(os.path.join(os.path.join(data_folder, frame_folder), image_name))
        # rotated_img = ndimage.rotate(data, 180)
        data = cv2.imread(os.path.join(os.path.join(data_folder, frame_folder), image_name))[:, :, ::-1]
        data = cv2.flip(cv2.rotate(data, cv2.ROTATE_180), 1)

        # to draw a point on co-ordinate (200,300)
        Xs.append(id * 1. / len(input_frames) * 1920)
        # print(Xs[-1])
        Ys.append(random.randint(70, 400))  # your accumulated motion of each frame
        # plt.bar(Xs, Ys, width=int(1920 // len(input_frames)), color="white")

        plt.plot(Xs, Ys, marker='o', color="white", markersize=6, alpha=0.5)
        plt.imshow(data, origin='lower')

        # plt.xlabel('Frame ID')
        # plt.ylabel('Accumulated Motion')
        # plt.xlim(0, 3000)
        # plt.legend(['Key frame'])
        plt.show()
















