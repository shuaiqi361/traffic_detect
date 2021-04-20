import numpy as np
import cv2
import os
import time


def select_key_frame(frame1, frame2, threshold=0.002, downsample=False, use_sum=False, ):
    """
    :param threshold: for motion detection
    :param frame1: numpy array (H, W, 3) or (H, W), color or grey image
    :param frame2: numpy array (H, W, 3) or (H, W)
    :param downsample:
    :param use_sum:
    :return:
    """
    assert frame1.shape == frame2.shape
    width = int(frame1.shape[1])
    height = int(frame1.shape[0])

    if downsample:
        frame1 = cv2.resize(frame1, dsize=(width // 2, height // 2))
        frame2 = cv2.resize(frame2, dsize=(width // 2, height // 2))
    if len(frame1.shape) == 3:
        diff_img = np.sum(np.abs(frame1 - frame2), axis=2).astype(np.uint8)
    elif len(frame1.shape) == 2:
        diff_img = np.abs(frame1 - frame2).astype(np.uint8)
    else:
        print('The input frames should be numpy array (H, W, 3) or (H, W)')
        raise NotImplementedError

    # bin_diff_img = (diff_img > threshold) * 1.
    if use_sum:
        global_max = np.sum(diff_img)
    else:
        diff_img = cv2.GaussianBlur(diff_img, (51, 51), 0)
        global_max = np.max(diff_img)

    return global_max


def main():
    root_folder = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/AIC21_Track1_Vehicle_Counting/Dataset_A/cam_11_frames'
    img_1 = cv2.imread(os.path.join(root_folder, 'img0001.png'))
    img_2 = cv2.imread(os.path.join(root_folder, 'img0002.png'))

    time_start = time.time()
    max_diff = select_key_frame(img_1, img_2)
    time_end = time.time()
    print('Runtime: {}, max difference: {}'.format(time_end - time_start, max_diff))


if __name__ == "__main__":
    main()














