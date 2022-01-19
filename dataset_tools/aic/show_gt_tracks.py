import json
import os
import random

import numpy as np
from dataset_tools.aic import videoInfo, camera_info
import cv2

# camera_name = 'cam_1'
dataset_root = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/AIC21_Track1_Vehicle_Counting'
Baidu_root = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results'
file_to_save = os.path.join(dataset_root, 'camera_config/camera_config_w_ours.json')

annotation_root = '/media/keyi/Data/Research/traffic/data/AIC2021/tracks_labelling_gui-master'

num_cameras = 25
num_points_max_per_track = 50
pixel_threshold = 10.


def track_resample(track, threshold=50):
    """
    :param track: input track numpy array (M, 2)
    :param threshold: default 20 pixel interval for neighbouring points
    :return:
    """
    assert track.shape[1] == 2
    accum_dist = 0
    index_keep = [0]
    for i in range(1, track.shape[0]):
        dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
        if dist_ >= 9:
            accum_dist += dist_
            if accum_dist >= threshold:
                index_keep.append(i)
                accum_dist = 0

    return track[index_keep, :]


# loop over all videos
for v in range(num_cameras):
    if v < 3:
        continue
    camera_name = 'cam_{}'.format(v + 1)  # camera id start from 1
    print(camera_name)

    anno_file = os.path.join(annotation_root, 'annotation/{}.csv'.format(camera_name))
    if not os.path.exists(anno_file):
        continue

    num_moi = camera_info[camera_name]['movement_num']
    camera_info[camera_name]['moi'] = {}
    img = cv2.imread(os.path.join(annotation_root, '{}.jpg'.format(camera_name)))
    # img_ = cv2.imread(os.path.join(annotation_root, '{}.jpg'.format(camera_name)))
    # img_ = cv2.imread('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/slides/gt_tracks_trim.png')

    # adding in our annotations:
    with open(anno_file, 'r') as f_csv:
        lines = f_csv.readlines()

    skip_first = True
    pre_track_id = -1
    pre_moi_id = -1
    if_the_first_to_show = True
    tr_ = []
    for line in lines:
        if skip_first:
            skip_first = False
            continue
        elements = line.strip('\n').split(',')
        frame_id = int(elements[1])
        track_id = int(elements[2])
        x, y, w, h = float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])
        moi_id = int(elements[-1])

        # print(frame_id, track_id, x, y, w, h, moi_id)
        # img = img_.copy()
        if track_id != pre_track_id and pre_track_id != -1:  # the last track ends
            if pre_moi_id == 9:
                # t = tr_
                t = track_resample(np.array(tr_))
                # t[:, 0] = t[:, 0] - 105
                # cv2.polylines(img, [t.reshape(-1, 2).astype(np.int32)], isClosed=False,
                #               color=(255, 255, 75), thickness=2)
                cv2.polylines(img, [t.reshape(-1, 2).astype(np.int32)], isClosed=False,
                              color=(255, 0, 75), thickness=2)
                #
                # print("{} points on this track.".format(len(t)))
                # cv2.putText(img, str(pre_moi_id), (int(t[-1][0]), int(t[-1][1])), cv2.FONT_HERSHEY_SIMPLEX,
                #                     2, (255, 100, 255), 3, cv2.LINE_AA)

                for pt in t:  # all the points for each trajectory
                    cv2.circle(img, (int(pt[0]), int(pt[1])), radius=5, color=(255, 0, 55), thickness=-1)
                # for pt in t:  # all the points for each trajectory
                #     cv2.circle(img, (int(pt[0]), int(pt[1])), radius=4, color=(255, 255, 75), thickness=-1)

                cv2.circle(img, (int(t[0][0]), int(t[0][1])), radius=8, color=(0, 250, 0), thickness=-1)
                cv2.circle(img, (int(t[-1][0]), int(t[-1][1])), radius=8, color=(0, 0, 250), thickness=-1)

                cv2.imshow('ROI MOI image', img)
                if cv2.waitKey() & 0xFF == ord('q'):
                    exit()

            tr_ = []  # start a new track
        else:
            new_pt = [x + w / 2, y + h / 2]
            if len(tr_) > 0:
                if np.sqrt((tr_[-1][0] - new_pt[0]) ** 2
                           + (tr_[-1][1] - new_pt[1]) ** 2) > pixel_threshold:
                    tr_.append(new_pt)
            else:
                tr_.append(new_pt)

        pre_track_id = track_id
        pre_moi_id = moi_id

    # print(camera_info[camera_name])

with open(file_to_save, 'w') as f_out:
    json.dump(camera_info, f_out)
