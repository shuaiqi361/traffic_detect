import os
import sys
import numpy as np
from dataset_tools.aic.utils import parseTXT, parseXLSX, compute_nwRMSE, compute_efficiency_score
from dataset_tools.aic import videoInfo, camera_info


old_file = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/2080_track_all_test1/track_all.txt'
new_file = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/2080_track_all_test1/track_all_corrected.txt'

with open(old_file, 'r') as f_pred:
    pred_lines = f_pred.readlines()

reverse_video_id = {}
for k, v in camera_info.items():
    videos_ = v['video_id']
    for k_, v_ in videos_.items():
        reverse_video_id[v_] = k_

print(reverse_video_id)

f = open(new_file, 'a')
for line in pred_lines:
    elements = line.strip('\n').split(' ')
    # print(elements)
    video_id = int(elements[1])
    frame_id = int(elements[2])
    moi_id = int(elements[3])
    class_id = int(elements[4])
    if class_id == 7:
        class_id = 2
    else:
        class_id = 1

    new_line = '{} {} {} {} {}\n'.format(elements[0], elements[1], elements[2], elements[3], str(class_id))
    # print(new_line)
    f.write(new_line)

f.close()
