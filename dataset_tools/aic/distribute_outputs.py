import os
import sys
import numpy as np
from dataset_tools.aic.utils import parseTXT, parseXLSX, compute_nwRMSE, compute_efficiency_score
from dataset_tools.aic import videoInfo, camera_info


NUM_SEGMENTS = 50
GT_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results/gt'
PRED_FILE = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/2080_track_hist_1/track1_hist.txt'  # should be the outputs .txt files
OUTPUT_FOLDER = '//media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/2080_track_hist_1/tracks/'

gt_filenames = os.listdir(GT_FOLDER)
with open(PRED_FILE, 'r') as f_pred:
    pred_lines = f_pred.readlines()

if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

# videos_to_be_eval = []  # evaluate predicted counts with existing gt
# for pred in pred_filenames:
#     if pred.split('.')[-1] == 'txt':
#         video_name = pred.split('.')[0]
#         if video_name in videoInfo.keys() and video_name + '.xlsx' in gt_filenames:
#             videos_to_be_eval.append(video_name)

reverse_video_id = {}
for k, v in camera_info.items():
    videos_ = v['video_id']
    for k_, v_ in videos_.items():
        reverse_video_id[v_] = k_

print(reverse_video_id)
flag_array = [0] * 31
f = None
for line in pred_lines:
    elements = line.strip('\n').split(' ')
    video_id = int(elements[1])
    frame_id = int(elements[2])
    moi_id = int(elements[3])
    class_id = int(elements[4])
    # if class_id == 7:
    #     class_id = 2
    # else:
    #     class_id = 1

    if flag_array[video_id - 1] == 0:
        if f is not None:
            f.close()
        file_path = os.path.join(OUTPUT_FOLDER, "{}.txt".format(reverse_video_id[video_id]))
        f = open(file_path, 'w')
        flag_array[video_id - 1] = 1

    new_line = '{} {} {} {}\n'.format(video_id, frame_id, moi_id, class_id)
    f.write(new_line)




