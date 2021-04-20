import os
import sys
import numpy as np
from dataset_tools.aic.utils import parseTXT, parseXLSX, compute_nwRMSE, compute_efficiency_score
from dataset_tools.aic import videoInfo


NUM_SEGMENTS = 50
GT_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results/gt'
PRED_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results/eval_code/yufei_result'  # should be the outputs .txt files
BASE_FACTOR = 0.428572  # Yufei's hardware
total_execution_time = 6789.  # specify total execution time of all 4 videos from yufei_result

gt_filenames = os.listdir(GT_FOLDER)
pred_filenames = os.listdir(PRED_FOLDER)

videos_to_be_eval = []  # evaluate predicted counts with existing gt
for gt_ in gt_filenames:
    if gt_.split('.')[-1] == 'xlsx':
        video_name = gt_.split('.')[0]
        if video_name in videoInfo.keys():
            videos_to_be_eval.append(video_name)

videos_to_be_eval = np.sort(videos_to_be_eval)
num_video = len(videos_to_be_eval)
print('Total number of videos to be tested: ', num_video)
for vId in range(num_video):
    video_name = videos_to_be_eval[vId]
    # print(video_name)
    video_info = videoInfo[video_name]
    num_frame = min(video_info["frame_num"], 3000)  # max 3000 frames are annotated
    num_movement = video_info["movement_num"]
    num_vehicle_type = 2  # car and freight-truck
    video_duration = video_info["frame_num"] / video_info["fps"]

    # parse gt and pred
    gt_counts = np.zeros((num_frame, num_movement, num_vehicle_type))
    gt_file_path = os.path.join(GT_FOLDER, video_name + '.xlsx')
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    else:
        parseXLSX(gt_file_path, gt_counts)

    gt_counts = np.zeros((num_frame, num_movement, num_vehicle_type))
    gt_file_path = os.path.join(GT_FOLDER, video_name + '.xlsx')
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    else:
        parseXLSX(gt_file_path, gt_counts)

    counts_MOI = np.sum(np.sum(gt_counts, axis=0), axis=1)
    print(video_name)
    print(counts_MOI, np.sum(counts_MOI) / num_movement)
    temp = ''
    for it in counts_MOI:
        temp += str(it) + ','

    print(temp)


