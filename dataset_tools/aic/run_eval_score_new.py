import os
import sys
import numpy as np
from dataset_tools.aic.utils import parseTXT, parseXLSX, compute_nwRMSE, compute_efficiency_score
from dataset_tools.aic import videoInfo


NUM_SEGMENTS = 20
GT_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results/gt'
PRED_FILE = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/track1.txt'  # should be the outputs .txt files
BASE_FACTOR = 0.428572  # Yufei's hardware
total_execution_time = 77.126

gt_filenames = os.listdir(GT_FOLDER)
with open(PRED_FILE, 'r') as f_pred:
    pred_lines = f_pred.readlines()

# videos_to_be_eval = []  # evaluate predicted counts with existing gt
# for pred in pred_filenames:
#     if pred.split('.')[-1] == 'txt':
#         video_name = pred.split('.')[0]
#         if video_name in videoInfo.keys() and video_name + '.xlsx' in gt_filenames:
#             videos_to_be_eval.append(video_name)

# num_video = len(videos_to_be_eval)
# nwRMSE_perVideo = np.zeros(num_video)
# vehicleCnt_perVideo = np.zeros(num_video)
# video_times = []


for line in pred_lines:
    elements = line.strip('\n').split(',')


for vId in range(num_video):
    video_name = videos_to_be_eval[vId]
    print(video_name)
    video_info = videoInfo[video_name]
    num_frame = min(video_info["frame_num"], 3000)  # max 3000 frames are annotated
    num_movement = video_info["movement_num"]
    num_vehicle_type = 2  # car and freight-truck
    video_duration = video_info["frame_num"] / video_info["fps"]
    video_times.append(video_duration)

    # parse gt and pred
    gt_counts = np.zeros((num_frame, num_movement, num_vehicle_type))
    gt_file_path = os.path.join(GT_FOLDER, video_name + '.xlsx')
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    else:
        parseXLSX(gt_file_path, gt_counts)

    pred_counts = np.zeros((num_frame, num_movement, num_vehicle_type))
    pred_file_path = os.path.join(PRED_FOLDER, video_name + '.txt')
    if not os.path.exists(pred_file_path):
        raise FileNotFoundError
    else:
        parseTXT(pred_file_path, pred_counts)
    print(np.sum(np.sum(pred_counts, axis=0), axis=1))
    continue

    nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
    vehicleCnt_perVideo[vId] = np.sum(vehicleCnt)
    nwRMSE_perVideo[vId] = np.sum(nwRMSE * vehicleCnt) / vehicleCnt_perVideo[vId]

    print('Effectiveness score of video {}: {:.6f}'.format(video_name, nwRMSE_perVideo[vId]))

effectiveness_score = np.sum(nwRMSE_perVideo * vehicleCnt_perVideo) / np.sum(vehicleCnt_perVideo)
print('Overall effectiveness score: {:.6f}'.format(effectiveness_score))

total_frames = 0
for k, v in videoInfo.items():
    total_frames += v['frame_num']

efficiency_score = compute_efficiency_score([total_execution_time], [np.sum(video_times)], BASE_FACTOR)
print('Overall efficiency score: {:.6f}'.format(efficiency_score))
print('Overall fps {}, {} ms. per frame'.format())

final_eval_score = 0.3 * efficiency_score + 0.7 * effectiveness_score

print('Final evaluation score: {:.6f}'.format(final_eval_score))




