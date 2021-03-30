import os
import matplotlib.pyplot as plt
import numpy as np
from dataset_tools.aic.utils import parseTXT, parseXLSX, compute_nwRMSE, compute_efficiency_score, \
    add_hit_to_counts, remove_hit_to_counts
from dataset_tools.aic import videoInfo
import copy


NUM_SEGMENTS = 10
GT_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results/gt'
PRED_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results/yufei/report/baidu_result'
BASE_FACTOR = 0.428572  # Yufei's hardware

gt_filenames = os.listdir(GT_FOLDER)
pred_filenames = os.listdir(PRED_FOLDER)

pred_file = 'cam_7.txt'
pred_file_path = os.path.join(PRED_FOLDER, pred_file)
gt_file_path = os.path.join(GT_FOLDER, pred_file.split('.')[0] + '.xlsx')

video_name = pred_file.split('.')[0]
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

pred_counts = np.zeros((num_frame, num_movement, num_vehicle_type))
pred_file_path = os.path.join(PRED_FOLDER, video_name + '.txt')
if not os.path.exists(pred_file_path):
    raise FileNotFoundError
else:
    parseTXT(pred_file_path, pred_counts)

# This is to add false negatives to predicted counts
# diff_counts = gt_counts - pred_counts
# num_miss = int(np.sum(diff_counts > 0))
# num_miss = 10
#
# all_nwRMSE_curves = []
# nwRMSEs_perVideo = []
# # current scores before perturbations
# nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
# nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
# nwRMSEs_perVideo.append(nwRMSE_score)
#
# print('Effectiveness score of video {} before perturbation: {:.6f}'.format(video_name, nwRMSE_score))
#
# for i in range(1, num_miss):
#     pred_counts = add_hit_to_counts(pred_counts, gt_counts, num_hit=1)
#     nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
#     nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
#     nwRMSEs_perVideo.append(nwRMSE_score)
#
# fig = plt.figure()
# plt.plot(range(0, len(nwRMSEs_perVideo)), nwRMSEs_perVideo, color='red', marker='^', linewidth=2, markersize=5)
# plt.title('Effectiveness score vs. Num TPs added')
# # plt.legend([r'$\lambda = 0.01$', r'$\lambda = 0.05$', r'$\lambda = 0.1$', r'$\lambda = 0.2$', r'$\lambda = 0.5$'])
# plt.xlabel('Number of True Positives added')
# plt.ylabel('Effectiveness')
# plt.show()


# This is to remove false positives in the predicted counts
diff_counts = gt_counts - pred_counts
num_miss = int(np.sum(diff_counts < 0))
num_miss = 50

all_nwRMSE_curves = []
nwRMSEs_perVideo = []
# current scores before perturbations
nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
nwRMSEs_perVideo.append(nwRMSE_score)

print('Effectiveness score of video {} before perturbation: {:.6f}'.format(video_name, nwRMSE_score))

for i in range(1, num_miss):
    if i % 2 == 0:
        pred_counts = remove_hit_to_counts(pred_counts, gt_counts, num_hit=1)
    else:
        pred_counts = add_hit_to_counts(pred_counts, gt_counts, num_hit=1)
    nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
    nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
    nwRMSEs_perVideo.append(nwRMSE_score)

fig = plt.figure()
plt.plot(range(0, len(nwRMSEs_perVideo)), nwRMSEs_perVideo, color='red', marker='^', linewidth=2, markersize=5)
plt.title('Effectiveness score vs. Num FP. removed')
plt.xlabel('Number of True Positives added')
plt.ylabel('Effectiveness')
plt.show()



# effectiveness_score = np.sum(nwRMSE_perVideo * vehicleCnt_perVideo) / np.sum(vehicleCnt_perVideo)
# print('Overall effectiveness score: {:.6f}'.format(effectiveness_score))
#
# total_execution_time = 6789.
# efficiency_score = compute_efficiency_score([total_execution_time], [np.sum(video_times)], BASE_FACTOR)
# print('Overall efficiency score: {:.6f}'.format(efficiency_score))
#
# final_eval_score = 0.3 * efficiency_score + 0.7 * effectiveness_score
#
# print('Final evaluation score: {:.6f}'.format(final_eval_score))




