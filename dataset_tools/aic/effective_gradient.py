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
video_duration = int(video_info["frame_num"] / video_info["fps"])

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
print('Total estimated counts: {} vs, gt counts {}'.format(np.sum(pred_counts), np.sum(gt_counts)))
# diff_counts = gt_counts - pred_counts
# num_miss = int(np.sum(diff_counts > 0))
num_add = 50
num_run = 50
all_nwRMSEs_perVideo = 0
all_gradients = []
for run in range(num_run):
    pred_counts = copy.deepcopy(gt_counts)
    all_nwRMSE_curves = []
    nwRMSEs_perVideo = []
    # current scores before perturbations
    nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
    nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
    nwRMSEs_perVideo.append(nwRMSE_score)
    print('Total gt counts: ', np.sum(gt_counts))
    print('Effectiveness score of video {} before perturbation: {:.6f}'.format(video_name, nwRMSE_score))

    for i in range(0, num_add):
        print('adding the {}-th hits.'.format(i + 1))
        add_hit_to_counts(pred_counts, gt_counts, num_hit=1)
        nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
        nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
        nwRMSEs_perVideo.append(nwRMSE_score)

    gradients = []
    for g in range(len(nwRMSEs_perVideo) - 1):
        delta_ = nwRMSEs_perVideo[i + 1] - nwRMSEs_perVideo[i]
        gradients.append(delta_)

    all_nwRMSEs_perVideo += np.array(nwRMSEs_perVideo)
    all_gradients.append(np.mean(gradients))

print('gradient of the curve: ', np.mean(all_gradients))
fig = plt.figure()
plt.plot(range(0, len(nwRMSEs_perVideo)), all_nwRMSEs_perVideo / num_run, color='red', marker='^', linewidth=2, markersize=5)
plt.title('Effectiveness score vs. Num FPs')
plt.xlabel('Number of False Positives')
plt.ylabel('nwRMSE Score')
plt.show()


# # This is to remove false positives in the predicted counts
# diff_counts = gt_counts - pred_counts
# print('Total estimated counts: {} vs, gt counts {}'.format(np.sum(pred_counts), np.sum(gt_counts)))
# # num_miss = int(np.sum(diff_counts < 0))
# num_miss = 50
# num_run = 50
# all_nwRMSEs_perVideo = 0
# all_gradients = []
# for run in range(num_run):
#     pred_counts = copy.deepcopy(gt_counts)
#     all_nwRMSE_curves = []
#     nwRMSEs_perVideo = []
#     # current scores before perturbations
#     nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
#     nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
#     nwRMSEs_perVideo.append(nwRMSE_score)
#     print('Total gt counts: ', np.sum(gt_counts))
#     print('Effectiveness score of video {} before perturbation: {:.6f}'.format(video_name, nwRMSE_score))
#
#     for i in range(0, num_miss):
#         print('removing the {}-th hits.'.format(i + 1))
#         remove_hit_to_counts(pred_counts, gt_counts, num_hit=1)
#         nwRMSE, vehicleCnt = compute_nwRMSE(NUM_SEGMENTS, pred_counts, gt_counts)
#         nwRMSE_score = np.sum(nwRMSE * vehicleCnt) / np.sum(vehicleCnt)
#         nwRMSEs_perVideo.append(nwRMSE_score)
#
#     gradients = []
#     for g in range(len(nwRMSEs_perVideo) - 1):
#         delta_ = nwRMSEs_perVideo[i + 1] - nwRMSEs_perVideo[i]
#         gradients.append(delta_)
#
#     all_nwRMSEs_perVideo += np.array(nwRMSEs_perVideo)
#     all_gradients.append(np.mean(gradients))
#
# print('gradient of the curve: ', np.mean(all_gradients))
# fig = plt.figure()
# plt.plot(range(0, len(nwRMSEs_perVideo)), all_nwRMSEs_perVideo / num_run, color='red', marker='^', linewidth=2, markersize=5)
# plt.title('Effectiveness score vs. Num FNs')
# plt.xlabel('Number of False Negatives')
# plt.ylabel('nwRMSE Score')
# plt.show()


# effectiveness_score = np.sum(nwRMSE_perVideo * vehicleCnt_perVideo) / np.sum(vehicleCnt_perVideo)
# print('Overall effectiveness score: {:.6f}'.format(effectiveness_score))

# total_execution_time = 6789.
# total_execution_time = video_duration
# total_execution_time = 1800 + 27000 / 15 + 300 + 14400 / 8
# video_duration = total_execution_time
# efficiency_score = compute_efficiency_score([total_execution_time], [video_duration], BASE_FACTOR)
# print('Overall efficiency score: {:.6f}'.format(efficiency_score))
# all_efficiency_scores = [efficiency_score]
# interv = 50
# for i in range(25):
#     total_execution_time += interv
#     all_efficiency_scores.append(compute_efficiency_score([total_execution_time], [video_duration], BASE_FACTOR))
#
# gradients = []
# for g in range(len(all_efficiency_scores) - 1):
#     delta_ = (all_efficiency_scores[i + 1] - all_efficiency_scores[i]) / interv
#     gradients.append(delta_)
#
# print('gradient of the curve: ', np.mean(gradients))
# print('gradient of the curve: ', (all_efficiency_scores[-1] - all_efficiency_scores[0]) / 25 / (interv - 1))
# print('Theoretical gradient: ', BASE_FACTOR / -1.1 / total_execution_time)
# fig = plt.figure()
# plt.plot(range(int(video_duration), int(video_duration + interv * 25 + 1), int(interv)), all_efficiency_scores, color='blue', marker='*', linewidth=2, markersize=5)
# plt.title('Efficiency score vs. Execution Time')
# plt.xlabel('Execution Time')
# plt.ylabel('Score')
# plt.show()


# fig = plt.figure()
# plt.plot(range(0, len(nwRMSEs_perVideo)), all_nwRMSEs_perVideo / num_run, color='red', marker='^', linewidth=2, markersize=5)
# plt.title('Effectiveness score vs. Num FNs')
# plt.xlabel('Number of False Negatives')
# plt.ylabel('nwRMSE Score')
# plt.show()

# final_eval_score = 0.3 * efficiency_score + 0.7 * effectiveness_score
#
# print('Final evaluation score: {:.6f}'.format(final_eval_score))




