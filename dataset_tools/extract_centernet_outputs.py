import os
import numpy as np
import matplotlib.pyplot as plt

# This is for a single prediction
"""prediction_file = '/media/keyi/Data/Research/traffic/detection/PointCenterNet_project/experiments/HG-104_detrac_001/demo/HG104_detrac_001_Hwy7_2k.txt'

with open(prediction_file, 'r') as f_in:
    predictions = f_in.readlines()


area_list = []
vehicle_counter = 0
for line in predictions:
    elements = line.strip('\n').split(',')
    frame_id, counter, x1, y1, w, h, score = elements
    vehicle_counter += 1
    area_list.append(np.sqrt(float(w) * float(h)))


# show the histogram of coefficients
fig = plt.figure()
plt.hist(area_list, bins=300, color='c', edgecolor='k', alpha=0.5)
plt.axvline(np.mean(area_list), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(area_list) * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.mean(area_list)))
plt.xlabel('Square root of BBox area')
plt.title('CenterNet predicted bbox sizes on Hwy7-2k with detrac-finetuned')
plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/coco/centernet_pred_obj_sizes_Hwy7_2k_detrac_ft.jpg')
plt.show()"""


# This is for a folder of a list of prediction files, like DETRAC
root_folder = '/media/keyi/Data/Research/traffic/detection/PointCenterNet_project/experiments/HG-104_exp_001/demo/eval_results/pruned'

file_list = os.listdir(root_folder)
area_list = []
vehicle_counter = 0
for file_name in file_list:
    prediction_file = os.path.join(root_folder, file_name)

    with open(prediction_file, 'r') as f_in:
        predictions = f_in.readlines()

    for line in predictions:
        elements = line.strip('\n').split(',')
        frame_id, counter, x1, y1, w, h, score = elements
        vehicle_counter += 1
        area_list.append(np.sqrt(float(w) * float(h)))

np.save('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/detrac/detrac_coco_bboxes_pruned.npy', area_list)

# show the histogram of coefficients
# fig = plt.figure()
# plt.hist(area_list, bins=300, color='c', edgecolor='k', alpha=0.5)
# plt.axvline(np.mean(area_list), color='k', linestyle='dashed', linewidth=1)
# min_ylim, max_ylim = plt.ylim()
# plt.text(np.mean(area_list) * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.mean(area_list)))
# plt.xlabel('Square root of BBox area')
# plt.title('CenterNet predicted bbox sizes on detrac with finetuning')
# plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/detrac/centernet_pred_obj_sizes_detrac_ft.jpg')
#
# plt.show()


