import os
import numpy as np
import matplotlib.pyplot as plt

root_path = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/detrac'
detrac_gt_hist = np.load(os.path.join(root_path, 'detrac_gt_bboxes.npy'))
detrac_coco_pred_hist = np.load(os.path.join(root_path, 'detrac_coco_bboxes.npy'))
detrac_finetune_pred_hist = np.load(os.path.join(root_path, 'detrac_finetuned_bboxes.npy'))
detrac_coco_pred_hist_pruned = np.load(os.path.join(root_path, 'detrac_coco_bboxes_pruned.npy'))

fig = plt.figure()
plt.hist(detrac_gt_hist, bins=200, color='g', alpha=0.5)
plt.hist(detrac_coco_pred_hist_pruned, bins=200, color='r', alpha=0.5)
plt.legend(['GT Labels', 'Prediction'])
plt.xlabel('Square root of BBox area')
plt.title('CenterNet predicted(pruned) with finetuning on DETRAC Test')
plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/detrac/centernet_pred_detrac_ft_2hist_pruned.jpg')

plt.show()







