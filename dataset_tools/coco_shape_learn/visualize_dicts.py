from pycocotools.coco import COCO
import numpy as np
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from dataset_tools.coco_utils.utils import close_contour

mask_size = 28
n_vertices = 360  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.01
contour_closed = True

n_atom_row = int(np.sqrt(n_coeffs))
n_atom_col = int(np.sqrt(n_coeffs))

# save_dict_root = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/sparse_shape_dict'
# out_dict = '{}/single_fromMask_Dict_m{}_n{}_v{}_a{:.2f}.npy'.format(save_dict_root, mask_size, n_coeffs, n_vertices, alpha)
save_dict_root = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/dictionary'
out_dict = '{}/train_scaled_ranked_dict_v{}_n{}_a{:.2f}.npy'.format(save_dict_root, n_vertices, n_coeffs, alpha)
learned_dict = np.load(out_dict)
# with np.load(out_dict) as data_meta:
#     learned_dict = data_meta['dictionary']
#     shape_mean = data_meta['mean']
#     shape_std = data_meta['std']

fig = plt.figure(figsize=(n_atom_row * 2, n_atom_col * 2))
# plt.title('A dictionary of 64 basis functions')
for i in range(n_coeffs):
    plt.subplot(n_atom_row, n_atom_col, i + 1)
    # plt.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    shape_basis = learned_dict[i, :].reshape((n_vertices, 2))
    if contour_closed:
        shape_basis = close_contour(shape_basis)
    max_x = np.max(np.abs(shape_basis[:, 0])) * 1.2
    max_y = np.max(np.abs(shape_basis[:, 1])) * 1.2
    plt.xlim((-max_x, max_x))
    plt.ylim((-max_y, max_y))
    plt.plot(shape_basis[:, 0], shape_basis[:, 1], c='C{}'.format(i % n_coeffs), lw=2.2)


plt.tight_layout()
plt.show()

# fig = plt.figure(1)
# show_mean = shape_mean.reshape((-1, 2))
# if contour_closed:
#     show_mean = close_contour(show_mean)
# plt.plot(show_mean[:, 0], show_mean[:, 1], c='C{}'.format(i % n_coeffs), lw=2.2)
# plt.show()
# exit()

