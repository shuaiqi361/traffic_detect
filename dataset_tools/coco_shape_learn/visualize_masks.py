from pycocotools.coco import COCO
import numpy as np
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from dataset_tools.coco_utils.utils import close_contour
from matplotlib import gridspec

mask_size = 28
n_vertices = 180  # predefined number of polygonal vertices
n_coeffs = 128
alpha = 0.20
contour_closed = True
dtm_type = 'standard'

n_atom_row = 8
# n_atom_col = int(n_coeffs // n_atom_row)
n_atom_col = 8

# save_dict_root = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/sparse_shape_dict'
save_dict_root = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/adet/modeling/SparseDTMInst/dictionary'
# out_dict = '{}/mask_fromMask_basis_m{}_n{}_a{:.2f}.npy'.format(save_dict_root, mask_size, n_coeffs, alpha)
# out_dict = '{}/mask_fromDTM_minusone_basis_m{}_n{}_a{:.2f}.npy'.format(save_dict_root, mask_size, n_coeffs, alpha)
# save_dict_root = '/media/keyi/Data/Research/traffic/data/cityscapes/dictionary'
# out_dict = '{}/train_scaled_ranked_dict_v{}_n{}_a{:.2f}.npy'.format(save_dict_root, n_vertices, n_coeffs, alpha)
out_dict = '{}/Centered_DTM_basis_m{}_n{}_a{:.2f}.npz'.format(save_dict_root, mask_size, n_coeffs, alpha)
# out_dict = '{}/Cityscapes_{}_DTM_basis_m{}_n{}_a{:.2f}.npz'.format(save_dict_root, dtm_type, mask_size, n_coeffs, alpha)
learned_dict = np.load(out_dict)
# with np.load(out_dict) as data_meta:
#     learned_dict = data_meta['dictionary']
#     shape_mean = data_meta['mean']
#     shape_std = data_meta['std']

parameters = np.load(out_dict)
learned_dict = parameters['shape_basis']
shape_mean = parameters['shape_mean']
shape_std = parameters['shape_std']

fig = plt.figure(figsize=(n_atom_row, n_atom_col))

for i in range(64):
    plt.subplot(n_atom_row, n_atom_col, i + 1)
    plt.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    shape_basis = learned_dict[i, :].reshape((mask_size, mask_size))
    plt.imshow(shape_basis, cmap='viridis')

# plt.colorbar()
# plt.title('Sparse DTM Basis')
plt.tight_layout()
plt.show()

# fig = plt.figure(1)
# show_mean = shape_mean.reshape((-1, 2))
# if contour_closed:
#     show_mean = close_contour(show_mean)
# plt.plot(show_mean[:, 0], show_mean[:, 1], c='C{}'.format(i % n_coeffs), lw=2.2)
# plt.show()
# exit()

