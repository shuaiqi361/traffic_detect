from pycocotools.coco import COCO
import numpy as np
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from dataset_tools.coco_utils.utils import close_contour
from matplotlib import gridspec
import seaborn as sns

mask_size = 28
n_vertices = 180  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.80
contour_closed = True

n_atom_row = 8
n_atom_col = int(n_coeffs // n_atom_row)

save_dict_root = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/sparse_shape_dict'
# out_dict = '{}/mask_fromMask_basis_m{}_n{}_a{:.2f}.npy'.format(save_dict_root, mask_size, n_coeffs, alpha)
out_dict = '{}/mask_fromDTM_minusone_basis_m{}_n{}_a{:.2f}.npy'.format(save_dict_root, mask_size, n_coeffs, alpha)
out_code = '{}/mask_fromDTM_minusone_code_m{}_n{}_a{:.2f}.npy'.format(save_dict_root, mask_size, n_coeffs, alpha)
learned_dict = np.load(out_dict)
learned_codes = np.load(out_code)

# rank the codes from the highest activation to lowest
code_frequency = np.sum(np.abs(learned_codes) > 1e-4, axis=0) / learned_codes.shape[0]
idx_codes = np.argsort(code_frequency)[::-1]
print('Code Frequency:', len(code_frequency), 'sorted code magnitude: ', code_frequency[idx_codes])
learned_dict = learned_dict[idx_codes, :]

# plot heatmap for active coefficeints, show frequency
# plot_data = np.sum(np.abs(codes), axis=0) / len(codes)
plot_data = code_frequency[idx_codes]
print(plot_data.shape, len(learned_codes))
ax = sns.heatmap(plot_data.reshape((8, 8)), linewidths=.5, annot=True, fmt=".2f")
plt.show()

# avg_codes = np.mean(np.abs(learned_codes), axis=0)
# idx_codes = np.argsort(avg_codes)[::-1]  # sort the learned basis according to the average coefficients
# print('Code Magnitudes:', len(avg_codes), 'sorted code magnitude: ', avg_codes[idx_codes])
# learned_dict = learned_dict[idx_codes, :]
# plot_data = avg_codes[idx_codes]
# ax = sns.heatmap(plot_data.reshape((8, 8)), linewidths=.5, annot=True, fmt=".2f")
# plt.show()


fig = plt.figure(figsize=(n_atom_row, n_atom_col))

for i in range(n_coeffs):
    plt.subplot(n_atom_row, n_atom_col, i + 1)
    plt.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    shape_basis = learned_dict[i, :].reshape((mask_size, mask_size))
    plt.imshow(shape_basis, cmap='viridis')

# plt.colorbar()
# plt.title('Sparse DTM Basis')
# plt.tight_layout()
plt.show()

# fig = plt.figure(1)
# show_mean = shape_mean.reshape((-1, 2))
# if contour_closed:
#     show_mean = close_contour(show_mean)
# plt.plot(show_mean[:, 0], show_mean[:, 1], c='C{}'.format(i % n_coeffs), lw=2.2)
# plt.show()
# exit()

