from pycocotools.coco import COCO
import numpy as np
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from dataset_tools.coco_utils.utils import close_contour

n_vertices = 32  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.1
contour_closed = True

n_atom_row = int(np.sqrt(n_coeffs))
n_atom_col = int(np.sqrt(n_coeffs))

save_dict_root = '/media/keyi/Data/Research/traffic/data/KINS/dictionary'
out_dict = '{}/train_dict_kins_v{}_n{}_a{:.2f}.npy'.format(save_dict_root, n_vertices, n_coeffs, alpha)
# out_dict = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/dictionary/train_scaled_dict_v{}_n{}_a{:.2f}.npy'.format(n_vertices, n_coeffs, alpha)
learned_dict = np.load(out_dict)

fig = plt.figure(figsize=(n_atom_row * 2, n_atom_col * 2))
# plt.title('A dictionary of 64 basis functions')
for i in range(n_coeffs):
    plt.subplot(n_atom_row, n_atom_col, i + 1)
    plt.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
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
exit()

