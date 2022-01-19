from pycocotools.coco import COCO
import numpy as np
import random
import json
import cv2
import copy
import matplotlib.pyplot as plt
import os
from dataset_tools.coco_utils.utils import close_contour
from matplotlib import gridspec

mask_size = 28
n_coeffs = 64
alpha = 0.2

n_atom_row = 8
n_atom_col = int(n_coeffs // n_atom_row)

save_dict_root = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17/sparse_shape_dict'
out_dict = '{}/Conditional_DTM_basis_m{}_n{}_a{:.2f}.json'.format(save_dict_root, mask_size, n_coeffs, alpha)
basis_out = '{}/Conditional_basis_m{}_n{}_a{:.2f}'.format(save_dict_root, mask_size, n_coeffs, alpha)

with open(out_dict, 'r') as fp:
    class_cond_dict = json.load(fp)

if not os.path.exists(basis_out):
    os.mkdir(basis_out)

for k, _ in class_cond_dict.items():
    fig = plt.figure(figsize=(n_atom_row, n_atom_col))
    learned_dict = class_cond_dict[k]['basis']
    print('Basis class: ', class_cond_dict[k]['name'])
    learned_dict = np.array(learned_dict)
    for i in range(n_coeffs):
        plt.subplot(n_atom_row, n_atom_col, i + 1)
        plt.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        shape_basis = learned_dict[i, :].reshape((mask_size, mask_size))
        plt.imshow(shape_basis, cmap='viridis')

    plt.tight_layout()
    # plt.show()

    figure_name = '{}_{}_basis.png'.format(k, class_cond_dict[k]['name'])
    plt.savefig(os.path.join(basis_out, figure_name))

