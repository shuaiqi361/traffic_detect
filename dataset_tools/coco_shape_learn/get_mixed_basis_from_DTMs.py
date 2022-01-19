import numpy as np
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista, learn_sparse_components
import os
from scipy import linalg


mask_size = 28
n_coeffs = 128
alpha = 0.2

# load the saved shapes for Cityscapes and COCO
dataDir = '/media/keyi/Data/Research/traffic/data/cityscapes'
dataType = 'train'  # options are train, val, test
annDir = os.path.join(dataDir, 'gtFine')
imgDir = os.path.join(dataDir, 'leftImg8bit_trainvaltest/leftImg8bit')

save_data_root = os.path.join(dataDir, 'dictionary')
# out_dict = '{}/Cityscapes_{}_DTM_basis_m{}_n{}_a{:.2f}.npz'.format(save_data_root, dtm_type, mask_size, n_coeffs, alpha)
cityscapes_out_resampled_shape_file = '{}/Cityscapes_{}_DTM_basis_fromDTM_m{}.npy'.format(save_data_root, dataType, mask_size)

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'sparse_shape_dict')
# out_dict = '{}/Centered_DTM_basis_m{}_n{}_a{:.2f}.npz'.format(save_data_root, mask_size, n_coeffs, alpha)
coco_out_resampled_shape_file = '{}/train_DTM_basis_fromDTM_m{}.npy'.format(save_data_root, mask_size)

out_dict = '{}/Centered_mixed_DTM_basis_m{}_n{}_a{:.2f}.npz'.format(save_data_root, mask_size, n_coeffs, alpha)


# concatenate shapes and jointly learn the dictionary
coco_shape_data = np.load(coco_out_resampled_shape_file)
# shape_mean = np.mean(coco_shape_data, axis=0, keepdims=True)
# shape_std = np.std(coco_shape_data, axis=0, keepdims=True) + 1e-6
print('Loading train2017 coco shape data: ', coco_shape_data.shape)

city_shape_data = np.load(cityscapes_out_resampled_shape_file)
# shape_mean = np.mean(city_shape_data, axis=0, keepdims=True)
# shape_std = np.std(city_shape_data, axis=0, keepdims=True) + 1e-6
print('Loading train Cityscapes shape data: ', city_shape_data.shape)

shape_data = np.concatenate([coco_shape_data, city_shape_data], axis=0)
shape_mean = np.mean(shape_data, axis=0, keepdims=True)
shape_std = np.std(shape_data, axis=0, keepdims=True) + 1e-6

n_shapes, n_feats = shape_data.shape
assert n_shapes == len(coco_shape_data) + len(city_shape_data)
assert n_feats == mask_size ** 2

# Start learning the dictionary
centered_shape_data = shape_data - shape_mean
learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(centered_shape_data,
                                                                    n_components=n_coeffs,
                                                                    alpha=alpha,
                                                                    batch_size=300,
                                                                    n_iter=500)

rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) + shape_mean - shape_data) ** 2 / shape_data.shape[0]
print('Training Reconstruction error:', rec_error)

code_active_rate = np.sum(np.abs(learned_codes) > 1e-4) / learned_codes.shape[0] / n_coeffs
print('Average Active rate:', code_active_rate)

# rank the codes from the highest activation to lowest
code_frequency = np.sum(np.abs(learned_codes) > 1e-4, axis=0) / learned_codes.shape[0]
idx_codes = np.argsort(code_frequency)[::-1]
print('Average Code Frequency:', len(code_frequency), 'sorted code magnitude: ', code_frequency[idx_codes])

ranked_dict = learned_dict[idx_codes, :]
np.savez(out_dict,
         shape_mean=shape_mean,
         shape_std=shape_std,
         shape_basis=ranked_dict)


