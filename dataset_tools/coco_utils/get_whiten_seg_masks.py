from pycocotools.coco import COCO
import numpy as np
from dataset_tools.coco_utils.utils import check_clockwise_polygon
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy import linalg
from dataset_tools.coco_utils.utils import intersect
from dataset_tools.coco_utils.utils import get_connected_polygon, turning_angle_resample, \
    get_connected_polygon_with_mask, uniformsample

n_vertices = 128  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.01

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'dictionary')
# out_dict = '{}/train_whiten_dict_v{}_n{}_a{:.2f}.npy'.format(save_data_root, n_vertices, n_coeffs, alpha)
out_resampled_shape_file = '{}/train_whiten_data_v{}.npy'.format(save_data_root, n_vertices)
out_dict_stats = '{}/train_whiten_stats_v{}_n{}_a{:.2f}.npz'.format(save_data_root, n_vertices, n_coeffs, alpha)  # contain other values, e.g. means, vars, ...

# coco = COCO(annFile)
# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# catIds = coco.getCatIds(catNms=nms)
# imgIds = coco.getImgIds(catIds=catIds)
# annIds = coco.getAnnIds(catIds=catIds)
# all_anns = coco.loadAnns(ids=annIds)
#
# counter_iscrowd = 0  # remove crowd annotated objects
# counter_total = 0  # total number of segments
# counter_valid = 0
# counter_poor = 0  # objects too small to extract the shape
#
# COCO_original_shape_objects = []  # all objects
# COCO_resample_shape_list = []
# COCO_resample_shape_matrix = np.zeros(shape=(0, n_vertices * 2))
# for annotation in all_anns:
#     if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
#         continue
#     # if random.random() > 0.4:  # randomly skip 70% of the objects
#     #     continue
#     counter_total += 1
#
#     if counter_total % 10000 == 0:
#         print('Processing {}/{} ...', counter_total, len(all_anns))
#     if annotation['iscrowd'] == 1:
#         counter_iscrowd += 1
#         continue
#
#     if len(annotation['segmentation']) > 1:
#         continue
#     img = coco.loadImgs(annotation['image_id'])[0]
#     image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
#     w_img = img['width']
#     h_img = img['height']
#     if w_img < 1 or h_img < 1:
#         continue
#
#     polygons = annotation['segmentation'][0]
#     # if len(annotation['segmentation']) > 1:
#     #     obj_contours = [np.array(s).reshape((-1, 2)).astype(np.int32) for s in annotation['segmentation']]
#     #     obj_contours = sorted(obj_contours, key=cv2.contourArea)
#     #     polygons = obj_contours[-1]
#     #     # print(obj_contours[-1])
#     #     # print([cv2.contourArea(s) for s in obj_contours])
#     #     # exit()
#     # else:
#     #     polygons = annotation['segmentation'][0]
#     if len(polygons) < 32 * 2:
#         counter_poor += 1
#         continue
#     else:
#         counter_valid += 1  # valid shape extracted
#
#     gt_bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
#     gt_x1, gt_y1, gt_w, gt_h = gt_bbox
#     cat_id = annotation['category_id']
#     cat_name = coco.loadCats([cat_id])[0]['name']
#
#     # construct data matrix
#     contour = np.array(polygons).reshape((-1, 2))
#     # if cv2.contourArea(contour.astype(np.int32)) < 150 or len(contour) < 16:
#     #     continue
#
#     # Flip if the polygon is not sorted in clockwise order
#     canonic_contour = uniformsample(contour, n_vertices)
#
#     clockwise_flag = check_clockwise_polygon(canonic_contour)
#     if not clockwise_flag:
#         canonic_contour = np.flip(canonic_contour, axis=0)
#
#     # Indexing from the left-most vertex
#     idx = np.argmin(canonic_contour[:, 0])
#     canonic_contour = np.concatenate((canonic_contour[idx:, :], canonic_contour[:idx, :]), axis=0)
#
#     # canonic_contour[:, 0] = np.clip(canonic_contour[:, 0], gt_x1, gt_x1 + gt_w)
#     # canonic_contour[:, 1] = np.clip(canonic_contour[:, 1], gt_y1, gt_y1 + gt_h)
#
#     updated_bbox = [np.min(canonic_contour[:, 0]), np.min(canonic_contour[:, 1]),
#                     np.max(canonic_contour[:, 0]), np.max(canonic_contour[:, 1])]
#     updated_width = np.max(canonic_contour[:, 0]) - np.min(canonic_contour[:, 0])
#     updated_height = np.max(canonic_contour[:, 1]) - np.min(canonic_contour[:, 1])
#
#     # Normalize the shapes
#     shifted_shape = canonic_contour - np.array([updated_bbox[0], updated_bbox[1]])
#     norm_shape = shifted_shape / np.array([updated_width, updated_height])
#
#     assert np.max(norm_shape[:, 0]) == 1 and np.max(norm_shape[:, 1]) == 1
#     assert np.min(norm_shape[:, 0]) == 0 and np.min(norm_shape[:, 1]) == 0
#
#     COCO_resample_shape_list.append(norm_shape.reshape((1, -1)))
#
#     # if len(COCO_resample_shape_list) >= 100000:
#     #     break
#
# COCO_resample_shape_matrix = np.concatenate(COCO_resample_shape_list, axis=0)
# # assert COCO_resample_shape_matrix.shape[0] == 80000 and COCO_resample_shape_matrix.shape[1] == n_vertices * 2
#
# print('Total valid shape: ', counter_valid)
# print('Poor shape: ', counter_poor)
# print('Is crowd: ', counter_iscrowd)
# print('Total number: ', counter_total)
# print('Size of shape matrix: ', COCO_resample_shape_matrix.shape)
# np.save(out_resampled_shape_file, COCO_resample_shape_matrix)

# Start learning the dictionary
shape_data = np.load(out_resampled_shape_file).astype(np.float32)
print('Loading train2017 coco shape data: ', shape_data.shape)
n_shapes, n_feats = shape_data.shape

# First whitening the shapes by zca
# print('Start performing zero component analysis ...')
shape_mean = np.mean(shape_data, axis=0)
shape_std = np.std(shape_data, axis=0) + 1e-5
X_norm = (shape_data - shape_mean) / shape_std
print('Normalization finished, data shape:', X_norm.shape)

# cov = np.cov(X_norm, rowvar=False)
# U, S, V = np.linalg.svd(cov)  # U:(256, 256), S:(256,)
# epsilon = 0.1
# X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm.T).T

# print('ZCA finished, shape_zca shape:', X_ZCA.shape)

learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(X_norm,
                                                                    n_components=n_coeffs,
                                                                    alpha=alpha,
                                                                    batch_size=400,
                                                                    n_iter=500)


print('Training error: ', error)
rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) * shape_std + shape_mean - shape_data) ** 2 / shape_data.shape[0]
print('Training Reconstruction error:', rec_error)
print('Outputing learned dictionary:', learned_dict.shape)

# np.save(out_dict, learned_dict)
np.savez(out_dict_stats,
         dictionary=learned_dict,
         mean=shape_mean,
         std=shape_std)

# count the number of self-intersections
total_counts = []
for i in range(n_coeffs):
    # for each shape basis, check every pair of edges in the polygon
    temp_basis = learned_dict[i, :].reshape((n_vertices, 2))
    temp_counts = 0
    for j in range(n_vertices):
        p1 = (temp_basis[j % n_vertices, 0], temp_basis[j % n_vertices, 1])
        p2 = (temp_basis[(j + 1) % n_vertices, 0], temp_basis[(j + 1) % n_vertices, 1])

        for k in range(j + 1, n_vertices):
            p3 = (temp_basis[k % n_vertices, 0], temp_basis[k % n_vertices, 1])
            p4 = (temp_basis[(k + 1) % n_vertices, 0], temp_basis[(k + 1) % n_vertices, 1])

            if intersect(p1, p2, p3, p4):
                temp_counts += 1

    total_counts.append(temp_counts - n_vertices)

print(total_counts)
print('Total intersections: {}, average {}'.format(sum(total_counts), np.mean(total_counts)))
