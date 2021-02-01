from pycocotools.coco import COCO
import numpy as np
import pickle
from dataset_tools.coco_utils.utils import check_clockwise_polygon
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from sklearn.decomposition import PCA
from scipy import linalg
from dataset_tools.coco_utils.utils import intersect
from dataset_tools.coco_utils.utils import get_connected_polygon, turning_angle_resample, \
    get_connected_polygon_with_mask


def learn_pca_components(shapes, n_components, whiten=False):
    """Learn sparse components from a dataset of shapes."""
    # Learn sparse components and predict coefficients for the dataset
    pca_learner = PCA(n_components=n_components, whiten=whiten)

    dicts = pca_learner.fit(shapes)  # n_samples, n_feats
    learned_dict_ = dicts.components_  # n_coeffs, n_feats
    learned_codes_ = np.matmul(shapes - dicts.mean_, learned_dict_.T)

    print('PCA learning... ', learned_dict_.shape, learned_codes_.shape)
    # error = np.mean((np.matmul(learned_codes_, learned_dict_) + dicts.mean_ - shapes) ** 2)
    # print('reconstruction error(frobenius norm): ', error)

    return learned_dict_, learned_codes_, dicts.explained_variance_, dicts.mean_


n_vertices = 32  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.01

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'dictionary')
out_dict = '{}/train_pca_components_v{}_n{}.npy'.format(save_data_root, n_vertices, n_coeffs)
out_resampled_shape_file = '{}/train_norm_data_v{}.npy'.format(save_data_root, n_vertices)

out_stat = {}

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
# COCO_resample_shape_matrix = np.zeros(shape=(0, n_vertices * 2))
# for annotation in all_anns:
#     if random.random() > 0.4:  # randomly skip 70% of the objects
#         continue
#     counter_total += 1
#
#     if counter_total % 10000 == 0:
#         print('Processing {}/{} ...', counter_total, len(all_anns))
#     if annotation['iscrowd'] == 1:
#         counter_iscrowd += 1
#         continue
#
#     img = coco.loadImgs(annotation['image_id'])[0]
#     image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
#     w_img = img['width']
#     h_img = img['height']
#     if w_img < 1 or h_img < 1:
#         continue
#
#     if len(annotation['segmentation']) > 1:
#         continue
#
#     polygons = annotation['segmentation'][0]
#     gt_bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
#     gt_x1, gt_y1, gt_w, gt_h = gt_bbox
#     cat_id = annotation['category_id']
#     cat_name = coco.loadCats([cat_id])[0]['name']
#
#     # obj = {'image_name': image_name, 'polygons': polygons, 'bbox': bbox, 'cat_name': cat_name}
#     # COCO_original_shape_objects.append(obj)
#
#     if len(polygons) < 32 * 2:
#         counter_poor += 1
#         continue
#     else:
#         counter_valid += 1  # valid shape extracted
#
#     # construct data matrix
#     contour = np.array(polygons).reshape((-1, 2))
#
#     # Flip if the polygon is not sorted in clockwise order
#     canonic_contour = resample(contour, num=n_vertices)
#     clockwise_flag = check_clockwise_polygon(canonic_contour)
#     if not clockwise_flag:
#         canonic_contour = np.flip(canonic_contour, axis=0)
#
#     # Indexing from the left-most vertex
#     idx = np.argmin(canonic_contour[:, 0])
#     canonic_contour = np.concatenate((canonic_contour[idx:, :], canonic_contour[:idx, :]), axis=0)
#
#     canonic_contour[:, 0] = np.clip(canonic_contour[:, 0], gt_x1, gt_x1 + gt_w)
#     canonic_contour[:, 1] = np.clip(canonic_contour[:, 1], gt_y1, gt_y1 + gt_h)
#
#     # Normalize the shapes
#     contour_mean = np.mean(canonic_contour, axis=0)
#     # contour_std = np.sqrt(np.sum(np.std(canonic_contour, axis=0) ** 2))
#     norm_shape = canonic_contour - contour_mean
#
#     # draw re-sampled points
#     # fig = plt.figure()
#     # plt.title(cat_name)
#     # plt.plot(indexed_shape[:, 0], indexed_shape[:, 1], '-o', c='C0', lw=3)
#     # plt.text(indexed_shape[0, 0], indexed_shape[0, 1], '0', fontsize=8)
#     # plt.text(indexed_shape[1, 0], indexed_shape[1, 1], '1', fontsize=8)
#     # plt.show()
#
#     COCO_resample_shape_matrix = np.concatenate((COCO_resample_shape_matrix, norm_shape.reshape((1, -1))), axis=0)
#
#     if len(COCO_resample_shape_matrix) >= 60000:
#         break
#
# print('Total valid shape: ', counter_valid)
# print('Poor shape: ', counter_poor)
# print('Is crowd: ', counter_iscrowd)
# print('Total number: ', counter_total)
# print('Size of shape matrix: ', COCO_resample_shape_matrix.shape)
# np.save(out_resampled_shape_file, COCO_resample_shape_matrix)

# Start learning the dictionary
shape_data = np.load(out_resampled_shape_file)
print('Loading train2017 coco shape data: ', shape_data.shape)
n_shapes, n_feats = shape_data.shape

learned_dict, learned_codes, variance_explained, mean_shape = learn_pca_components(shape_data, n_coeffs)

rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) + mean_shape - shape_data) ** 2 / shape_data.shape[0]
print('Training Reconstruction error:', rec_error)
print('Outputing learned dictionary:', learned_dict.shape)

out_stat = {'mean': mean_shape, 'dict': learned_dict}

f = open(out_dict, "wb")
np.savez(f, dict=learned_dict, mean=mean_shape)
f.close()

# # count the number of self-intersections
# total_counts = []
# for i in range(n_coeffs):
#     # for each shape basis, check every pair of edges in the polygon
#     temp_basis = learned_dict[i, :].reshape((n_vertices, 2))
#     temp_counts = 0
#     for j in range(n_vertices):
#         p1 = (temp_basis[j % n_vertices, 0], temp_basis[j % n_vertices, 1])
#         p2 = (temp_basis[(j + 1) % n_vertices, 0], temp_basis[(j + 1) % n_vertices, 1])
#
#         for k in range(j + 1, n_vertices):
#             p3 = (temp_basis[k % n_vertices, 0], temp_basis[k % n_vertices, 1])
#             p4 = (temp_basis[(k + 1) % n_vertices, 0], temp_basis[(k + 1) % n_vertices, 1])
#
#             if intersect(p1, p2, p3, p4):
#                 temp_counts += 1
#
#     total_counts.append(temp_counts - n_vertices)
#
# print(total_counts)
# print('Total intersections: {}, average {}'.format(sum(total_counts), np.mean(total_counts)))
