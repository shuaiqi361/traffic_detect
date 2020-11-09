from pycocotools.coco import COCO
import numpy as np
from dataset_tools.coco_utils.utils import check_clockwise_polygon
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy import linalg
from dataset_tools.coco_utils.utils import intersect

n_vertices = 32  # predefined number of polygonal vertices
n_coeffs = 128
alpha = 0.01

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'dictionary')
out_dict = '{}/train_scaled_dict_v{}_n{}_a{:.2f}.npy'.format(save_data_root, n_vertices, n_coeffs, alpha)
out_resampled_shape_file = '{}/train_scaled_norm_data_v{}.npy'.format(save_data_root, n_vertices)

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
#     if random.random() > 0.3:  # randomly skip 70% of the objects
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
#     polygons = annotation['segmentation'][0]
#     gt_bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
#     gt_x1, gt_y1, gt_w, gt_h = gt_bbox
#     cat_id = annotation['category_id']
#     cat_name = coco.loadCats([cat_id])[0]['name']
#
#     # obj = {'image_name': image_name, 'polygons': polygons, 'bbox': bbox, 'cat_name': cat_name}
#     # COCO_original_shape_objects.append(obj)
#
#     if len(polygons) < 24 * 2 or len(polygons) > 96 * 2:
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
#
#     # Indexing from the left-most vertex, argmin x-axis
#     idx = np.argmin(canonic_contour[:, 0])
#     indexed_shape = np.concatenate((canonic_contour[idx:, :], canonic_contour[:idx, :]), axis=0)
#
#     clockwise_flag = check_clockwise_polygon(indexed_shape)
#     if not clockwise_flag:
#         fixed_contour = np.flip(indexed_shape, axis=0)
#     else:
#         fixed_contour = indexed_shape.copy()
#
#     # correct the resampled vertices to be in the image
#     fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
#     fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)
#
#     # calculate the bounding box
#     x1, y1, x2, y2 = min(fixed_contour[:, 0]), min(fixed_contour[:, 1]), \
#                      max(fixed_contour[:, 0]), max(fixed_contour[:, 1])
#     bbox_width, bbox_height = x2 - x1, y2 - y1
#     bbox = [x1, y1, bbox_width, bbox_height]
#     bbox_center = np.array([(x1 + x2) / 2., (y1 + y2) / 2.])
#
#     # visualize resampled points in image
#     # img = cv2.imread(image_name)
#     # cv2.polylines(img, [fixed_contour.astype(np.int32)], True, (0, 0, 255))
#     # cv2.imshow('Poly', img)
#     # cv2.waitKey(1)
#
#     # Normalize the shapes with shape scaling in the range of [-1, 1]
#     norm_shape = (fixed_contour - bbox_center) / np.array([bbox_width / 2., bbox_height / 2.])
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

learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(shape_data,
                                                                    n_components=n_coeffs,
                                                                    alpha=alpha,
                                                                    batch_size=500,
                                                                    n_iter=500)


print('Training error: ', error)
rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) - shape_data) ** 2 / shape_data.shape[0]
print('Training Reconstruction error:', rec_error)
print('Outputing learned dictionary:', learned_dict.shape)

np.save(out_dict, learned_dict)


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
print('Total intersections: ', sum(total_counts))

