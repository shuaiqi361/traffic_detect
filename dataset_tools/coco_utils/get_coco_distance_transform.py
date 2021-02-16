from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
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
from dataset_tools.coco_utils.utils import get_connected_polygon, turning_angle_resample, \
    get_connected_polygon_with_mask, uniformsample

mask_size = 32
n_coeffs = 64
alpha = 0.01

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'distance_transforms')
out_dict = '{}/dt_dict_m{}_n{}_a{:.2f}.npy'.format(save_data_root, mask_size, n_coeffs, alpha)
out_resampled_shape_file = '{}/train_norm_mask_m{}.npy'.format(save_data_root, mask_size)


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_iscrowd = 0  # remove crowd annotated objects
counter_total = 0  # total number of segments
counter_valid = 0
counter_poor = 0  # objects too small to extract the shape

COCO_original_shape_objects = []  # all objects
COCO_resample_shape_matrix = np.zeros(shape=(0, mask_size * mask_size))
for annotation in all_anns:
    if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
        continue
    # if random.random() > 0.3:  # randomly skip 70% of the objects
    #     continue
    counter_total += 1

    if counter_total % 10000 == 0:
        print('Processing {}/{} ...', counter_total, len(all_anns))
    if annotation['iscrowd'] == 1:
        counter_iscrowd += 1
        continue

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
    w_img = img['width']
    h_img = img['height']
    if w_img < 1 or h_img < 1:
        continue

    # if annotation['area'] > 60 or annotation['area'] < 5:
    #     continue
    if annotation['area'] < 150:
        continue

    # if len(annotation['segmentation']) > 1:
    #     obj_contours = [np.array(s).reshape((-1, 2)).astype(np.int32) for s in annotation['segmentation']]
    #     obj_contours = sorted(obj_contours, key=cv2.contourArea)
    #     polygons = obj_contours[-1]
    # else:
    #     polygons = annotation['segmentation'][0]

    gt_bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    rles = cocomask.frPyObjects(annotation['segmentation'], h_img, w_img)
    rle = cocomask.merge(rles)  # ['counts'].decode('ascii')
    m = cocomask.decode(rle).astype(np.uint8)
    m_bbox = m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)]
    # rle_new = encode_mask(m.astype(np.uint8))
    cv2.imshow('bbox', m_bbox * 255)
    image = cv2.imread(image_name)
    cv2.rectangle(image, pt1=(int(gt_x1), int(gt_y1)), pt2=(int(gt_x1 + gt_w), int(gt_y1 + gt_h)), color=(0, 255, 0))
    dist_bbox = cv2.distanceTransform(m_bbox, distanceType=cv2.DIST_L2, maskSize=5)
    dist_bbox = dist_bbox / np.max(dist_bbox) * 255

    # Show the images and masks
    cv2.imshow('image', image)
    cv2.imshow('dist bbox', dist_bbox.astype(np.uint8))
    cv2.waitKey()

    # draw re-sampled points
    # fig = plt.figure()
    # plt.title(cat_name)
    # plt.plot(indexed_shape[:, 0], indexed_shape[:, 1], '-o', c='C0', lw=3)
    # plt.text(indexed_shape[0, 0], indexed_shape[0, 1], '0', fontsize=8)
    # plt.text(indexed_shape[1, 0], indexed_shape[1, 1], '1', fontsize=8)
    # plt.show()

    # COCO_resample_shape_matrix = np.concatenate((COCO_resample_shape_matrix, norm_shape.reshape((1, -1))), axis=0)
    #
    # if len(COCO_resample_shape_matrix) >= 50000:
    #     break


# print('Total valid shape: ', counter_valid)
# print('Poor shape: ', counter_poor)
# print('Is crowd: ', counter_iscrowd)
# print('Total number: ', counter_total)
# print('Size of shape matrix: ', COCO_resample_shape_matrix.shape)
# np.save(out_resampled_shape_file, COCO_resample_shape_matrix)
#
# # Start learning the dictionary
# shape_data = np.load(out_resampled_shape_file)
# print('Loading train2017 coco shape data: ', shape_data.shape)
# n_shapes, n_feats = shape_data.shape
#
# learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(shape_data,
#                                                                     n_components=n_coeffs,
#                                                                     alpha=alpha,
#                                                                     batch_size=300,
#                                                                     n_iter=400)
#
#
# print('Training error: ', error)
# rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) - shape_data) ** 2 / shape_data.shape[0]
# print('Training Reconstruction error:', rec_error)
# print('Outputing learned dictionary:', learned_dict.shape)
#
# np.save(out_dict, learned_dict)
#
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
