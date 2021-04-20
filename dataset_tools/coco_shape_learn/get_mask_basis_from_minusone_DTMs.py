from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
from dataset_tools.coco_utils.utils import check_clockwise_polygon
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista, learn_sparse_components
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy import linalg
import seaborn as sns
from dataset_tools.coco_utils.utils import intersect
from dataset_tools.coco_utils.utils import turning_angle_resample, get_connected_polygon_coco_mask, \
    get_connected_polygon_using_mask, get_connected_polygon_with_mask, uniform_sample_segment, uniformsample, \
    get_connected_polys_with_measure, close_contour

mask_size = 48
n_vertices = 180
n_coeffs = 256
alpha = 0.2

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'sparse_shape_dict')
out_dict = '{}/mask_fromDTM_minusone_basis_m{}_n{}_a{:.2f}.npy'.format(save_data_root, mask_size, n_coeffs, alpha)
out_code = '{}/mask_fromDTM_minusone_code_m{}_n{}_a{:.2f}.npy'.format(save_data_root, mask_size, n_coeffs, alpha)
out_resampled_shape_file = '{}/train_mask_basis_minusone_fromDTM_m{}.npy'.format(save_data_root, mask_size)


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
n_classes = 80
counter_max = {}  # ensure that shapes from all categories are equally drawn
shape_per_cat = 500

COCO_original_shape_objects = []  # all objects
COCO_resample_shape_matrix = []
for annotation in all_anns:
    if sum([c for _, c in counter_max.items()]) == shape_per_cat * n_classes:
        break
    if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
        counter_iscrowd += 1
        continue

    counter_total += 1

    if counter_total % 10000 == 0:
        print('Processing {}/{} ...', counter_total, len(all_anns))

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/coco/%s/%s' % (dataDir, dataType, img['file_name'])
    w_img = img['width']
    h_img = img['height']
    if w_img < 1 or h_img < 1:
        continue

    # filter out small shapes and disconnected shapes for learning the dictionary
    if annotation['area'] < 150:
        continue

    # if the current shape reach its max in the counter list, skip
    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    if cat_name not in counter_max.keys():
        counter_max[cat_name] = 1
    else:
        if counter_max[cat_name] == shape_per_cat:
            continue
        counter_max[cat_name] += 1

    gt_bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    gt_x1, gt_y1, gt_w, gt_h = [int(ss) for ss in gt_bbox]
    if gt_w < 2 or gt_h < 2:
        continue

    original_rles = cocomask.frPyObjects(annotation['segmentation'], h_img, w_img)
    rle = cocomask.merge(original_rles)
    m = cocomask.decode(rle).astype(np.uint8)

    m_bbox = m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)] * 255  # crop the mask according to the bbox
    # m_bbox = np.pad(m_bbox, 1, mode='constant')

    resized_dist_bbox = cv2.resize(m_bbox, dsize=(mask_size, mask_size))
    resized_dist_bbox = np.where(resized_dist_bbox >= 255 * 0.5, 1, 0).astype(np.uint8)

    dist_bbox_in = cv2.distanceTransform(resized_dist_bbox, distanceType=cv2.DIST_L2, maskSize=3)
    dist_bbox_in = dist_bbox_in / np.max(dist_bbox_in)

    dist_bbox = np.where(dist_bbox_in > 0, dist_bbox_in, -1.)  # in the range of -1, (0, 1)

    assert dist_bbox.shape[0] == dist_bbox.shape[1] == mask_size
    dist_bbox = (dist_bbox + 1) * 255 / 2.

    # Show the images and masks
    dist_bbox_show = (dist_bbox > 0) * 255
    dtm_show = dist_bbox_in * 255
    # original_comp = np.concatenate([m_bbox * 255, dist_bbox.astype(np.uint8)], axis=1)
    # cv2.imshow('original_comp', original_comp)
    # recovered_mask = np.where(dist_bbox > 124, 255, 0).astype(np.uint8)
    # original_comp = np.concatenate([resized_dist_bbox * 255, dist_bbox_show.astype(np.uint8), dtm_show.astype(np.uint8)], axis=1)
    # cv2.imshow('resize', original_comp)
    output_image = cv2.imread(image_name)
    # bbox = [gt_x1, gt_y1, gt_x1 + gt_w, gt_y1 + gt_h]
    # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
    #               pt2=(int(bbox[2]), int(bbox[3])),
    #               color=(0, 255, 0), thickness=2)
    # cv2.imshow('image', output_image)
    # cv2.imshow('resize', resized_dist_bbox * 255)
    # cv2.imshow('dtm', dtm_show.astype(np.uint8))
    cv2.imshow('complement', resized_dist_bbox * 255 - dtm_show.astype(np.uint8))
    if cv2.waitKey() & 0xFF == ord('q'):
        exit()

    # recovered_mask = np.where(dist_bbox >= 0, 255, 0).astype(np.uint8)
    # dist_bbox = (dist_bbox + 1) / 2 * 255
    # original_comp = np.concatenate([resized_dist_bbox * 255, dist_bbox.astype(np.uint8), recovered_mask], axis=1)
    # cv2.imshow('resize', original_comp)
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     exit()

    COCO_resample_shape_matrix.append(dist_bbox.reshape((1, -1)).astype(np.float16))
#
# COCO_resample_shape_matrix = np.concatenate(COCO_resample_shape_matrix, axis=0)
# print('Total valid shape: ', counter_valid)
# print('Poor shape: ', counter_poor)
# print('Is crowd: ', counter_iscrowd)
# print('Total number: ', counter_total)
# print('Size of shape matrix: ', COCO_resample_shape_matrix.shape)
# print('Length of the counter max:', len(counter_max))
# np.save(out_resampled_shape_file, COCO_resample_shape_matrix)

# Start learning the dictionary
shape_data = np.load(out_resampled_shape_file)
print('Loading train2017 coco shape data: ', shape_data.shape)
n_shapes, n_feats = shape_data.shape

learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(shape_data,
                                                                    n_components=n_coeffs,
                                                                    alpha=alpha,
                                                                    batch_size=400,
                                                                    n_iter=500)

# learned_dict, learned_codes = learn_sparse_components(shape_data, n_components=n_coeffs,
#                                                       lmbda=alpha, batch_size=400, n_iter=500)


# print('Training error: ', error)
rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) - shape_data) ** 2 / shape_data.shape[0]
print('Training Reconstruction error:', rec_error)
print('Outputing learned dictionary:', learned_dict.shape)

np.save(out_dict, learned_dict)
np.save(out_code, learned_codes)

code_active_rate = np.sum(np.abs(learned_codes) > 1e-4) / learned_codes.shape[0] / n_coeffs
print('Average Active rate:', code_active_rate)

# rank the codes from the highest activation to lowest
# code_frequency = np.sum(np.abs(learned_codes) > 1e-4, axis=0) / learned_codes.shape[0]
# idx_codes = np.argsort(code_frequency)[::-1]
# print('Average Code Frequency:', len(code_frequency), 'sorted code magnitude: ', code_frequency[idx_codes])
#
# ranked_dict = learned_dict[idx_codes, :]
# np.save(out_dict, ranked_dict)
#
# # plot heatmap for active coefficeints, show frequency
# # plot_data = np.sum(np.abs(codes), axis=0) / len(codes)
# plot_data = code_frequency[idx_codes]
# print(plot_data.shape, len(learned_codes))
# ax = sns.heatmap(plot_data.reshape((8, 8)), linewidths=.5, annot=True, fmt=".2f")
# plt.show()
#
# avg_codes = np.mean(np.abs(learned_codes), axis=0)
# idx_codes = np.argsort(avg_codes)[::-1]  # sort the learned basis according to the average coefficients
# print('Code Magnitudes:', len(avg_codes), 'sorted code magnitude: ', avg_codes[idx_codes])
#
# plot_data = avg_codes[idx_codes]
# ax = sns.heatmap(plot_data.reshape((8, 8)), linewidths=.5, annot=True, fmt=".2f")
# plt.show()
