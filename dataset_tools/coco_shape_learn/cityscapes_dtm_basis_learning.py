import json

import numpy as np
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista, learn_sparse_components
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy import linalg

CITYSCAPES_CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

mask_size = 28
n_coeffs = 128
alpha = 0.2
dtm_type = 'standard'  # options are standard dtm, reciprocal, and complement

dataDir = '/media/keyi/Data/Research/traffic/data/cityscapes'
dataType = 'train'  # options are train, val, test
annDir = os.path.join(dataDir, 'gtFine')
imgDir = os.path.join(dataDir, 'leftImg8bit_trainvaltest/leftImg8bit')

save_data_root = os.path.join(dataDir, 'dictionary')
out_dict = '{}/Cityscapes_{}_DTM_basis_m{}_n{}_a{:.2f}.npz'.format(save_data_root, dtm_type, mask_size, n_coeffs, alpha)
out_resampled_shape_file = '{}/Cityscapes_{}_DTM_basis_fromDTM_m{}.npy'.format(save_data_root, dataType, mask_size)

counter_total = 0  # total number of segments
counter_valid = 0
counter_poor = 0  # objects too small to extract the shape
n_classes = len(CITYSCAPES_CLASSES)
counter_max = {}  # ensure that shapes from all categories are equally drawn
shape_per_cat = 600

original_shape_objects = []  # all objects
resampled_shape_matrix = []

img_root = os.path.join(imgDir, dataType)
img_scene_list = os.listdir(img_root)
counter = 1

# color image: frankfurt_000000_000576_leftImg8bit.png
# image annotation: frankfurt_000000_000576_gtFine_polygons.json

for scene in img_scene_list:
    print('parsing data for scene: ', scene, '({}/{})'.format(counter, len(img_scene_list)))
    image_folder = os.path.join(img_root, scene)
    all_images = os.listdir(image_folder)
    counter += 1
    for img_name in all_images:
        if not img_name.endswith('.png'):
            continue

        img_file = os.path.join(image_folder, img_name)
        ann_folder = os.path.join(os.path.join(annDir, dataType), scene)
        ann_name = img_name.split('_')
        ann_name = ann_name[0] + '_' + ann_name[1] + '_' + ann_name[2] + '_gtFine_polygons.json'
        ann_file = os.path.join(ann_folder, ann_name)

        with open(ann_file, 'r') as f_ann:
            ann_json = json.load(f_ann)
        img = cv2.imread(img_file)

        h_img = int(ann_json['imgHeight'])
        w_img = int(ann_json['imgWidth'])
        if h_img < 2 or w_img < 2:
            continue

        all_objects = ann_json['objects']
        for obj in all_objects:
            counter_total += 1
            cat_name = obj['label']
            if cat_name not in CITYSCAPES_CLASSES:
                continue

            counter_valid += 1
            instance_polygon = np.array(obj['polygon'])
            instance_polygon[:, 0] = np.clip(instance_polygon[:, 0], 0, w_img - 1)
            instance_polygon[:, 1] = np.clip(instance_polygon[:, 1], 0, h_img - 1)

            gt_x1, gt_y1, gt_x2, gt_y2 = int(min(instance_polygon[:, 0])), int(min(instance_polygon[:, 1])), \
                                         int(max(instance_polygon[:, 0])), int(max(instance_polygon[:, 1]))

            m = np.zeros((h_img, w_img), dtype=np.uint8)
            vertices = instance_polygon.astype(np.int32)
            cv2.drawContours(m, [vertices], color=255, contourIdx=0, thickness=-1)

            # cv2.imshow('Instance mask', m)
            #
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break

            # show ground truth
            # gt_image = img.copy()
            # gt_blend_mask = np.zeros(shape=gt_image.shape, dtype=np.uint8)
            #
            # cv2.polylines(gt_image, [instance_polygon.astype(np.int32)], True,
            #               color=(0, 155, 0), thickness=2)
            # cv2.drawContours(gt_blend_mask, [instance_polygon.astype(np.int32)], contourIdx=-1,
            #                  color=(0, 255, 0), thickness=-1)
            # cv2.rectangle(gt_image, (gt_x1, gt_y1), (gt_x2, gt_y2), color=(0, 255, 0), thickness=2)
            #
            # gt_dst_img = cv2.addWeighted(gt_image, 0.4, gt_blend_mask, 0.6, 0)
            # gt_dst_img[gt_blend_mask == 0] = gt_image[gt_blend_mask == 0]
            #
            # cat_image = np.concatenate([img, gt_dst_img], axis=1)
            #
            # cv2.imshow('Image vs. Segmentation', cat_image)
            #
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break

            if cat_name not in counter_max.keys():
                counter_max[cat_name] = 1
            else:
                if counter_max[cat_name] == shape_per_cat:
                    continue
                counter_max[cat_name] += 1

            gt_w, gt_h = gt_x2 - gt_x1, gt_y2 - gt_y1
            if gt_w * gt_h < 500 or gt_w < 10 or gt_h < 10:
                counter_poor += 1
                continue

            m_bbox = m[int(gt_y1):int(gt_y2), int(gt_x1):int(gt_x2)]  # crop the mask according to the bbox
            # print(m_bbox)
            # cv2.imshow('Instance mask', m_bbox)
            #
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     break

            resized_dist_bbox = cv2.resize(m_bbox, dsize=(mask_size, mask_size))
            resized_dist_bbox = np.where(resized_dist_bbox >= 255 * 0.5, 1, 0).astype(np.uint8)

            dist_bbox_in = cv2.distanceTransform(resized_dist_bbox, distanceType=cv2.DIST_L2, maskSize=3)
            dist_bbox_in = dist_bbox_in / (np.max(dist_bbox_in) + 1e-5)

            dist_bbox = np.where(dist_bbox_in > 0, np.clip(dist_bbox_in, 1e-5, 1), -1.)  # in the range of -1, (0, 1)
            if np.sum(np.isinf(dist_bbox)) > 0 or np.sum(np.isnan(dist_bbox)) > 0:
                print('shape with NaN or Inf.')
                continue

            resampled_shape_matrix.append(dist_bbox.reshape((1, -1)).astype(np.float16))

Cityscapes_resampled_shape_matrix = np.concatenate(resampled_shape_matrix, axis=0)
print('Total valid shape: ', counter_valid)
print('Poor shape: ', counter_poor)

print('Total number: ', counter_total)
print('Size of shape matrix: ', Cityscapes_resampled_shape_matrix.shape)
print('Length of the counter max:', len(counter_max))
np.save(out_resampled_shape_file, Cityscapes_resampled_shape_matrix)

# Start learning the dictionary
shape_data = np.load(out_resampled_shape_file)
shape_mean = np.mean(shape_data, axis=0, keepdims=True)
shape_std = np.std(shape_data, axis=0, keepdims=True) + 1e-5
print('Loading Cityscape train split shape data: ', shape_data.shape)

n_shapes, n_feats = shape_data.shape

centered_shape_data = shape_data - shape_mean
learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(centered_shape_data,
                                                                    n_components=n_coeffs,
                                                                    alpha=alpha,
                                                                    batch_size=400,
                                                                    n_iter=200)

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

