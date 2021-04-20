from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista, learn_sparse_components
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
import json
from scipy import linalg

mask_size = 28

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'sparse_shape_dict')
out_class_conditional_masks = '{}/class_conditional_masks_m{}_id_key.json'.format(save_data_root, mask_size)


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
class_conditional_masks = {}

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
    if annotation['area'] < 100:
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

    resized_dist_bbox = cv2.resize(m_bbox, dsize=(mask_size, mask_size))
    resized_dist_bbox = np.where(resized_dist_bbox >= 255 * 0.5, 1, 0).astype(np.float32).reshape(1, mask_size, mask_size)

    # cv2.imshow('image', output_image)
    # cv2.imshow('resize', resized_dist_bbox * 255)
    # cv2.imshow('dtm', dtm_show.astype(np.uint8))
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     exit()

    if cat_id not in class_conditional_masks.keys():
        class_conditional_masks[cat_id] = [resized_dist_bbox]
    else:
        class_conditional_masks[cat_id].append(resized_dist_bbox)

print(len(class_conditional_masks.keys()), " classes in total.")
assert len(class_conditional_masks.keys()) == 80


for k, v in class_conditional_masks.items():
    temp = np.concatenate(class_conditional_masks[k], axis=0)
    temp = np.mean(temp, axis=0)
    temp_show = temp > 0.5
    temp_show = temp_show.astype(np.uint8) * 255
    class_conditional_masks[k] = temp.tolist()

    # cv2.imshow('class cond. mask', temp_show)
    # print('Class: ', k)
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     continue


with open(out_class_conditional_masks, 'w') as f_det:
    json.dump(class_conditional_masks, f_det)


