from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista
from dataset_tools.coco_utils.utils import check_clockwise_polygon
import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt
import os


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


mask_size = 28

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'sparse_shape_dict')
out_class_conditional_masks = '{}/class_conditional_masks_m{}.json'.format(save_data_root, mask_size)

with open(out_class_conditional_masks, 'r') as f_out:
    class_conditional_masks = json.load(f_out)

coco = COCO(annFile)

# Load all annotations
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_obj = 0
counts_codes = []

det_results = []
seg_results = []
all_img_ids = []
mean_IoUs = []

for annotation in all_anns:
    if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
        continue

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
    w_img = int(img['width'])
    h_img = int(img['height'])
    if w_img < 1 or h_img < 1:
        continue

    counter_obj += 1
    if annotation['image_id'] not in all_img_ids:
        all_img_ids.append(annotation['image_id'])

    gt_bbox = annotation['bbox']
    gt_x1, gt_y1, gt_w, gt_h = [int(ss) for ss in gt_bbox]
    if gt_w < 2 or gt_h < 2:
        continue

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    original_rles = cocomask.frPyObjects(annotation['segmentation'], h_img, w_img)
    rle = cocomask.merge(original_rles)
    m = cocomask.decode(rle).astype(np.uint8) * 255

    m_bbox = m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)]

    recon_contour = np.clip(np.array(class_conditional_masks[cat_name]) * 255., 0, 255)
    recon_contour = recon_contour.astype(np.uint8)
    recon_ori_masks = cv2.resize(recon_contour, dsize=(int(gt_w), int(gt_h)))
    recon_ori_masks = np.where(recon_ori_masks >= 0.5 * 255, 1, 0).astype(np.uint8)

    # convert reconstructed masks to original rle masks
    recon_m = np.zeros((h_img, w_img), dtype=np.uint8)
    recon_m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)] = recon_ori_masks
    rle_new = encode_mask(recon_m.astype(np.uint8))

    # compute mean iou
    cropped_gt_rle = encode_mask(m_bbox.astype(np.uint8) // 255)
    cropped_recon_rle = encode_mask(recon_ori_masks.astype(np.uint8))
    iou = cocomask.iou([cropped_gt_rle], [cropped_recon_rle], [annotation['iscrowd']])
    mean_IoUs.append(iou[0][0])

    seg = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'segmentation': rle_new
    }
    seg_results.append(seg)

with open('{}/results/{}_class_conditional_predict_m{}.json'.format(dataDir, dataType, mask_size), 'w') as f_seg:
    json.dump(seg_results, f_seg)

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_class_conditional_predict_m{}.json'.format(dataDir, dataType, mask_size))
coco_eval = COCOeval(coco, coco_pred, 'segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print('Average mIoU of all instances: ', np.mean(mean_IoUs))
