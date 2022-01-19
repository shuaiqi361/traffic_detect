import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt

from dataset_tools.pycocotools.coco import COCO
from dataset_tools.pycocotools import mask as cocomask
from dataset_tools.pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


mask_size = 28
num_code = 128
alpha = 0.2

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
result_json = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/3090_Res_50_DTMRInst_007/inference/coco_instances_results.json'

coco = COCO(annFile)

# Load all annotations
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes(result_json)
coco_eval = COCOeval(coco, coco_pred, 'segm')
coco_eval.params.imgIds = [480985]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# print(coco_eval.eval)
# print('--------------------------------------------------------1')
# print(coco_eval.evalImgs)
print('--------------------------------------------------------2')
print(coco_eval.stats)
# print('--------------------------------------------------------3')
# print(coco_eval.ious)
