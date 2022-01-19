import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt

from dataset_tools.pycocotools.coco import COCO
from dataset_tools.pycocotools import mask as cocomask
from dataset_tools.pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista


mask_size = 28
num_code = 128
alpha = 0.2

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
result_json = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/3090_Res_50_DTMRInst_007/inference/coco_instances_results.json'
out_AP_dict = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/3090_Res_50_DTMRInst_007/inference/DTMRInst_007_perImgAPs.json'

coco = COCO(annFile)

# Load all annotations
cats = coco.loadCats(coco.getCatIds())
imgIds = coco.getImgIds()

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes(result_json)
result_dict = {}

# start per img evaluation
for image_id in imgIds:
    coco_eval = COCOeval(coco, coco_pred, 'segm')
    coco_eval.params.imgIds = [image_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print('Image id: {}, AP: {}'.format(image_id, coco_eval.stats[0]))
    result_dict[image_id] = '{:.3f}'.format(coco_eval.stats[0])

with open(out_AP_dict, 'w') as fp:
    json.dump(result_dict, fp)
