import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt
import pickle

from dataset_tools.pycocotools.coco import COCO
from dataset_tools.pycocotools import mask as cocomask
from dataset_tools.pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista


baseline_results = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/BlendMask_R_50_3x/results/BlendMask_perImgAPs.json'
our_method_results = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/3090_Res_50_DTMRInst_007/inference/DTMRInst_007_perImgAPs.json'
failure_modes_APs = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/3090_Res_50_DTMRInst_007/inference/BlendMask_fail_perImgAPs.json'

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

with open(baseline_results, 'r') as f:
    other_dict = json.load(f)

with open(our_method_results, 'r') as f:
    our_dict = json.load(f)

print(our_dict)
print('----------------------------------------------------------------------1')
# Load all annotations
coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
imgIds = coco.getImgIds()

compare_AP_difference_dict = {}  # our APs - others, find the smallest value (negative), which is worst failure modes

for image_id in imgIds:
    image_id = str(image_id)
    if not (image_id in other_dict.keys() and image_id in our_dict.keys()):
        continue

    compare_AP_difference_dict[image_id] = -float(our_dict[image_id]) + float(other_dict[image_id])

sort_APs = sorted(compare_AP_difference_dict.items(), key=lambda x: x[1], reverse=False)

# for i in sort_APs:
# 	print(i[0], i[1])

# print(compare_AP_difference_dict)
print('----------------------------------------------------------------------2')
print(sort_APs)

with open(failure_modes_APs, 'w') as fp:
    json.dump(sort_APs, fp)

# with open(failure_modes_APs, "wb") as fp:   #Pickling
#     pickle.dump(sort_APs, fp)
