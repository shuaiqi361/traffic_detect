from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)
bbox_width = []
bbox_height = []
bbox_area = []
mask_area = []

# Load all annotations
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_obj = 0

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
    # print(annotation)
    gt_bbox = annotation['bbox']
    gt_x1, gt_y1, gt_w, gt_h = [int(ss) for ss in gt_bbox]
    if gt_w < 2 or gt_h < 2:
        continue

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    gt_area = annotation['area']

    bbox_width.append(gt_w)
    bbox_height.append(gt_h)
    bbox_area.append(gt_w * gt_h)
    mask_area.append(gt_area)


bbox_width = np.array(bbox_width)
bbox_height = np.array(bbox_height)
bbox_area = np.array(bbox_area)
mask_area = np.array(mask_area)
print(bbox_width.shape)
print(bbox_height.shape)
print(bbox_area.shape)
print(mask_area.shape)

# # plot 2d histogram of bbox sizes
# plt.figure()
# plt.hist2d(bbox_width, bbox_height, bins=(20, 20), density=False, cmap=plt.cm.nipy_spectral)  # plt.cm.Reds
# # arr = plt.hist2d(bbox_width, bbox_height, bins=(10, 10), density=True, cmap=plt.cm.Reds)
# plt.xlabel('Width')
# plt.ylabel('Height')
# plt.title('Histogram of bbox width and height')
# plt.colorbar()
#
# # show plot
# # plt.tight_layout()
# plt.show()
# exit()


# plot histogram of bbox and mask areas

bin_box = [0, 32 * 32 / 1000., 64 * 64 / 1000., 96 * 96 / 1000., 128 * 128 / 1000.]
fig = plt.figure()
plt.hist(bbox_area / 1000., bins=bin_box, color='g', density=True, stacked=True, edgecolor='black')
plt.xlabel('Bbox area (x 1000)')
plt.ylabel('Percentage')
plt.title('Histogram of bbox area')
plt.show()

fig = plt.figure()
plt.hist(mask_area / 1000., bins=bin_box, color='r', density=True, stacked=True, edgecolor='black')
plt.xlabel('Mask area (x 1000)')
plt.ylabel('Percentage')
plt.title('Histogram of mask area')
plt.show()
exit()
