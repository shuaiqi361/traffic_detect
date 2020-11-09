from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import json


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

# Load all annotations
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_obj = 0

det_results = []
seg_results = []
all_img_ids = []

for annotation in all_anns:
    if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
        continue

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/images/%s/%s' % (dataDir, dataType, img['file_name'])
    w_img = int(img['width'])
    h_img = int(img['height'])
    # if w_img < 1 or h_img < 1:
    #     continue

    counter_obj += 1
    if annotation['image_id'] not in all_img_ids:
        all_img_ids.append(annotation['image_id'])

    # polygons = annotation['segmentation'][0]
    bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    det = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'bbox': bbox
    }
    det_results.append(det)

    # convert polygons to masks, then to RLE
    polygons = annotation['segmentation']
    if len(polygons) > 1:
        bg = np.zeros((h_img, w_img, 1), dtype=np.uint8)
        for poly in polygons:
            len_poly = len(poly)
            vertices = np.zeros((1, len_poly // 2, 2), dtype=np.int32)
            for i in range(len_poly // 2):
                vertices[0, i, 0] = int(poly[2 * i])
                vertices[0, i, 1] = int(poly[2 * i + 1])
            # cv2.fillPoly(bg, vertices, color=(255))
            cv2.drawContours(bg, vertices, color=(255), contourIdx=-1, thickness=-1)

        pads = 5
        while True:
            kernel = np.ones((pads, pads), np.uint8)
            bg_closed = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
            obj_contours, _ = cv2.findContours(bg_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(obj_contours) > 1:
                pads += 5
            else:
                polygons = np.ndarray.flatten(obj_contours[0]).tolist()
                break
    else:
        # continue
        polygons = annotation['segmentation'][0]

    # len_poly = len(polygons)
    # vertices = np.zeros((1, len_poly // 2, 2), dtype=np.int32)
    # for i in range(len_poly // 2):
    #     vertices[0, i, 0] = int(polygons[2 * i])
    #     vertices[0, i, 1] = int(polygons[2 * i + 1])
    #
    # bg = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    # cv2.fillPoly(bg, vertices, color=(255, 255, 255))
    # rle = encode_mask((bg[:, :, 0] // 255).astype(np.uint8))

    # convert polygons to rle masks
    rles = cocomask.frPyObjects([polygons], h_img, w_img)
    rle = cocomask.merge(rles)  # ['counts'].decode('ascii')
    m = cocomask.decode(rle)
    rle_new = encode_mask(m.astype(np.uint8))

    seg = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'segmentation': rle_new
    }
    seg_results.append(seg)

    # take object with multiple parts as multiple objects
    # for poly in polygons:
    #     rles = cocomask.frPyObjects([poly], h_img, w_img)
    #     rle = cocomask.merge(rles)
    #     m = cocomask.decode(rle)
    #     rle_new = encode_mask(m.astype(np.uint8))
    #
    #     seg = {
    #         'image_id': annotation['image_id'],
    #         'category_id': cat_id,
    #         'score': 1.,
    #         'segmentation': rle_new
    #     }
    #     seg_results.append(seg)

with open('{}/results/{}_det_results_original.json'.format(dataDir, dataType), 'w') as f_det:
    json.dump(det_results, f_det)
with open('{}/results/{}_seg_results_original.json'.format(dataDir, dataType), 'w') as f_seg:
    json.dump(seg_results, f_seg)

# run COCO detection evaluation
print('Running COCO detection val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_det_results_original.json'.format(dataDir, dataType))
imgIds = sorted(coco.getImgIds())
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_seg_results_original.json'.format(dataDir, dataType))
coco_eval = COCOeval(coco, coco_pred, 'segm')
coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
