from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import json
from scipy.signal import resample
from dataset_tools.coco_utils.utils import turning_angle_resample, get_connected_polygon_coco_mask, \
    get_connected_polygon_using_mask, get_connected_polygon_with_mask, uniform_sample_segment, uniformsample, \
    get_connected_polys_with_measure, close_contour


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

num_vertices = 128

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
    if w_img < 1 or h_img < 1:
        continue

    counter_obj += 1
    if annotation['image_id'] not in all_img_ids:
        all_img_ids.append(annotation['image_id'])

    if len(annotation['segmentation']) > 1:
        obj_contours = [np.array(s).reshape((-1, 2)).astype(np.int32) for s in annotation['segmentation']]
        obj_contours = sorted(obj_contours, key=cv2.contourArea)
        polygons = obj_contours[-1]
    else:
        polygons = annotation['segmentation'][0]

    contour = np.array(polygons).reshape((-1, 2))
    # if cv2.contourArea(contour.astype(np.int32)) > 50:
    #     continue

    bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    gt_x1, gt_y1, gt_w, gt_h = bbox

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    fixed_contour = uniformsample(contour, num_vertices)

    # fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
    # fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

    x1, y1, x2, y2 = min(fixed_contour[:, 0]), min(fixed_contour[:, 1]), \
                     max(fixed_contour[:, 0]), max(fixed_contour[:, 1])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox))
    det = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'bbox': bbox_out
    }
    det_results.append(det)

    # visualize resampled points in image side by side
    # print('contour {} : {} -- area {:.2f}'.format(len(contour), len(fixed_contour), cv2.contourArea(contour.astype(np.int32))))
    # img = cv2.imread(image_name)
    # img_ref = cv2.imread(image_name)
    # img_final = cv2.imread(image_name)
    # cv2.polylines(img_ref, [contour.astype(np.int32)], True, (10, 10, 255), thickness=1)
    # for point in contour.astype(np.int32):
    #     cv2.circle(img_ref, tuple(point), 3, (0, 0, 255))
    # # cv2.polylines(img, [fixed_contour_.astype(np.int32)], True, (10, 10, 255), thickness=2)
    # cv2.polylines(img_final, [fixed_contour.astype(np.int32)], True, (10, 10, 255), thickness=1)
    # for point in fixed_contour.astype(np.int32):
    #     cv2.circle(img_final, tuple(point), 3, (0, 0, 255))
    # im_cat = np.concatenate((img_ref, img_final), axis=1)
    # cv2.imshow('Poly Original vs. Resampled vs Aligned', im_cat)
    # cv2.waitKey()

    # convert polygons to rle masks
    poly = np.ndarray.flatten(fixed_contour, order='C').tolist()  # row major flatten
    rles = cocomask.frPyObjects([poly], h_img, w_img)
    rle = cocomask.merge(rles)
    m = cocomask.decode(rle)
    rle_new = encode_mask(m.astype(np.uint8))

    seg = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'segmentation': rle_new
    }
    seg_results.append(seg)

with open('{}/results/{}_det_results_v{}.json'.format(dataDir, dataType, num_vertices), 'w') as f_det:
    json.dump(det_results, f_det)
with open('{}/results/{}_seg_results_v{}.json'.format(dataDir, dataType, num_vertices), 'w') as f_seg:
    json.dump(seg_results, f_seg)

# run COCO detection evaluation
print('Running COCO detection val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_det_results_v{}.json'.format(dataDir, dataType, num_vertices))
imgIds = sorted(coco.getImgIds())
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_seg_results_v{}.json'.format(dataDir, dataType, num_vertices))
coco_eval = COCOeval(coco, coco_pred, 'segm')
# coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
