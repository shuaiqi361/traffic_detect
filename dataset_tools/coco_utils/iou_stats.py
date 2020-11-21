from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from scipy.signal import resample
from dataset_tools.coco_utils.utils import turning_angle_resample, align_original_polygon, get_connected_polygon


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)

num_vertices = 32

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
obj_ious = []

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

    polygons = get_connected_polygon(annotation['segmentation'], (h_img, w_img))
    contour = np.array(polygons).reshape((-1, 2))
    bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    gt_x1, gt_y1, gt_w, gt_h = bbox

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    # Downsample the contour to fix number of vertices
    if len(contour) > num_vertices:
        fixed_contour = resample(contour, num=num_vertices)
        # fixed_contour = align_original_polygon(fixed_contour_, contour)
    else:
        fixed_contour = turning_angle_resample(contour, num_vertices)
    assert len(fixed_contour) == num_vertices
    fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
    fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

    det = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'bbox': bbox
    }
    det_results.append(det)

    # convert polygons to rle masks
    poly = np.ndarray.flatten(fixed_contour, order='C').tolist()  # row major flatten
    rles = cocomask.frPyObjects([poly], h_img, w_img)
    rle = cocomask.merge(rles)
    m = cocomask.decode(rle)
    rle_new = encode_mask(m.astype(np.uint8))
    rle_resampled = cocomask.encode(m.astype(np.uint8))

    seg = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'segmentation': rle_new
    }
    seg_results.append(seg)

    # calculate iou between masks
    original_rles = cocomask.frPyObjects([polygons], h_img, w_img)
    rle = cocomask.merge(original_rles)
    m = cocomask.decode(rle)
    rle_original = cocomask.encode(m.astype(np.uint8))
    iou = cocomask.iou([rle_resampled], [rle_original], np.zeros((1,), dtype=np.uint8))
    obj_ious.append(iou[0][0])

    # visualize resampled points in image side by side
    # if iou[0][0] < 0.6:
    #     img = cv2.imread(image_name)
    #     img_ref = cv2.imread(image_name)
    #     cv2.polylines(img_ref, [contour.astype(np.int32)], True, (10, 10, 255), thickness=2)
    #     cv2.polylines(img, [fixed_contour.astype(np.int32)], True, (10, 10, 255), thickness=2)
    #     image = cv2.putText(img, 'IoU:{:.3f}'.format(iou[0][0]), (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #                         fontScale=1, color=(0, 0, 200), thickness=2)
    #     im_cat = np.concatenate((img_ref, img), axis=1)
    #     cv2.imshow('Original vs. Resampled', im_cat)
    #     cv2.waitKey()


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
coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


# Show statistics of IoUs between objects
print('IoU stats of {} instances, mean {:.3f}, std {:.3f}'.format(len(obj_ious), np.mean(obj_ious), np.std(obj_ious)))
fig = plt.figure()
plt.hist(obj_ious, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], color='g')
plt.xlabel('Intersection over Unions')
plt.ylabel('Number of Instances')
plt.title('IoUs of resampled instances on COCO val17')
plt.ylim((0, 1000))
plt.show()
