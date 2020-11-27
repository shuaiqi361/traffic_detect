from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from scipy.signal import resample
from copy import deepcopy
from dataset_tools.coco_utils.utils import turning_angle_resample, get_connected_polygon_using_mask, \
    get_connected_polygon_with_mask


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
counter_multiparts = 0
counter_complicated = 0
counter_simple = 0

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

    # polygons, is_simple = get_connected_polygon_using_mask(annotation['segmentation'], (h_img, w_img),
    #                                                        n_vertices=num_vertices, closing_max_kernel=50)
    polygons = get_connected_polygon_with_mask(annotation['segmentation'], (h_img, w_img),
                                                           n_vertices=num_vertices, closing_max_kernel=50)
    # if is_simple == 0:
    #     counter_multiparts += 1
    # elif is_simple == 1:
    #     counter_simple += 1
    # else:
    #     counter_complicated += 1
    # polygons = get_connected_polygon(annotation['segmentation'], (h_img, w_img))
    contour = np.array(polygons).reshape((-1, 2))
    bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    gt_x1, gt_y1, gt_w, gt_h = bbox

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    # if cat_name != 'bus' and cat_name != 'dog':
    #     continue

    # Downsample the contour to fix number of vertices
    # fixed_contour = resample(contour, num=num_vertices)
    # print('0-- ', annotation['segmentation'][0][0:2])
    # print('1-- ', contour[0])
    if len(contour) > num_vertices:
        fixed_contour = resample(contour, num=num_vertices)
        # fixed_contour = align_original_polygon(fixed_contour_, contour)
    else:
        fixed_contour = turning_angle_resample(contour, num_vertices)
    assert len(fixed_contour) == num_vertices

    fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
    fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

    # print('2-- ', fixed_contour[0])

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
    original_rles = cocomask.frPyObjects(annotation['segmentation'], h_img, w_img)
    rle = cocomask.merge(original_rles)
    m = cocomask.decode(rle)
    rle_original = cocomask.encode(m.astype(np.uint8))
    iou = cocomask.iou([rle_resampled], [rle_original], np.zeros((1,), dtype=np.uint8))
    obj_ious.append(iou[0][0])

    # visualize resampled points in image side by side
    # if iou[0][0] < 0.95:
    #     print('Image id:', annotation['image_id'])
    #     img = cv2.imread(image_name)
    #     img_ref = cv2.imread(image_name)
    #     # plot the original gt polygons
    #     for po in annotation['segmentation']:
    #         tmp_contour = np.array(po).reshape((-1, 2))
    #         cv2.polylines(img_ref, [tmp_contour.astype(np.int32)], True, (10, 10, 255), thickness=2)
    #     cv2.polylines(img, [fixed_contour.astype(np.int32)], True, (10, 10, 255), thickness=2)
    #     image = cv2.putText(img, 'IoU:{:.3f}'.format(iou[0][0]), (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #                         fontScale=1, color=(0, 0, 200), thickness=2)
    #     im_cat = np.concatenate((img_ref, img), axis=1)
    #     cv2.imshow('Original vs. Resampled', im_cat)
    #     cv2.waitKey()

    # cv2.imshow('mask m', m * 255)
    # print('Values in m: ', np.sum(m))
    # cv2.waitKey()

    # visualize resampled points in image side by side for small objects
    # print('3-- ', fixed_contour[0])
    # exit()
    # if iou[0][0] < 0.45 and len(annotation['segmentation']) == 1:
    #     print('Image id:', annotation['image_id'])
    #     # print(m.shape)
    #     fig = plt.figure()
    #     # plot the original gt polygons
    #     for po in annotation['segmentation']:
    #         tmp_contour = np.array(po).reshape((-1, 2))
    #         plt.plot(tmp_contour[:, 0], tmp_contour[:, 1], color='green', marker='o', linestyle='--',
    #                  linewidth=2, markersize=6)
    #     plt.plot(fixed_contour[:, 0], fixed_contour[:, 1], color='blue', marker='+', linestyle='-',
    #              linewidth=1, markersize=6)
    #     # resampled_poly = deepcopy(fixed_contour)
    #     # resampled_poly = resampled_poly.astype(np.uint8)
    #     # plt.plot(resampled_poly[:, 0], resampled_poly[:, 1], color='red', marker='*', linestyle='-',
    #     #          linewidth=1, markersize=6)
    #     plt.title('original versus resampled(iou:{:.3f})'.format(iou[0][0]))
    #     plt.legend(['GT', 'Resampled', 'Quant'])
    #     plt.show()


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
print('OBJECTS: multi {}, simple {}, complicated {}'.format(counter_multiparts, counter_simple, counter_complicated))
print('IoU stats of {} instances, mean {:.3f}, std {:.3f}'.format(len(obj_ious), np.mean(obj_ious), np.std(obj_ious)))
print('Number of objects(total {}) above threshold 0.75: {}/{}'.format(len(obj_ious),
                                                                      np.sum(np.array(obj_ious) < 0.75),
                                                                      1 - np.sum(np.array(obj_ious) < 0.75) * 1. / len(obj_ious)))
fig = plt.figure()
arr = plt.hist(obj_ious, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], color='g')
plt.rcParams.update({'font.size': 8})
for i in range(10):
    plt.text(arr[1][i], arr[0][i], '{:.2f}%'.format(arr[0][i] * 100. / len(obj_ious)))
plt.xlabel('Intersection over Unions')
plt.ylabel('Number of Instances')
plt.title('IoUs of {}-resampled instances on COCO val17'.format(num_vertices))
plt.show()


