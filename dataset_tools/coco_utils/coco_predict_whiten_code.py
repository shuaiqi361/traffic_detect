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
from scipy.signal import resample
from dataset_tools.coco_utils.utils import get_connected_polygon, turning_angle_resample, \
    get_connected_polygon_with_mask, uniformsample


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


num_vertices = 128
alpha = 0.01
n_coeffs = 64

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# dictFile = '{}/dictionary/train_scaled_dict_v{}_n{}_a{:.2f}.npy'.format(dataDir, num_vertices, n_coeffs, alpha)
statFile = '{}/dictionary/train_whiten_stats_v{}_n{}_a{:.2f}.npz'.format(dataDir, num_vertices, n_coeffs, alpha)  # code mean and std
with np.load(statFile) as data_meta:
    learned_dict = data_meta['dictionary']
    shape_mean = data_meta['mean']
    shape_std = data_meta['std']

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

    gt_bbox = annotation['bbox']
    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    contour = np.array(polygons).reshape((-1, 2))

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    # Downsample the contour to fix number of vertices
    fixed_contour = uniformsample(contour, num_vertices)

    clockwise_flag = check_clockwise_polygon(fixed_contour)
    if not clockwise_flag:
        fixed_contour = np.flip(fixed_contour, axis=0)
    else:
        fixed_contour = fixed_contour.copy()

    # Indexing from the left-most vertex, argmin x-axis
    idx = np.argmin(fixed_contour[:, 0])
    indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

    # # draw re-sampled points
    # fig = plt.figure()
    # plt.title(cat_name)
    # plt.plot(indexed_shape[:, 0], -indexed_shape[:, 1], '-o', c='C0', lw=2)
    # plt.plot(contour[:, 0], -contour[:, 1], '-o', c='r', lw=2)
    # for i in range(num_vertices):
    #     plt.text(indexed_shape[i, 0], -indexed_shape[i, 1], '{}'.format(i), fontsize=12)
    # # plt.text(indexed_shape[1, 0], indexed_shape[1, 1], '1', fontsize=8)
    # plt.show()

    # visualize original resampled points in image
    # img = cv2.imread(image_name)
    # cv2.polylines(img, [np.round(fixed_contour).astype(np.int32)], True, (0, 0, 255))
    # cv2.imshow('Poly', img)
    # cv2.waitKey()

    # x1, y1, x2, y2 = min(fixed_contour[:, 0]), min(fixed_contour[:, 1]), \
    #                  max(fixed_contour[:, 0]), max(fixed_contour[:, 1])
    #
    # bbox_width, bbox_height = x2 - x1, y2 - y1
    # bbox = [x1, y1, bbox_width, bbox_height]
    # bbox_center = np.array([(x1 + x2) / 2., (y1 + y2) / 2.])
    # shape_center = np.mean(indexed_shape, axis=0)

    updated_bbox = [np.min(indexed_shape[:, 0]), np.min(indexed_shape[:, 1]),
                    np.max(indexed_shape[:, 0]), np.max(indexed_shape[:, 1])]
    updated_width = np.max(indexed_shape[:, 0]) - np.min(indexed_shape[:, 0])
    updated_height = np.max(indexed_shape[:, 1]) - np.min(indexed_shape[:, 1])

    shifted_shape = indexed_shape - np.array([updated_bbox[0], updated_bbox[1]])
    norm_shape = shifted_shape / np.array([updated_width, updated_height])

    # normalize across shapes in the training set
    norm_shape = (norm_shape.reshape((1, -1)) - shape_mean) / shape_std

    # sparsing coding using pre-learned dict
    learned_val_codes, _ = fast_ista(norm_shape, learned_dict, lmbda=alpha, max_iter=80)
    recon_contour = np.matmul(learned_val_codes, learned_dict) * shape_std + shape_mean
    recon_contour = recon_contour.reshape((-1, 2)) * np.array([updated_width, updated_height]) + np.array([updated_bbox[0], updated_bbox[1]])

    counts_codes.append(np.sum(learned_val_codes != 0))

    x1, y1, x2, y2 = min(recon_contour[:, 0]), min(recon_contour[:, 1]), \
                     max(recon_contour[:, 0]), max(recon_contour[:, 1])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox))
    det = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 1.,
        'bbox': bbox_out
    }
    det_results.append(det)

    # visualize reconstructed resampled points in image
    # img = cv2.imread(image_name)
    # cv2.polylines(img, [recon_contour.astype(np.int32)], True, (0, 0, 255))
    # cv2.polylines(img, [indexed_shape.astype(np.int32)], True, (0, 255, 0))
    # cv2.imshow('Poly', img)
    # cv2.waitKey()
    #
    # fig = plt.figure()
    # plt.hist(learned_val_codes[0], bins=50, color='g', alpha=0.5)
    # plt.xlabel('Coefficients')
    # plt.title('Sparse Coding of a {}'.format(cat_name))
    # plt.show()

    # convert polygons to rle masks
    poly = np.ndarray.flatten(recon_contour, order='C').tolist()  # row major flatten
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

with open('{}/results/{}_scaled_det_results_v{}.json'.format(dataDir, dataType, num_vertices), 'w') as f_det:
    json.dump(det_results, f_det)
with open('{}/results/{}_scaled_seg_results_v{}.json'.format(dataDir, dataType, num_vertices), 'w') as f_seg:
    json.dump(seg_results, f_seg)

# run COCO detection evaluation
print('Running COCO detection val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_scaled_det_results_v{}.json'.format(dataDir, dataType, num_vertices))
imgIds = sorted(coco.getImgIds())
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_scaled_seg_results_v{}.json'.format(dataDir, dataType, num_vertices))
coco_eval = COCOeval(coco, coco_pred, 'segm')
coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print('Average active codes: ', np.mean(counts_codes) / n_coeffs)
