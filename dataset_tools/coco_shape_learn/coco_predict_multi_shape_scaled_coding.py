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


mask_size = 40
num_vertices = 360
alpha = 0.01
n_coeffs = 64

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
dictFile = '{}/sparse_shape_dict/multi_fromMask_dict_wild_m{}_n{}_v{}_a{:.2f}.npy'.format(dataDir, mask_size, n_coeffs, num_vertices, alpha)
# statFile = '{}/dictionary/train_stat_v{}_n{}_a{:.2f}.npy'.format(dataDir, num_vertices, n_coeffs, alpha)  # code mean and std
learned_dict = np.load(dictFile)

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

    if annotation['iscrowd'] == 1:
        rle = cocomask.merge(annotation['segmentation'])
        m = cocomask.decode(rle).astype(np.uint8) * 255
    else:
        original_rles = cocomask.frPyObjects(annotation['segmentation'], h_img, w_img)
        rle = cocomask.merge(original_rles)
        m = cocomask.decode(rle).astype(np.uint8) * 255

    # m = np.zeros((h_img, w_img), dtype=np.uint8)
    # for poly in annotation['segmentation']:
    #     vertices = np.round(np.array(poly).reshape(1, -1, 2)).astype(np.int32)
    #     cv2.drawContours(m, vertices, color=255, contourIdx=-1, thickness=-1)

    m_bbox = m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)]  # crop the mask according to the bbox
    # m_bbox = np.pad(m_bbox, 1, mode='constant')

    dist_bbox = cv2.resize(m_bbox, dsize=(mask_size, mask_size))  # rescale to fixed size masks
    dist_bbox = np.where(dist_bbox >= 255 * 0.5, 255, 0).astype(np.uint8)

    pads = 5
    obj_contours, _ = cv2.findContours(dist_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    while len(obj_contours) > 1:
        kernel = np.ones((pads, pads), np.uint8)
        dist_bbox_closed = cv2.morphologyEx(dist_bbox, cv2.MORPH_CLOSE, kernel)
        obj_contours, _ = cv2.findContours(dist_bbox_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(obj_contours) > 1:
            if pads <= mask_size // 2:
                pads += 5
            else:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)  # get the largest masks
                break

    contour = obj_contours[-1].reshape(-1, 2)
    uni_contour = uniformsample(contour, num_vertices)
    # norm_shape = uni_contour * 1. / mask_size
    norm_shape = uni_contour * 1.

    # sparsing coding using pre-learned dict
    learned_val_codes, _ = fast_ista(norm_shape.reshape((1, -1)), learned_dict, lmbda=alpha, max_iter=100)
    recon_contour = np.matmul(learned_val_codes, learned_dict)
    # recon_contour = np.floor(np.clip(recon_contour, 0, 0.999) * mask_size).astype(np.int32)
    recon_contour = np.round(np.clip(recon_contour, 0, mask_size - 1)).astype(np.int32)

    poly_rc = np.ndarray.flatten(recon_contour, order='C').tolist()  # row major flatten
    rles = cocomask.frPyObjects([poly_rc], mask_size, mask_size)
    rle = cocomask.merge(rles)
    m = cocomask.decode(rle).astype(np.uint8) * 255

    recon_ori_masks = cv2.resize(m, dsize=(int(gt_w), int(gt_h)))
    recon_ori_masks = np.where(recon_ori_masks > 0.5 * 255, 1, 0).astype(np.uint8)

    recon_m = np.zeros((h_img, w_img), dtype=np.uint8)
    recon_m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)] = recon_ori_masks
    rle_new = encode_mask(recon_m.astype(np.uint8))

    show_img = np.concatenate([m_bbox, recon_ori_masks * 255], axis=1)
    print(m_bbox.shape, recon_ori_masks.shape)
    cv2.imshow('cat', show_img)
    if cv2.waitKey() & 0xFF == ord('q'):
        exit()

    # img = cv2.imread(image_name)
    # show_img = np.concatenate([dist_bbox, recon_contour], axis=1)
    # cv2.imshow('cat', show_img.astype(np.uint8) * 255)
    # cv2.waitKey()
    # print(m_bbox.shape, recon_ori_masks.shape)
    # show_img = np.concatenate([m_bbox, recon_ori_masks.astype(np.uint8) * 255], axis=1)
    # cv2.imshow('cat', show_img)
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     break

    counts_codes.append(np.sum(learned_val_codes != 0))

    bbox_out = list(map(lambda x: float("{:.2f}".format(x)), gt_bbox))
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
    # cv2.imshow('Poly', img)
    # cv2.waitKey()

    # fig = plt.figure()
    # plt.hist(learned_val_codes[0], bins=100, color='g', alpha=0.5)
    # plt.xlabel('Coefficients')
    # plt.title('Sparse Coding of a {}'.format(cat_name))
    # plt.show()

    # convert reconstructed masks to original rle masks
    recon_m = np.zeros((h_img, w_img), dtype=np.uint8)
    recon_m[int(gt_y1):int(gt_y1 + gt_h), int(gt_x1):int(gt_x1 + gt_w)] = recon_ori_masks
    rle_new = encode_mask(recon_m.astype(np.uint8))

    # original_rles = cocomask.frPyObjects(annotation['segmentation'], h_img, w_img)
    # rle = cocomask.merge(original_rles)
    # m = cocomask.decode(rle)
    # rle_original = cocomask.encode(m.astype(np.uint8))
    # iou = cocomask.iou([rle_new], [rle_original], np.zeros((1,), dtype=np.uint8))
    # iou = cocomask.iou([rle_new], [rle_original], [annotation['iscrowd']])
    # iou = cocomask.iou([rle_original], [rle_original], [annotation['iscrowd']])
    cropped_gt_rle = encode_mask(m_bbox.astype(np.uint8) // 255)
    cropped_recon_rle = encode_mask(recon_ori_masks.astype(np.uint8))
    iou = cocomask.iou([cropped_gt_rle], [cropped_recon_rle], [annotation['iscrowd']])
    mean_IoUs.append(iou[0][0])

    # if iou[0][0] < 0.75:
    #     print(m_bbox.shape, recon_ori_masks.shape)
    #     print('current iou: ', iou[0][0])
    #
    #     cropped_gt_rle = encode_mask(m_bbox.astype(np.uint8) // 255)
    #     cropped_recon_rle = encode_mask(recon_ori_masks.astype(np.uint8))
    #     print('cropped iou: ', cocomask.iou([cropped_gt_rle], [cropped_recon_rle], [annotation['iscrowd']])[0][0])
    #     show_img = np.concatenate([m_bbox, recon_ori_masks.astype(np.uint8) * 255], axis=1)
    #     cv2.imshow('cat', show_img)
    #     if cv2.waitKey() & 0xFF == ord('q'):
    #         exit()

    seg = {
        'image_id': annotation['image_id'],
        'category_id': cat_id,
        'score': 0.99,
        'segmentation': rle_new
    }
    seg_results.append(seg)

with open('{}/results/{}_scaled_det_results_v{}.json'.format(dataDir, dataType, num_vertices), 'w') as f_det:
    json.dump(det_results, f_det)
with open('{}/results/{}_scaled_seg_results_v{}.json'.format(dataDir, dataType, num_vertices), 'w') as f_seg:
    json.dump(seg_results, f_seg)

# run COCO detection evaluation
# print('Running COCO detection val17 evaluation ...')
# coco_pred = coco.loadRes('{}/results/{}_scaled_det_results_v{}.json'.format(dataDir, dataType, num_vertices))
# coco_eval = COCOeval(coco, coco_pred, 'bbox')
# coco_eval.evaluate()
# coco_eval.accumulate()
# coco_eval.summarize()

print('---------------------------------------------------------------------------------')
print('Running COCO segmentation val17 evaluation ...')
coco_pred = coco.loadRes('{}/results/{}_scaled_seg_results_v{}.json'.format(dataDir, dataType, num_vertices))
coco_eval = COCOeval(coco, coco_pred, 'segm')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print('Average active codes: ', np.mean(counts_codes) / n_coeffs)
print('Average mIoU of all instances: ', np.mean(mean_IoUs))
