from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista
from dataset_tools.coco_utils.utils import check_clockwise_polygon
import numpy as np
import cv2
import json
from scipy.signal import resample
from dataset_tools.coco_utils.utils import get_connected_polygon_using_mask, turning_angle_resample, \
    get_connected_polygon_with_mask
import matplotlib.pyplot as plt
import torch


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


num_vertices = 32
# alpha = 0.1
# n_coeffs = 128

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# dictFile = '{}/dictionary/train_dict_v{}_n{}_a{:.2f}.npy'.format(dataDir, num_vertices, n_coeffs, alpha)
# learned_dict = np.load(dictFile)

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

    polygons = get_connected_polygon_with_mask(annotation['segmentation'], (h_img, w_img),
                                                           n_vertices=num_vertices, closing_max_kernel=50)
    gt_bbox = annotation['bbox']
    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    contour = np.array(polygons).reshape((-1, 2))

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    # Downsample the contour to fix number of vertices
    if len(contour) > num_vertices:
        fixed_contour = resample(contour, num=num_vertices)
    else:
        fixed_contour = turning_angle_resample(contour, num_vertices)

    fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
    fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

    clockwise_flag = check_clockwise_polygon(fixed_contour)
    if not clockwise_flag:
        fixed_contour = np.flip(fixed_contour, axis=0)
    else:
        fixed_contour = fixed_contour.copy()

    # Indexing from the left-most vertex, argmin x-axis
    idx = np.argmin(fixed_contour[:, 0])
    indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

    x1, y1, x2, y2 = min(fixed_contour[:, 0]), min(fixed_contour[:, 1]), \
                     max(fixed_contour[:, 0]), max(fixed_contour[:, 1])
    bbox = [x1, y1, x2 - x1, y2 - y1]
    shape_center = np.array([x1, y1])
    shape_scale = np.array([bbox[2], bbox[3]])
    # shape_center = np.mean(indexed_shape, axis=0)

    centered_shape = indexed_shape - shape_center

    # FFT to get Fourier Descriptors, consider each point as x + yj
    # p_complex = np.empty(centered_shape.shape[:-1], dtype=complex)
    # p_complex.real = centered_shape[:, 0]
    # p_complex.imag = centered_shape[:, 1]
    #
    # fourier_result = np.fft.fft(p_complex) / 32
    # # print(np.abs(fourier_result))
    # print(fourier_result)
    # print(fourier_result.real)
    # print('Length of descriptors: ', len(fourier_result))
    # plt.plot(np.real(fourier_result), 'g')
    # plt.plot(np.imag(fourier_result), 'r')
    # plt.legend(['real', 'imaginary'])
    # plt.show()

    # reconstruct = np.fft.ifft(fourier_result * 32)
    # reconstruct = np.array([reconstruct.real, reconstruct.imag])
    # reconstruct = np.fft.ifftn(np.transpose(fourier_result))
    # recon_contour = np.transpose(reconstruct) + shape_center

    # perform fft to x and y-axis independently
    obj_shape = torch.FloatTensor(centered_shape)
    fourier = torch.fft(obj_shape, 1) / num_vertices
    # fourier_y = torch.fft(obj_shape[:, 1], 0) / num_vertices
    # fourier_x = np.fft.fft(centered_shape[:, 0]) / num_vertices
    # fourier_y = np.fft.fft(centered_shape[:, 1]) / num_vertices
    # print(centered_shape.shape)
    # print(fourier.size(), obj_shape.size(), fourier)
    # print(fourier_x.real)
    # print('Length of descriptors: ', len(fourier_x))
    plt.plot(fourier[:, 0], 'g')
    plt.plot(fourier[:, 1], 'r')
    # plt.plot(np.imag(fourier_y))
    # plt.plot(np.imag(fourier_y))
    plt.legend(['real', 'imag'])
    plt.show()

    # fourier = torch.reshape(fourier, shape=(1, num_vertices, 2))
    # print(torch.ifft(fourier * 32, 1).size())
    recon_contour = torch.ifft(fourier * 32, 1).numpy() + shape_center
    # recon_contour[:, 1] = torch.ifft(torch.from_numpy(fourier_y.real * 32)) + shape_center[1]

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

    # visualize resampled points in image
    # img = cv2.imread(image_name)
    # img_recon = cv2.imread(image_name)
    # cv2.polylines(img, [indexed_shape.astype(np.int32)], True, (10, 255, 10), thickness=2)
    # cv2.polylines(img_recon, [recon_contour.astype(np.int32)], True, (10, 10, 255), thickness=2)
    # im_cat = np.concatenate((img, img_recon), axis=1)
    # cv2.imshow('Poly Original vs. Fourier', im_cat)
    # cv2.waitKey(1000)

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

# print('Average activated codes per shape: ', np.mean(counts_codes) / n_coeffs)
