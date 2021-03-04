from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
from dataset_tools.coco_utils.utils import check_clockwise_polygon
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy import linalg
from dataset_tools.coco_utils.utils import intersect
from dataset_tools.coco_utils.utils import turning_angle_resample, get_connected_polygon_coco_mask, \
    get_connected_polygon_using_mask, get_connected_polygon_with_mask, uniform_sample_segment, uniformsample, \
    get_connected_polys_with_measure, close_contour

mask_size = 40
n_vertices = 360
n_coeffs = 64
alpha = 0.01

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'sparse_shape_dict')
out_dict = '{}/single_fromMask_Dict_m{}_n{}_v{}_a{:.2f}.npy'.format(save_data_root, mask_size, n_coeffs, n_vertices, alpha)
out_resampled_shape_file = '{}/train_single_fromMask_m{}_v{}.npy'.format(save_data_root, mask_size, n_vertices)


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_iscrowd = 0  # remove crowd annotated objects
counter_total = 0  # total number of segments
counter_multiple = 0
counter_valid = 0
counter_poor = 0  # objects too small to extract the shape
n_classes = 80
counter_max = {}  # ensure that shapes from all categories are equally drawn
shape_per_cat = 500

COCO_original_shape_objects = []  # all objects
# COCO_resample_shape_matrix = np.zeros(shape=(0, n_vertices * 2))
COCO_resample_shape_matrix = []
for annotation in all_anns:
    if sum([c for _, c in counter_max.items()]) == shape_per_cat * n_classes:
        break
    if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
        counter_iscrowd += 1
        continue

    counter_total += 1

    if counter_total % 10000 == 0:
        print('Processing {}/{} ...', counter_total, len(all_anns))

    img = coco.loadImgs(annotation['image_id'])[0]
    image_name = '%s/coco/%s/%s' % (dataDir, dataType, img['file_name'])
    w_img = img['width']
    h_img = img['height']
    if w_img < 1 or h_img < 1:
        continue

    # filter out small shapes and disconnected shapes for learning the dictionary
    if annotation['area'] < 150 or len(annotation['segmentation']) > 1:
        continue

    # if the current shape reach its max in the counter list, skip
    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']
    if cat_name not in counter_max.keys():
        counter_max[cat_name] = 1
    else:
        if counter_max[cat_name] == shape_per_cat:
            continue
        counter_max[cat_name] += 1

    if len(annotation['segmentation']) > 1:
        obj_contours = [np.array(s).reshape((-1, 2)).astype(np.int32) for s in annotation['segmentation']]
        obj_contours = sorted(obj_contours, key=cv2.contourArea)
        polygons = obj_contours[-1]
        counter_multiple += 1
    else:
        polygons = annotation['segmentation'][0]

    # gt_bbox = annotation['bbox']  # top-left corner coordinates, width and height convention
    # gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    gt_shape = np.array(polygons).reshape((-1, 2))
    gt_x1, gt_y1, gt_x2, gt_y2 = int(np.min(gt_shape[:, 0])), int(np.min(gt_shape[:, 1])), \
                                 int(np.max(gt_shape[:, 0])), int(np.max(gt_shape[:, 1]))
    gt_w, gt_h = gt_x2 - gt_x1, gt_y2 - gt_y1

    rles = cocomask.frPyObjects([polygons], h_img, w_img)
    rle = cocomask.merge(rles)  # ['counts'].decode('ascii')
    m = cocomask.decode(rle).astype(np.uint8) * 255  # in image domain
    # m = np.zeros((h_img, w_img), dtype=np.uint8)
    # for poly in annotation['segmentation']:
    #     vertices = np.round(np.array(poly).reshape(1, -1, 2)).astype(np.int32)
    #     cv2.drawContours(m, vertices, color=255, contourIdx=-1, thickness=-1)

    m_bbox = m[int(gt_y1):int(gt_y1 + gt_h + 1), int(gt_x1):int(gt_x1 + gt_w + 1)]  # crop the mask according to the bbox
    # m_bbox = np.pad(m_bbox, 1, mode='constant')
    # original_polygons = np.array(annotation['segmentation'][0]).reshape((-1, 2))
    # original_polygons = original_polygons - np.array([gt_x1, gt_y1])

    # cv2.imshow('bbox', m_bbox)
    # cv2.waitKey()
    # image = cv2.imread(image_name)
    # cv2.rectangle(image, pt1=(int(gt_x1), int(gt_y1)), pt2=(int(gt_x1 + gt_w), int(gt_y1 + gt_h)), color=(0, 255, 0))
    dist_bbox = cv2.resize(m_bbox, dsize=(mask_size, mask_size))  # rescale to fixed size masks
    dist_bbox = np.where(dist_bbox >= 0.5 * 255, 255, 0).astype(np.uint8)
    # obj_contours, _ = cv2.findContours(dist_bbox, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    obj_contours, _ = cv2.findContours(dist_bbox, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(obj_contours) > 1:
        obj_contours = sorted(obj_contours, key=cv2.contourArea)  # get the largest masks

    contour = obj_contours[-1].reshape(-1, 2)
    uni_contour = uniformsample(contour, n_vertices)

    # clockwise_flag = check_clockwise_polygon(uni_contour)
    # if not clockwise_flag:
    #     fixed_contour = np.flip(uni_contour, axis=0)
    # else:
    #     fixed_contour = uni_contour.copy()
    #
    # # Indexing from the left-most vertex, argmin x-axis
    # idx = np.argmin(fixed_contour[:, 0])
    # indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

    # dist_bbox = dist_bbox / np.max(dist_bbox) * 255.

    # Show the images and masks
    # cv2.imshow('image', image)
    # cv2.imshow('dist bbox', dist_bbox.astype(np.uint8))
    # cv2.waitKey()
    #
    # # draw re-sampled points
    # print(contour.shape)
    # fig = plt.figure()
    # plt.title(cat_name)
    # plt.plot(contour[:, 0], -contour[:, 1], '-o', c='green', lw=2.5)
    # plt.plot(uni_contour[:, 0], -uni_contour[:, 1], '-*', c='red', lw=1)
    # # plt.text(contour[0, 0], contour[0, 1], '0', fontsize=8)
    # # plt.text(contour[1, 0], contour[1, 1], '1', fontsize=8)
    # plt.show()

    # normalize the shape and store
    norm_shape = uni_contour * 1. / mask_size
    # recovered_shape = norm_shape * np.array([gt_w, gt_h])

    # # draw re-sampled points
    # fig = plt.figure()
    # print(contour.shape)
    # plt.title(cat_name)
    # plt.plot(norm_shape[:, 0], -norm_shape[:, 1], '-o', c='green', lw=2.5)
    # # plt.text(contour[0, 0], contour[0, 1], '0', fontsize=8)
    # # plt.text(contour[1, 0], contour[1, 1], '1', fontsize=8)
    # plt.show()

    fig = plt.figure()
    print(contour.shape)
    plt.title(cat_name)
    plt.plot(norm_shape[:, 0], -norm_shape[:, 1], '-o', c='green', lw=2.5)
    plt.text(norm_shape[0, 0], -norm_shape[0, 1], '0', fontsize=6)
    # plt.plot(recovered_shape[:, 0], -recovered_shape[:, 1], '--', c='red', lw=1.5)
    plt.show()

    COCO_resample_shape_matrix.append(norm_shape.reshape((1, -1)).astype(np.float16))
    # COCO_resample_shape_matrix = np.concatenate((COCO_resample_shape_matrix, norm_shape.reshape((1, -1))), axis=0)

COCO_resample_shape_matrix = np.concatenate(COCO_resample_shape_matrix, axis=0)
print('Total valid shape: ', counter_valid)
print('Poor shape: ', counter_poor)
print('Disconnected shape: ', counter_multiple)
print('Is crowd: ', counter_iscrowd)
print('Total number: ', counter_total)
print('Size of shape matrix: ', COCO_resample_shape_matrix.shape)
print('Length of the counter max:', len(counter_max))
np.save(out_resampled_shape_file, COCO_resample_shape_matrix)

# Start learning the dictionary
shape_data = np.load(out_resampled_shape_file)
print('Loading train2017 coco shape data: ', shape_data.shape)
n_shapes, n_feats = shape_data.shape

learned_dict, learned_codes, losses, error = iterative_dict_learning_fista(shape_data,
                                                                    n_components=n_coeffs,
                                                                    alpha=alpha,
                                                                    batch_size=400,
                                                                    n_iter=400)


print('Training error: ', error)
rec_error = 0.5 * linalg.norm(np.matmul(learned_codes, learned_dict) - shape_data) ** 2 / shape_data.shape[0]
print('Training Reconstruction error:', rec_error)
print('Outputing learned dictionary:', learned_dict.shape)

np.save(out_dict, learned_dict)

# count the number of self-intersections
total_counts = []
for i in range(n_coeffs):
    # for each shape basis, check every pair of edges in the polygon
    temp_basis = learned_dict[i, :].reshape((n_vertices, 2))
    temp_counts = 0
    for j in range(n_vertices):
        p1 = (temp_basis[j % n_vertices, 0], temp_basis[j % n_vertices, 1])
        p2 = (temp_basis[(j + 1) % n_vertices, 0], temp_basis[(j + 1) % n_vertices, 1])

        for k in range(j + 1, n_vertices):
            p3 = (temp_basis[k % n_vertices, 0], temp_basis[k % n_vertices, 1])
            p4 = (temp_basis[(k + 1) % n_vertices, 0], temp_basis[(k + 1) % n_vertices, 1])

            if intersect(p1, p2, p3, p4):
                temp_counts += 1

    total_counts.append(temp_counts - n_vertices)

print(total_counts)
print('Total intersections: {}, average {}'.format(sum(total_counts), np.mean(total_counts)))
