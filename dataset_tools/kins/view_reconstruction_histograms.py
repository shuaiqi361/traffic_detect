from pycocotools.coco import COCO
import numpy as np
from dataset_tools.coco_utils.utils import check_clockwise_polygon, uniformsample, close_contour
from sparse_coding.utils import fast_ista, iterative_dict_learning_fista
import random
import cv2
import copy
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy import linalg
from dataset_tools.coco_utils.utils import intersect
from dataset_tools.coco_utils.utils import get_connected_polygon, turning_angle_resample, \
    get_connected_polygon_with_mask

n_vertices = 32  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.05

src_root = '/media/keyi/Data/Research/traffic/data/KINS'
src_img_path = "/media/keyi/Data/Research/traffic/data/KINS/data_object_image_2/training/image_2"
src_gt_path = "/media/keyi/Data/Research/traffic/data/KINS/tools/update_train_2020.json"
coco = COCO(src_gt_path)

save_data_root = os.path.join(src_root, 'dictionary')
out_dict = '{}/train_dict_kins_v{}_n{}_a{:.2f}.npy'.format(save_data_root, n_vertices, n_coeffs, alpha)
learned_dict = np.load(out_dict)

coco = COCO(src_gt_path)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_iscrowd = 0  # remove crowd annotated objects
counter_total = 0  # total number of segments
counter_valid = 0
counter_poor = 0  # objects too small to extract the shape
active_rate = []

COCO_original_shape_objects = []  # all objects
COCO_resample_shape_matrix = np.zeros(shape=(0, n_vertices * 2))
for annotation in all_anns:

    # if counter_total % 10000 == 0:
    #     print('Processing {}/{} ...', counter_total, len(all_anns))

    img = coco.loadImgs(annotation['image_id'])[0]
    image_path = os.path.join(src_img_path, img['file_name'])
    w_img = img['width']
    h_img = img['height']
    if w_img < 1 or h_img < 1:
        continue

    polygons = annotation['a_segm'][0]

    if len(polygons) < 24 * 2:
        continue

    gt_bbox = annotation['a_bbox']  # top-left corner coordinates, width and height convention
    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    if cat_name != 'cyclist':
        continue
    # if cat_name == 'person-siting':
    #     continue

    # construct data matrix
    contour = np.array(polygons).reshape((-1, 2))
    fixed_contour = uniformsample(contour, n_vertices)

    clockwise_flag = check_clockwise_polygon(fixed_contour)
    if not clockwise_flag:
        canonic_contour = np.flip(fixed_contour, axis=0)
    else:
        canonic_contour = fixed_contour.copy()

    # Indexing from the left-most vertex
    idx = np.argmin(canonic_contour[:, 0])
    canonic_contour = np.concatenate((canonic_contour[idx:, :], canonic_contour[:idx, :]), axis=0)

    canonic_contour[:, 0] = np.clip(canonic_contour[:, 0], gt_x1, gt_x1 + gt_w)
    canonic_contour[:, 1] = np.clip(canonic_contour[:, 1], gt_y1, gt_y1 + gt_h)

    # Normalize the shapes
    contour_mean = np.mean(canonic_contour, axis=0)
    contour_std = np.sqrt(np.sum(np.std(canonic_contour, axis=0) ** 2))
    norm_shape = (canonic_contour - contour_mean) / contour_std

    learned_codes, _ = fast_ista(norm_shape.reshape((1, -1)), learned_dict, lmbda=alpha, max_iter=100)
    recon_contour = np.matmul(learned_codes, learned_dict).reshape((-1, 2))

    active_counts = np.sum((np.abs(learned_codes) > 1e-3))
    # print(active_counts)
    active_rate.append(active_counts / n_coeffs)

    # draw reconstruction results
    fig = plt.figure()
    plt.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    plt.rcParams.update({'font.size': 22})
    closed_norm = close_contour(norm_shape)
    closed_recon = close_contour(recon_contour)
    plt.title('shape reconstruction')
    plt.plot(closed_norm[:, 0], -closed_norm[:, 1], '-', c='green', lw=2.5)
    plt.plot(closed_recon[:, 0], -closed_recon[:, 1], '-', c='red', lw=2.5)
    # plt.legend(['ground truth', 'reconstructed'])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    fig = plt.figure()
    plt.hist(learned_codes.reshape((-1,)), bins=30, color='blue', density=False)
    # plt.rcParams.update({'font.size': 38})
    plt.xlabel('Sparse coefficients', fontsize=32)
    plt.ylabel('Counts', fontsize=32)
    plt.title('Histogram', fontsize=38)
    plt.show()

print('overall activate rates on training KINS', np.mean(active_rate))
