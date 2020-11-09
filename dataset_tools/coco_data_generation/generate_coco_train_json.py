from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
from sparse_coding.utils import fast_ista
from dataset_tools.coco_utils.utils import check_clockwise_polygon
import numpy as np
import cv2
import json
import os
from scipy.signal import resample

n_vertices = 32  # predefined number of polygonal vertices
n_coeffs = 64
alpha = 0.01

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

save_data_root = os.path.join(dataDir, 'my_annotation')
out_dict = '{}/train_scaled_dict_v{}_n{}_a{:.2f}.npy'.format(save_data_root, n_vertices, n_coeffs, alpha)
# out_resampled_shape_file = '{}/train_norm_shape_v{}.json'.format(save_data_root,
#                                                                  n_vertices)  # the normalized resampled shapes
out_resampled_train_json = '{}/train2017_shape_instance_v{}.json'.format(save_data_root, n_vertices)  # training annotations
dictFile = '{}/dictionary/train_scaled_dict_v{}_n{}_a{:.2f}.npy'.format(dataDir, n_vertices, n_coeffs, alpha)
learned_dict = np.load(dictFile)

# Load all annotations
coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)

counter_obj = 0
count_nonzero = []

all_annotations = {}
all_shapes = {}

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

    polygons = annotation['segmentation'][0]  # need to be revised
    gt_bbox = annotation['bbox']
    gt_x1, gt_y1, gt_w, gt_h = gt_bbox
    contour = np.array(polygons).reshape((-1, 2))

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    # Downsample the contour to fix number of vertices
    fixed_contour = resample(contour, num=n_vertices)

    # Indexing from the left-most vertex, argmin x-axis
    idx = np.argmin(fixed_contour[:, 0])
    indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

    clockwise_flag = check_clockwise_polygon(indexed_shape)
    if not clockwise_flag:
        fixed_contour = np.flip(indexed_shape, axis=0)
    else:
        fixed_contour = indexed_shape.copy()

    fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
    fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

    x1, y1, x2, y2 = min(fixed_contour[:, 0]), min(fixed_contour[:, 1]), \
                     max(fixed_contour[:, 0]), max(fixed_contour[:, 1])

    bbox_width, bbox_height = x2 - x1, y2 - y1
    bbox = [x1, y1, bbox_width, bbox_height]
    bbox_center = np.array([(x1 + x2) / 2., (y1 + y2) / 2.])

    contour_mean = np.mean(fixed_contour, axis=0)
    obj_mean = contour_mean.tolist()
    contour_std = np.sqrt(np.sum(np.std(fixed_contour, axis=0) ** 2))
    if contour_std < 1e-6 or contour_std == np.inf or contour_std == np.nan:
        print('Shape with std: ', contour_std, ' skipped.')
        continue

    obj_std = np.std(fixed_contour, axis=0).tolist()
    norm_shape = (fixed_contour - contour_mean) / contour_std

    # sparsing coding using pre-learned dict
    learned_val_codes, _ = fast_ista(norm_shape.reshape((1, -1)), learned_dict, lmbda=alpha, max_iter=200)
    recon_contour = np.matmul(learned_val_codes, learned_dict).reshape((-1, 2))
    recon_contour = recon_contour * contour_std + contour_mean
    count_nonzero.append(np.sum(np.nonzero(learned_val_codes)))

    # visualize reconstructed resampled points in image
    # img = cv2.imread(image_name)
    # cv2.polylines(img, [recon_contour.astype(np.int32)], True, (0, 0, 255))
    # cv2.imshow('Poly', img)
    # cv2.waitKey()

    if annotation['image_id'] not in all_annotations.keys():
        all_annotations[annotation['image_id']] = {'image_id': annotation['image_id'],
                                                   'codes': [learned_val_codes.tolist()],
                                                   'bbox': [[x1, y1, x2, y2]],
                                                   'cat_id': [cat_id],
                                                   'cat_name': [cat_name],
                                                   'image_name': img['file_name']}
    else:
        # all_annotations[annotation['image_id']]['contour_mean'].append(obj_mean)
        # all_annotations[annotation['image_id']]['contour_std'].append(obj_std)
        all_annotations[annotation['image_id']]['codes'].append(learned_val_codes.tolist())
        all_annotations[annotation['image_id']]['bbox'].append([x1, y1, x2, y2])
        all_annotations[annotation['image_id']]['cat_id'].append(cat_id)
        all_annotations[annotation['image_id']]['cat_name'].append(cat_name)

    if annotation['image_id'] not in all_shapes.keys():
        all_shapes[annotation['image_id']] = {'image_id': annotation['image_id'],
                                              'shape': [np.ndarray.flatten(fixed_contour, order='C').tolist()],
                                              'image_name': img['file_name']}
    else:
        all_shapes[annotation['image_id']]['shape'].append(np.ndarray.flatten(fixed_contour, order='C').tolist())

    # print(all_annotations[annotation['image_id']])
    # exit()
    # if counter_obj == 3:
    #     break

print('Total training coco images: ', len(all_annotations))
obj_counts = 0
for _, obj in all_annotations.items():
    obj_counts += len(obj['cat_id'])
print('Total annotated instances: ', obj_counts)

# save json files
with open(out_resampled_train_json, 'w') as f_train:
    json.dump(all_annotations, f_train)
with open(out_resampled_shape_file, 'w') as f_shape:
    json.dump(all_shapes, f_shape)
