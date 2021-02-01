import os
import numpy as np
import cvbase as cvb
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb
import copy


def polys_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def vis_mask(img, a_m, i_m, pause=0):
    cv2.namedWindow("a_i_m")
    a_i = copy.deepcopy(img)
    i_i = copy.deepcopy(img)

    a_m = a_m.astype(np.uint8) * 255
    i_m = i_m.astype(np.uint8) * 255
    a_m = np.stack((a_m, a_m, a_m), axis=2)
    i_m = np.stack((i_m, i_m, i_m), axis=2)

    a_m_w = cv2.addWeighted(a_i, 0.3, a_m, 0.7, 0)
    i_m_w = cv2.addWeighted(i_i, 0.3, i_m, 0.7, 0)
    a_i_m = np.concatenate((a_m_w, i_m_w), axis=0)

    cv2.imshow("a_i_m", a_i_m)
    cv2.waitKey(pause)


def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


is_train = True

if is_train:
    base_img_path = "/media/keyi/Data/Research/traffic/data/KINS/data_object_image_2/training/image_2"
    base_ann_path = "/media/keyi/Data/Research/traffic/data/KINS/tools/update_train_2020.json"
else:
    base_img_path = "/media/keyi/Data/Research/traffic/data/KINS/data_object_image_2/testing/image_2"
    base_ann_path = "/media/keyi/Data/Research/traffic/data/KINS/tools/update_test_2020.json"

anns = cvb.load(base_ann_path)
imgs_info = anns['images']
anns_info = anns["annotations"]

imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)

for img_id in anns_dict.keys():
    img_name = imgs_dict[img_id]
    # print(imgs_dict[img_id])
    # print('---------------------')
    # print(anns_dict[img_id])
    # print('---------------------')
    # print(anns_dict[img_id][0].keys())
    # exit()

    img_path = os.path.join(base_img_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    anns = anns_dict[img_id]

    for ann in anns:
        # if ann["a_area"] < 5000 or ann["a_area"] < 2000:  # or ann['category_id'] != 1:
        #     continue
        if not (ann["a_area"] > 6000 and 2500 < ann["i_area"] < 5000):
            continue
        # a_mask = polys_to_mask(ann["a_segm"], height, width)
        # i_mask = polys_to_mask(ann["i_segm"], height, width)
        a_poly = np.array(ann["a_segm"][0]).reshape((-1, 2))
        i_poly = np.array(ann["i_segm"][0]).reshape((-1, 2))
        a_bbox = ann["a_bbox"]  # x1, y1, w, h

        a_poly_shifted = a_poly - np.array([a_bbox[0], a_bbox[1]])
        i_poly_shifted = i_poly - np.array([a_bbox[0], a_bbox[1]])
        print(a_poly_shifted.shape, a_poly_shifted[None, :, :].shape)
        # a_votes = polys_to_mask([np.ndarray.flatten(a_poly_shifted, order='C').tolist()], a_bbox[3], a_bbox[2]) * 1.
        a_votes = np.zeros((a_bbox[3], a_bbox[2]), dtype=np.uint8)
        cv2.drawContours(a_votes, a_poly_shifted[None, :, :].astype(np.int32), color=255, contourIdx=-1, thickness=-1)
        # print(np.max(a_votes), a_votes.dtype)

        i_votes = polys_to_mask([np.ndarray.flatten(i_poly_shifted, order='C').tolist()], a_bbox[3], a_bbox[2]) * 1.
        vote_map = (a_votes * 1. + i_votes * 255.) / 2

        cropped_img = img[a_bbox[1]:a_bbox[1] + a_bbox[3], a_bbox[0]:a_bbox[0] + a_bbox[2], :]
        cropped_votes = np.repeat(vote_map.reshape((vote_map.shape[0], vote_map.shape[1], 1)), 3, axis=2).astype(np.uint8)
        # cropped_votes = np.zeros(shape=cropped_img.shape, dtype=cropped_img.dtype)
        # cv2.drawContours(cropped_votes, a_poly_shifted.astype(np.int32), color=255, contourIdx=-1, thickness=-1)

        # img_show = np.concatenate([cropped_img, cropped_votes], axis=1)
        print(vote_map.shape)
        img_show = vote_map.astype(np.uint8)
        print(np.max(vote_map))

        cv2.imshow('votes', img_show)
        # cv2.imshow('vector', cv2.resize(cropped_votes, dsize=(17, 17), interpolation=cv2.INTER_AREA))
        cv2.imshow('vector', cv2.resize(vote_map.astype(np.uint8), dsize=(17, 17), interpolation=cv2.INTER_LINEAR))
        cv2.waitKey()

