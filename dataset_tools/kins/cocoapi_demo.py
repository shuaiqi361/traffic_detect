import os
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np


def polys_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


if __name__ == '__main__':
    src_img_path = "/media/keyi/Data/Research/traffic/data/KINS/data_object_image_2/training/image_2"
    src_gt_path = "/media/keyi/Data/Research/traffic/data/KINS/tools/update_train_2020.json"

    coco = COCO(src_gt_path)
    imgIds = coco.getImgIds()

    count = 0

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]
        img_name = img['file_name']

        img_path = os.path.join(src_img_path, img_name)
        image = cv2.imread(img_path)
        h, w, c = image.shape

        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)

        m = np.zeros(shape=(h, w), dtype=np.uint8)
        for an in anns:
            cat_id = an['category_id']
            cat_name = coco.loadCats([cat_id])[0]['name']
            print('Label: {} - {}'.format(cat_name, cat_id))

            polygons = an['i_segm'][0]

            m += polys_to_mask(an['a_segm'], h, w)
            contour = np.array(polygons).reshape((-1, 2))
            cv2.polylines(image, [np.round(contour).astype(np.int32)], True, (0, 0, 255), thickness=2)

        m_img = m.astype(np.float32) / np.max(m) * 255
        m_img = cv2.cvtColor(m_img.astype(np.uint8), code=cv2.COLOR_GRAY2BGR)
        cat_img = np.concatenate([m_img, image], axis=0)
        cv2.imshow('segm mask', cat_img)
        cv2.waitKey()


