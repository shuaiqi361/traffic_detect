import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


detrac_root = '/media/keyi/Data/Research/traffic/data/DETRAC'
dataType = 'Test'
test_images = list()
test_objects = list()

annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
annotation_path = os.path.join(detrac_root, annotation_folder)
if not os.path.exists(annotation_path):
    print('annotation_path not exist')
    raise FileNotFoundError

veh_counter = 0
area_list = []
for video in os.listdir(annotation_path):
    if video.endswith('.xml'):
        label_file = os.path.join(annotation_path, video)
        tree = ET.parse(label_file)
        root = tree.getroot()
        object_list = list()
        image_paths = list()

        for obj in root.iter('frame'):
            boxes = list()

            target_list = obj.find('target_list')
            for target in target_list:
                bbox = target.find('box').attrib
                left = float(bbox['left'])
                top = float(bbox['top'])
                width = float(bbox['width'])
                height = float(bbox['height'])
                veh_counter += 1
                area_list.append(np.sqrt(width * height))


print('Total vehicles in DETRAC train set:', veh_counter)
# np.save('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/detrac/detrac_gt_bboxes.npy', area_list)
# fig = plt.figure()
# plt.hist(area_list, bins=300, color='c', edgecolor='k', alpha=0.5)
# plt.axvline(np.mean(area_list), color='k', linestyle='dashed', linewidth=1)
# min_ylim, max_ylim = plt.ylim()
# plt.text(np.mean(area_list) * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.mean(area_list)))
# plt.xlabel('Square root of BBox area')
# plt.title('DETRAC vehicle sizes in training data')
# plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/detrac/detrac_train_vehicle_sizes.jpg')
#
# plt.show()





