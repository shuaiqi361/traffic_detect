import os
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt


coco_root_path = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'
annotation_folder = os.path.join(coco_root_path, 'annotations')
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(coco_root_path, dataType)
coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
coco_labels = [cat['name'] for cat in cats]  # list of labels

coco_label_map = {k: v + 1 for v, k in enumerate(coco_labels)}  # this is a customized map, with background class added
coco_label_map['background'] = 0
rev_coco_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping

catIds = coco.getCatIds(catNms=coco_labels)
annIds = coco.getAnnIds(catIds=catIds)
all_anns = coco.loadAnns(ids=annIds)  # get all annotations

all_images_dict = {}
count_obj = 0
count_veh = 0
overall_areas_list = []
vehicle_areas_list = []
DETRAC_aligned_class = ['car', 'truck', 'bus']

for annotation in all_anns:
    if annotation['iscrowd'] == 1:
        continue
    bbox = annotation['bbox']
    if bbox[2] < 2 or bbox[3] < 2:
        # print('Eliminate small objects for training < 2px.')
        continue

    count_obj += 1
    overall_areas_list.append(np.sqrt(bbox[2] * bbox[3] * 1.))

    cat_id = annotation['category_id']
    cat_name = coco.loadCats([cat_id])[0]['name']

    if cat_name in DETRAC_aligned_class:
        count_veh += 1
        vehicle_areas_list.append(np.sqrt(bbox[2] * bbox[3] * 1.))

print('Total objects {}, total vehicles {}'.format(count_obj, count_veh))
print(len(overall_areas_list), len(vehicle_areas_list))

# show the histogram of coefficients
fig = plt.figure()
plt.hist(overall_areas_list, bins=300, color='c', edgecolor='k', alpha=0.5)
plt.axvline(np.mean(overall_areas_list), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(overall_areas_list) * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.mean(overall_areas_list)))
plt.xlabel('Square root of BBox area')
plt.title('COCO object sizes in training data')
plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/coco/coco_obj_sizes.jpg')

plt.show()

fig = plt.figure()
plt.hist(vehicle_areas_list, bins=200, color='c', edgecolor='k', alpha=0.5)
plt.axvline(np.mean(vehicle_areas_list), color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(vehicle_areas_list) * 1.1, max_ylim * 0.9, 'Mean: {:.3f}'.format(np.mean(vehicle_areas_list)))
plt.xlabel('Square root of BBox area')
plt.title('COCO vehicle sizes in training data')
plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/coco/coco_vehicle_sizes.jpg')

plt.show()

