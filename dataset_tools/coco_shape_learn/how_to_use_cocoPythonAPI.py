from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

dataDir = '/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17'  # This will be the path to the images
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)  # path to the json file for the labels

# Load coco dataset and labels with coco API
coco = COCO(annFile)
imgIds = coco.getImgIds()

img_counter = 0
valid_ids = []
det_results = []
for img_id in imgIds:
    img = coco.loadImgs(img_id)[0]
    image_path = '%s/train/%s' % (args.data_path, img['file_name'].split('/')[-1])
    if not os.path.exists(image_path):
        continue
    else:
        img_counter += 1
        valid_ids.append(img_id)  # evaluate on the 3.5K images

    # # plot ground truth cars for ground truth verification
    # output_image = cv2.imread(image_path)
    # ann_ids = coco.getAnnIds(imgIds=img_id)
    # gt_anns = coco.loadAnns(ids=ann_ids)

    # for ann_ in gt_anns:
    # x1, y1, w, h = ann_['bbox']
    # bbox = [x1, y1, x1 + w, y1 + h]
    # text = 'class: ' + str(ann_["category_id"])
    # label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
    # text_location = [int(bbox[0]) + 1, int(bbox[1]) + 1,
    #                  int(bbox[0]) + 1 + label_size[0][0],
    #                  int(bbox[1]) + 1 + label_size[0][1]]
    # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
    #               pt2=(int(bbox[2]), int(bbox[3])),
    #               color=nice_colors[ann_["category_id"]], thickness=2)
    # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
    #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.5,
    #             color=nice_colors[ann_["category_id"]])

    # cv2.imshow('GT bbox', output_image)
    # if cv2.waitKey() & 0xFF == ord('q'):
    #     exit()
    #
    # continue

    # # use PIL, to be consistent with evaluation
    img = read_image(image_path, format="BGR")

    predictions = predictor(img)  # this is to run a forward path of the deep network

    for result in predictions["instances"]:
        # mapping coco class to the aic dataset classes
        category_id = result["category_id"]
        _id = reverse_id_mapping[category_id]
        if _id == 3:  # cars
            result["category_id"] = 1
        elif _id == 6:  # buses
            result["category_id"] = 1
        elif _id == 8:  # trucks
            result["category_id"] = 2
        else:
            continue

        # prepare each prediction in the following structure
        det = {
            'image_id': img_id,
            'category_id': result["category_id"],
            'score': result["score"],
            'bbox': result["bbox"]  # should be [x1, y1, w, h]
        }
        det_results.append(result)

# save and evaluate
with open('{}/results/{}_baidu_bbox_results.json'.format(args.result_dir, args.data_type), 'w') as f_det:
    json.dump(det_results, f_det)

print('---------------------------------------------------------------------------------')
print('Running bbox evaluation on Baidu finetune data ...')
coco_pred = coco.loadRes('{}/results/{}_baidu_bbox_results.json'.format(args.result_dir, args.data_type))
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = valid_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
