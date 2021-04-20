import json
import os
import numpy as np
import cv2

dataset_root = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting' \
               '/AIC21_Track1_Vehicle_Counting/camera_config'
config_json_file = os.path.join(dataset_root, "camera_config_resample.json")
ROI_image_root = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/' \
                 'AIC21_Track1_Vehicle_Counting/screen_shot_with_roi_and_movement'

num_cameras = 20
with open(config_json_file, 'r') as f:
    camera_info = json.load(f)

# loop over all videos
for v in range(num_cameras):
    camera_name = 'cam_{}'.format(v + 1)  # camera id start from 1
    # if camera_name != 'cam_5':
    #     continue
    # load all information
    print('Showing MOI for ', camera_name)
    num_movement = camera_info[camera_name]['movement_num']  # --> int
    fps = camera_info[camera_name]['fps']  # --> int
    roi = camera_info[camera_name]['roi']  # --> List:[List[x, y]]
    moi = camera_info[camera_name]['moi']  # --> Dict{id: List:[List:[List[x, y]]} all MOIs for the current camera
    image_path = os.path.join(ROI_image_root, '{}.jpg'.format(camera_name))

    for id_ in range(num_movement):
        """
        Each moi will have multiple ground truth trajectories, 
        each trajectory is a list of points, and each point is a 2-List with [x, y] coordinates,
        [x, y]'s are the bounding box center locations, in order to use Baidu's label, it has to be the center
        """
        img = cv2.imread(image_path)

        trajectories = moi[str(id_ + 1)]  # --> List:[List:[List[x, y]]

        # show ROI and MOI
        cv2.polylines(img, [np.array(roi).astype(np.int32)], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        for t in trajectories:  # t could be multiple trajectories
            cv2.polylines(img, [np.array(t).reshape(-1, 2).astype(np.int32)], isClosed=False,
                          color=(255, 0, 25), thickness=2)

            # print("{} points on this track.".format(len(t)))
            image = cv2.putText(img, str(id_ + 1), (int(t[-1][0]), int(t[-1][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (255, 100, 255), 3, cv2.LINE_AA)

            for pt in t:  # all the points for each trajectory
                cv2.circle(img, (int(pt[0]), int(pt[1])), radius=5, color=(255, 10, 55), thickness=-1)

        cv2.imshow('ROI MOI image', img)
        if cv2.waitKey() & 0xFF == ord('q'):
            exit()
