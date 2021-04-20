import os
import cv2


root_path = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/' \
            'AIC21_Track1_Vehicle_Counting/screen_shot_with_roi_and_movement'

num_cam = 20

for i in range(num_cam):
    img_path = os.path.join(root_path, 'cam_{}.jpg'.format(i + 1))

    img = cv2.imread(img_path)

    cv2.imshow('MOIs', img)
    print('Currently, showing camera: ', i + 1)
    if cv2.waitKey() & 0xFF == ord('q'):
        continue













