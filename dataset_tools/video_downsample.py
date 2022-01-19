import cv2
import os


original_frame_folder = ''
down_video_folder = ''

all_frames = os.listdir(original_frame_folder)
for img_name in all_frames:
    img_path = os.path.join(original_frame_folder, img_name)
    img = cv2.imread(img_path)
    down_img = cv2.resize(img, (1920, 1080))

    cv2.imwrite(os.path.join(down_video_folder, img_name), down_img)
