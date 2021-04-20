import os
import cv2
from dataset_tools.aic import videoInfo, camera_info

NUM_CAM = 20
OUTPUT_FOLDER = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/' \
                '2080_yolo_track_4/tracks/'
VIDEO_FOLDER = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/' \
               'AIC21_Track1_Vehicle_Counting/Dataset_A'
DEMO_FOLDER = '/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/' \
              '2080_yolo_track_4/demo/'

if not os.path.exists(DEMO_FOLDER):
    os.mkdir(DEMO_FOLDER)

LAG = 3
visual_lag = -1
for i in range(NUM_CAM):
    cam_name = 'cam_{}'.format(i + 1)
    cam_info = camera_info[cam_name]
    video_ids = cam_info['video_id']  # a dict of video ids, keys are video names, will loop over keys
    movement_num = cam_info['movement_num']
    fps = cam_info['fps']
    mark_locations = cam_info['movement_location']

    for v_name in video_ids.keys():
        print('processing video:', v_name)
        result_path = os.path.join(OUTPUT_FOLDER, v_name + '.txt')
        video_path = os.path.join(VIDEO_FOLDER, v_name + '.mp4')

        with open(result_path, 'r') as f_result:
            result_lines = f_result.readlines()

        counting_results = {}
        for line in result_lines:
            elements = line.strip('\n').split(' ')
            frame_id = int(elements[1])
            moi_id = int(elements[2])
            type_id = int(elements[3])
            counting_results[str(frame_id)] = {'moi_id': moi_id, 'type_id': type_id}

        # print(counting_results.keys())

        # start the video and plot the results
        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video_out = cv2.VideoWriter(os.path.join(DEMO_FOLDER, '{}_result.mp4'.format(cam_name)),
                                    cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps,
                                    (width, height))

        frame_id = 0
        car_counter = 0
        truck_counter = 0
        while frame_id < num_frames:
            # for frame_id in tqdm(range(num_frames)):
            frame_id += 1
            ret, frame = video.read()

            if not ret:
                print('Error: skipping frame ', frame_id)
                continue

            if str(frame_id) in counting_results.keys():
                visual_lag = LAG
                lagged_frame_id = frame_id
                moi_id = counting_results[str(frame_id)]['moi_id']
                type_id = counting_results[str(frame_id)]['type_id']
                if type_id == 1:
                    car_counter += 1
                    vehicle = 'Car'
                else:
                    truck_counter += 1
                    vehicle = 'Truck'

                plot_loc = mark_locations[str(moi_id)]  # x, y coordinates to show the text
                track_text = '{}-{}'.format(moi_id, vehicle)
                cv2.putText(frame, track_text, org=(int(plot_loc[0]), int(plot_loc[1])),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=1.2,
                            color=(0, 240, 240))
            else:
                if visual_lag > 0:
                    visual_lag -= 1
                    moi_id = counting_results[str(lagged_frame_id)]['moi_id']
                    type_id = counting_results[str(lagged_frame_id)]['type_id']
                    if type_id == 1:
                        car_counter += 1
                        vehicle = 'Car'
                    else:
                        truck_counter += 1
                        vehicle = 'Truck'

                    plot_loc = mark_locations[str(moi_id)]  # x, y coordinates to show the text
                    track_text = '{}-{}'.format(moi_id, vehicle)
                    cv2.putText(frame, track_text, org=(int(plot_loc[0]), int(plot_loc[1])),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=1.2,
                                color=(0, 240, 240))

            counter_text = 'Car: {}  Truck: {}'.format(car_counter, truck_counter)
            text_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
            cv2.putText(frame, counter_text, org=(int(width // 2 - text_size[0][0] // 1.5), 18 + int(text_size[0][1])),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=1,
                        color=(0, 0, 240))

            video_out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                exit()


        exit()














