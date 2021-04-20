videoInfo = {"cam_1": {"frame_num": 3000, "movement_num": 4, "fps": 10},
             "cam_1_dawn": {"frame_num": 3000, "movement_num": 4, "fps": 10},
             "cam_1_rain": {"frame_num": 2961, "movement_num": 4, "fps": 10},
             "cam_2": {"frame_num": 18000, "movement_num": 4, "fps": 10},
             "cam_2_rain": {"frame_num": 3000, "movement_num": 4, "fps": 10},
             "cam_3": {"frame_num": 18000, "movement_num": 4, "fps": 10},
             "cam_3_rain": {"frame_num": 3000, "movement_num": 4, "fps": 10},
             "cam_4": {"frame_num": 27000, "movement_num": 12, "fps": 15},
             "cam_4_dawn": {"frame_num": 4500, "movement_num": 12, "fps": 15},
             "cam_4_rain": {"frame_num": 3000, "movement_num": 12, "fps": 10},
             "cam_5": {"frame_num": 18000, "movement_num": 12, "fps": 10},
             "cam_5_dawn": {"frame_num": 3000, "movement_num": 12, "fps": 10},
             "cam_5_rain": {"frame_num": 3000, "movement_num": 12, "fps": 10},
             "cam_6": {"frame_num": 18000, "movement_num": 12, "fps": 10},
             "cam_6_snow": {"frame_num": 3000, "movement_num": 12, "fps": 10},
             "cam_7": {"frame_num": 14400, "movement_num": 12, "fps": 8},
             "cam_7_dawn": {"frame_num": 2400, "movement_num": 12, "fps": 8},
             "cam_7_rain": {"frame_num": 3000, "movement_num": 12, "fps": 10},
             "cam_8": {"frame_num": 3000, "movement_num": 6, "fps": 10},
             "cam_9": {"frame_num": 3000, "movement_num": 12, "fps": 10},
             "cam_10": {"frame_num": 2111, "movement_num": 3, "fps": 10},
             "cam_11": {"frame_num": 2111, "movement_num": 3, "fps": 10},
             "cam_12": {"frame_num": 1997, "movement_num": 3, "fps": 10},
             "cam_13": {"frame_num": 1966, "movement_num": 3, "fps": 10},
             "cam_14": {"frame_num": 3000, "movement_num": 2, "fps": 10},
             "cam_15": {"frame_num": 3000, "movement_num": 2, "fps": 10},
             "cam_16": {"frame_num": 3000, "movement_num": 2, "fps": 10},
             "cam_17": {"frame_num": 3000, "movement_num": 2, "fps": 10},
             "cam_18": {"frame_num": 3000, "movement_num": 2, "fps": 10},
             "cam_19": {"frame_num": 3000, "movement_num": 2, "fps": 10},
             "cam_20": {"frame_num": 3000, "movement_num": 2, "fps": 10}}

# The camera_info dictionary will be used to store everything about the camera and scene
camera_info = {
    "cam_1": {"movement_num": 4, "fps": 10, "video_id": {"cam_1": 1, "cam_1_dawn": 2, "cam_1_rain": 3},
              "movement_location": {"1": [15, 148], "2": [963, 83], "3": [1088, 221], "4": [1130, 607]}},
    "cam_2": {"movement_num": 4, "fps": 10, "video_id": {"cam_2": 4, "cam_2_rain": 5},
              "movement_location": {"1": [10, 449], "2": [949, 15], "3": [11, 12], "4": [1175, 67]}},
    "cam_3": {"movement_num": 4, "fps": 10, "video_id": {"cam_3": 6, "cam_3_rain": 7},
              "movement_location": {"1": [17, 203], "2": [1127, 107], "3": [1129, 530], "4": [1146, 299]}},
    "cam_4": {"movement_num": 12, "fps": 15, "video_id": {"cam_4": 8, "cam_4_dawn": 9, "cam_4_rain": 10},
              "movement_location": {"1": [7, 120], "2": []}},
    "cam_5": {"movement_num": 12, "fps": 10, "video_id": {"cam_5": 11, "cam_5_dawn": 12, "cam_5_rain": 13}},
    "cam_6": {"movement_num": 12, "fps": 10, "video_id": {"cam_6": 14, "cam_6_snow": 15}},
    "cam_7": {"movement_num": 12, "fps": 8, "video_id": {"cam_7": 16, "cam_7_dawn": 17, "cam_7_rain": 18}},
    "cam_8": {"movement_num": 6, "fps": 10, "video_id": {"cam_8": 19}},
    "cam_9": {"movement_num": 12, "fps": 10, "video_id": {"cam_9": 20}},
    "cam_10": {"movement_num": 3, "fps": 10, "video_id": {"cam_10": 21}},
    "cam_11": {"movement_num": 3, "fps": 10, "video_id": {"cam_11": 22}},
    "cam_12": {"movement_num": 3, "fps": 10, "video_id": {"cam_12": 23}},
    "cam_13": {"movement_num": 3, "fps": 10, "video_id": {"cam_13": 24}},
    "cam_14": {"movement_num": 2, "fps": 10, "video_id": {"cam_14": 25}},
    "cam_15": {"movement_num": 2, "fps": 10, "video_id": {"cam_15": 26}},
    "cam_16": {"movement_num": 2, "fps": 10, "video_id": {"cam_16": 27}},
    "cam_17": {"movement_num": 2, "fps": 10, "video_id": {"cam_17": 28}},
    "cam_18": {"movement_num": 2, "fps": 10, "video_id": {"cam_18": 29}},
    "cam_19": {"movement_num": 2, "fps": 10, "video_id": {"cam_19": 30}},
    "cam_20": {"movement_num": 2, "fps": 10, "video_id": {"cam_20": 31}}
}

if __name__ == "__main__":
    total_time = 0
    for k, v in videoInfo.items():
        t = v['frame_num'] / v['fps']
        total_time += t

    print('total video time: ', total_time)
