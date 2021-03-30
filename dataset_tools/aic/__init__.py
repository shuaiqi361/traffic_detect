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


if __name__ == "__main__":
    total_time = 0
    for k, v in videoInfo.items():
        t = v['frame_num'] / v['fps']
        total_time += t

    print('total video time: ', total_time)