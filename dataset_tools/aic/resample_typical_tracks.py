import json
import os
import numpy as np
from dataset_tools.aic import videoInfo, camera_info
from scipy.signal import resample


def track_resample(track, threshold=20):
    """
    :param track: input track numpy array (M, 2)
    :param threshold: default 20 pixel interval for neighbouring points
    :return:
    """
    assert track.shape[1] == 2
    accum_dist = 0
    index_keep = [0]
    for i in range(1, track.shape[0]):
        dist_ = np.sqrt(np.sum((track[i] - track[i - 1]) ** 2))
        if dist_ >= 1.1:
            accum_dist += dist_
            if accum_dist >= threshold:
                index_keep.append(i)
                accum_dist = 0

    return track[index_keep, :]


dataset_root = '/media/keyi/Data/Research/traffic/data/AIC2021/Dataset/AIC21_Track1_Vehicle_Counting/AIC21_Track1_Vehicle_Counting'
Baidu_root = '/media/keyi/Data/Research/traffic/data/AIC2021/Baidu_results'
file_to_save = os.path.join(dataset_root, 'camera_config/camera_config_resample.json')

annotation_root = '/media/keyi/Data/Research/traffic/data/AIC2021/tracks_labelling_gui-master'

num_cameras = 20
num_points_max_per_track = 50
pixel_threshold = 1.

# loop over all videos
for v in range(num_cameras):
    camera_name = 'cam_{}'.format(v + 1)  # camera id start from 1
    print(camera_name)

    anno_file = os.path.join(annotation_root, 'annotation/{}.csv'.format(camera_name))
    if not os.path.exists(anno_file):
        print('skipping file', anno_file)
        continue

    # load and save the ROI
    roi_file = os.path.join(dataset_root, 'ROIs/{}.txt'.format(camera_name))
    with open(roi_file, 'r') as f_roi:
        lines = f_roi.readlines()
    roi = []
    for line in lines:
        elements = line.strip('\n').split(',')
        roi.append([int(elements[0]), int(elements[1])])

    camera_info[camera_name]['roi'] = roi

    # load the baidu annotation for typical trajectory
    ty_traj_file = os.path.join(Baidu_root, 'Research-master/CV/VehicleCounting/cam-configs/{}.json'.format(camera_name))
    with open(ty_traj_file, 'r') as f:
        baidu_ty_traj = json.load(f)

    num_moi = camera_info[camera_name]['movement_num']
    camera_info[camera_name]['moi'] = {}
    for i in range(num_moi):
        baidu_key = 'movement_{}'.format(i + 1)
        all_tracks = baidu_ty_traj[baidu_key]['tracklets']  # List[List[List]]]
        resulting_tracks = []
        for j in range(len(all_tracks)):
            # remove points overlaps on each track
            resampled_track = track_resample(np.array(all_tracks[j])).tolist()
            resulting_tracks.append(resampled_track)
        camera_info[camera_name]['moi'][i + 1] = resulting_tracks

    # adding in our annotations:
    with open(anno_file, 'r') as f_csv:
        lines = f_csv.readlines()

    skip_first = True
    pre_track_id = -1
    pre_moi_id = -1
    tr_ = []
    for line in lines:
        if skip_first:
            skip_first = False
            continue
        elements = line.strip('\n').split(',')
        frame_id = int(elements[1])
        track_id = int(elements[2])
        x, y, w, h = float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])
        moi_id = int(elements[-1])
        # print(frame_id, track_id, x, y, w, h, moi_id)

        if track_id != pre_track_id and pre_track_id != -1:  # the last track ends
            resampled_track = track_resample(np.array(tr_)).tolist()
            camera_info[camera_name]['moi'][pre_moi_id].append(resampled_track)
            tr_ = []  # start a new track
        else:
            new_pt = [x + w / 2, y + h / 2]
            if len(tr_) > 0:
                if np.sqrt((tr_[-1][0] - new_pt[0]) ** 2
                           + (tr_[-1][1] - new_pt[1]) ** 2) > pixel_threshold:
                    tr_.append(new_pt)
            else:
                tr_.append(new_pt)

        pre_track_id = track_id
        pre_moi_id = moi_id

    # print(camera_info[camera_name])

with open(file_to_save, 'w') as f_out:
    json.dump(camera_info, f_out)






