import matlab
import matlab.engine
import numpy as np
import random
import time


def calculate_cmm(traj_a, traj_b, eng=None):
    """
    :param eng: matlab engine by calling matlab.engine.start_matlab()
    :param traj_a: These has to be numpy array or tensors with shape (M, 2)
    :param traj_b: These has to be numpy array or tensors with shape (N, 2)
    :return: a scalar, which is the average Eucliean distance between matched pairs
    """
    assert traj_a.shape[-1] == 2 and traj_b.shape[-1] == 2
    if eng is None:
        eng = matlab.engine.start_matlab()

    ia = matlab.double(traj_a[:, 0].tolist())
    ja = matlab.double(traj_a[:, 1].tolist())
    ib = matlab.double(traj_b[:, 0].tolist())
    jb = matlab.double(traj_b[:, 1].tolist())

    ret, t = eng.getCMM(ia, ja, ib, jb, nargout=2)

    return ret, eng, t


def test_runtime():
    video_info = {
        "cam_2": {"frame_num": 18000, "movement_num": 4, "fps": 10, "pred_track": 1595},
        "cam_4": {"frame_num": 27000, "movement_num": 12, "fps": 15, "pred_track": 1877},
        "cam_5_dawn": {"frame_num": 3000, "movement_num": 12, "fps": 10, "pred_track": 185},
        "cam_7": {"frame_num": 14400, "movement_num": 12, "fps": 8, "pred_track": 1866}
    }
    num_point_per_target_trajectory = 5
    num_point_per_track = 5
    # typical_traj = np.random.randint(low=1, high=640, size=(num_point_per_target_trajectory, 2))
    # track_traj = np.random.randint(low=1, high=640, size=(num_point_per_track, 2))

    eng = matlab.engine.start_matlab()
    cmm_time = 0
    start_time = time.time()
    for k, v in video_info.items():
        for _ in range(v["movement_num"] * v["pred_track"]):
            typical_traj = np.random.randint(low=1, high=640, size=(num_point_per_target_trajectory, 2))
            track_traj = np.random.randint(low=1, high=640, size=(num_point_per_track, 2))
            ret, _, t_ = calculate_cmm(typical_traj, track_traj, eng)
            cmm_time += t_

    end_time = time.time()
    dur_time = end_time - start_time
    print('Total matching time: ', dur_time)
    print('Matlab time: ', cmm_time)
    eng.quit()


def main():
    # T_1 = np.random.random(size=(20, 2))
    # T_2 = np.random.random(size=(30, 2))

    T_2 = np.array([[1., 2, 3, 4, 5], [1., 2, 3., 4, 5]]).transpose()
    T_1 = np.array([[-1., 2, 3, 4, 5, 6, 8, 9], [-1., 0, 3., 4, 5, 7, 11, 12]]).transpose()
    print("Start computing cmm error for curves ...")
    eng = matlab.engine.start_matlab()
    # ret = eng.getCMM(ia, ja, ib, jb)
    ret, _, _ = calculate_cmm(T_1, T_2, eng)
    eng.quit()
    print('CMM Measure: ', ret)


if __name__ == "__main__":
    main()
    # test_runtime()
