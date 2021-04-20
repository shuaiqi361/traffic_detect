import ctypes
import numpy as np
import glob
import time


def initfunc_name(name):
    try:
        suffix = b'_' + name.encode('ascii')
    except UnicodeEncodeError:
        suffix = b'U_' + name.encode('punycode').replace(b'-', b'_')
    return b'PyInit' + suffix


def calulate_cmm(traj_a, traj_b, dll_handle):
    """
    :param traj_a: numpy array of shape (M, 2)
    :param traj_b: numpy array of shape (N, 2)
    :param dll_handle: dll object opened and configured by python
    :return: a single number, cmm measure
    """
    assert traj_a.shape[1] == 2 and traj_b.shape[1] == 2
    traj_a, traj_b = traj_a.astype(np.float64), traj_b.astype(np.float64)
    if traj_a.shape[0] >= traj_b.shape[0]:
        c = dll_handle(traj_a[:, 0], traj_a[:, 1], traj_b[:, 0], traj_b[:, 1], traj_a.shape[0], traj_b.shape[0])
    else:
        c = dll_handle(traj_b[:, 0], traj_b[:, 1], traj_a[:, 0], traj_a[:, 1], traj_b.shape[0], traj_a.shape[0])
    return c


# 0. find the shared library, you need to do first: "python setup.py build", and you will get it from the build folder
libfile = glob.glob('build/*/cmm*.so')[0]

# 1. open the shared library
mylib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function cmm and cmm_truncate_sides
# mylib.cmm_truncate_sides.restype = ctypes.c_double
# mylib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
#                                      np.ctypeslib.ndpointer(dtype=np.float64),
#                                      np.ctypeslib.ndpointer(dtype=np.float64),
#                                      np.ctypeslib.ndpointer(dtype=np.float64),
#                                      ctypes.c_int, ctypes.c_int]

# This may work for win if you fixed the import problem
# my_func = getattr(mylib, initfunc_name('cmm_truncate_sides'))
my_func = getattr(mylib, 'cmm_truncate_sides')
my_func.restype = ctypes.c_double
my_func.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                    np.ctypeslib.ndpointer(dtype=np.float64),
                    np.ctypeslib.ndpointer(dtype=np.float64),
                    np.ctypeslib.ndpointer(dtype=np.float64),
                    ctypes.c_int, ctypes.c_int]

"""
Test the running time
"""
video_info = {
    "cam_2": {"frame_num": 18000, "movement_num": 4, "fps": 10, "pred_track": 1595},
    "cam_4": {"frame_num": 27000, "movement_num": 12, "fps": 15, "pred_track": 1877},
    "cam_5_dawn": {"frame_num": 3000, "movement_num": 12, "fps": 10, "pred_track": 185},
    "cam_7": {"frame_num": 14400, "movement_num": 12, "fps": 8, "pred_track": 1866}
}
num_point_per_target_trajectory = 20
num_point_per_track = 20
# counter = 0
start_time = time.time()
for k, v in video_info.items():
    for _ in range(v["movement_num"] * v["pred_track"]):
        typical_traj = np.random.randint(low=1, high=640, size=(num_point_per_target_trajectory, 2))
        track_traj = np.random.randint(low=1, high=640, size=(num_point_per_track, 2))
        cc = calulate_cmm(track_traj, typical_traj, my_func)
# counter += 1

end_time = time.time()
dur_time = end_time - start_time
print('Total matching time: ', dur_time)
