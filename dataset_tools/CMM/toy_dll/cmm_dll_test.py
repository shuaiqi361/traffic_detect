import ctypes
import numpy as np
import glob

# find the shared library
libfile = glob.glob('build/*/cmm*.so')[0]

# 1. open the shared library
mylib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function cmm
mylib.cmm.restype = ctypes.c_double
mylib.cmm.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                      np.ctypeslib.ndpointer(dtype=np.float64),
                      np.ctypeslib.ndpointer(dtype=np.float64),
                      np.ctypeslib.ndpointer(dtype=np.float64),
                      ctypes.c_int, ctypes.c_int]

mylib.cmm_truncate_sides.restype = ctypes.c_double
mylib.cmm_truncate_sides.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                                     np.ctypeslib.ndpointer(dtype=np.float64),
                                     np.ctypeslib.ndpointer(dtype=np.float64),
                                     np.ctypeslib.ndpointer(dtype=np.float64),
                                     ctypes.c_int, ctypes.c_int]

# Test case 1:
T_1 = np.array([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]]).transpose().astype(np.float64)
T_2 = np.array([[0, 1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1, 1]]).transpose().astype(np.float64)

# Test case 2:
T_1 = np.array([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]]).transpose().astype(np.float64)
T_2 = np.array([[2, 3, 4], [1, 1, 1]]).transpose().astype(np.float64)

# Test case 3:
T_1 = np.array([[0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0, 0]]).transpose().astype(np.float64)
T_2 = np.array([[2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 1]]).transpose().astype(np.float64)

print(T_1.shape, T_2.shape)
n1 = T_1.shape[0]
n2 = T_2.shape[0]

# 3. call function original cmm
cmm_measure = mylib.cmm(T_1[:, 0], T_1[:, 1], T_2[:, 0], T_2[:, 1], n1, n2)
print('CMM measure: {}'.format(cmm_measure))

# 3. call function truncated cmm
cmm_truncate_measure = mylib.cmm_truncate_sides(T_1[:, 0], T_1[:, 1], T_2[:, 0], T_2[:, 1], n1, n2)
print('CMM truncated measure: {}'.format(cmm_truncate_measure))
