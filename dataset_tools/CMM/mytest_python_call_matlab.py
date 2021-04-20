import numpy as np
from dataset_tools.CMM.CMM_utils import calculate_cmm


T_1 = np.array([[1., 2, 3, 4, 5], [1., 2, 3., 4, 5]]).transpose()
T_2 = np.array([[-1., 2, 3, 4, 5, 6], [-1., 0, 3., 4, 5, 7]]).transpose()


print("Start computing cmm error for curves ...")
eng = None

ret, eng = calculate_cmm(T_1, T_2, eng)

print('CMM Measure: ', ret)
