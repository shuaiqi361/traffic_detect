# import cppimport.import_hook
import cppimport
import numpy as np

# import optMapMaeCode
optMapMaeCode = cppimport.imp_from_filepath("optMapMaeCode.cpp")

T_1 = np.array([[-1., 2, 3, 4, 5, 6], [-1., 0, 3., 4, 5, 7]], dtype=np.double).transpose()
T_2 = np.array([[1., 2, 3, 4, 5], [1., 2, 3., 4, 5]], dtype=np.double).transpose()

result = 0
n1 = T_1.shape[0]
n2 = T_2.shape[0]

optMapMaeCode.optMapMae(T_1[:, 0], T_1[:, 1], T_2[:, 0], T_2[:, 1], n1, n2, result)

print(result)
