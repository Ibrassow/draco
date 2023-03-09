import numpy as np
import matplotlib.pyplot as plt 

from kinematics.mapping import exp_map_se3, vec_to_se3, se3_to_vec
import modern_robotics as mr 

V = np.array([1, 2, 3, 4, 5, 6])

mat = np.array([[0,          0,           0,          0],
                [0,          0, -1.57079632, 2.35619449],
                [0, 1.57079632,           0, 2.35619449],
                [0,          0,           0,          0]])

V = mr.se3ToVec(mat)

#mat_mm = mr.VecTose3(V)
mm = mr.MatrixExp6(mat)
ii = exp_map_se3(V)

print("Now comparison")
print(mm)
print(ii)

