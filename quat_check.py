from kinematics.transformations import L, R, conj
import numpy as np

q1 = np.array([1,0,0,0])
q2 = np.array([0.7071068, 0, 0.7071068, 0])

print(L(q1).T @ q2)
print(R(q2) @ conj(q2))


print(conj(L(q2)))
print(L(conj(q2)))