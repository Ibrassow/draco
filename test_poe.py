import numpy as np
import matplotlib.pyplot as plt 

from kinematics.transformations import mat_exp_se3

import modern_robotics as mr 
"""
Exemple 

3R robot - rotating base + link of 2 + rotating joint + link of 2 + rotating joint + link of 2 + EE (body frame)

"""

# Transformation matrix in the zero configuration (b frame in s frame)
M = np.array([[1,0,0,0],
              [0,1,0,4],
              [0,0,1,2],
              [0,0,0,1]])
print(M)
# Screw axis of the joints
S1 = np.array([0,0,1,0,0,0])
S2 = np.array([1,0,0,0,2,0])
S3 = np.array([1,0,0,0,2,-2])

# Joint value µ
# TODO something is wrong here
q1 = 75*np.pi/180
q2 = 32*np.pi/180
q3 = 28*np.pi/180

print("Twists")
V1=S1*q1
V2=S2*q2
V3=S3*q3
print(V1)
print(V2)
print(V3)




# Final configuration of the B frame 
print("Tb")
Tb = mat_exp_se3(S1*q1) @ mat_exp_se3(S2*q2) @ mat_exp_se3(S3*q3) @ M
print(Tb)


M1 = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]])


M2 = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,2],
              [0,0,0,1]])


M3 = np.array([[1,0,0,0],
              [0,1,0,2],
              [0,0,1,2],
              [0,0,0,1]])


T_01 = M1
T_02 = mat_exp_se3(S1*q1) @ M2
T_03 = mat_exp_se3(S1*q1) @ mat_exp_se3(S2*q2) @ M3




# base (0,0,0)
X = np.array([0, T_01[0,3], T_02[0,3], T_03[0,3], Tb[0,3]])
Y = np.array([0, T_01[1,3], T_02[1,3], T_03[1,3], Tb[1,3]])
Z = np.array([0, T_01[2,3], T_02[2,3], T_03[2,3], Tb[2,3]])



print("X",X)
print("Y",Y)
print("Z",Z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 


ax.set_xlim3d([-6, 6])
ax.set_ylim3d([-6, 6])
ax.set_zlim3d([0, 8])


ax.plot3D(X, Y, Z, '-bo', linewidth=2)


Slist = np.array([S1,S2,S3])
print('SLIST', Slist)
thetalist = np.array([q1,q2,q3])

Tb_mr = mr.FKinSpace(M, Slist[:3].T, thetalist[:3])
print("Check MR - Tb")
print(Tb_mr)

ax.scatter(Tb_mr[0,3],Tb_mr[1,3],Tb_mr[2,3],c='g', marker='o', s=50)


from kinematics.kinematics import space_forward_kinematics as fk

V = np.array([V1,V2,V3])

print((Tb_mr-fk(M, V)).round())


plt.show()
