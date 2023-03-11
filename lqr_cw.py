import numpy as np
import matplotlib.pyplot as plt 
from models.astrodynamics import d_clohessy_wilthsire, get_cw_discrete_dynamics

from control.lqr import dlqr


## Simulation parameters
dt = 0.1          # seconds
tspan = 5.0 * 60.0  # seconds
steps = int(tspan/dt)

## Clohessy-Wiltshire state
cw_d = np.zeros((steps, 6,1))

IC_relative = np.zeros((6,1))
IC_relative[0] = -1200  #x
IC_relative[1] = -74  #y
IC_relative[2] = -160  #z
IC_relative[3] = -1.2 #xdot
IC_relative[4] = 0.25  #ydot
IC_relative[5] = 1.1 #zdot


cw_d[0] = IC_relative
zero_control = np.array([0,0,0])
A, B = get_cw_discrete_dynamics(dt)
Q = np.eye(A.shape[0])
R = np.eye(B.shape[1])
Q[0,0] = 50
Q[2,2] = 30
Q[5,5] = 30
R[2,2] = 40
R[1,1] = 10
R[0,0] = 240
"""print(A)
print(B)
print(Q)
print(R)"""

K = dlqr(A, B, Q, R)
ctrl = 0

for step in range(steps-1):
    #cw_d[step+1] = d_clohessy_wilthsire(cw_d[step], zero_control, dt)
    ctrl = (- K @ cw_d[step])
    cw_d[step+1] = d_clohessy_wilthsire(cw_d[step], ctrl, dt)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

ax.scatter(0,0,0,c='g',label="Client position", marker='o', s=50)
ax.plot3D(cw_d[:,0].flatten(), cw_d[:,1].flatten(), cw_d[:,2].flatten(), c='r', label="clohessy_wiltshire discrete")


plt.legend()
plt.show()