import numpy as np
import matplotlib.pyplot as plt 
from models.astrodynamics import d_clohessy_wilthsire, clohessy_wilthsire
from models.integrator import rk4


## Simulation parameters
dt = 0.1          # seconds
tspan = 5.0 * 60.0  # seconds
steps = int(tspan/dt)

## Clohessy-Wiltshire state
cw_d = np.zeros((steps, 6))
cw_c = np.zeros((steps, 6))

IC_relative = np.zeros(6)
IC_relative[0] = 0  #x
IC_relative[1] = 0  #y
IC_relative[2] = 0  #z
IC_relative[3] = 0.01 #xdot
IC_relative[4] = 0.01  #ydot
IC_relative[5] = 0.01 #zdot



cw_d[0] = IC_relative
cw_c[0] = IC_relative
zero_control = np.array([0,0,0])



for step in range(steps-1):
    #cw[step+1] = rk4(d_clohessy_wilthsire, cw[step], 0, dt)
    cw_d[step+1] = d_clohessy_wilthsire(cw_d[step], zero_control, dt)
    cw_c[step+1] = clohessy_wilthsire(cw_c[step], zero_control, dt)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

ax.scatter(0,0,0,c='g',label="Client position", marker='o', s=50)
ax.plot3D(cw_c[:,0].flatten(), cw_c[:,1].flatten(), cw_c[:,2].flatten(), c='b', label="clohessy_wiltshire continuous")
ax.plot3D(cw_d[:,0].flatten(), cw_d[:,1].flatten(), cw_d[:,2].flatten(), c='r', label="clohessy_wiltshire discrete")


plt.legend()
plt.show()