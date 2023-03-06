import numpy as np

from control import dlqr
from models.orbital_models import d_clohessy_wilthsire
from dynamics.integrator import rk4
import numpy as np
import matplotlib.pyplot as plt 



## Simulation parameters
dt = 0.1          # seconds
tspan = 5.0 * 60.0  # seconds
steps = int(tspan/dt)

## Clohessy-Wiltshire state
cw = np.zeros((steps, 6))


IC_relative = np.zeros(6)
IC_relative[0] = 0  #x
IC_relative[1] = 0  #y
IC_relative[2] = 0  #z
IC_relative[3] = 0.01 #xdot
IC_relative[4] = 0.01  #ydot
IC_relative[5] = 0.01 #zdot



cw[0] = IC_relative
control = np.array([0,0,0])


for step in range(steps-1):
    cw[step+1] = rk4(d_clohessy_wilthsire, cw[step], 0, dt)
    



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(0,0,0,c='g',label="Client vehicle", marker='o', s=700)
ax.plot3D(cw[:,0], cw[:,1], cw[:,2], c='r', label="Clohessy-Wiltshire")
lgd = plt.legend()
lgd.legendHandles[0]._sizes = [30]
plt.show()
