""" Setup_Gowdy_AMR.py

    Created by Elliot Marshall 2025-08-04

    This file performs the setup to solve vacuum Einstein equations
    in Gowdy symemtry using AMR. 
     
"""

### Import python libraries
import numpy as np
from matplotlib import pyplot as plt
import time
# import cProfile
# import pstats



### Import my packages
from Euler_AMR import Euler_AMR
from grid_classes import make_basegrid
from TimeStepping import Euler_rk_AMR, rk2_AMR, rk4_AMR
from Boundary_Conditions import spherical_symmetry_BC, outflow_BC
from simulation import AMR_simulation


# pr = cProfile.Profile()
# pr.enable()


### Construct "Shadow" Grid
### This is a coarse grid which is only used for error estimation for the base grid

interval = (0,10)
Npoints = 400
Nghosts = 2

shadow_grid = make_basegrid(interval,Npoints,Nghosts)

### Construct Base Grid

Npoints = 800  # Double the resolution of shadow grid
Nghosts = 2

base_grid = make_basegrid(interval,Npoints,Nghosts)


### Set up system

K = 0.8
sigma = 0.0

system = Euler_AMR(base_grid,spherical_symmetry_BC,K,sigma)

# plt.plot(system.Initial_Data(),'x')
# plt.show()
# breaks


### Set up simulation
t_interval = (1,20)
sim = AMR_simulation(system,rk4_AMR,t_interval,shadow_grid,base_grid,cfl=0.4,max_levels=2,threshold=0.01)


### Evolve in time

t, solution_data, grid_merged, soln_merged = sim.Evolve() # Output only the finest available grid at each timestep


# ### Plot solution

sim.animate(t,solution_data)


fig, axis = plt.subplots(1,1)
fig.tight_layout()
for i in range(np.shape(t)[0]):

    # axis.plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
    axis.plot(grid_merged[i][:],soln_merged[i][0][:])
    # axis.set_ylim([0.8,1.2])
    # axis.set_ylim([-0.05,1.5])
    plt.title("t = " +str(t[i]))
    plt.draw()
    plt.pause(0.01)
    axis.cla()
