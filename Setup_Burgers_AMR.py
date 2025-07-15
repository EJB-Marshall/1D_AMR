""" Setup_Burgers_AMR.py

    Created by Elliot Marshall 2025-07-13

    This file performs the setup to solve Burgers equation
     using AMR. 
     
"""

### Import python libraries
import numpy as np
from matplotlib import pyplot as plt
import time
# import cProfile
# import pstats



### Import my packages
from Burgers_AMR import Burgers_AMR
from grid_classes import make_basegrid
from TimeStepping import Euler_rk_AMR, rk2_AMR, rk4_AMR
from Boundary_Conditions import periodic_BC, outflow_BC
from simulation import AMR_simulation


# pr = cProfile.Profile()
# pr.enable()


def compute_integral_abs(f,dx):
    """This function outputs a vector whose 
    ith entry is the absolute value of spatial integral at timestep i
    """
    int_eval = np.zeros_like(f[:,1])
    for i in range(0,np.shape(f)[0]):
        int_eval[i] = np.abs(dx*np.sum(f[i,:]))
    return int_eval



### Construct "Shadow" Grid
### This is a coarse grid which is only used for error estimation for the base grid

interval = (0,2*np.pi)
Npoints = 50
Nghosts = 2

shadow_grid = make_basegrid(interval,Npoints,Nghosts)

### Construct Base Grid

Npoints = 100  # Double the resolution of shadow grid
Nghosts = 2

base_grid = make_basegrid(interval,Npoints,Nghosts)


### Set up system

system = Burgers_AMR(base_grid,periodic_BC)

# plt.plot(system.Initial_Data(),'x')
# plt.show()
# breaks


### Set up simulation
t_interval = (0,8)
sim = AMR_simulation(system,rk4_AMR,t_interval,shadow_grid,base_grid,cfl=0.2,max_levels=1)

sim2 = AMR_simulation(system,rk4_AMR,t_interval,shadow_grid,base_grid,cfl=0.2,max_levels=3) 

sim3 = AMR_simulation(system,rk4_AMR,t_interval,shadow_grid,base_grid,cfl=0.2,max_levels=6) 

### Evolve in time

t, solution_data, grid_merged, soln_merged = sim.Evolve() # Output only the finest available grid at each timestep
t2, solution_data2, grid_merged2, soln_merged2 = sim2.Evolve() 
t3, solution_data3, grid_merged3, soln_merged3 = sim3.Evolve() 


# pr.disable()
# stats = pstats.Stats(pr)
# stats.sort_stats('cumulative').print_stats(20)


### Compute the absolute integral of the base_grid solution
# soln = np.array(soln_merged)
# int_abs = compute_integral_abs(soln[:,:] - soln[0,:],base_grid.dx)


# plt.rcParams.update({"text.usetex": True,
#                 "font.family": "serif",
#                 "font.serif": "Computer Modern",
#                 "savefig.bbox": "tight",
#                 "savefig.format": "pdf"})
# plt.rc('font', size=16)

# plt.plot(t,int_abs)
# # plt.yscale('log')
# plt.ylim([-2e-4,2e-4])
# # plt.title("Conservation of u without refluxing")
# plt.ylabel(r'$|\int_{\mathcal{D}} u \; dx|$')
# plt.xlabel('Time')
# plt.tight_layout()
# plt.show()

# ### Plot solution

# sim.animate(t,solution_data)



# fig, axis = plt.subplots(3,1)
# fig.tight_layout()
# i = 5
# # axis[0].plot(grid_merged[i][:],soln_merged[i][:],markersize=1)
# axis[0].plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
# # axis[0].set_title('Single fixed grid')
# # axis[1].plot(grid_merged2[i+4][:],soln_merged2[i+4][:],markersize=1)
# axis[1].plot(grid_merged2[i][:],soln_merged2[i][:],'x',linestyle='none',markersize=1)
# # axis[1].set_title('Up to 3 grid levels')
# # axis[2].plot(grid_merged3[i+4][:],soln_merged3[i+4][:],markersize=1)
# axis[2].plot(grid_merged3[i][:],soln_merged3[i][:],'x',linestyle='none',markersize=1)
# # axis[2].set_title('Up to 6 grid levels')
# axis[0].set_ylim([-2,2])
# axis[1].set_ylim([-2,2])
# axis[2].set_ylim([-2,2])
# plt.show()
# print(t[i],t2[i],t3[i])







# fig, axis = plt.subplots(3,1)
# fig.tight_layout()
# for i in range(np.shape(t)[0]):

#     # plt.plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
#     # plt.plot(grid_merged2[i][:],soln_merged2[i][:],'x',linestyle='none',markersize=1)
#     # plt.plot(grid_merged3[i][:],soln_merged3[i][:],'x',linestyle='none',markersize=1)

#     # axis[0].plot(grid_merged[i][:],soln_merged[i][:],markersize=1)
#     axis[0].plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
#     axis[0].set_title('Single fixed grid')
#     # axis[1].plot(grid_merged2[i][:],soln_merged2[i][:],markersize=1)
#     axis[1].plot(grid_merged2[i][:],soln_merged2[i][:],'x',linestyle='none',markersize=1)
#     axis[1].set_title('Up to 3 grid levels')
#     # axis[2].plot(grid_merged3[i][:],soln_merged3[i][:],markersize=1)
#     axis[2].plot(grid_merged3[i][:],soln_merged3[i][:],'x',linestyle='none',markersize=1)
#     axis[2].set_title('Up to 6 levels')
#     axis[0].set_ylim([-2,2])
#     axis[1].set_ylim([-2,2])
#     axis[2].set_ylim([-2,2])
#     plt.draw()
#     plt.pause(0.01)
#     axis[0].cla()
#     axis[1].cla()
#     axis[2].cla()
#     plt.cla()


# fig, axis = plt.subplots(1,1)
# fig.tight_layout()
# for i in range(np.shape(t)[0]):

#     axis.plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
#     axis.set_ylim([-3,3])
#     plt.draw()
#     plt.pause(0.01)
#     axis.cla()
