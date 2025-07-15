""" Setup_Wave_AMR.py

    Created by Elliot Marshall 2025-07-12

    This example class evolves the wave equation
     using adaptive mesh refinement (AMR) """

### Import python libraries
import numpy as np
from matplotlib import pyplot as plt


### Import my packages
from Wave_AMR import Wave_AMR
from grid_classes import make_basegrid
from TimeStepping import Euler_rk_AMR, rk2_AMR, rk4_AMR
from Boundary_Conditions import periodic_BC, outflow_BC
from simulation import AMR_simulation
from grid_classes import patch


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

interval = (-10,10)
# interval = (0,2*np.pi)
Npoints = 400
Nghosts = 2

shadow_grid = make_basegrid(interval,Npoints,Nghosts)

### Construct Base Grid

Npoints = 800      # Double the resolution of shadow grid
Nghosts = 2

base_grid = make_basegrid(interval,Npoints,Nghosts)



### Set up system

system = Wave_AMR(base_grid,outflow_BC)


### Set up simulation class
t_interval = (0,15)

sim = AMR_simulation(system,Euler_rk_AMR,t_interval,shadow_grid,base_grid,cfl=0.3,max_levels=1)

# Evolve simulation
t, solution_data, grid_merged, soln_merged = sim.Evolve() 


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
# plt.yscale('log')
# plt.ylim([-1e-5,1e-3])
# plt.title("Conservation of u without refluxing")
# plt.ylabel(r'$|\int_{\mathcal{D}} u \; dx|$')
# plt.xlabel('Time')
# plt.tight_layout()
# plt.show()

### Plot solution

# sim.animate(t,solution_data)

# fig, axis = plt.subplots(1,1)
# fig.tight_layout()
# for i in range(np.shape(t)[0]):

#     # plt.plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
#     # plt.plot(grid_merged2[i][:],soln_merged2[i][:],'x',linestyle='none',markersize=1)
#     # plt.plot(grid_merged3[i][:],soln_merged3[i][:],'x',linestyle='none',markersize=1)

#     # axis[0].plot(grid_merged[i][:],soln_merged[i][:],markersize=1)
#     axis.plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
#     axis.set_title('Single fixed grid')
#     axis.set_ylim([-1,1])
#     plt.draw()
#     plt.pause(0.01)
#     axis.cla()
#     # plt.cla()


fig, axis = plt.subplots(1,1)
fig.tight_layout()
for i in range(0,np.shape(t)[0],3):

    axis.plot(grid_merged[i][:],soln_merged[i][:],'x',linestyle='none',markersize=1)
    # axis.set_ylim([0,1.5])
    plt.title("t = " +str(t[i]))
    plt.draw()
    plt.pause(0.01)
    axis.cla()