import numpy as np
from matplotlib import pyplot as plt
from DiffOp import FD_Backward_2, FD_Central_2, FD_Central_4


""" Advection_AMR.py

    Created by Elliot Marshall 2025-07-11

    This file contains an example system class for 
    an AMR evolution of the 1D advection equation. 

"""


class Advection_AMR(object):

    def __init__(self, advection_speed, grid, Boundary_Conditions):
        self.Nvars = 1
        self.advec_vel = advection_speed
        self.grid = grid
        self.Npoints = self.grid.Npoints
        self.Ngz = self.grid.Nghosts
        self.BCs = Boundary_Conditions
        self.dt_fixed = True


    #####################################################
    # Convert between Conserved and Primitive Variables
    #####################################################


    def Conserved_to_Primitive():
        
        return
    
    def Primitive_to_Conserved():

        return


    ##############################################################
    # Evolution Equation(s)
    ##############################################################

    def rhs(self,t,f):

        u = f[0,:]


        ###############################
        # Evolution Equation
        ##############################


        # dtu = -self.advec_vel*FD_Backward_2(u,self.grid)
        dtu = -self.advec_vel*FD_Central_2(u,self.grid)


        ### Ghost Points not updated using time integrator:

        dtu[:self.Ngz] = 0  
        dtu[-self.Ngz:] = 0 

        ### This is probably not necessary, but should stop any NaNs etc. occuring
        ### from spurious data at ghost points.


        return np.array([dtu])
    

    
    ############################################################
    # Evolution Equation(s)
    ############################################################
        
    def Initial_Data(self):

        x = self.grid.interior_coordinates()
        # ID_interior = 0.3*np.exp(-5*(x)**2) 
        # ID_interior[np.abs(ID_interior)<1e-15] = 0

        ID_interior = np.sin(x)


        ### Pad array with appropriate number of ghost points
        ID = np.array([np.concatenate((self.Ngz*[0],ID_interior,self.Ngz*[0]))]) 


        ### Update ghost points using boundary conditions
        ID = self.BCs(ID,self.Npoints,self.Ngz)

        return ID
