import sys
sys.path.insert(0, '/Users/elliotmarshall/Desktop/PDE_Test/')
import numpy as np
from matplotlib import pyplot as plt
from DiffOp import Minmod, MC, Linear_Reconstruct, Local_Lax_Friedrichs
from numba import njit


""" Burgers_AMR.py

    Created by Elliot Marshall 2025-07-13

    This file contains an example system class for 
    Burger's equation in 1D

"""


class Burgers_AMR(object):

    def __init__(self, grid, Boundary_Conditions):
        self.Nvars = 1
        self.grid = grid
        self.Npoints = self.grid.Npoints
        self.Ngz = self.grid.Nghosts
        self.BCs = Boundary_Conditions
        self.dt_fixed = False


    #####################################################
    # Convert between Conserved and Primitive Variables
    #####################################################

    def Conserved_to_Primitive(self,u):
        
        return u 
    
    def Primitive_to_Conserved(self,u):

        return u
    

    #####################################################
    # Compute Fluxes + Characteristic Speeds
    #####################################################

    # @staticmethod
    # @njit
    def compute_flux(self,t,f):

        u = f[0,:]

        U_plus, U_minus = Linear_Reconstruct(u,Minmod)

        CS = self.Characteristic_Speeds(U_plus,U_minus)

        # CS = np.fmax(np.abs(U_plus),np.abs(U_minus)) # If using Numba, we can't call other class methods

        flux = Local_Lax_Friedrichs(0.5*U_plus**2,0.5*U_minus**2,U_plus,U_minus,CS) 

        flux = np.reshape(flux,(1,np.shape(flux)[0]))

        return flux
    
    # @staticmethod
    # @njit
    def Characteristic_Speeds(self,u_plus,u_minus):

        """ Compute characteristic speeds for
            local Lax-Friedrichs flux function"""

        a = np.fmax.reduce(np.abs([u_plus,u_minus]))

        return a
    
    def Char_Speed_TimeStep(self,u,t):

        """ Compute largest characteristic speed over 
        the domain for adjusting time-step."""

        CS = np.max(np.abs(u))

        return CS


    ##############################################################
    # Evolution Equation(s)
    ##############################################################

    def rhs(self,t,f):

        u = f[0,:]

        ###############################
        # Evolution Equation
        ##############################

        dtu = np.zeros_like(u)

        flux = self.compute_flux(t,f)

        dtu[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[0,self.Ngz:-self.Ngz] - flux[0,self.Ngz-1:-self.Ngz-1])


        return np.array([dtu])
    

    
    ############################################################
    # Evolution Equation(s)
    ############################################################
        
    def Initial_Data(self):

        x = self.grid.interior_coordinates()
        n = np.shape(x)[0]
        ID_interior = 2*np.exp(-20*(x-self.grid.interval[1]/4)**2) 
        # ID_interior[np.abs(ID_interior)<1e-15] = 0
        # ID_interior = 0.5*np.sin(x)
        # ID_interior = x*0
        # ID_interior[:int(n/2)] = 1


        ### Pad array with appropriate number of ghost points
        ID = np.array([np.concatenate((self.Ngz*[0],ID_interior,self.Ngz*[0]))]) 
        ID = np.array([np.concatenate((self.Ngz*[0],ID_interior,self.Ngz*[0]))]) 


        ### Update ghost points using boundary conditions
        ID = self.BCs(ID,self.Npoints,self.Ngz)

        return ID
