import numpy as np
from matplotlib import pyplot as plt
from DiffOp import Linear_Reconstruct, Local_Lax_Friedrichs, Minmod, VA


""" Wave_AMR.py

    Created by Elliot Marshall 2025-07-11

    This file contains an example system class for 
    an AMR evolution of the 1D wave equation
    in first order form

"""


class Wave_AMR(object):

    def __init__(self, grid, Boundary_Conditions):
        self.Nvars = 3
        self.grid = grid
        self.Npoints = self.grid.Npoints
        self.Ngz = self.grid.Nghosts
        self.BCs = Boundary_Conditions
        self.dt_fixed = True


    #####################################################
    # Convert between Conserved and Primitive Variables
    #####################################################


    def Conserved_to_Primitive(self,u):
        
        return u 
    
    def Primitive_to_Conserved(self,u):

        return u
    
    #####################################################
    # Compute flux
    #####################################################

    def compute_flux(self,t,f):

        u, ux, ut = f[:]

        Ux_plus, Ux_minus = Linear_Reconstruct(ux,Minmod)
        Ut_plus, Ut_minus = Linear_Reconstruct(ut,Minmod)

        CS = np.ones_like(Ux_plus)

        flux_u = np.zeros_like(u)
        flux_ux = Local_Lax_Friedrichs(-Ut_plus,-Ut_minus,Ux_plus,Ux_minus,CS) # Minus sign in flux
        flux_ut = Local_Lax_Friedrichs(-Ux_plus,-Ux_minus,Ut_plus,Ut_minus,CS)


        return np.array([flux_u,flux_ux,flux_ut])
    

    ##############################################################
    # Evolution Equation(s)
    ##############################################################

    def rhs(self,t,f):

        u,ux,ut = f


        ###############################
        # Evolution Equation
        ##############################

        dtu = np.copy(ut)

        dtux = np.zeros_like(ux)

        dtut = np.zeros_like(ut)
        
        flux = self.compute_flux(t,f)


        dtux[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[1,self.Ngz:-self.Ngz] - flux[1,self.Ngz-1:-self.Ngz-1])

        dtut[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[2,self.Ngz:-self.Ngz] - flux[2,self.Ngz-1:-self.Ngz-1])
        

        return np.array([dtu, dtux, dtut])
    

    
    ############################################################
    # Evolution Equation(s)
    ############################################################
        
    def Initial_Data(self):

        x = self.grid.interior_coordinates()
        a = 10
        b = 0.3
        u_interior = b*np.exp(-a*(x)**2) 
        u_interior[np.abs(u_interior)<1e-15] = 0
        ux_interior = -2*a*b*x*np.exp(-a*(x)**2)

        # u_interior = np.sin(x)
        # ux_interior = -np.cos(x)

        ut_interior = 0*np.sin(x)


        ### Pad array with appropriate number of ghost points
        u_ID = np.concatenate((self.Ngz*[0],u_interior,self.Ngz*[0]))
        ux_ID = np.concatenate((self.Ngz*[0],ux_interior,self.Ngz*[0]))
        ut_ID = np.concatenate((self.Ngz*[0],ut_interior,self.Ngz*[0])) 

        ID = np.reshape(np.array([u_ID, ux_ID, ut_ID]),(self.Nvars,x.shape[0]+4))

        ID = self.BCs(ID,self.Npoints,self.Ngz)


        return ID
