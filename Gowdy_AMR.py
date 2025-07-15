import numpy as np
from matplotlib import pyplot as plt
from DiffOp import Linear_Reconstruct, Local_Lax_Friedrichs, Minmod, MC


""" Gowdy_AMR.py

    Created by Elliot Marshall 2025-08-04

    This file contains an example system class for 
    an AMR evolution of the vacuum Einstein equations
    in Gowdy symmetry

"""


class Gowdy_AMR(object):

    def __init__(self, grid, Boundary_Conditions):
        self.Nvars = 6
        self.grid = grid
        self.Npoints = self.grid.Npoints
        self.Ngz = self.grid.Nghosts
        self.BCs = Boundary_Conditions
        self.dt_fixed = True

    
    #####################################################
    # Compute flux
    #####################################################

    def compute_flux(self,t, f):

        P, P1, P0, Q, Q1, Q0 = f[:]

        P1_plus, P1_minus = Linear_Reconstruct(P1,Minmod)
        P0_plus, P0_minus = Linear_Reconstruct(P0,Minmod)
        Q1_plus, Q1_minus = Linear_Reconstruct(Q1,Minmod)
        Q0_plus, Q0_minus = Linear_Reconstruct(Q0,Minmod)

        CS = np.zeros_like(P1_plus) + np.exp(-t)

        flux_P = np.zeros_like(Q)
        flux_Q = np.zeros_like(Q)
        flux_P1 = Local_Lax_Friedrichs(-P0_plus,-P0_minus,P1_plus,P1_minus,CS) 
        flux_P0 = Local_Lax_Friedrichs(-np.exp(-2*t)*P1_plus,-np.exp(-2*t)*P1_minus,P0_plus,P0_minus,CS) 
        flux_Q1 = Local_Lax_Friedrichs(-Q0_plus,-Q0_minus,Q1_plus,Q1_minus,CS) 
        flux_Q0 = Local_Lax_Friedrichs(-np.exp(-2*t)*Q1_plus,-np.exp(-2*t)*Q1_minus,Q0_plus,Q0_minus,CS) 


        return np.array([flux_P,flux_P1,flux_P0,flux_Q,flux_Q1,flux_Q0])
    
    
    def Char_Speed_TimeStep(self,f,t):

        """ Compute largest characteristic speed over 
        the domain for adjusting time-step."""

        CS = np.max(np.abs(np.exp(-t)))

        return CS
    

    ##############################################################
    # Evolution Equation(s)
    ##############################################################

    def rhs(self,t,f):

        P, P1, P0, Q, Q1, Q0 = f


        ###############################
        # Evolution Equation
        ##############################

        ### ODE Equations
        dtP = np.copy(P0)

        dtQ = np.copy(Q0)


        ### Wave Equations
        dtP1 = np.zeros_like(P1)

        dtP0 = np.zeros_like(P1)

        dtQ1 = np.zeros_like(P1)

        dtQ0 = np.zeros_like(P1)
        
        flux = self.compute_flux(t,f)


        dtP1[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[1,self.Ngz:-self.Ngz] - flux[1,self.Ngz-1:-self.Ngz-1]) 


        dtP0[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[2,self.Ngz:-self.Ngz] - flux[2,self.Ngz-1:-self.Ngz-1]) \
                                    + np.exp(2*P[self.Ngz:-self.Ngz])*(Q0[self.Ngz:-self.Ngz]**2 - np.exp(-2*t)*Q1[self.Ngz:-self.Ngz]**2)


        dtQ1[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[4,self.Ngz:-self.Ngz] - flux[4,self.Ngz-1:-self.Ngz-1])


        dtQ0[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*(flux[5,self.Ngz:-self.Ngz] - flux[5,self.Ngz-1:-self.Ngz-1]) \
                                    + 2*(np.exp(-2*t)*P1[self.Ngz:-self.Ngz]*Q1[self.Ngz:-self.Ngz] - P0[self.Ngz:-self.Ngz]*Q0[self.Ngz:-self.Ngz])

        

        return np.array([dtP, dtP1, dtP0, dtQ, dtQ1, dtQ0])
    

    
    ############################################################
    # Evolution Equation(s)
    ############################################################
        
    def Initial_Data(self):

        x = self.grid.interior_coordinates()
        P_interior = 0*x
        P1_interior = 0*x
        P0_interior = 5*np.cos(x)

        Q_interior = np.cos(x)
        Q1_interior = -np.sin(x)
        Q0_interior = 0*x



        ### Pad array with appropriate number of ghost points
        P_ID = np.concatenate((self.Ngz*[0],P_interior,self.Ngz*[0]))
        P1_ID = np.concatenate((self.Ngz*[0],P1_interior,self.Ngz*[0]))
        P0_ID = np.concatenate((self.Ngz*[0],P0_interior,self.Ngz*[0])) 
        Q_ID = np.concatenate((self.Ngz*[0],Q_interior,self.Ngz*[0]))
        Q1_ID = np.concatenate((self.Ngz*[0],Q1_interior,self.Ngz*[0]))
        Q0_ID = np.concatenate((self.Ngz*[0],Q0_interior,self.Ngz*[0])) 

        ID = np.reshape(np.array([P_ID, P1_ID, P0_ID, Q_ID, Q1_ID, Q0_ID]),(self.Nvars,x.shape[0]+4))

        ID = self.BCs(ID,self.Npoints,self.Ngz)

        return ID
