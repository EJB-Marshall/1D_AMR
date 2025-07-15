import numpy as np
from matplotlib import pyplot as plt
from DiffOp import Minmod, MC, Linear_Reconstruct, Local_Lax_Friedrichs
from numba import njit
import sys


""" Euler_AMR.py

    Created by Elliot Marshall 2025-08-07

    This file contains the system class for the 
    spherically symmetric relativistic 
    Euler equations on a fixed FLRW background
    in terms of conformal time.

"""


class Euler_AMR(object):

    def __init__(self, grid, Boundary_Conditions,K,sigma):
        self.Nvars = 2
        self.grid = grid
        self.Npoints = self.grid.Npoints
        self.Ngz = self.grid.Nghosts
        self.BCs = Boundary_Conditions
        self.dt_fixed = False
        self.K = K
        self.sigma = sigma


    #####################################################
    # Convert between Conserved and Primitive Variables
    #####################################################

    def Conserved_to_Primitive(self,Pi,Phi):

        Floor = 1e-14 # Ideally should be able to use a positivity 
                      # preserving limiter to avoid the floor condition
                      # which is a bit ambiguous...

        K = self.K

        Phi[np.where(Phi<Floor)] = Floor
        Pi[np.where(Pi<Floor)] =  Floor


        beta = 1/4*(2-(K+1))
        mu = 1/K *(-beta*(Pi+Phi) + np.sqrt(beta**2*(Phi+Pi)**2 + K*Phi*Pi))
        w = (Pi-Phi)/(Pi + Phi + 2*K*mu)

        if np.any(w>=1):
            print("Unphysical value of w!")
            sys.exit()
        
        return np.array([w,mu])
    
    
    def Primitive_to_Conserved(self,u): # Don't this is ever needed...

        return u
    

    #####################################################
    # Compute Fluxes + Characteristic Speeds
    #####################################################

    # @staticmethod
    # @njit
    def compute_flux(self,t,f,arg=0): # Need to modify to get the correct refluxing behaviour...

        Pi, Phi = f

        K = self.K
        sigma = self.sigma


        # Get the values at grid points
        grid_points = self.grid.interior_coordinates() + 0.5*self.grid.dx
        grid_points = np.concatenate(([0.0],grid_points))

        # Values at grid points, including ghost points
        grid_points2 = self.grid.coordinates() + 0.5*self.grid.dx 
        grid_points2[1] = 0.0

        #################################
        # Construct Fluxes
        ################################

        w, mu = self.Conserved_to_Primitive(Pi,Phi)


        w_plus, w_minus = Linear_Reconstruct(w,Minmod)
        mu_plus, mu_minus = Linear_Reconstruct(mu,Minmod)
        Pi_plus, Pi_minus = Linear_Reconstruct(Pi,Minmod)
        Phi_plus, Phi_minus = Linear_Reconstruct(Phi,Minmod)


        CS = self.Characteristic_Speeds(w_plus, w_minus,t)

        flux_Pi_tmp_plus = 0.5*((Pi_plus - Phi_plus)*(1+w_plus))
        flux_Pi_tmp_minus = 0.5*((Pi_minus - Phi_minus)*(1+w_minus))

        flux_Phi_tmp_plus = 0.5*((Pi_plus - Phi_plus)*(1-w_plus))
        flux_Phi_tmp_minus = 0.5*((Pi_minus - Phi_minus)*(1-w_minus))

        
        flux_Pi_1 = Local_Lax_Friedrichs(flux_Pi_tmp_plus,flux_Pi_tmp_minus,Pi_plus,Pi_minus,CS) 
        flux_Phi_1 = Local_Lax_Friedrichs(flux_Phi_tmp_plus,flux_Phi_tmp_minus,Phi_plus,Phi_minus,CS) 

        flux_Pi_2 = 0.5*K*(mu_plus + mu_minus)
        flux_Phi_2 = -0.5*K*(mu_plus + mu_minus)


        flux_Pi_total = 3*flux_Pi_1*grid_points2**2 + flux_Pi_2
        flux_Phi_total = 3*flux_Phi_1*grid_points2**2 + flux_Phi_2

        

        if arg == 0:
            return np.array([flux_Pi_total,flux_Phi_total])*0
        
        if arg == 1:
            return np.array([flux_Pi_1,flux_Pi_2,flux_Phi_1,flux_Phi_2])
    
    # @staticmethod
    # @njit
    def Characteristic_Speeds(self,w_plus,w_minus,t):

        """ Compute characteristic speeds for
            local Lax-Friedrichs flux function"""
        
        sigma = self.sigma
        K = self.K
        
        lam1_plus = ((-1+K)*w_plus + np.sqrt(K)*(1-w_plus**2))/(-1 + K*w_plus**2)
        lam1_minus = ((-1+K)*w_minus + np.sqrt(K)*(1-w_minus**2))/(-1 + K*w_minus**2)

        lam2_plus = ((-1+K)*w_plus - np.sqrt(K)*(1-w_plus**2))/(-1 + K*w_plus**2)
        lam2_minus =  ((-1+K)*w_minus - np.sqrt(K)*(1-w_minus**2))/(-1 + K*w_minus**2)

        a = np.fmax.reduce(np.abs([lam1_plus,lam1_minus,lam2_plus,lam2_minus]))

        return a
    
    def Char_Speed_TimeStep(self,f,t):

        """ Compute largest characteristic speed over 
        the domain for adjusting time-step."""

        sigma = self.sigma
        K = self.K

        Pi, Phi = f
        w, mu = self.Conserved_to_Primitive(Pi,Phi)

        lam1 = ((-1+K)*w + np.sqrt(K)*(1-w**2))/(-1 + K*w**2)
        lam2 = ((-1+K)*w - np.sqrt(K)*(1-w**2))/(-1 + K*w**2)

        CS = np.max(np.abs([lam1,lam2]))

        return CS


    ##############################################################
    # Evolution Equation(s)
    ##############################################################

    def rhs(self,t,f):

        Pi, Phi = f

        Floor = 1e-14
        Phi[np.where(Phi<Floor)] = Floor
        Pi[np.where(Pi<Floor)] =  Floor

        K = self.K
        sigma = self.sigma

        dtPi = np.zeros_like(Pi)
        dtPhi = np.zeros_like(Phi)

        # Get the values at grid points
        grid_points = self.grid.interior_coordinates() + 0.5*self.grid.dx
        grid_points = np.concatenate(([grid_points[0]-self.grid.dx],grid_points))

        # Values at grid points, including ghost points
        grid_points2 = self.grid.coordinates() + 0.5*self.grid.dx 
        # grid_points2[1] = 0.0

        #################################
        # Construct Fluxes
        ################################

        w, mu = self.Conserved_to_Primitive(Pi,Phi)


        w_plus, w_minus = Linear_Reconstruct(w,Minmod)
        mu_plus, mu_minus = Linear_Reconstruct(mu,Minmod)
        Pi_plus, Pi_minus = Linear_Reconstruct(Pi,Minmod)
        Phi_plus, Phi_minus = Linear_Reconstruct(Phi,Minmod)


        CS = self.Characteristic_Speeds(w_plus, w_minus,t)

        flux_Pi_tmp_plus = 0.5*((Pi_plus - Phi_plus)*(1+w_plus))
        flux_Pi_tmp_minus = 0.5*((Pi_minus - Phi_minus)*(1+w_minus))

        flux_Phi_tmp_plus = 0.5*((Pi_plus - Phi_plus)*(1-w_plus))
        flux_Phi_tmp_minus = 0.5*((Pi_minus - Phi_minus)*(1-w_minus))

        
        flux_Pi_1 = Local_Lax_Friedrichs(flux_Pi_tmp_plus,flux_Pi_tmp_minus,Pi_plus,Pi_minus,CS) 
        flux_Phi_1 = Local_Lax_Friedrichs(flux_Phi_tmp_plus,flux_Phi_tmp_minus,Phi_plus,Phi_minus,CS) 

        flux_Pi_2 = 0.5*K*(mu_plus + mu_minus)
        flux_Phi_2 = -0.5*K*(mu_plus + mu_minus)

        # flux_Pi_1, flux_Pi_2, flux_Phi_1, flux_Phi_2 = self.compute_flux(t,f,arg=1)

        #################################
        # Construct Source Terms
        ################################

        Pi_source = ((1+K)*(-1+3*K)*sigma*w*mu)/((1-sigma)*t*(1-w))

        Phi_source = ((1+K)*(-1+3*K)*sigma*w*mu)/((-1 + sigma)*t*(1+w))

        ###############################
        # Evolution Equations
        ##############################

        dr3 = np.diff(grid_points**3) # This is the `step size' for the d_{r^{3}} derivative terms


        dtPi[self.Ngz:-self.Ngz] = -3/(dr3)*(grid_points2[self.Ngz:-self.Ngz]**2*flux_Pi_1[self.Ngz:-self.Ngz] - grid_points2[self.Ngz-1:-self.Ngz-1]**2*flux_Pi_1[self.Ngz-1:-self.Ngz-1])\
               -1/(self.grid.dx)*(flux_Pi_2[self.Ngz:-self.Ngz] - flux_Pi_2[self.Ngz-1:-self.Ngz-1]) \
                + Pi_source[self.Ngz:-self.Ngz]
        
        dtPhi[self.Ngz:-self.Ngz] = -3/(dr3)*(grid_points2[self.Ngz:-self.Ngz]**2*flux_Phi_1[self.Ngz:-self.Ngz] -  grid_points2[self.Ngz-1:-self.Ngz-1]**2*flux_Phi_1[self.Ngz-1:-self.Ngz-1])\
                -1/(self.grid.dx)*(flux_Phi_2[self.Ngz:-self.Ngz] - flux_Phi_2[self.Ngz-1:-self.Ngz-1]) \
                + Phi_source[self.Ngz:-self.Ngz]
        

        # dtPi[self.Ngz:-self.Ngz] = -1/(self.grid.dx)*1/(self.grid.interior_coordinates()**2)*(grid_points2[self.Ngz:-self.Ngz]**2*flux_Pi_1[self.Ngz:-self.Ngz] - grid_points2[self.Ngz-1:-self.Ngz-1]**2*flux_Pi_1[self.Ngz-1:-self.Ngz-1])\
        #        -1/(self.grid.dx)*(flux_Pi_2[self.Ngz:-self.Ngz] - flux_Pi_2[self.Ngz-1:-self.Ngz-1]) \
        #         + Pi_source[self.Ngz:-self.Ngz]
        
        # dtPhi[self.Ngz:-self.Ngz] = -1/(self.grid.dx)**1/(self.grid.interior_coordinates()**2)*(grid_points2[self.Ngz:-self.Ngz]**2*flux_Phi_1[self.Ngz:-self.Ngz] -  grid_points2[self.Ngz-1:-self.Ngz-1]**2*flux_Phi_1[self.Ngz-1:-self.Ngz-1])\
        #         -1/(self.grid.dx)*(flux_Phi_2[self.Ngz:-self.Ngz] - flux_Phi_2[self.Ngz-1:-self.Ngz-1]) \
        #         + Phi_source[self.Ngz:-self.Ngz]
        

        return np.array([dtPi, dtPhi])
    

    
    ############################################################
    # Evolution Equation(s)
    ############################################################
        
    def Initial_Data(self):

        x = self.grid.interior_coordinates()
        n = np.shape(x)[0]

        K = self.K

        w_interior = 0.00*np.exp(-20*(x-(self.grid.interval[1]+self.grid.interval[0])/2)**2)  # Need |w| < 1!
        w_interior[np.abs(w_interior)<1e-15] = 0.0
        mu_interior = np.ones_like(x) + 0.5*x**3*np.exp(-20*(x-(self.grid.interval[1]+self.grid.interval[0])/2)**2)
        Gamma2 = 1/(1-w_interior**2)

        tau_interior = (K+1)*mu_interior*Gamma2 - K*mu_interior
        S_interior = (K+1)*mu_interior*Gamma2*w_interior

        Pi_interior = tau_interior + S_interior
        Phi_interior = tau_interior - S_interior

        # Pi_interior = (K+1)*Gamma2*(1+w_interior)*mu_interior - K*mu_interior
        # Phi_interior = (K+1)*Gamma2*(1-w_interior)*mu_interior - K*mu_interior


        ### Pad array with appropriate number of ghost points
        Pi_ID = np.array([np.concatenate((self.Ngz*[0],Pi_interior,self.Ngz*[0]))]) 
        Phi_ID = np.array([np.concatenate((self.Ngz*[0],Phi_interior,self.Ngz*[0]))]) 

        ID = np.reshape(np.array([Pi_ID,Phi_ID]),(2,n+4))

        ### Update ghost points using boundary conditions
        ID = self.BCs(ID,self.Npoints,self.Ngz)


        return ID
