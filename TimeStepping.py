import numpy as np
from matplotlib import pyplot as plt


""" TimeStepping.py

    Created by Elliot Marshall 2025-07-07
    
    This file contains all the routines for timestepping
    via the method of lines (MOL).

    Currently, only simple Runge-Kutta methods are implemented. 
    
"""





#######################################################################################################
#                                    FIXED MESH TIMESTEPPING ROUTINES
#######################################################################################################


##############################################
# Second-Order Runge Kutta (Heun's Method)
##############################################
def rk2(system,tl,dt): 

    """ Inputs:
    
        system: The PDE system to solve
        tl: Current timelevel
        dt: Time step"""
    
    rhs = system.rhs ### RHS of PDE
    BCs = system.BCs ### External Boundary Conditions
    t0 = tl.t
    y0 = tl.soln

    k1 = rhs(t0,y0)
    k1 = BCs(k1,system.Npoints,system.Ngz) ### Update ghost points

    k2 = rhs(t0 + dt, y0 + dt*k1)
    k2 = BCs(k2,system.Npoints,system.Ngz)

    y0 = y0 + dt*(0.5*k1 + 0.5*k2)

    return y0


##############################################
# Third-Order SSP Runge Kutta (Shu-Osher?)
##############################################
def rk3(system,tl,dt): 

    """ Inputs:
    
        system: The PDE system to solve
        tl: Current timelevel
        dt: Time step"""
    
    rhs = system.rhs ### RHS of PDE
    BCs = system.BCs ### External Boundary Conditions
    t0 = tl.t
    y0 = tl.soln

    k1 = rhs(t0, y0)
    k1 = BCs(k1,system.Npoints,system.Ngz) 

    k2 = rhs(t0 + dt, y0 + dt*k1)
    k2 = BCs(k2,system.Npoints,system.Ngz) 

    k3 = rhs(t0 + 0.5*dt, y0 + dt*(0.25*k1+0.25*k2))
    k3 = BCs(k3,system.Npoints,system.Ngz) 

    y0 = y0 + dt*(1/6*k1 + 1/6*k2 + 2/3*k3)


    return y0


##############################################
# Fourth-Order "Classic" Runge Kutta 
##############################################
def rk4(system,tl,dt): 

    """ Inputs:
    
        system: The PDE system to solve
        tl: Current timelevel
        dt: Time step"""
    
    rhs = system.rhs ### RHS of PDE
    BCs = system.BCs ### External Boundary Conditions
    t0 = tl.t
    y0 = tl.soln


    k1 = rhs(t0,y0)
    y1 = y0 + 0.5*dt*k1
    y1 = BCs(y1,system.Npoints,system.Ngz)  ### Updates ghost points using physical boundaries


    k2 = rhs(t0 + 0.5*dt, y1)
    y2 = y0 + 0.5*dt*k2
    y2 = BCs(y2,system.Npoints,system.Ngz) 


    k3 = rhs(t0 + 0.5*dt, y2)
    y3 = y0 + dt*k3
    y3 = BCs(y3,system.Npoints,system.Ngz)  

    k4 = rhs(t0 + dt, y3)

    y0 = y0 + dt*(1/6*k1 + 1/3*k2 +1/3*k3 +1/6*k4)

    y0 = BCs(y0,system.Npoints,system.Ngz) 


    return y0



#######################################################################################################
#                                   ADAPTIVE MESH TIMESTEPPING ROUTINES
#######################################################################################################


#----------------------------------------------
# First-Order Runge Kutta (Euler)
#----------------------------------------------

def Euler_rk_AMR(patch, dt, level,iteration_steps): 

    """ Inputs:
    
        patch: The current patch
        dt: Time step
        level: The current level
        iteration_steps: The iteration step that each level takes
        
    """
    
    rhs = patch.system.rhs ### RHS of PDE
    BCs = patch.system.BCs ### External Boundary Conditions
    interpolate_child_boundary = patch.interpolate_child_boundary

    tl = patch.tl_previous ### Recall that tl_previous is updated to tl_current at the start of each grid loop
    t0 = np.copy(tl.t)
    y0 = np.copy(tl.soln)


    if level >= 1:
        flux = patch.system.compute_flux(t0,y0) 
        patch.update_flux_register_fine(flux)
        if patch.children:
            patch.update_flux_register_coarse(flux)
                  


    k1 = rhs(t0,y0)

    y0 = y0 + dt*k1

    return y0


#----------------------------------------------
# Second-Order Runge Kutta (Heun's Method)
#----------------------------------------------

def rk2_AMR(patch, dt, level,iteration_steps): 

    """ Inputs:
    
        patch: The current patch
        dt: Time step
        level: The current level
        iteration_steps: The iteration step that each level takes
        
    """
    
    rhs = patch.system.rhs ### RHS of PDE
    BCs = patch.system.BCs ### External Boundary Conditions
    interpolate_child_boundary = patch.interpolate_child_boundary

    tl = patch.tl_previous ### Recall that tl_previous is updated to tl_current at the start of each grid loop
    t0 = np.copy(tl.t)
    y0 = np.copy(tl.soln)

    ### Compute first stage of flux for flux register. 
    ### NB: We must weight the fluxes by the RK substep weights
    if level >= 1:
        flux_1 = patch.system.compute_flux(t0,y0) 
        patch.update_flux_register_fine(0.5*flux_1)
        if patch.children:
            patch.update_flux_register_coarse(0.5*flux_1)

    # It is important to note that we only use interpolation on intermediate states!
    k1 = rhs(t0,y0)

    y1 = y0 + dt*k1
    if level < 2:
        y1 = BCs(y1,patch.Npoints,patch.Nghosts) ### Updates ghost points using physical boundaries
    
    ### Update the internal (AMR) boundaries:
    tplus = 1.0
    if level > 1: 
        # y1 = BCs(y1,patch.Npoints,patch.Nghosts) # This doesn't work if using periodic BCs
        y1 = interpolate_child_boundary(y1,tplus,iteration_steps,level)

  
    if level >= 1:
        flux_2 = patch.system.compute_flux(t0+dt,y1) 
        patch.update_flux_register_fine(0.5*flux_2)
        if patch.children:
            patch.update_flux_register_coarse(0.5*flux_2)



    k2 = rhs(t0 + dt, y1)


    y0 = y0 + dt*(0.5*k1 + 0.5*k2) 


    return y0



#----------------------------------------------
# Fourth-Order Runge Kutta ("Classic RK4")
#----------------------------------------------

def rk4_AMR(patch, dt, level,iteration_steps): 

    """ Inputs:
    
        patch: The current patch
        dt: Time step
        level: The current level
        iteration_steps: The iteration step that each level takes
        
    """
   
    rhs = patch.system.rhs ### RHS of PDE
    BCs = patch.system.BCs ### External Boundary Conditions
    interpolate_child_boundary = patch.interpolate_child_boundary

    tl = patch.tl_previous ### Recall that tl_previous is updated to tl_current at the start of each grid loop
    t0 = np.copy(tl.t)
    y0 = np.copy(tl.soln)


    ### Compute first stage of flux for flux register. 
    ### NB: We must weight the fluxes by the RK substep weights
    if level >= 1:
        flux_1 = patch.system.compute_flux(t0,y0) 
        patch.update_flux_register_fine(1/6*flux_1)
        if patch.children:
            patch.update_flux_register_coarse(1/6*flux_1)

    # It is important to note that we only use interpolation on intermediate states!
    k1 = rhs(t0,y0)

    y1 = y0 + 0.5*dt*k1
    if level < 2:
        y1 = BCs(y1,patch.Npoints,patch.Nghosts) ### Updates ghost points using physical boundaries

    tplus = 0.5
    if level > 1: 
        y1 = BCs(y1,patch.Npoints,patch.Nghosts) # This doesn't work if using periodic BCs
        y1 = interpolate_child_boundary(y1,tplus,iteration_steps,level)

    ### Second Flux Register Update
    if level >= 1:
        flux_2 = patch.system.compute_flux(t0+0.5*dt,y1) 
        patch.update_flux_register_fine(1/3*flux_2)
        if patch.children:
            patch.update_flux_register_coarse(1/3*flux_2)

    k2 = rhs(t0 + 0.5*dt, y1)

    y2 = y0 + 0.5*dt*k2
    if level < 2:
        y2 = BCs(y2,patch.Npoints,patch.Nghosts)

    tplus = 0.5
    if level > 1: 
        y2 = BCs(y2,patch.Npoints,patch.Nghosts) 
        y2 = interpolate_child_boundary(y2,tplus,iteration_steps,level)

    
    ### Third Flux Register Update
    if level >= 1:
        flux_3 = patch.system.compute_flux(t0+0.5*dt,y2)  
        patch.update_flux_register_fine(1/3*flux_3)
        if patch.children:
            patch.update_flux_register_coarse(1/3*flux_3)

    k3 = rhs(t0 + 0.5*dt, y2)

    y3 = y0 + dt*k3
    if level < 2:
        y3 = BCs(y3,patch.Npoints,patch.Nghosts) 

    tplus = 1.0
    if level > 1: 
        y3 = BCs(y3,patch.Npoints,patch.Nghosts) 
        y3 = interpolate_child_boundary(y3,tplus,iteration_steps,level)

    ### Fourth Flux Register Update
    if level >= 1:
        flux_4 = patch.system.compute_flux(t0+dt,y3)  
        patch.update_flux_register_fine(1/6*flux_4)
        if patch.children:
            patch.update_flux_register_coarse(1/6*flux_4)


    k4 = rhs(t0 + dt, y3)


    y0 = y0 + dt*(1/6*k1 + 1/3*k2 +1/3*k3 +1/6*k4)

    
    return y0

