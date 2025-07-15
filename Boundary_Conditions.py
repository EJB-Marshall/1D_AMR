import numpy as np

""" Boundary_Conditions.py 

    Created by Elliot Marshall 2025-07-08
    
    This file contains some simple functions for setting up 
    external boundary conditions using ghost points. 

"""


###########################################
# Boundary Conditions
##########################################

def periodic_BC(u, Npoints, Ngz): # Periodic BCs have to be on boths sides

    ### If using periodic boundaries, currently need to update both
    ### the timestepping and interpolate_child_boundary routines (see comments in those files)

    """ Inputs:
    
        u - The array to be updated
        Npoints - The number of internal grid points
        Ngz - The number of ghost points
    """

    
    ### For 2 ghost points (testing)
    # u[:,0] = u[:,-4]
    # u[:,1] = u[:,-3]

    # u[:,-2] = u[:,2]
    # u[:,-1] = u[:,3]

    ### For arbitrary ghost points
    u[:,:Ngz] = u[:,-2*Ngz:-Ngz]
    u[:,-Ngz:] = u[:,Ngz:2*Ngz]

    return u


def outflow_BC(u, Npoints, Ngz): # Outflow boundaries on both sides

    for i in range(Ngz):
        u[:,i] = u[:,Ngz]
        u[:,Npoints+Ngz+i] = u[:,Npoints+Ngz-1]


    return u



def spherical_symmetry_BC(u, Npoints, Ngz): 

    """ These boundary conditions are used when evolving the relativisitic Euler
        equations in spherical symmetry.
    
    
        At the origin r=0, we use symmetry conditions on Pi and Phi to fill in the ghost points.
        See Hawke-Stewart "The Dynamics of Primordial Black-Hole Formation", 2002
        and Neilsen-Choptuik "Ultrarelativistic Fluid Dynamics" 1999 for details.

        The right boundary uses first-order extrapolation boundary conditions.

    """


    # Left Boundary
    u[0,0] = u[1,Ngz+1]
    u[0,1] = u[1,Ngz]

    u[1,0] = u[0,Ngz+1]
    u[1,1] = u[0,Ngz]


    # Right Boundary
    for i in range(Ngz):
        u[:,Npoints+Ngz+i] = u[:,Npoints+Ngz-1]


    return u
