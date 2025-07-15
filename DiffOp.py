import numpy as np
# from numba import njit

""" DiffOp.py

    Created by Elliot Marshall 2025-07-07
    
    This file contains the functions for discretising derivatives.

"""


##############################################################
# Standard Finite Difference Operators
##############################################################

def FD_Central_2(f, grid):

    """ This function computes a second-order central 
    finite difference approximation for the first derivative.
    
    Inputs:
    
        f - The function to be differentiated
        grid - The grid instance
    
    """

    df = np.zeros_like(f)
    n = np.shape(f)[0]
    Ngz = grid.Nghosts


    df[Ngz:-Ngz] = (-f[Ngz-1:-Ngz-1] + f[Ngz+1:-Ngz+1])/(2*grid.dx)

    # df[2:-2] = (-f[1:-3] + f[3:-1])/(2*grid.dx)

 
    return df # Ghost points set to zero.


def FD_Central_4(f, grid):

    """ This function computes a fourth-order central 
    finite difference approximation for the first derivative.
    
    Inputs:
    
        f - The function to be differentiated
        grid - The grid instance
    
    """

    df = np.zeros_like(f)
    n = np.shape(f)[0]
    Ngz = grid.Nghosts


    df[Ngz:-Ngz] = (f[Ngz-2:-Ngz-2]-8*f[Ngz-1:-Ngz-1]+8*f[Ngz+1:-Ngz+1]-f[Ngz+2:])/(12*grid.dx)
 

    return df # Ghost points set to zero.



def FD_Forward_2(f, grid):

    """ This function computes a second-order forward
    finite difference approximation for the first derivative.
    
    Inputs:
    
        f - The function to be differentiated
        grid - The grid instance
    
    """

    df = np.zeros_like(f)
    n = np.shape(f)[0]
    Ngz = grid.Nghosts

    df[Ngz:-Ngz] = (-3/2*f[Ngz:-Ngz] + 2*f[Ngz+1:-Ngz+1] - 0.5*f[Ngz+2:])/(grid.dx)


    return df ### Derivative at ghost points is set to zero.


def FD_Backward_2(f, grid):

    """ This function computes a second-order backward 
    finite difference approximation for the first derivative.
    
    Inputs:
    
        f - The function to be differentiated
        grid - The grid instance
    
    """

    df = np.zeros_like(f)
    n = np.shape(f)[0]
    Ngz = grid.Nghosts

    # df[Ngz:-Ngz] = (3/2*f[Ngz:-Ngz] - 2*f[Ngz-1:-Ngz-1] + 0.5*f[Ngz-2:-Ngz-2])/(grid.dx)

    df[Ngz:-Ngz] = (f[Ngz:-Ngz] - f[Ngz-1:-Ngz-1])/(grid.dx)


    return df ### Derivative at ghost points is set to zero.




def FD2_Central_2(f, grid):

    """ This function computes a second-order central 
    finite difference approximation for the second derivative.
    
    Inputs:
    
        f - The function to be differentiated
        grid - The grid instance
    
    """

    df = np.zeros_like(f)
    n = np.shape(f)[0]
    Ngz = grid.Nghosts


    df[Ngz:-Ngz] = (f[Ngz-1:-Ngz-1] -2*f[Ngz:-Ngz] + f[Ngz+1:-Ngz+1])/(grid.dx)



    return df # Ghost points set to zero.


##############################################################
# Slope-Limited Methods
##############################################################

#----------------------------------
# Flux Limiters 
#----------------------------------
# @njit(cache=True)
def Minmod(r):

    df = np.zeros(np.shape(r))

    df = np.fmax(0,np.fmin(1,r))

    return df

def MC(r):

    df = np.zeros(np.shape(r))

    df = np.fmax(0,np.fmin(np.fmin(2*r,0.5*(1+r)),2))

    return df

def superbee(r):
    df = np.zeros(np.shape(r))

    df = np.fmax(0,np.fmax(np.fmin(2*r,1),np.fmin(r,2)))

    return df

def VA(r):
    df = np.zeros(np.shape(r))

    df = (r**2 +r)/(r**2+1)

    return df


#-----------------------------------------------------
# Piecewise Linear Reconstruction
#-----------------------------------------------------

# @njit(cache=True)
def Linear_Reconstruct(u,limiter):

    u_plus = np.zeros_like(u) ### "Plus" = Reconstruction approaching from the right
    u_minus = np.zeros_like(u) ### "Minus" = Reconstruction approaching from the left
    ratio = np.zeros_like(u)
    n = np.shape(u)[0]

    
    ### Compute the ratio of slopes for the limiter
    ratio[1:n-1] = (u[1:n-1]-u[0:n-2])/(u[2:n]-u[1:n-1]+1e-16)

    ### NB: We add small number to the denominator to 
    ### stop NaN issues when neighbouring grid points are close to equal


    ### Compute the slope-limited linear reconstruction:

    u_minus[0:n-1] = u[0:n-1] + 0.5*limiter(ratio)[0:n-1]*(u[1:n]-u[0:n-1])

    u_plus[0:n-2] = u[1:n-1] - 0.5*limiter(ratio)[1:n-1]*(u[2:n]-u[1:n-1])


    return u_plus, u_minus

# @njit(cache=True)
def Local_Lax_Friedrichs(flux_plus,flux_minus,cons_plus,cons_minus,char_speed):

    """ Inputs:
        
        flux_plus: Reconstructed flux approaching from the right.
        flux_minus: Reconstructed flux approaching from the left.
        cons_plus: Reconstructed conserved variables approaching from the right.
        cons_minus: Reconstructed conserved variables approaching from the left.
        char_speed: The maximum of the absolute value of the characteristic speeds in each cell. (Using reconstructed values)
    """

    flux_LF = 0.5*( (flux_plus + flux_minus)  - (char_speed)*(cons_plus - cons_minus))

    return flux_LF


##############################################################
# SBP Operators
##############################################################

### To Do
