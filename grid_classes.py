import numpy as np
import copy
import sys
from matplotlib import pyplot as plt

""" grid_classes.py
    Created by Elliot Marshall 2025-07-07
        
    This file contains the classes for constructing boxes, grids, timelevels, and patches.

    
    Box: A box defines the location of boundaries with respect to a parent grid. 
        It also distinguishes between external (physical) and internal 
        (i.e. those caused by constructing a refined grid) boundaries
        
    Grid: The grid class constructs arrays filled with each cell centre in the domain.
    
    Timelevel: A timelevel is a grid with the solution data at one time step.

    Patch: A patch is a grid which contains two timelevels of solution data.
            The patch class is the main object used in an AMR evolution
            and contains functions for interpolation and re-gridding etc.
    
"""


###########################################
# Box and Grid classes
###########################################

class box(object):

    def __init__(self,boundaries,boundary_type,Nghosts):
        """ Inputs:

            boundaries: Integer tuple, indices of boundary wrt parent. (Left_bnd, Right_bnd) (Python indexing, so Right_bnd not included)
            boundary_type: Boolean list, True for external bdy, false for internal bdy
        
        """
        self.boundaries = boundaries #
        self.boundary_type = boundary_type 
        self.Nghosts = Nghosts
        self.Npoints = 2 * ((boundaries[1]-boundaries[0])) # The number of cell centres


class grid(object):

    """Inputs:
    
        interval: A tuple (a,b) of the grid boundaries.
        grid_box: A grid_box instance
        
    """
    
    def __init__(self,interval, grid_box):
        self.interval = interval
        self.box = grid_box
        self.Npoints = grid_box.Npoints
        self.Nghosts = grid_box.Nghosts
        self.dx = (self.interval[1]-self.interval[0])/self.Npoints ### Spatial Step Size
    

    def coordinates(self):  ### Location of cell centres, including ghost cells

        x_start = self.interval[0] + (0.5 - self.Nghosts)*self.dx
        x_end = self.interval[1] + (self.Nghosts - 0.5)*self.dx

        return np.linspace(x_start, x_end, self.Npoints + 2*self.Nghosts)
   
    
    def interior_coordinates(self): ### Location of cell centres, excluding ghost cells

        x_start = self.interval[0] + 0.5*self.dx
        x_end = self.interval[1] - 0.5*self.dx

        return np.linspace(x_start, x_end, self.Npoints)
    
    

###########################################
# Set up initial box and grid instances
###########################################

def make_basebox(Npoints, Ngz):
    b = box((0,0),[True,True], Ngz)
    b.Npoints = Npoints
    b.boundaries = (None, None)

    return b

def make_basegrid(interval, Npoints, Nghosts):
    b = make_basebox(Npoints,Nghosts)

    return grid(interval,b)


###########################################
# Timelevel Class
###########################################
    
class timelevel(object):

    """ Inputs:
        
        grid: A grid class
        system: A system class (i.e. essentially the rhs of PDE we are solving)
        t: Current time
        Load_ID: True if loading initial data for first time, false otherwise
    """

    def __init__(self, grid, system, t, Load_ID=False):

        self.grid = grid
        self.system = system
        self.t = t
        self.Nvars = system.Nvars
        self.Npoints = grid.Npoints
        self.Nghosts = grid.Nghosts
        self.soln = np.zeros((self.Nvars, self.Npoints + 2*self.Nghosts)) ### Array for solution data
        if Load_ID:
            self.soln = self.system.Initial_Data() 



###########################################################
# AMR Classes/Functions
##########################################################


#----------------------------------
# Minmod Slope Limiter
#----------------------------------

def minmod(u):

    ### Note, we don't divide by dx because this term cancels out in the interpolation formula

    slope_l = np.zeros_like(u)
    slope_r = np.zeros_like(u)

    slope_l[1:] = np.diff(u)
    slope_r[:-1] = np.diff(u)

    
    return 0.5*(np.sign(slope_l) + np.sign(slope_r))*np.minimum(np.abs(slope_l), np.abs(slope_r))

    # return slope_r ### Linear reconstruction without slope limiting


#----------------------------------
# Flux Limiters 
#----------------------------------
def Minmod(r):

    df = np.zeros(np.shape(r))

    df = np.fmax(0,np.fmin(1,r))

    return df

def MC(r):

    df = np.zeros(np.shape(r))

    df = np.fmax(0,np.fmin(np.fmin(2*r,0.5*(1+r)),2))


    return df


#-----------------------------------------------------
# Piecewise Linear Interpolation 
#-----------------------------------------------------

def Linear_Reconstruct(u,limiter): # Only used for flux-limiter style reconstruction!

    u_left = np.zeros_like(u) 
    u_right = np.zeros_like(u) 
    ratio = np.zeros_like(u)
    n = np.shape(u)[0]

    ### Compute the ratio of slopes for the limiter
    ratio[1:n-1] = (u[1:n-1]-u[0:n-2])/(u[2:n]-u[1:n-1]+1e-16)

    ### NB: We add small number to the denominator to 
    ### stop NaN issues when neighbouring grid points are close to equal

    ### Compute the slope-limited linear reconstruction:
    ### Since our new cell centres are ∆x/4 away from the previous centre we must have 0.25 in front of limiter terms!
    u_right[0:n-1] = u[0:n-1] + 0.25*limiter(ratio)[0:n-1]*(u[1:n]-u[0:n-1]) 

    u_left[0:n-1] = u[0:n-1] - 0.25*limiter(ratio)[0:n-1]*(u[1:n]-u[0:n-1])


    return u_left, u_right


#------------------------------------------------------
# Piecewise Parabolic Interpolation (No limiters!)
#------------------------------------------------------

def Parabolic_Reconstruct(u):

    u_x = np.zeros_like(u)
    u_xx = np.zeros_like(u)
    u_left = np.zeros_like(u)
    u_right = np.zeros_like(u)
    # n = np.shape(u)[0]

    # These are second-order approximations of derivatives! 
    # For PPM reconstruction should be using (at least) 3rd-order accurate approximations here...
    # This would require at least *three* ghost points to be able to extrapolate at the first ghost point of parent grid
    u_x[1:-1] =  (u[2:] - u[0:-2])/2 # Not dividing by dx!
    u_xx[1:-1] = (u[2:] - 2*u[1:-1] +  u[0:-2])


    u_left[1:-1] = u[1:-1] - 1/4*u_x[1:-1] + 1/4*u_xx[1:-1]
    u_right[1:-1] = u[1:-1] + 1/4*u_x[1:-1] + 1/4*u_xx[1:-1]


    return u_left, u_right




# Compute L2 norm at a single timestep (for error estimation)
def computel2norm(u,dx):

    norm = np.sqrt(dx*np.sum(u[:]**2))

    return norm


class patch(object):

    """ The patch class is for running AMR simulations.

        It is essentially a grid with timelevel data available to it. 
        
        Inputs: 

        grid: The current (sub)grid we are working with
        system: The PDE system
        t: The current time
        parent: The parent of this subgrid. Set to "None" if the grid has no parent
        children: A list of the child patches. 

    """

    def __init__(self, grid, system, t, parent=None, children=[]):
        self.grid = grid
        self.parent = parent
        self.system = copy.deepcopy(system) # This ensures that we do not affect the system instance of the other patches
        self.Npoints = grid.Npoints
        self.Nghosts = grid.Nghosts
        self.local_error = np.zeros(self.grid.Npoints + 2 * self.grid.Nghosts) 
        self.t = t # Why do we have time for both timelevels and the patch itself. This is annoying and should be fixed...
        self.iterations = 0 # How many time steps this patch has taken
        self.children = children

        ### Update the system to use the correct grid quantities for each new patch
        self.system.grid = grid
        self.system.Npoints = self.Npoints
        self.system.Ngz = self.Nghosts

        
        if self.parent:
            self.interpolate_to_child() ### Construct fine solution data from parent grid
        else:
            self.tl_current = timelevel(self.grid,self.system,self.t,Load_ID=True) # Create two time levels of data with initial data from system class
            self.tl_previous = timelevel(self.grid,self.system,self.t,Load_ID=True)

        ### Construct flux register + flux store
        self.flux_register = np.zeros((self.system.Nvars,2))


    def swap_timelevels(self):

        """ Swaps the timelevel at the start of each timestep.
        
        """

        self.tl_current, self.tl_previous = self.tl_previous, self.tl_current

        self.tl_current.t = self.tl_previous.t #Update times

    

    def Kreiss_Oliger_Diss(self):

        """ Adds 3rd order (Kreiss-Oliger dissipation (i.e. using 4th derivatives)
            to all grid functions. 
             
            This seems to be necessary for a finite difference scheme to ensure high frequency 
            oscillations don't build up at grid refinment boundaries over time.
             
            (This is a 2nd order approximation of the 4th derivative, 
             why is it called 3rd order dissipation?)  
             
        """
        ### NB: No factor of dx in dissipation operator since 
        ### we are adding it directly to solution and not the evolution equations! 
        eps = 1.0
        soln = np.copy(self.tl_current.soln)
        soln2 = np.copy(soln)
        for Nvars in range(np.shape(soln)[0]):
            soln2[Nvars,self.Nghosts:-self.Nghosts] -= (eps/16) * (soln[Nvars,self.Nghosts-2:-self.Nghosts-2] \
                                                - 4*soln[Nvars,self.Nghosts-1:-self.Nghosts-1] \
                                                + 6*soln[Nvars,self.Nghosts:-self.Nghosts] \
                                                - 4*soln[Nvars,self.Nghosts+1:-self.Nghosts+1] \
                                                + soln[Nvars,self.Nghosts+2:]) 
            

            
        self.tl_current.soln = soln2

    #################################################
    # Interpolation Operator
    #################################################

    def interpolate_to_child(self): 
        """Fill in child data using slope-limited 
           piecewise linear interpolation of parent data

        """
        self.tl_current = timelevel(self.grid,self.system,self.parent.tl_current.t)
        parent_slopes = np.zeros_like(self.parent.tl_current.soln)
        u_left = np.zeros_like(self.parent.tl_current.soln)
        u_right = np.zeros_like(self.parent.tl_current.soln)

        for Nvars in range(np.shape(parent_slopes)[0]): # Fill in the internal points *and* the ghost points 

            ### Piecewise Linear Reconstruction

            # Slope-Limiter Style (e.g. Leveque 2002, Section 6.9)
            # parent_slopes[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2] = \
            #     minmod(self.parent.tl_current.soln[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2]) 

            # Flux-Limiter Style (e.g. Leveque 2002, Section 6.11)
            u_left[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2],u_right[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2] = \
                Linear_Reconstruct(self.parent.tl_current.soln[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2],Minmod)
            
            # Parabolic Reconstruction - Need to be more careful about pointvalue/cell average distinction if seriously trying to use this...
            # u_left[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2],u_right[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2] = \
            #     Parabolic_Reconstruct(self.parent.tl_current.soln[Nvars,self.grid.box.boundaries[0]-2:self.grid.box.boundaries[1]+2])


            for parent_index in range(self.grid.box.boundaries[0]-1, self.grid.box.boundaries[1]+1): # This fills in the internal points *and* the ghost points
                child_index = 2 * (parent_index - self.grid.box.boundaries[0]) + self.grid.Nghosts


                # Slope-Limiter Style, see Levque 1992, Section 16.3 for the derivation of this formula. 
                # Since our new cell centres are ∆x/4 away from the previous centre we must have 0.25 in front of limiter terms!
                # self.tl_current.soln[Nvars,child_index] = self.parent.tl_current.soln[Nvars,parent_index]  - 0.25*parent_slopes[Nvars,parent_index] # Left interpolation
                # self.tl_current.soln[Nvars,child_index + 1] = self.parent.tl_current.soln[Nvars,parent_index]  + 0.25*parent_slopes[Nvars,parent_index] #Right interpolation

                # Flux-Limiter Style
                self.tl_current.soln[Nvars,child_index] = u_left[Nvars,parent_index] # Left interpolation
                self.tl_current.soln[Nvars,child_index + 1] = u_right[Nvars,parent_index] #Right interpolation





        self.tl_previous = copy.deepcopy(self.tl_current)



    #################################################
    # Restriction Operator
    #################################################
    
    
    def restrict_patch(self): 
        """ Updates parent grid with data from child and
            computes the truncation error for re-grdding process.

        """
            
        self.local_error[:] = 0.0 ### Reset local error estimate
        local_error_tmp = np.zeros((self.system.Nvars,self.grid.Npoints + 2 * self.grid.Nghosts))
        for Nvars in range(np.shape(self.tl_current.soln)[0]):

            local_error_tmp[Nvars,2:-2] = np.abs(-self.tl_current.soln[Nvars,1:-3] + self.tl_current.soln[Nvars,3:-1]) # Error estimate from Clawpack
            for parent_index in range(self.grid.box.boundaries[0], self.grid.box.boundaries[1]): 
                child_index = 2 * (parent_index - self.grid.box.boundaries[0]) + self.grid.Nghosts


                ### Conservative update
                update_coarse = 0.5*(self.tl_current.soln[Nvars,child_index] + self.tl_current.soln[Nvars,child_index+1])

                ### Estimate truncation error:
                # local_error_tmp[Nvars,child_index:child_index+2] = np.abs(update_coarse - self.parent.tl_current.soln[Nvars,parent_index])\
                #                                                       /np.maximum(1,computel2norm(self.tl_current.soln[Nvars,:],self.grid.dx))
                # local_error_tmp[Nvars,child_index:child_index+2] = np.abs(update_coarse - self.parent.tl_current.soln[Nvars,parent_index])
                # local_error_tmp[Nvars,child_index:child_index+2] = np.abs(update_coarse - self.parent.tl_current_noupdate[Nvars,parent_index])\
                #                                                     /np.maximum(1,computel2norm(self.tl_current_noupdate[Nvars,:],self.grid.dx))


                self.parent.tl_current.soln[Nvars,parent_index] = update_coarse

        if self.system.Nvars == 1:
            self.local_error = local_error_tmp[0,:]
        else:
            self.local_error = np.fmax.reduce(local_error_tmp) # If more than 1 variable, define the local error in a cell to be largest error for any variable


    #################################################
    # Refluxing Functions
    #################################################

    # To ensure that conservation is preserved across fine/coarse grid boundaries
    # we need to correct the boundary cell averages on the coarse grid using the flux from the fine grids.
    # This was first done in the 1989 Berger-Collela AMR paper "Local Adaptive Mesh Refinement for Shock Hydrodynamics"
    # I think a better explanation of refluxing is given in the paper
    # "Adaptive Mesh Refinement Using Wave-Propagation Algorithms For Hyperbolic Systems", 1998
    # by  Berger-LeVeque. 

    def update_flux_register_fine(self, flux):

        """ The child grid updates its flux register at its parent 
            coarse/fine boundary
            
        """

        # Left Boundary
        if self.grid.box.boundaries[0] != self.Nghosts: 

            for Nvar in range(self.system.Nvars):
                

                self.flux_register[Nvar,0] += 0.5*flux[Nvar,self.Nghosts-1]


        # Right Boundary
        if self.grid.box.boundaries[1] != self.Nghosts + self.parent.Npoints:

            for Nvar in range(self.system.Nvars):

                self.flux_register[Nvar,1] += 0.5*flux[Nvar,-self.Nghosts-1]




    def update_flux_register_coarse(self, flux):

        """ The parent grid updates the flux register of its children at each of its child
            coarse/fine boundaries
            
        """

        for child in self.children:

            # Left Boundary
            if child.grid.box.boundaries[0] != child.Nghosts:

                i = child.grid.box.boundaries[0]-1

                for Nvar in range(self.system.Nvars):
                    
                    child.flux_register[Nvar,0] -= flux[Nvar,i]

            # Right Boundary
            if child.grid.box.boundaries[1] != child.Nghosts + child.parent.Npoints:

                i = child.grid.box.boundaries[1]

                for Nvar in range(self.system.Nvars):

                    child.flux_register[Nvar,1] -= flux[Nvar,i-1]
      

    def reflux_boundaries(self):


        # Left Boundary of Child Grid
        if self.grid.box.boundaries[0] != self.Nghosts:

            for Nvar in range(self.system.Nvars):
                
                i = self.grid.box.boundaries[0]-1
                self.parent.tl_current.soln[Nvar,i] -= \
                    (self.parent.dt/self.parent.grid.dx)*self.flux_register[Nvar,0]
                

        # Right Boundary of Child Grid
        if self.grid.box.boundaries[1] != self.Nghosts + self.parent.Npoints:

            for Nvar in range(self.system.Nvars):
                
                i = self.grid.box.boundaries[1]
                self.parent.tl_current.soln[Nvar,i] += \
                    (self.parent.dt/self.parent.grid.dx)*self.flux_register[Nvar,1]
              
                
        # Reset Flux Register    
        self.flux_register[:] = 0
            




    #################################################
    # Helper Functions for Regridding
    #################################################

    def update_error_from_child(self,threshold):
        """ Suppose the grids at level N are being regridded. This occurs after the grids 
        at level N+1 have been regridded and potentially created child grids at level N+2
        (i.e grandchild grids for level N). 
        
        To ensure the grandchild grids remain within one parent grid, 
        we copy error flags from level N+1 to level N.
            
        """ 
        for child in self.children:

            for parent_index in range(child.grid.box.boundaries[0],child.grid.box.boundaries[1]): # Can probably replace this loop with numpy functions...
                    child_index = 2*(parent_index - child.grid.box.boundaries[0]) + self.grid.Nghosts

                    if np.any(child.local_error[child_index:child_index+2]> threshold):
                        self.local_error[parent_index] = np.maximum(child.local_error[child_index], child.local_error[child_index+1])




    def update_parent_of_child(self, new_patches):
        """ New grids at level N must be re-linked to their children (and vice versa).
            This is necessary because we regrid from fine to coarse!
            
        """

        # This loop is unecessary if everything works, but useful for debugging 
        # if a parent grid is not assigned...
        for child in self.children:

                for grandchild in child.children:
                    grandchild.parent = None

        for new_patch in new_patches: ### All of these loops are really slow...
            new_patch.children = []

            for child in self.children:

                for grandchild in child.children:

                    
                    # Since a child grid should only have one parent, we check if the grandchild grids are contained in any of the new patches 
                    if (grandchild.grid.interior_coordinates()[0]  > new_patch.grid.interior_coordinates()[0] - 0.5*new_patch.grid.dx) and\
                    (grandchild.grid.interior_coordinates()[-1] < new_patch.grid.interior_coordinates()[-1] + 0.5*new_patch.grid.dx): 
                        
                        grandchild.parent = new_patch # Update child's parent
                        new_patch.children.append(grandchild) # Update the list of children for the new patch


            
        
        
    def correct_child_box_boundaries(self):
        """ If we are re-gridding level N (i.e. creating grids at level N+1)
            we may shift the parent grids (at level N+1) of grids at level N+2 (i.e. the grandchild grids.)

            This function updates the boundaries of grandchild grids 
            during the re-gridding stage.
            
        """
        for child in self.children:


            offset = int(np.ceil((child.grid.interior_coordinates()[0] - self.grid.interior_coordinates()[0])/self.grid.dx + self.Nghosts))
            child.grid.box.boundaries = (offset, offset + int(child.Npoints/2)) # Recall the right boundary is excluded (i.e. python indexing)




    
    def copy_data_from_old_grid(self,new_patches):
        """ Copies the solution data from the old grids at level N
            to the newly created grids at level N (if they overlap).

            If this function is not used, new grids are created using 
            interpolation only. (Not using this function seems to destroy the conservation though!)
            
        """

        for new_patch in new_patches: # Again, should remove loops here if possible

            new_patch_bnds = np.arange(new_patch.grid.box.boundaries[0],new_patch.grid.box.boundaries[1],1)
           
            for child in self.children:

                child_patch_bnds = np.arange(child.grid.box.boundaries[0],child.grid.box.boundaries[1],1)
                
                overlap_parent_indices = np.intersect1d(new_patch_bnds,child_patch_bnds)


                if np.shape(overlap_parent_indices)[0] >0:

                    child_start = 2 * (overlap_parent_indices[0] - child.grid.box.boundaries[0]) + child.grid.Nghosts
                    new_patch_start = 2 * (overlap_parent_indices[0]- new_patch.grid.box.boundaries[0]) + new_patch.grid.Nghosts
                    child_end = 2 * (overlap_parent_indices[-1] - child.grid.box.boundaries[0]) + child.grid.Nghosts + 1
                    new_patch_end = 2 * (overlap_parent_indices[-1]- new_patch.grid.box.boundaries[0]) + new_patch.grid.Nghosts + 1

                    # plt.plot(child.grid.interior_coordinates(),child.tl_current.soln[2,2:-2],'x',markersize=20)
                    # plt.plot(new_patch.grid.interior_coordinates(),new_patch.tl_current.soln[2,2:-2],'.',markersize=20)
                    # plt.plot(child.grid.coordinates()[child_start:child_end+1],child.tl_current.soln[2,child_start:child_end+1],'.',color='tab:purple')
                    # plt.plot(new_patch.grid.coordinates()[new_patch_start:new_patch_end+1],new_patch.tl_current.soln[2,new_patch_start:new_patch_end+1],'x',color='tab:red')
                    # plt.show()


                    for Nvars in range(self.system.Nvars):
                        new_patch.tl_current.soln[Nvars,new_patch_start:new_patch_end+1] = child.tl_current.soln[Nvars,child_start:child_end+1]
                        new_patch.tl_previous.soln[Nvars,new_patch_start:new_patch_end+1] = child.tl_current.soln[Nvars,child_start:child_end+1]




                

    #################################################
    # Regridding Function
    #################################################   

    def regrid_patch(self,threshold): ### This was modified from Ian Hawke's python AMR example code
        """ Flag points where re-gridding needs to occur, construct child grids if necessary.

            Inputs: 

            Threshold: The truncation error threshold. 
                       Points with errors above this value are flagged for refinement.

        """

        ### First, we copy the error from any child grids.
        ### At every index where the child grid is above error threshold, the parent grid should also be flagged. 
        ### (The local_error for children is not reset until the next restriction step)
        ### This ensures that any grandchild grids created in the previous re-gridding 
        ### are still contained in the new child grids created in this step.
        if self.children:
                self.update_error_from_child(threshold) # This should only be called if the patch has grandchildren, need to fix this


        error_flag = self.local_error > threshold # Create Boolean array with flagged cells
        error_flag2 = np.copy(error_flag)


        ### Reset the ghost cells to false, so they are not included in grid refinement.
        error_flag2[:self.Nghosts] = False
        error_flag2[-self.Nghosts:] = False


        # Flag error if any cells within Nghosts of i are flagged 
        # This creates what Pretorius calls the "buffer zone". 
        # For Nghosts = 2, we flag two buffer cells (on each boundary) which corresponds to 8 child buffer cells. This might be too many or too few, idk...
        for i in range(self.Nghosts, self.Npoints + self.Nghosts): 
            error_flag2[i] = (error_flag[i] or np.any(error_flag[i-self.Nghosts:i+self.Nghosts+1])) # Add 1 to account for python indexing...



        ### Reset error flags for the ghost cells.
        ### If parent boundary is internal reset flag 1 cell away from boundary of parent.
        ### This ensures no child grid entirely refines its parent except when touching an external boundary.
        ### This is stated to occur in step (2) of the Berger-Colella regridding algorithm,
        ### see pg. 73 of Berger-Colella 1989. 
        ### East et al.  do something similar by not allowing a single cell to be both type A and B,
        ### see East et al. "Hydrodynamics in Full General Relativity with Conservative AMR" 2012.
        if self.grid.box.boundary_type[0]:
            error_flag2[:self.Nghosts] = False # If parent boundary is external, we allow refinement to the outer boundary
        else: 
            error_flag2[:self.Nghosts+1] = False # If parent boundary is internal, insert 1 cell buffer between new child grid and parent boundary

        if self.grid.box.boundary_type[1]:
            error_flag2[-self.Nghosts:] = False
        else: 
            error_flag2[-1-self.Nghosts:] = False

        ### Find boundaries of boxes for refinement
        starts = []
        ends = []
        for i in range(self.Nghosts, self.Npoints + self.Nghosts): 
            if (not(error_flag2[i-1]) and error_flag2[i]):
                starts.append(i)
            elif (error_flag2[i] and not(error_flag2[i+1])):
                ends.append(i+1)


        ### Gather into boxes
        boundary_tuples = zip(starts,ends) # This combines the lists 'starts' and 'ends' and groups them into tuples (starts[i],ends[i]) which we can loop over

        
        ### Create the subgrids
        grids = []
        for boundaries in boundary_tuples:
            boundary_type = [False, False]


            ### Check if the child grid contains external boundaries
            if boundaries[0] == self.Nghosts:
                boundary_type[0] = self.grid.box.boundary_type[0]
            if boundaries[1] == self.Npoints + self.Nghosts:
                boundary_type[1] = self.grid.box.boundary_type[1]
            # Create the box
            sub_box = box(boundaries, boundary_type, self.Nghosts)
            interval = (self.grid.interval[0] + (boundaries[0] - self.Nghosts) * self.grid.dx,
                        self.grid.interval[0] + (boundaries[1] - self.Nghosts) * self.grid.dx)

            g = grid(interval, sub_box)

            grids.append(g)

        ### Create the new child patches
        patches = []
        for g in grids:
            patches.append(patch(g, self.system, self.t, self)) # New data is created only from interpolation from parent grid (at first). 
    
        for p in patches:
            p.iterations = np.copy(p.parent.iterations) # Re-gridding only occurs when child and parent are synchronised in time



        ### Update new child grid data using old child grids where they overlap
        ### It seems this is essential to preserving the conservation of the AMR scheme.
        self.copy_data_from_old_grid(patches) 
        
        ### Update the parent-child links for the new patches
        self.update_parent_of_child(patches)

        
        self.children = patches # Define the new child grids of current patch


        if self.children:
            for child in self.children:
                if child.children:
                    child.correct_child_box_boundaries() # Correct the box boundaries for grandchildren
        
        return patches


    ###########################################
    # Boundary Interpolation
    ###########################################


    def interpolate_child_boundary(self,child_soln,tplus,iteration_steps,level): 
        """ Updates the patch boundaries for child grids using linear interpolation 
            in time and space from the parent grid.

        Inputs:

        child_soln: The child solution array to be updated
        tplus: The child solution is at child.t + child.dt*tplus
        iteration_steps: The amount of iterations each grid level takes to complete one time step for its parent
        level: The refinement level of the current grid
        
        """

        ### Define parent timelevels for time interpolation
        ### NB: Since, the parent level always finishes its timesteps 
        ### before the child, tl_previous *is* the *previous* step
        ### and tl_current *is* the *current* step for the parent solution. (i.e. don't worry about the swap timelevels stuff)
        tl_previous = self.parent.tl_previous 
        tl_current = self.parent.tl_current

        ### Compute the weights for linear interpolation in time
        ### These formulas can be computed from the point gradient form a line etc.
        if self.iterations == self.parent.iterations:
            if tplus != 0:
                print("Parent and child grids are aligned in time but tplus is not zero!")
                print("Exiting simulation...")
                sys.exit()
            
            current_weight = 1.0
            previous_weight = 0.0

        elif self.iterations == self.parent.iterations-iteration_steps[level]:
            
            current_weight = 0.5 * (1.0 + tplus)
            previous_weight = 0.5 * (1 - tplus)

        elif self.iterations == self.parent.iterations-iteration_steps[level-1]:
            
            current_weight = 0.5 * tplus
            previous_weight = 0.5 * (2 - tplus)

        else: 
            print("Error when computing boundary interpolation!")
            sys.exit()


        ### This is the time interpolation weights from the Chombo docuemntation:
        ### I think the way I currently update time means I shouldn't use this...
        # child_t = self.tl_current.t + self.dt * tplus
        # current_weight = (child_t-self.tl_previous.t) / self.parent.dt
        # previous_weight = 1 - current_weight




        
        ####################################################
        # Check whether boundary is external or internal
        ####################################################

        ### Note, for Periodic BCs we always have to use interpolation for child grids!
        ### Ideally, you would like to use some sort of communication between grids at each level
        ### so periodic BCs can be enforced without having to use interpolation...
        
        do_left = not(self.grid.box.boundary_type[0]) # The reversed logic here is really dumb! This should be fixed.
        do_right = not(self.grid.box.boundary_type[1]) 

        # Set both boundaries to true when using periodic BCs
        # do_left = True
        # do_right = True




        ####################################################
        # Update boundaries using linear interpolation
        ####################################################

        if self.Nghosts != 2: 
            print("AMR is only implemented for 2 ghost points!")
            print("Exiting simulation...")
            sys.exit()

        for Nvars in range(np.shape(self.tl_current.soln)[0]):

            ### Update left boundary 
            if do_left:

                parent_index = self.grid.box.boundaries[0]-1 # Subtract 1 for the index of ghost cell adjacent to boundary

                ### Compute the linear slopes for current time step of parent grid:

                slope_l = tl_current.soln[Nvars,parent_index] - tl_current.soln[Nvars,parent_index-1] 
                slope_r = tl_current.soln[Nvars,parent_index+1] - tl_current.soln[Nvars,parent_index] 

                ### Minmod reconstruction of slope:
                slope = 0.5*(np.sign(slope_l) + np.sign(slope_r))*np.fmin.reduce(np.abs([slope_l, slope_r]))

                ### Ghost points
                child_Ngz_current = tl_current.soln[Nvars,parent_index] - 0.25*slope
                child_Ngz2_current = tl_current.soln[Nvars,parent_index] + 0.25*slope


                ### Compute the linear slopes for previous time step of parent grid:

                slope_l = tl_previous.soln[Nvars,parent_index] - tl_previous.soln[Nvars,parent_index-1] 
                slope_r = tl_previous.soln[Nvars,parent_index+1] - tl_previous.soln[Nvars,parent_index] 

                ### Minmod reconstruction of slope:
                slope = 0.5*(np.sign(slope_l) + np.sign(slope_r))*np.fmin.reduce(np.abs([slope_l, slope_r]))

                ### Ghost points
                child_Ngz_previous = tl_previous.soln[Nvars,parent_index] - 0.25*slope
                child_Ngz2_previous = tl_previous.soln[Nvars,parent_index] + 0.25*slope


                ### Interpolate in time:
                child_Ngz = current_weight * child_Ngz_current + previous_weight * child_Ngz_previous
                child_Ngz2 = current_weight * child_Ngz2_current + previous_weight * child_Ngz2_previous


                ### Fill in child data
                child_soln[Nvars,0] = child_Ngz
                child_soln[Nvars,1] = child_Ngz2

                

            ### Update right boundary 
            if do_right:

                parent_index = self.grid.box.boundaries[1]

                ### Compute the linear slopes for current time step of parent grid:

                slope_l = tl_current.soln[Nvars,parent_index] - tl_current.soln[Nvars,parent_index-1] 
                slope_r = tl_current.soln[Nvars,parent_index+1] - tl_current.soln[Nvars,parent_index] 

                ### Minmod reconstruction of slope:
                slope = 0.5*(np.sign(slope_l) + np.sign(slope_r))*np.fmin.reduce(np.abs([slope_l, slope_r]))

                ### Ghost points
                child_Ngz_current = tl_current.soln[Nvars,parent_index] + 0.25*slope
                child_Ngz2_current = tl_current.soln[Nvars,parent_index] - 0.25*slope 

                ### Compute the linear slopes for previous time step of parent grid:

                slope_l = tl_previous.soln[Nvars,parent_index] - tl_previous.soln[Nvars,parent_index-1] 
                slope_r = tl_previous.soln[Nvars,parent_index+1] - tl_previous.soln[Nvars,parent_index] 

                ### Minmod reconstruction of slope:
                slope = 0.5*(np.sign(slope_l) + np.sign(slope_r))*np.fmin.reduce(np.abs([slope_l, slope_r]))

                ### Ghost points
                child_Ngz_previous = tl_previous.soln[Nvars,parent_index] + 0.25*slope
                child_Ngz2_previous = tl_previous.soln[Nvars,parent_index] - 0.25*slope


                ### Interpolate in time:
                child_Ngz = current_weight * child_Ngz_current + previous_weight * child_Ngz_previous
                child_Ngz2 = current_weight * child_Ngz2_current + previous_weight * child_Ngz2_previous


                ### Fill in child data
                child_soln[Nvars,-1] = child_Ngz
                child_soln[Nvars,-2] = child_Ngz2
                

        return child_soln
    

   





    

        





        
        
        
                
    
    

    

    














