import numpy as np
from matplotlib import pyplot as plt
from grid_classes import grid, timelevel, patch
import copy
import sys


""" Simulation.py
    Created by Elliot Marshall 2025-07-07


    This file contains the simulation class for unigrid evolutions
    and the AMR_simulation class for AMR evolutions (work in progress!).

    Based on simple tests (e.g. wave equation, Burgers' equation)
    the AMR code appears to work for arbitrary levels of refinement, 
    and preserve conservation.

    It remains to test convergence etc. 
    
    
"""


#######################################################################
# Single Mesh Simulation Class
########################################################################


class simulation(object):

    """ Inputs:

        system: The PDE system to be evolved
        TimeIntegrator: The method used for integrating in time
        t_interval: The start and end times for the simulation (tuple)
        grid: The base grid
        cfl: Value of CFL number, default = 0.5
    
        """

    def __init__(self, system, TimeIntegrator, t_interval, grid, cfl=0.5):
        self.system = system
        self.TimeIntegrator = TimeIntegrator
        self.t_interval = t_interval
        self.grid = grid
        self.cfl = cfl
        self.t = t_interval[0]
        self.tl = timelevel(self.grid,self.system,self.t,Load_ID=True) ### Set up initial timelevel instance
        



    #####################################################
    # Evolution Routine
    #####################################################

    def Evolve(self):

        soln_store = [] 
        t_store = []

        ### Append Initial data
        soln_store.append(self.tl.soln[:,self.grid.Nghosts:-self.grid.Nghosts])
        # soln_store.append(self.tl.soln[:,:])
        t_store.append(self.t) 

        i = 0
        k = 0
        while self.t < self.t_interval[1]:

            i += 1

            if self.system.dt_fixed:
                dt = self.cfl*self.grid.dx**(4/3)
                # dt = self.cfl*self.grid.dx

            else:

                ### Adjust timestep based on characteristic speeds
                dt = (self.cfl*self.grid.dx)/(self.system.Characteristic_Speeds_TimeStep(self.tl.soln)) 

                ### If timestep becomes larger than normal CFL condition use CFL timestep instead.
                if dt >(self.cfl*self.grid.dx): 
                    dt = (self.cfl*self.grid.dx)
                

            if self.t + dt > self.t_interval[1]:
                dt = self.t_interval[1] - self.t ### Adjust time step at end to hit correct end point
                print("Simulation ending, adjusting timestep to dt = ",dt)

            self.tl.soln = self.TimeIntegrator(self.system, self.tl, dt)
            self.t += dt

            ### Store Solution
            if i == 20:
                soln_store.append(self.tl.soln[:,self.grid.Nghosts:-self.grid.Nghosts])
                # soln_store.append(self.tl.soln[:,:])
                t_store.append(self.t) 
                i = 0
                


        return [np.array(t_store),np.array(soln_store)]







#######################################################################
# Adaptive Mesh Simulation Class
########################################################################


class AMR_simulation(object):

    """ Inputs:

        system: The PDE system to be evolved
        TimeIntegrator: The method used for integrating in time
        t_interval: The start and end times for the simulation (tuple)
        grid: The base grid
        cfl: Value of CFL number, default = 0.5
        max_levels: Number of levels of mesh refinement
        threshold: The error threshold for refinining a grid
    
    """

    def __init__(self, system, TimeIntegrator, t_interval, shadow_grid, base_grid, cfl=0.5,max_levels=3,threshold=0.01):
        self.system = system
        self.TimeIntegrator = TimeIntegrator
        self.t_interval = t_interval
        self.shadow_grid = shadow_grid
        self.base_grid = base_grid
        self.base_coordinates = base_grid.coordinates()
        self.cfl = cfl
        self.t = t_interval[0]
        self.soln_merged_store = []
        self.grid_merged_store = []
        self.t_store = []
        self.solution_store = []
        self.max_levels = max_levels
        

        self.threshold = threshold
        # self.iterations = np.zeros(1+max_levels)
        self.iteration_steps = 2**np.arange(1+max_levels-1, -1, -1) ### Grid level N should check for refinement every 2*(N-1) steps

        ### Create initial patches
        shadow_patch = patch(self.shadow_grid,system,self.t)
        base_patch = patch(self.base_grid,system,self.t)

        # Create Parent-Child links
        shadow_patch.children = [base_patch]
        base_patch.parent = shadow_patch
        base_patch.grid.box.boundaries = [2,self.shadow_grid.Npoints+self.shadow_grid.Nghosts]

        # Create initial list of patches
        self.patches =[[shadow_patch],[base_patch]] ### Need two patches to start off the AMR


        #####################################################
        #  Testing for Fixed Mesh Refinement 
        ######################################################

        ### Need to turn off the regridding call in Evolve_One_Step()
        ### if you want to test with one level of fixed refinement.

        ### Flag some regions for refinement
        
        # base_patch.local_error[5:10] = 1
        # base_patch.local_error[150:650] = 1
        # child_patches= base_patch.regrid_patch(0.5)

        # child_patch = child_patches[0]
        # child_patch2 = child_patches[1]
        # child_patch.local_error[:] = 1
        # grandchild_patch = child_patch.regrid_patch(0.5)[0]

        # plt.rcParams.update({"text.usetex": True,
        #                 "font.family": "serif",
        #                 "font.serif": "Computer Modern",
        #                 "savefig.bbox": "tight",
        #                 "savefig.format": "pdf"})
        # plt.rc('font', size=16)

        # plt.plot(base_patch.grid.coordinates()[2:-2], base_patch.tl_current.soln[0,2:-2],'.')
        # # plt.plot(base_patch.grid.coordinates()[3:12], base_patch.tl_current.soln[0,3:12],'x')
        # plt.plot(child_patch.grid.coordinates()[2:-2], child_patch.tl_current.soln[0,2:-2],'.')
        # # plt.plot(base_patch.grid.coordinates()[353],base_patch.tl_current.soln[0,353],'.')
        # # plt.xlabel(r'$x$')
        # # plt.ylabel(r'$\bar{w}$')
        # plt.show()
        # breaks
        
        ## Add new patch(es) to the list
        # self.patches =[[shadow_patch],[base_patch],[child_patch]] 
        # self.patches =[[shadow_patch],[base_patch],[child_patch, child_patch2], [grandchild_patch]] 


    #####################################################
    # Save Evolution Data
    #####################################################

    def Save_Grids(self):

        """ Returns a list of dictionaries
            with the patch data at a single timestep
            
        """

        Grid_Data = []

        for level, patches in enumerate(self.patches):
            
            if level == 0: # Skip shadow grid
                continue

            level_data = []
            for patch in patches:
                data = {
                "level": level,
                "coordinates": patch.grid.coordinates().copy(),
                "soln": patch.tl_current.soln[:, :].copy(), 
                "boundaries": list(patch.grid.box.boundaries),
                }
            
                level_data.append(data)
            Grid_Data.append(level_data)



        return Grid_Data
    
    def construct_finest_solution(self):

        """ Construct solution (and the corresponding grid) array 
            with only the finest solution data available 
            in each cell.
            
        """

        soln_merged = self.patches[1][0].tl_current.soln[0,2:-2].tolist() # Remove ghost points
        grid_merged = self.patches[1][0].grid.interior_coordinates()[:].tolist()
        
        # Currently this converts everything to python lists and merges, which is less than ideal...
        # In the future, everything should be done using only numpy arrays/functions.
        for level, p_list in enumerate(self.patches): 

            if level in [0,1]: # Skip shadow and base grids
                continue

            for p in p_list:

                sort = np.searchsorted(grid_merged,p.grid.interior_coordinates())
                sort_idx_start = sort[0]
                sort_idx_end = sort[-1]

                grid_merged[sort_idx_start:sort_idx_end] = p.grid.interior_coordinates().tolist()
                soln_merged[sort_idx_start:sort_idx_end] = p.tl_current.soln[0,2:-2].tolist() # Remove ghost points



        return grid_merged, soln_merged
    
    
    def check_synchronisation(self):

        patch = self.patches[len(self.patches)-1][0]

        while patch.parent:

            if patch.iterations == patch.parent.iterations:
                patch = patch.parent
            else:
                return False
        
        return True





    #####################################################
    # Evolution Routine
    #####################################################

    def Evolve_One_Step(self,dt,level): 

        global count

        """ AMR Evolution Routine.
        
            This is the recursive Berger-Oliger 
            timestepping algorithm.
            
        """

        # print("Level " + str(level) + " evolves")
        # print()

        for patch in self.patches[level]: # Evolve all grids at this level first

            
            ### Switch timelevels (i.e. tl_current is now tl_previous and vice versa)
            patch.swap_timelevels()
            patch.dt = dt

            
            ### Evolve in time
            patch.tl_current.soln = self.TimeIntegrator(patch, dt, level, self.iteration_steps)

            patch.tl_current.t += dt
            patch.t += dt

                            
            ### Update boundaries
            if level >1:
                patch.tl_current.soln = patch.system.BCs(patch.tl_current.soln,patch.Npoints,patch.Nghosts)
                patch.tl_current.soln = patch.interpolate_child_boundary(patch.tl_current.soln,1.0,self.iteration_steps,level)
            else:
                patch.tl_current.soln = patch.system.BCs(patch.tl_current.soln,patch.Npoints,patch.Nghosts)

            ### Update patch iteration number
            patch.iterations += self.iteration_steps[level]

            # if level == 1:
            #     # print(dt)
            #     plt.plot(patch.tl_current.soln[1,:])
            #     plt.show()
                # plt.draw()
                # plt.pause(0.01)
                # plt.cla()
        

            patch.tl_current_noupdate = np.copy(patch.tl_current.soln) # This saves the value of the solution before injection - Better for error estimation?


        ### Check if subgrids need to be evolved
        if level < len(self.patches)-1:
            self.Evolve_One_Step(dt/2,level+1)
            self.Evolve_One_Step(dt/2,level+1)

        

    
        ### If level N has taken 2*(N-1) steps then restrict patch and check for re-gridding:
        if (level > 0) and (patch.iterations == patch.parent.iterations): # This only checks the last patch on a level and its parent are synchronised which is a bit sketchy...
            
        
            for patch in self.patches[level]:


                ### Restriction of fine grid to coarse:
                # patch.Kreiss_Oliger_Diss() # Add dissipation to all grid functions
                patch.restrict_patch() # Inject fine grid data to coarse parent, calculate error estimate


                ### Update physical BCs after restriction? This doesn't seem to do anything...
                # if (level-1 == 0) or (level-1 == 1):
                #     patch.parent.system.BCs(patch.parent.tl_current.soln,patch.parent.system.Npoints,patch.parent.system.Ngz)  

                ### Reflux grid functions to ensure flux conservation is preserved across coarse/fine boundaries
                patch.reflux_boundaries()

            
            ### Check for re-gridding:   
            if (0 < level < self.max_levels ): # This could probably be improved...

                refine_list = [] 
                for patch in self.patches[level]:
                        refine_list +=  patch.regrid_patch(self.threshold)
                
                
                if level == len(self.patches)-1:
                    self.patches.insert(level+1,refine_list)
                    if len(self.patches[level+1]) == 0: # If no points were flagged, the grids can be deleted.
                        self.patches.pop(level+1)
                        
                else:
                    self.patches[level + 1] = refine_list
                    if len(self.patches[level+1]) == 0: 
                        self.patches.pop(level+1)

            
            

    def Evolve(self):
        global count

        count = 0


        """ Starts off evolution routine for the base grid."""

        i = 0
        while self.t < self.t_interval[1]:
            
            count +=1

            if self.system.dt_fixed:
                dt = self.cfl*self.shadow_grid.dx
            else:
                dt = self.cfl*self.base_grid.dx/\
                    (self.patches[1][0].system.Char_Speed_TimeStep(self.patches[1][0].tl_current.soln,self.patches[1][0].tl_current.t))*2
                                
                # if dt > self.cfl*self.shadow_grid.dx:
                #     dt = self.cfl*self.shadow_grid.dx


            if self.t + dt > self.t_interval[1]:
                dt = self.t_interval[1] - self.t ### Adjust time step at end to hit correct end point
                print("Simulation ending, adjusting timestep to dt = ", dt)

            self.Evolve_One_Step(dt,0)
            self.t += dt # Simulation time

            i+=1
            

            if i == 1:
                self.solution_store.append(self.Save_Grids())
                grid_merged, soln_merged = self.construct_finest_solution()
                self.soln_merged_store.append(soln_merged)
                self.grid_merged_store.append(grid_merged)
                self.t_store.append(self.t)
                i = 0
            
            # if count == 1:
            #     count = 0

        

        return [np.array(self.t_store), self.solution_store, self.grid_merged_store, self.soln_merged_store]
    
        # return [np.array(self.t_store), self.solution_store]
    
        # return [np.array(self.t_store), self.grid_merged_store, self.soln_merged_store]
        


    ####################################################################
    # Plotting Functions
    ####################################################################


    def plotter(self, t_store, solution_store, plot_index):

        colours = ['g','r','b']

        for level, patches in enumerate(solution_store[plot_index]):

            if level == 0: # Skip shadow grid. Must be a better way to do this... (i.e. shouldn't need to save the shadow grid data at all...)
                continue

            for patch in patches:

                soln = patch['soln'][0]
                coordinates = patch['coordinates']
                # bnds = patch['boundaries']

                ### Remove ghost points
                soln_phys = soln[2:-2] # Fix this so it works for arbitrary ghost points
                coord_phys = coordinates[2:-2]


                plt.plot(coord_phys,soln_phys,label='Level ' + str(level), color = colours[level],marker='x',markersize=1.5)

    
        
        plt.legend()
        plt.title("AMR Solution at t = " + str(t_store[plot_index]))
        plt.ylim([-1.5,1.5])
        plt.show()



    def animate(self, t_store, solution_store):

        colours = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:pink','black']

        for i in range(np.shape(t_store)[0]):

            for level, patches in enumerate(solution_store[i]):


                for patch in patches:

                    soln = patch['soln'][0]
                    coordinates = patch['coordinates']
                    # bnds = patch['boundaries']

                    ### Remove ghost points
                    soln_phys = soln[2:-2] # Fix this so it works for arbitrary ghost points
                    coord_phys = coordinates[2:-2]


                    plt.plot(coord_phys,soln_phys,label='Level ' + str(level+1), color = colours[level],markersize=1.0)
                    # plt.plot(coord_phys,soln_phys, color = colours[level-1],marker='.',markersize=1.0)
                    # plt.plot(coord_phys,soln_phys,label='Level ' + str(level),marker='.',markersize=1.5, linestyle='none')
                    
                    plt.title("t = " +str(t_store[i]))

        
            
            plt.legend()
            # plt.ylim([0.8,1.2])
            # plt.ylim([-0.4,0.4])
            plt.draw()
            plt.pause(0.01)
            plt.cla()








