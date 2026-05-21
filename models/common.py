#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

common functions that can be used by many models

"""

def uniformGrid(params, grid):
    '''
    Calculates the new grid spacings based on the current xsize and ysize.
    
    This default version implements a fixed, uniform grid.

    Parameters
    ----------
    params : Parameters Class
        Parameters object containing all simulation parameters for the system.
    grid : OBJ
        The grid object into which the new node positions will be written.
        
    Returns
    -------
    None.

    '''
    
    
    xnum = grid.xnum
    ynum = grid.ynum
    
    dx = params.xsize/(xnum-1)
    dy = params.ysize/(ynum-1)
    
    # Simple, uniform grid
    # horizontal grid
    grid.x[0] = 0
    for i in range(1,xnum):
        grid.x[i] = grid.x[i-1] + dx
        
    # vertical grid
    grid.y[0] = 0
    for i in range(1,ynum):
        grid.y[i] = grid.y[i-1] + dy


def updateGrid(params, grid, t_curr, timestep, BC_bot):
    '''
    Calculates the new grid point spacings based on the current xsize and ysize.

    Parameters
    ----------
    params : Parameters Class
        Parameters object containing all simulation parameters for the system.
    grid : OBJ
        The grid object into which the new node positions will be written.
    t_curr : FLOAT
        The current simulation time, to determine whether to set up grid from scratch
        or extend an existing one.
    timestep : FLOAT
        The current timestep size.
    BC_bot : ARRAY
        The boundary condition which should also be updated by whatever changes
        are made to the grid, in this case the bottom.

    Returns
    -------
    None.

    '''
    
    if (t_curr > 0 and params.const==1):
        # we don't need to recalculate the grid, return here!
        return
    
    xnum = grid.xnum
    ynum = grid.ynum
    
    # pull out the required parameters from params object
    bx = params.bx
    by = params.by
    Nx = params.Nx
    Ny = params.Ny
    
    non_uni_xsize = params.non_uni_xsize
    non_uni_ysize = params.non_uni_ysize
    N_left = params.N_left
    N_right = params.N_right
    b_end = params.b_end
    Ny_end = params.Ny_end
    Ny_start = params.Ny_start
    by_end = params.by_end
    
    ###############################################################################
    # Horizontal grid
    
    # xsize - region of fixed grid at the end
    xsize_norm = params.xsize - N_right*b_end
    
    # grid point number at which the non-uniform grid ends
    xnum_ad = xnum - N_right

    if (t_curr==0):
        # set the points in the high res area
        # this only needs doing on the initial setup
        grid.x[Nx+N_left] = non_uni_xsize 
        for i in range(Nx+N_left+1,xnum_ad-Nx):  #xnum
            grid.x[i] = grid.x[i-1] + bx
            
        # size of the non-uniform region 
        D = xsize_norm - grid.x[xnum_ad-Nx-1]
        print(D)

    else:
        
        # update grid positions based on extension
        params.ysize += -params.v_ext/params.xsize*params.ysize*timestep
        params.xsize += params.v_ext*timestep
        
        # set the new position of the first node,
        # and the size of the non-uniform region
        grid.x[0] = grid.x[int(xnum/2)] - params.xsize/2
        D = params.xsize/2 - (grid.x[xnum-Nx-1] - grid.x[int(xnum/2)])
        
        # if we have changing grid, we also need to update bottom BC
        if (abs(params.v_ext)>0):
            BC_bot[:,2] = -params.v_ext/params.xsize*params.ysize
    
    # define factor of grid spacing to increase to the right of high res area
    # need to only do this for non-uniform def, otherwise div by 0!
    if (Nx > 0):
        F = 1.1
        # iteratively solve for F
        for i in range(0,200):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)
    
        # define grid points to the right of the high-res region
        for i in range(xnum_ad-Nx, xnum_ad):   #was xnum
            grid.x[i] = grid.x[i-1] + bx*F**(i-(xnum_ad-Nx-1))
            
        # we have a set of fixed resolution points at the upper edge of the grid 
        for i in range(xnum_ad, xnum):
            grid.x[i] = grid.x[i-1] + b_end
        
        if (t_curr==0):
            grid.x[xnum-1] = params.xsize
            grid.x[xnum_ad-1] = xsize_norm
    
        # now do the same going leftward
        #Add regular grid to the left side aswell
        for i in range(1,N_left+1):
            grid.x[i] = grid.x[i-1] + b_end
        
        D = grid.x[Nx+N_left] - grid.x[N_left] # think this should still work for inital case?
        # Print statement to check whether the D is correct estimated
        print(grid.x[Nx+N_left], grid.x[N_left], D)
        
        F = 1.1
        for i in range(0,200):
            F = (1 + D/bx*(1 - 1/F))**(1/Nx)

        # set the points left of the high res region
        for i in range(N_left+1,Nx+N_left):
            grid.x[i] = grid.x[i-1] + bx*F**(Nx+N_left+1-i)
        
        # Print statement to check if there are no holes in the grid
        print(grid.x)
        
    ###########################################################################
    # Vertical grid
    # one-sided, there is high resolution at the top of the grid and then a decreasing region below

    # ysize - region of fixed grid at the end
    ysize_norm = params.ysize - Ny_end*by_end

    # ysize - region of fixed grid at the start
    ysize_norm_top = Ny_start*by_end

    # grid point number at which the non-uniform grid ends
    ynum_ad = ynum - Ny_end

    # set the high resolution area, assumes y[0] = 0
    grid.y[Ny+Ny_start] = non_uni_ysize 
    for i in range(Ny_start+Ny+1,ynum_ad-Ny):
        grid.y[i] = grid.y[i-1] + by
      
    
    if (Ny > 0):
        # size of the non-uniform region at the bottom
        D = ysize_norm - grid.y[ynum_ad-Ny-1]
       
        # solve iteratively for scaling factor
        F = 1.1
        for i in range(0,100):
            F = (1 + D/by*(1 - 1/F))**(1/Ny)
            # set the grid points below the high-res region
        for i in range(ynum_ad-Ny, ynum_ad):
            grid.y[i] = grid.y[i-1] + by*F**(i-(ynum_ad-Ny-1))
        #we have a set of fixed resolution points between 200e3 and 400e3 m
        for i in range(ynum_ad, ynum):
            grid.y[i] = grid.y[i-1] + by_end

        # size of the non-unifortm region at the top
        D = grid.y[Ny+Ny_start] - ysize_norm_top

        # solve iteratively for scaling factor
        F = 1.1
        for i in range(0,100):
            F = (1+D/by*(1-1/F))**(1/Ny)
        for i in range(1, Ny_start+1):
            grid.y[i] = grid.y[i-1]+by_end
        for i in range(Ny_start+1, Ny_start+Ny):
            grid.y[i] = grid.y[i-1]+by*F**(i-(Ny_start-1))
        
        print(grid.y)
        
        # fix the end position if this is the first step
        if (t_curr==0):
            grid.y[ynum-1] = params.ysize
            grid.y[ynum_ad-1] = ysize_norm
