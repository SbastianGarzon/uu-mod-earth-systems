#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common visualisation routines 

"""

from matplotlib import figure
import numpy as np
from numba import jit

from solver.physics.grid_fns import basicGridVelocities

###############################################################################
# fns for plotting from the Grid

def plotTemperature(grid, params, ntstp, t_curr, xlims, ylims, aspect_ratio=None, height=None, plot_vel_arrows=False, arrow_stp = 5):
    '''
    Plot a single variable as a colormap, here it is the temperature but this serves as 
    a template for making simple plots of simulation output. It includes the option to
    annotate the field with arrows showing velocity.

    Parameters
    ----------
    grid : Grid object
        Grid containing all simulation variables.
    params : Parameters object
        Simulation's parameters object.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).
    aspect_ratio : FLOAT (OPTIONAL)
        Option to manually set the aspect ratio of the plots.
    height : FLOAT (OPTIONAL)
        Option to manually set height of an axis.
    vel_arrows : BOOL (OPTIONAL)
        Flag indicating if velocity arrows should be added to the plot.  Defualt is False.
    arrow_stp : INT (OPTIONAL)
        Number of grid points between each velocity arrow. Default is 5.

    Returns
    -------
    None.

    '''
    
    X, Y = np.meshgrid(grid.x, grid.y)
    
    # get aspect ratio
    xlen = abs(xlims[1] - xlims[0])
    ylen = abs(ylims[1] - ylims[0])
    
    if aspect_ratio == None:
        asp_rat = xlen/ylen
    else:
        asp_rat = aspect_ratio
    
    # set ysize of each figure
    if height==None:
        y_fig = 3
    else:
        y_fig=height
        
    if asp_rat <= 1:
        # add some padding bc. of axis/cb taking up space in x
        pad = 2.0
    else:
        pad = 0.0
    
    # number of plots vertically
    y_plots = 1
    
    # create figure
    fig = figure.Figure(figsize=(asp_rat*y_fig + pad, y_fig*y_plots), constrained_layout=True)
    axs = fig.subplots(1,1, sharex=True, sharey=True)

    # plot the temperature as colormap
    im = axs.pcolor(X, Y, grid.T-273, shading='nearest', vmin=0, vmax=1400)
    
    # if flag is true, plot the velocity arrows
    if plot_vel_arrows:
        vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)
        axs.quiver(X[::arrow_stp, ::arrow_stp], Y[::arrow_stp, ::arrow_stp],\
                   vxb[:grid.ynum,:][::arrow_stp, ::arrow_stp], np.flip(-vyb[:,:grid.xnum],0)[::arrow_stp, ::arrow_stp])	
        
    # set limits, titles etc.
    fig.colorbar(im, ax=axs,pad=0.0)                                             # display colorbar
    axs.set_title('Temperature (C)')                                             # set plot title
    axs.set(ylabel='y (m)', xlabel='x (m)', xlim=xlims, ylim=ylims)  # label the y-axis and x-axis
    #axs.invert_yaxis() 
    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    
    # save to file 
    fig.savefig('%s/%s/temp_%i.png'%(params.output_path, params.output_name, ntstp))


def plotSummary(grid, params, ntstp, t_curr, xlims, ylims, aspect_ratio=None, plotTempContours=False, temp_levels=None):
    '''
    Plots several simulation variables in a grid, here we use density, viscosity
    and pressure with velocity arrows on the density plot and optional temperature contours.

    Parameters
    ----------
    grid : Grid object
        Grid containing all simulation variables.
    params : Parameters object
        Simulation's parameters object.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).
    aspect_ratio : FLOAT (OPTIONAL)
        Option to manually set the aspect ratio of the plots.
    plotTempContours : BOOL (OPTIONAL)
        switch to say if temperature contours should be added to plots
    temp_levels : LIST (OPTIONAL)
        If plotTempContours == True, this list specifies the temperature values
        at which to plot contours.

    Returns
    -------
    None.

    '''
    
    X, Y = np.meshgrid(grid.x, grid.y)
    
    # get aspect ratio
    xlen = abs(xlims[1] - xlims[0])
    ylen = abs(ylims[1] - ylims[0])
    
    if aspect_ratio == None:
        asp_rat = xlen/ylen
    else:
        asp_rat = aspect_ratio
    
    if asp_rat <= 1:
        # add some padding bc. of axis/cb taking up space in x
        pad = 2.0
    else:
        pad = 0.0
    
    
    # set ysize of each figure
    y_fig = 6
    
    # number of plots vertically
    y_plots = 3
    
    # create figure
    fig = figure.Figure(figsize=(asp_rat*y_fig + pad, y_fig*y_plots), constrained_layout=True)
    axs = fig.subplots(3,1, sharex=True, sharey=True)
    
    # check that if temp contours are switched on, values have been provided
    if plotTempContours and temp_levels==None:
        raise ValueError("Plot temperature contours was set to true but no contour values were provided.  Please set temp_levels to a list of temperaure values at which contours should be plotted")
    

    ###########################################################################
    # density
    im = axs[0].pcolor(X, Y, grid.rho, shading='nearest', vmin=2200, vmax=3500)
    fig.colorbar(im, ax=axs[0],pad=0.0)        # display colorbar
    axs[0].set_title('Density (kg/m3) ')       # set plot title
    axs[0].set(ylabel='y (m)', xlim=xlims, ylim=ylims)                 # label the y-axis (shared axis for x)
    
    # add velocity arrows, not at every cell, step sets the spacing
    arrow_stp = 5
    vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)
    axs[0].quiver(X[::arrow_stp, ::arrow_stp], Y[::arrow_stp, ::arrow_stp],\
                   vxb[:grid.ynum,:][::arrow_stp, ::arrow_stp], np.flip(-vyb[:,:grid.xnum],0)[::arrow_stp, ::arrow_stp])	    
    
    # Add temperature contours
    if plotTempContours:
        cs = axs[0].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
        axs[0].clabel(cs, inline=True, fontsize=8, fmt='%d C')
    
    # plot increasing depth downward! (only needed for the annotations)
    axs[0].invert_yaxis()

    ###########################################################################
    # Viscosity
    im = axs[1].pcolor(X, Y, np.log10(grid.eta_n),vmin=18, vmax=28)
    fig.colorbar(im, ax=axs[1],pad=0.0)                 # display colorbar
    axs[1].set(ylabel='y (m)', xlim=xlims, ylim=ylims)                          # label the y-axis (shared axis for x)
    axs[1].set_title('Viscosity log10(Pa s)')           # set plot title
    
    # Add temperature contours
    if plotTempContours:
        cs = axs[1].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
        axs[1].clabel(cs, inline=True, fontsize=8, fmt='%d C')

    ###########################################################################
	# Pressure
    im = axs[2].pcolor(X, Y, grid.P, shading='flat',vmin=0.1e9,vmax=9e9)
    fig.colorbar(im, ax=axs[2],pad=0.0)                 # display colorbar
    axs[2].set(ylabel='y (m)', xlabel='x (m)', xlim=xlims, ylim=ylims)          # label the x and y-axis
    axs[2].set_title('Pressure (Pa)')                   # set plot title
    
    # Add temperature contours
    if plotTempContours:
        cs = axs[2].contour(X, Y, grid.T-273, levels=temp_levels, colors='w', linewidths=0.8)
        axs[2].clabel(cs, inline=True, fontsize=8, fmt='%d C')


    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('%s/%s/summary_%i.png'%(params.output_path, params.output_name, ntstp))
    

###############################################################################
# fns for plotting from markers

@jit(nopython=True)
def getMarkerPixelGrid(params, markers, grid, xres):
    '''
    maps the markers to a pixel grid, where each pixel records it's nearest marker's number.
    These marker indexes can then be used to plot a specific field from the markers.

    Parameters
    ----------
    params : Parameters object
        Simulation's parameters object.
    markers : Markers object
        Contains all the marker values for each variable.
    grid : Grid object
        Contains all the grid variables at the current time.
    xres : INT
        x-direction resolution of the pixel grid, y will be set from this to match
        physical size aspect ratio.

    Returns
    -------
    mark_nums : ARRAY
        Pixel grid of values of the nearest marker's index to that pixel.

    '''
    
    # define the image resolution - set by xres
    # but proportional to the grid size in each direction
    yres = max(int(params.ysize/params.xsize*xres) + 1, 300) #newcode
    
    
    ngrid = 2
    
    sxstp = params.xsize/(xres - 1)
    systp = params.ysize/(yres - 1)
    
    # create marker visualization arrays
    mark_nums = np.ones((yres, xres))*np.nan
    mark_dis = np.ones((yres, xres))*1e20

    # loop through markers
    for m in range(0,markers.num):
        
        # define pixel cell that this marker is in
        m1 = int((markers.x[m] - grid.x[0])/sxstp)
        m2 = int((markers.y[m] - grid.y[0])/systp)
        
        if (m1<0):
            m1 = 0
        elif (m1>xres-2):
            m1 = xres-2
 
        if (m2<0):
            m2 = 0
        elif (m2>yres-2):
            m2 = yres-2
        
        # get surrounding indicies
        m1min = m1-ngrid
        if (m1min<0):
            m1min = 0
        
        m1max = m1 + ngrid + 1
        if (m1max>xres-1):
            m1max = xres - 1
        
        m2min = m2-ngrid
        if (m2min < 0):
            m2min = 0
        
        m2max = m2 + ngrid + 1
        if (m2max>yres-1):
            m2max = yres-1
        
        # update value of all pixels around marker
        for m10 in range(m1min, m1max):
            for m20 in range(m2min, m2max):
                # check distance to current cell
                dx = (markers.x[m] - grid.x[0]) - m10*sxstp
                dy = (markers.y[m] - grid.y[0]) - m20*systp

                dd = np.sqrt(dx**2 + dy**2)
                
                # if we are closer to this cell than any previous marker, 
                # change to new marker number, so that each cell will record its closest marker
                if (dd<mark_dis[m20, m10]):
                    mark_nums[m20, m10] = m
                    mark_dis[m20, m10] = dd

    return mark_nums


def getMarkerField(mark_nums, markers_field):
    """
    Given the array of marker mappings to pixels and a marker field, returns
    a pixel array of the required field for plotting

    Parameters
    ----------
    mark_nums : ARRAY
        Array of pixels with the index of the nearest marker as the values, 
        produced by markerVisGrid.
    markers_field : ARRAY
        Field from the markers object to be extracted.

    Returns
    -------
    pixel_vals : ARRAY
        Pixel array of the chosen marker field, ready for plotting.

    """
    xres, yres = np.shape(mark_nums)
    pixel_vals = np.zeros((xres,yres))
    
    for i in range(0,xres):
        for j in range(0,yres):
            # set the pixel val to the value from the nearest marker
            if np.isnan(mark_nums[i,j]):
                #TODO: handle case where a pixel is never reached/ figure out why
                pixel_vals[i,j] = np.nan
            else:
                pixel_vals[i,j] = markers_field[int(mark_nums[i,j])]
    
    return pixel_vals



def plotMarkers_lithology(params, markers, grid, ntstp, t_curr, xlims, ylims, aspect_ratio=None, height=None, plot_vel_arrows=False):
    '''
    Plot the lithology/material ID recorded by the markers.

    Parameters
    ----------
    params : Parameters object
        Simulation's parameters object.
    markers : Markers object
        Contains all the marker values for each variable.
    grid : Grid object
        Contains all the grid variables at the current time.
    ntstp : INT
        Current timestep number.
    t_curr : FLOAT
        Current time (s).
    xlims : TUPLE
        Limits of plotting area in the x-direction.
    ylims : TUPLE
        Limits of plotting area in the y-direction.
    aspect_ratio : FLOAT (OPTIONAL)
        Option to manually set the aspect ratio of the plots.
    height : FLOAT (OPTIONAL)
        Option to manually set height of an axis.
    plot_vel_arrows : BOOL (Optional)
        Option to plot velocity arrows on the image, default is False.
    
    Returns
    -------
    None.

    '''

    # get the mapping of markers to pixel positions
    marker_map = getMarkerPixelGrid(params, markers, grid, 401) #newcode
    mark_com = getMarkerField(marker_map, markers.id)

    # get aspect ratio
    xlen = abs(xlims[1] - xlims[0])
    ylen = abs(ylims[1] - ylims[0])
    
    if aspect_ratio == None:
        asp_rat = xlen/ylen
    else:
        asp_rat = aspect_ratio
        
    if asp_rat <= 1:
        # add some padding bc. of axis/cb taking up space in x
        pad = 2.0
    else:
        pad = 0.0
    
    # set ysize of each figure
    if height==None:
        y_fig = 3
    else:
        y_fig=height
    
    # number of plots vertically
    y_plots = 1

    # create figure
    fig = figure.Figure(figsize=(asp_rat*y_fig + pad, y_fig*y_plots), constrained_layout=True)
    axs = fig.subplots(1,1, sharex=True, sharey=True)
    
    # create image grid and temperature contour levels - not in use here?
    X, Y = np.meshgrid(grid.x, grid.y)

    # plot the lithology as colormap
    im = axs.imshow(mark_com, origin='upper', aspect='auto', extent=[xlims[0], xlims[1], ylims[0],ylims[1]])
    fig.colorbar(im, ax=axs,pad=0.0)
    axs.set_title('Lithology') 
    axs.set(ylabel='y (m)', xlabel ='x (m)', xlim=xlims, ylim=ylims)               

    # add velocity arrows, not at every cell, step sets the spacing
    if plot_vel_arrows:
        arrow_stp = 5
        vxb, vyb = basicGridVelocities(grid.vx, grid.vy, grid.xnum, grid.ynum)
        axs.quiver(X[::arrow_stp, ::arrow_stp], Y[::arrow_stp, ::arrow_stp],\
                      vxb[:grid.ynum,:][::arrow_stp, ::arrow_stp], np.flip(-vyb[:,:grid.xnum],0)[::arrow_stp, ::arrow_stp])	            
    
    fig.suptitle('Time: %.3f Myr'%(t_curr*1e-6/(365.25*24*3600)))
    fig.savefig('%s/%s/litho_%i.png'%(params.output_path, params.output_name, ntstp))
    
    
