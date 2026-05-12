#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation routines for the lakeSeiche model.
"""

import numpy as np
from matplotlib import figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from output.visualisation import getMarkerPixelGrid, getMarkerField


def _basicGridVelocities(grid):
    """Interpolate staggered vx/vy to basic nodes for plotting."""
    xnum = grid.xnum
    ynum = grid.ynum
    vxb = np.zeros((ynum, xnum))
    vyb = np.zeros((ynum, xnum))
    for i in range(ynum):
        for j in range(xnum):
            vxb[i, j] = (grid.vx[i, j] + grid.vx[i+1, j]) / 2
            vyb[i, j] = (grid.vy[i, j] + grid.vy[i, j+1]) / 2
    return vxb, vyb


def makePlots(grid, markers, params, ntstp, t_curr):
    """
    Plot density, pressure, temperature, Vx, Vy and viscosity
    in a 2x3 grid with a fixed figure size (no aspect-ratio scaling).
    """

    xnum = grid.xnum
    ynum = grid.ynum

    X,  Y  = np.meshgrid(grid.x, grid.y)
    XP, YP = np.meshgrid(grid.cx[:xnum-1], grid.cy[:ynum-1])

    vxb, vyb = _basicGridVelocities(grid)

    fig = figure.Figure(figsize=(18, 12), constrained_layout=True)
    axs = fig.subplots(2, 3, sharex=True, sharey=True)

    # --- Density ---
    im = axs[0, 0].pcolor(X, Y, grid.rho, shading='nearest', vmin=0, vmax=3300)
    fig.colorbar(im, ax=axs[0, 0], pad=0.0)
    axs[0, 0].set_title('Density (kg/m3)')
    axs[0, 0].set(ylabel='y (m)')
    axs[0, 0].quiver(grid.x, grid.y, vxb[:ynum, :], np.flip(-vyb[:, :xnum], 0))
    axs[0, 0].invert_yaxis()

    # --- Pressure ---
    im = axs[0, 1].pcolor(XP, YP, grid.P, shading='nearest', vmin=0, vmax=1e6)
    fig.colorbar(im, ax=axs[0, 1], pad=0.0)
    axs[0, 1].set_title('Pressure (Pa)')

    # --- Temperature ---
    im = axs[0, 2].pcolor(X, Y, grid.T, shading='nearest', vmin=273, vmax=300)
    fig.colorbar(im, ax=axs[0, 2], pad=0.0)
    axs[0, 2].set_title('Temperature (K)')

    # --- Vx ---
    im = axs[1, 0].pcolor(X, Y, vxb, shading='nearest')
    fig.colorbar(im, ax=axs[1, 0], pad=0.0)
    axs[1, 0].set_title('Vx (m/s)')
    axs[1, 0].set(ylabel='y (m)', xlabel='x (m)')

    # --- Vy ---
    im = axs[1, 1].pcolor(X, Y, vyb, shading='nearest')
    fig.colorbar(im, ax=axs[1, 1], pad=0.0)
    axs[1, 1].set_title('Vy (m/s)')
    axs[1, 1].set(xlabel='x (m)')

    # --- Viscosity ---
    im = axs[1, 2].pcolor(X, Y, np.log10(grid.eta_s), shading='nearest', vmin=5, vmax=25)
    fig.colorbar(im, ax=axs[1, 2], pad=0.0)
    axs[1, 2].set_title('Viscosity log10(Pa s)')
    axs[1, 2].set(xlabel='x (m)')
    #axs[0,0].set_ylim(15,0)

    fig.suptitle('Time: %.3f s' % t_curr)
    fig.savefig('%s/%s/summary_%i.png' % (params.output_path, params.output_name, ntstp))


def makeLithologyPlot(grid, markers, params, ntstp, t_curr):
    """
    Plot lithology (material ID) from markers using a discrete 3-colour colormap.
    Air = light blue, Water = blue, Rock = brown.
    """
    # map markers to pixel grid
    marker_map = getMarkerPixelGrid(params, markers, grid, 401)
    mark_ids   = getMarkerField(marker_map, markers.id)

    # discrete colormap: 0=air, 1=water, 2=rock
    cmap = ListedColormap(['#cce5ff', '#1a6faf', '#8B6347'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig = figure.Figure(figsize=(18, 4), constrained_layout=True)
    ax  = fig.subplots(1, 1)

    im = ax.imshow(mark_ids, origin='upper', aspect='auto', cmap=cmap, norm=norm,
                   extent=[0, params.xsize, params.ysize, 0])

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], pad=0.0)
    cbar.ax.set_yticklabels(['Air', 'Water', 'Rock'])

    ax.set(xlabel='x (m)', ylabel='y (m)', title='Lithology')
    fig.suptitle('Time: %.3f s' % t_curr)
    fig.savefig('%s/%s/litho_%i.png' % (params.output_path, params.output_name, ntstp))
