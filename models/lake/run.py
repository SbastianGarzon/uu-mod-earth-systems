#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example script for running a model

"""

# external library imports
from time import time
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# internal imports
sys.path.append("../../") # required so that we can find the rest of the code from here!
from solver.dataStructures import Grid
from solver.main import step
from setup import initializeModel
from visualisation import makePlots, makeLithologyPlot


###############################################################################
# step 0 : start timer (for tracking performance) and set debug flags
###############################################################################
strt = time()

# if debugging, this should be 1 AND jitclass tags in solver/dataStructures and Parameters must be commented out!
os.environ["NUMBA_DISABLE_JIT"] = "0"
# if 1 prints out extra statements at various places in the timeloop
debug = 1


###############################################################################
# step 1 : initialize the model run
###############################################################################
params, grid, materials, markers, BC = initializeModel()

# initialize grid0 for old values
grid0 = Grid(grid.xnum, grid.ynum)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max

# if figures directory doesn't already exist, add it
if (os.path.exists(f"{params.output_path}/{params.output_name}")==False):
    os.makedirs(f"{params.output_path}/{params.output_name}")

# buoys every 500m, skipping rock margins (x < 1500m and x > 8500m)
dx = params.xsize / (grid.xnum - 1)
buoy_xs = np.arange(1500, 9000, 500)          # [1500, 2000, ..., 8500]
buoy_js = [int(round(x / dx)) for x in buoy_xs]
buoy_times = []
buoy_surfaces = [[] for _ in buoy_js]


###############################################################################
# step 2: time loop
###############################################################################
for nt in range(0, params.ntstp_max):
    
    ###########################################################################
    # do a timestep
    step(params, grid, materials, markers, BC, timestep, nt, grid0, debug)


    ###########################################################################
    # visualization
    if (nt%(params.save_fig)==0):
        print('plotting')
        
        # wrapper for calling whatever custom plots are defined in setup.py
        makePlots(grid, markers, params, nt, time_curr)
        makeLithologyPlot(grid, markers, params, nt, time_curr)
        

    ###########################################################################
    # record buoys: find shallowest water cell at each buoy position
    for b, j in enumerate(buoy_js):
        for i in range(grid.ynum):
            if grid.rho[i, j] > 500:
                buoy_surfaces[b].append(grid.y[i])
                break
    buoy_times.append(time_curr)

    ###########################################################################
    # advance timestep

    time_curr += timestep
    print('Time: %.3f s' % time_curr)

    
    ###########################################################################
    # Make any model-specific adjustments 
    # eg. for a moving grid, update the grid positions
    
        
    ###########################################################################
    # exit if final time is reached
    if (time_curr >= params.t_end):
        # we have reached the max time specified, exit loop
        print('t_end reached, exiting loop')
        break

end = time() - strt
print('time elapsed: %f'%(end))

# save buoy data to CSV
header = 'time,' + ','.join('x%.0f' % grid.x[j] for j in buoy_js)
data = np.column_stack([buoy_times] + buoy_surfaces)
np.savetxt('%s/%s/buoy.csv' % (params.output_path, params.output_name),
           data, delimiter=',', header=header, comments='')

# Hovmoller diagram: time on x-axis, buoy position on y-axis, surface height as colour
buoy_matrix = np.array(buoy_surfaces)   # shape (n_buoys, n_times)
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.pcolor(buoy_times, buoy_xs, buoy_matrix, shading='nearest', cmap='RdBu_r')
fig.colorbar(im, ax=ax, label='Water surface height (m)')
ax.set(xlabel='Time (s)', ylabel='x position (m)', title='Buoy records — Hovmöller diagram')
fig.tight_layout()
fig.savefig('%s/%s/buoy.png' % (params.output_path, params.output_name)) 
