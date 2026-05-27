#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for running a model.
Usage:
    python run.py [perturbation_type]

    perturbation_type : 'ricker' (default) or 'ellipse'
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
from visualisation import makePlots, makeLithologyPlot, makeMarkerMaterialPlot
from plot_hovmoller import plot as plot_hovmoller
from plot_surface_animation import plot as plot_surface_animation


###############################################################################
# step 0 : start timer (for tracking performance) and set debug flags
###############################################################################
strt = time()

# if debugging, this should be 1 AND jitclass tags in solver/dataStructures and Parameters must be commented out!
os.environ["NUMBA_DISABLE_JIT"] = "0"
# if 1 prints out extra statements at various places in the timeloop
debug = 1

# parse perturbation type from command line (default: 'ricker')
perturbation_type = sys.argv[1] if len(sys.argv) > 1 else 'ricker'
print('Perturbation type: %s' % perturbation_type)


###############################################################################
# step 1 : initialize the model run
###############################################################################
params, grid, materials, markers, BC = initializeModel(perturbation_type)

# initialize grid0 for old values
grid0 = Grid(grid.xnum, grid.ynum)

# initialize timesteping
time_curr = 0
timestep = params.tstp_max

# if figures directory doesn't already exist, add it
if (os.path.exists(f"{params.output_path}/{params.output_name}")==False):
    os.makedirs(f"{params.output_path}/{params.output_name}")

MAT_SURFACE = 3
buoy_xs = np.arange(200, 5801, 100, dtype=float)  # flat basin only, avoid margins and slopes
buoy_times = []
buoy_surfaces = [[] for _ in buoy_xs]


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
        #makeLithologyPlot(grid, markers, params, nt, time_curr)
        makeMarkerMaterialPlot(markers, params, nt, time_curr)


    ###########################################################################
    # record buoys: find the closest MAT_SURFACE marker to each buoy x position
    mx  = np.asarray(markers.x[:markers.num])
    my  = np.asarray(markers.y[:markers.num])
    mid = np.asarray(markers.id[:markers.num])
    surf_mx = mx[mid == MAT_SURFACE]
    surf_my = my[mid == MAT_SURFACE]
    band = 50.0  # m either side of buoy x
    for b, bx in enumerate(buoy_xs):
        in_band = np.abs(surf_mx - bx) <= band
        if in_band.any():
            buoy_surfaces[b].append(float(np.mean(surf_my[in_band])))
        elif buoy_surfaces[b]:
            buoy_surfaces[b].append(buoy_surfaces[b][-1])
        else:
            buoy_surfaces[b].append(float('nan'))
    buoy_times.append(time_curr)
    time_curr += timestep
    print('Time: %.3f s' % time_curr)


    ###########################################################################
    # exit if final time is reached
    if (time_curr >= params.t_end):
        # we have reached the max time specified, exit loop
        print('t_end reached, exiting loop')
        break

elapsed = time() - strt
print('time elapsed: %f' % elapsed)

###############################################################################
# save buoy data to CSV
###############################################################################
csv_path = '%s/%s/buoy.csv' % (params.output_path, params.output_name)
header = 'time,' + ','.join('x%.0f' % bx for bx in buoy_xs)
data = np.column_stack([buoy_times] + buoy_surfaces)
np.savetxt(csv_path, data, delimiter=',', header=header, comments='')

###############################################################################
# save timing summary
###############################################################################
timing_path = '%s/%s/timing.txt' % (params.output_path, params.output_name)
with open(timing_path, 'w') as f:
    f.write('perturbation_type: %s\n' % perturbation_type)
    f.write('tstp_max:          %.4f s\n' % params.tstp_max)
    f.write('ntstp_max:         %d\n' % params.ntstp_max)
    f.write('steps_run:         %d\n' % (nt + 1))
    f.write('t_end_reached:     %.3f s\n' % time_curr)
    f.write('elapsed:           %.2f s\n' % elapsed)
print('Timing saved to', timing_path)

###############################################################################
# post-processing plots
###############################################################################
print('Generating Hovmoller diagram...')
plot_hovmoller(csv_path,
               '%s/%s/hovmoller.png' % (params.output_path, params.output_name))

print('Generating surface animation...')
plot_surface_animation(csv_path,
                       '%s/%s/surface_animation.gif' % (params.output_path, params.output_name))
