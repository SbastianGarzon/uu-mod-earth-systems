#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Stoke's flow test with constant visc, T and vertical density contrast
"""
import numpy as np

from solver.dataStructures import Markers, Grid, Materials, ViscBox
from solver.physics.boundaryConditions import BCs
from models.common import uniformGrid

from numba import float64, int64, typeof
from numba.types import unicode_type
from numba.experimental import jitclass

def initializeModel():
    '''
    Sets up the initial state of the model, including BCs and output settings.

    Returns
    -------
    params : Parameters object
        Model physical and numerical parameters.
    grid : Grid object
        Initialised Grid object.
    materials : Materials
        Materials object initialsed with required material properties.
    markers : Markers
        Initialized markers object.
    BC : BCs Class
        Object containing all boundary condition arrays for velocity, pressure and temperature 

    '''
    
    # instantiate a pre-populated parameters object
    params = Parameters()

    # set resolution
    xnum = 161
    ynum = 241

    # instantiate/load material properties object
    matData = np.loadtxt('./material_properties.txt', delimiter=",")
    materials = Materials(matData)

    ###########################################################################    
    # Boundary conditions
    
    BC = BCs(xnum, ynum)
    
    # pressure BCs
    BC.P_first[1] = 1e5

    # velocity BCs
    BC.set_top_BC("free slip")
    BC.set_bottom_BC("no slip")
    BC.set_left_BC("free slip")
    BC.set_right_BC("free slip")

    
    # temperature BCs
    # upper and lower  - insulating
    BC.set_top_T_BC("insulating")
    BC.set_bottom_T_BC("insulating")

    # left and right = insulating
    BC.set_left_T_BC("insulating")
    BC.set_right_T_BC("insulating")

    ###########################################################################
    # create grid object
    grid = Grid(xnum, ynum)
    
    # define grid points for evenly spaced grid
    uniformGrid(params, grid)

    ############################################################################
    # create markers object
    mnumx = 400
    mnumy = 1200
    markers = Markers(mnumx, mnumy)

    # initialize markers
    initialize_markers(markers, materials, params)
    
    return params, grid, materials, markers, BC
           
           

def initialize_markers(markers, materials, params):
    '''
    Initialize the positions, material ID and temperature of the markers.

    Parameters
    ----------
    markers : Markers Object
        An empty markers object, which will be filled by this function.
    materials : Materials object
        Materials object filled from file with the required materials.
    params : Parameters Object
        Contains the simulation parameters

    Returns
    -------
    None.

    '''

    np.random.seed(1337)

    mxstp = params.xsize / markers.xnum
    mystp = params.ysize / markers.ynum

    mm = 0
    uniform_temp = 283.15

    # material IDs
    MAT_AIR = 0
    MAT_WATER = 1
    MAT_ROCK = 2

    # basin geometry
    border_lake_left = 1000
    border_lake_right = 100
    slope_zone_width = 1000
    air_metres = 5
    depth_lake = 50

    # free-surface setup
    base_water_level = 7.0

    # perturbation
    pert_amp    = 5.0      # height of displacement (m)
    pert_center = 5000.0   # centre of disturbance (m)
    zone_width  = 2000.0   # total width of disturbance (m) — zero outside this

    for j in range(markers.xnum):
        for i in range(markers.ynum):
            markers.x[mm] = (j + np.random.random()) * mxstp
            markers.y[mm] = (i + np.random.random()) * mystp

            x = markers.x[mm]
            y = markers.y[mm]

            markers.id[mm] = MAT_AIR
            markers.T[mm] = uniform_temp

            # rock margins
            if (x <= border_lake_left and y > air_metres) or (x >= (params.xsize - border_lake_right) and y > air_metres):
                markers.id[mm] = MAT_ROCK

            # left slope
            if (x > border_lake_left) and (x < border_lake_left + slope_zone_width):
                lake_floor = air_metres + depth_lake * ((x - border_lake_left) / slope_zone_width)
                if y >= lake_floor:
                    markers.id[mm] = MAT_ROCK

            # deep basin
            if (x > border_lake_left + slope_zone_width) and (x < params.xsize - border_lake_right - slope_zone_width):
                lake_floor = depth_lake + air_metres
                if y >= lake_floor:
                    markers.id[mm] = MAT_ROCK

            # right slope
            if (x > params.xsize - border_lake_right - slope_zone_width) and (x < params.xsize - border_lake_right):
                lake_floor = depth_lake + air_metres - (
                    depth_lake * (x - (params.xsize - border_lake_right - slope_zone_width)) / slope_zone_width
                )
                if y >= lake_floor:
                    markers.id[mm] = MAT_ROCK

            # perturbed initial water surface — sine taper (zero outside zone_width)
            dist = (x - pert_center) / (zone_width / 2)
            if abs(dist) < 1.0:
                water_level = base_water_level - pert_amp * np.sin(np.pi * dist)
            else:
                water_level = base_water_level

            if y > water_level and markers.id[mm] != MAT_ROCK:
                markers.id[mm] = MAT_WATER

            mm += 1
     

    
###############################################################################
# parameters
spec_par = [
    ('gx', float64),
    ('gy', float64),
    ('Rgas', float64),
    ('xsize', float64),
    ('ysize', float64),
    ('T_min', float64),
    ('eta_min', float64),
    ('eta_max', float64),
    ('stress_min', float64),
    ('eta_wt', float64),
    ('max_pow_law', float64),
    ('t_end', float64),
    ('ntstp_max', int64),
    ('Temp_stp_max', int64),
    ('tstp_max', float64),
    ('marker_max', float64),
    ('marker_sch', int64),
    ('movemode', int64),
    ('dsubgrid', float64),
    ('dsubgridT', float64),
    ('frict_yn', float64),
    ('adia_yn', float64),
    ('save_output', int64),
    ('save_fig', int64),
    ('output_name', unicode_type),
    ('output_path', unicode_type),
    ('viscbox', typeof(ViscBox(0)))
]
@jitclass(spec_par)
class Parameters():
    '''
    Class which holds the values of various physical and numerical parameters
    required in the simulation.
    
    Attributes
    ----------
    gx : FLOAT
        x-direction component of gravitational acceleration.
    gy : FLOAT
        y-direction component of gravitational acceleration.
    Rgas : FLOAT
        Ideal gas constant.
    xsize : FLOAT
        physical x-size of the grid.
    ysize : FLOAT
        physical y-size of the grid.
    T_min : FLOAT
        Minimum allowed temperature.
    eta_min : FLOAT
        Minimum allowed viscosity.
    eta_max : FLOAT
        Maximum allowed viscosity.
    eta_wt : FLOAT
        Weighting value for a (potentially not in use) visco-plastic model.
    max_pow_law : FLOAT
        Maximum allowed exponent in the power law viscosity model.
    t_end : FLOAT
        Time at which to end the simulation (if number of tsteps is less than ntstp_max).
    ntstp_max : INT
        Maximum number of timesteps to take.
    Temp_stp_max : INT
        Maximum number of temperature sub-timesteps to take.
    tstp_max : FLOAT
        Maximum size of timestep.
    marker_max : FLOAT
        Maximum fraction of average grid cell that a marker can move per timestep.
    marker_sch : INT
        Choice of advection scheme for markers, 1 = Euler, 4 = RK4
    movemode : INT
        Choice of how velocities are calculated, 0 = Stokes, no others implemented at present.
    dsubgrid : FLOAT
        Subgrid stress coefficient.
    dsubgridT : FLOAT
        Subgrid temperature diffusion coefficient.
    frict_yn : FLOAT
        Flag to apply friction heating.
    adia_yn : FLOAT
        Flag to apply adiabatic heating.
    save_output : INT
        Number of steps between output, not currently implemented.
    save_fig : INT
        Number of steps between plotting of figures.
    output_name : STR
        The name of the folder to write the output/figures to.  This will be located
        in models/{chosen_model}/figures/output_name.
    output_path : STR
        The output path where the result should be written to specified relative to the run.py file's location.
    viscbox : ViscBox Class
        Object containing parameters for controlling the optional high viscosity box
    
    '''
    
    
    # creates parameters object
    def __init__(self):
        '''
        Constructor for the parameters class

        Returns
        -------
        None.

        '''
        # main parameters, required by all simulations
        
        # physical constants
        self.gx = 0.0                           # x-direction gravitational acc
        self.gy = 9.81                          # y-direction gravitational acc
        self.Rgas = 8.314                       # gas constant
        
        # physical model setup
        self.xsize = 10000.0                        # physical x-size of model, m
        self.ysize = 60                      # physical y-size of model, m
        
        self.T_min = 273                        # Minimum allowed temperature in the simulation
        
        # viscosity model
        self.eta_min = 1e2                      # minimum viscosity
        self.eta_max = 1e25                     # maximum viscosity
        self.stress_min = 1e4                   # minimum stress
        self.eta_wt = 0                         # viscosity weighting, for (old?) visco-plastic model
        self.max_pow_law = 150                  # maximum power law exponent in visc model
        
        
        # timestepping
        self.t_end = 100000                     # end time
        self.ntstp_max = 600                   # maximum number of timesteps
        self.Temp_stp_max = 1                  # maximum number of temperature substeps
        
        self.tstp_max = 10      # maximum timestep
        
        # marker options
        self.marker_max = 0.3                   # maximum marker movement per timestep (fraction of av. grid step)
        self.marker_sch = 4                     # marker scheme 0 = no movement, 1 = Euler, 4=RK4
        
        self.movemode = 0                       # velocity calculation 0 = momentum eqn, 1 = solid body (not working currently)
        
        # subgrid diffusion
        self.dsubgrid = 1                       # subgrid stress coeff (none if zero)
        self.dsubgridT = 1                      # subgrid diffusion coeff(none if zero)
        
        # switches for heating terms
        self.frict_yn = 0                      # use friction heating?
        self.adia_yn = 0                       # use adiabatic heating?
        
        
        # output options
        self.save_output = 50                   # number of steps between output files
        self.save_fig = 5                      # number of steps between figure output
        self.output_name = "lakeSeiche_sine"         # name of the folder to write data to (within the main figures directory)
        self.output_path = "../../Results/figures"
        
        self.viscbox = ViscBox(0)               # high viscosity box, switched off


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # parameters from initialize_markers
    xsize          = 10000.0
    base_water_level = 7.0
    pert_amp       = 5.0
    pert_center    = 5000.0
    pert_width     = 500.0
    pert_offset    = 500.0

    x = np.linspace(0, xsize, 1000)
    left_lobe  = np.exp(-((x - (pert_center - pert_offset))**2) / (2.0 * pert_width**2))
    right_lobe = np.exp(-((x - (pert_center + pert_offset))**2) / (2.0 * pert_width**2))
    water_level = base_water_level + pert_amp * (left_lobe - right_lobe)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, water_level)
    ax.axhline(base_water_level, color='grey', linestyle='--', label='undisturbed level (%.1f m)' % base_water_level)
    ax.set(xlabel='x (m)', ylabel='Water surface height (m)', title='Initial water surface perturbation')
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    plt.show()
