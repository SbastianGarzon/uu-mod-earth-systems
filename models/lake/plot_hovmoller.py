#!/usr/bin/env python3
"""
Hovmoller diagram from saved buoy.csv data.
Can be imported and called from run.py, or run standalone from the lakeV2 directory.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot(csv_file, output_file):
    data = np.genfromtxt(csv_file, delimiter=',', names=True)

    times      = data['time']
    buoy_names = [name for name in data.dtype.names if name != 'time']
    buoy_xs    = np.array([float(name[1:]) for name in buoy_names])
    buoy_matrix = np.array([data[name] for name in buoy_names])  # (n_buoys, n_times)

    center = 17.0
    step   = 0.25
    half   = np.ceil(max(center - np.nanmin(buoy_matrix),
                         np.nanmax(buoy_matrix) - center) / step) * step
    vmin   = center - half
    vmax   = center + half
    levels = np.arange(vmin - step/2, vmax + step, step)
    norm   = mcolors.BoundaryNorm(levels, ncolors=plt.cm.RdBu_r.N)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.pcolormesh(times, buoy_xs, buoy_matrix, cmap='RdBu_r', norm=norm, shading='nearest')

    cbar = fig.colorbar(im, ax=ax, label='Water surface height (m)',
                        ticks=np.arange(vmin, vmax + step, step))
    cbar.ax.set_yticklabels(['%.2f m' % v for v in np.arange(vmin, vmax + step, step)])

    # theoretical wave speed lines from perturbation centre
    c     = np.sqrt(9.81 * 40)   # shallow-water wave speed (m/s), h = depth_lake = 40 m
    x0    = 3000.0                # perturbation centre (m)
    t_arr = np.array([times[0], times[-1]])
    ax.plot(t_arr, x0 + c * t_arr, color='black', linewidth=1.5, linestyle='--', label=f'c = {c:.1f} m/s')
    ax.plot(t_arr, x0 - c * t_arr, color='black', linewidth=1.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=9)

    ax.set(xlabel='Time (s)', ylabel='x position (m)', title='Buoy records — Hovmöller diagram')
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    print('Saved to', output_file)


if __name__ == '__main__':
    csv_file    = '../../Results/figures/lake_tstpmax_0_01/buoy.csv'
    output_file = '../../Results/figures/lake_tstpmax_0_01/hovmoller.png'
    plot(csv_file, output_file)
