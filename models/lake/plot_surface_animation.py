#!/usr/bin/env python3
"""
Surface deviation animation from saved buoy.csv data.
Can be imported and called from run.py, or run standalone from the lakeV2 directory.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot(csv_file, output_file, equilibrium=17.0):
    data = np.genfromtxt(csv_file, delimiter=',', names=True)

    times       = data['time']
    buoy_names  = [name for name in data.dtype.names if name != 'time']
    buoy_xs     = np.array([float(name[1:]) for name in buoy_names])
    buoy_matrix = np.array([data[name] for name in buoy_names])  # (n_buoys, n_times)

    deviation = equilibrium - buoy_matrix  # positive = above equilibrium

    ymax = np.nanmax(np.abs(deviation)) * 1.1

    fig, ax = plt.subplots(figsize=(14, 5))
    line, = ax.plot([], [], color='steelblue', linewidth=2)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, label='equilibrium')
    ax.set(xlabel='x position (m)', ylabel='Deviation from equilibrium (m)',
           title='Water surface deviation',
           xlim=(buoy_xs[0], buoy_xs[-1]),
           ylim=(-ymax, ymax))
    ax.legend()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top')
    fig.tight_layout()

    def update(t_idx):
        line.set_data(buoy_xs, deviation[:, t_idx])
        time_text.set_text('Time: %.3f s' % times[t_idx])
        return line, time_text

    dt = times[1] - times[0]
    anim = FuncAnimation(fig, update, frames=len(times), interval=dt * 1000, blit=True)
    anim.save(output_file, writer='pillow', fps=10, dpi=100)
    plt.close(fig)
    print('Saved to', output_file)


if __name__ == '__main__':
    csv_file    = '../../Results/figures/lake_tstpmax_0_01/buoy.csv'
    output_file = '../../Results/figures/lake_tstpmax_0_01/surface_animation.gif'
    plot(csv_file, output_file)
