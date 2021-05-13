# %%
# %matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from neuron import h
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.lfp import _LFPElectrode
from hnn_core.cells_default import pyramidal


def grid_array(xmin, xmax, ymin, ymax, step, posz=10):
    el_array = list()
    for posx in np.arange(xmin, xmax, step):
        el_array_row = list()
        for posy in np.arange(ymin, ymax, step):
            el_array_row.append(_LFPElectrode((posx, posy, posz), sigma=sigma,
                                              pc=None, cvode=_CVODE,
                                              method=method))
        el_array.append(el_array_row)
    return el_array


def simulation_time():
    print('Simulation time: {0} ms...'.format(round(h.t, 2)))


sigma = 0.3  # S / m
method = 'psa'

load_custom_mechanisms()

params = dict(tstop=300, dt=0.025, burnin=200)

h.load_file("stdrun.hoc")
h.tstop = params['tstop'] + params['burnin']
h.dt = params['dt']
h.celsius = 37

_CVODE = h.CVode()
_CVODE.active(0)
_CVODE.use_fast_imem(1)

# Why isn't this working??
for tt in range(0, int(h.tstop), 50):
    _CVODE.event(tt, simulation_time)

# %%
silence_hh = {'L5Pyr_soma_gkbar_hh2': 0.0,
              'L5Pyr_soma_gnabar_hh2': 0.0,
              'L5Pyr_dend_gkbar_hh2': 0.0,
              'L5Pyr_dend_gnabar_hh2': 0.0}
l5p = pyramidal(pos=(0, 0, 0), cell_name='L5Pyr', override_params=silence_hh)

xmin, xmax = -1e4, 1e4
ymin, ymax = -1e4, 1e4
step = 1e3
grid_lfp = grid_array(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                      step=step, posz=20)

# %%
syn_deep = l5p.synapses['basal_1_nmda']
syn_superf = l5p.synapses['apical_2_nmda']

stim_deep = h.NetStim()  # Make a new stimulator
stim_deep.number = 1
stim_deep.start = 49 + params['burnin']  # ms
ncstim_deep = h.NetCon(stim_deep, syn_deep)
ncstim_deep.delay = 10
ncstim_deep.weight[0] = 0.02  # NetCon weight is a vector.

stim_superf = h.NetStim()  # Make a new stimulator
stim_superf.number = 2
stim_superf.start = 199 + params['burnin']  # ms
ncstim_superf = h.NetCon(stim_superf, syn_superf)
ncstim_superf.delay = 1
ncstim_superf.weight[0] = 0.02  # NetCon weight is a vector.

h.finitialize()
h.fcurrent()
print(f'Running simulation with {len(grid_lfp) * len(grid_lfp[0])} electrodes')
h.run()

# %%
times_lfp = np.array(grid_lfp[0][0].lfp_t.to_python())
X_p = np.arange(xmin, xmax, step) / 1000
Y_p = np.arange(ymin, ymax, step) / 1000
idt_deep = np.argmin(np.abs(times_lfp - 260.))
idt_superf = np.argmin(np.abs(times_lfp - 420.))

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plt.subplots_adjust(wspace=0.005, left=0.07)
for ax, idt in zip(axs, [idt_deep, idt_superf]):
    phi_p = np.zeros((len(X_p), len(Y_p)))
    for ii, row in enumerate(grid_lfp):
        for jj, col in enumerate(row):
            phi_p[ii][jj] = col.lfp_v[idt] * 1e3  # uV to nV

    pcm = ax.pcolormesh(X_p, Y_p, phi_p.T,
                        norm=SymLogNorm(linthresh=1e-1, linscale=1.,
                                        vmin=-5e2, vmax=5e2, base=10),
                        cmap='BrBG_r', shading='flat')
    ax.set_xlabel('Distance from soma in X (mm)')
    ax.set_aspect('equal', 'box')
axs[0].set_ylabel('Distance from soma in Y (mm)')
axs[1].set_yticklabels(())
axs[0].set_title('Deep synapse active')
axs[1].set_title('Superficial synapse active')
axins = inset_axes(axs[1],
                   width="5%",  # width = 5% of parent_bbox width
                   height="80%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.175, 0.1, 1, 1),
                   bbox_transform=axs[1].transAxes,
                   borderpad=0,
                   )
cbh = fig.colorbar(pcm, cax=axins, extend='both')
cbh.ax.yaxis.set_ticks_position('left')
cbh.ax.set_ylabel('Potential (nV)')

plt.show()

# %%
