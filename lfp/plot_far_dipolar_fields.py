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


sigma = 0.3  # S / m
method = 'psa'  # or 'lsa', should be identical far away

# LFP Grid
xmin, xmax = -1e4, 1e4
ymin, ymax = -1e4, 1e4
step = 1e3
posz = 20.

tstop, dt, burnin = 300, 0.025, 200

load_custom_mechanisms()

h.load_file("stdrun.hoc")
h.tstop = tstop + burnin
h.dt = dt
h.celsius = 37

_PC = h.ParallelContext(1)
_PC.set_maxstep(10)

_CVODE = h.CVode()
_CVODE.active(0)
_CVODE.use_fast_imem(1)

# %% Create Cell and LFP grid
silence_hh = {'L5Pyr_soma_gkbar_hh2': 0.0,
              'L5Pyr_soma_gnabar_hh2': 0.0,
              'L5Pyr_dend_gkbar_hh2': 0.0,
              'L5Pyr_dend_gnabar_hh2': 0.0}
l5p = pyramidal(pos=(0, 0, 0), cell_name='L5Pyr', override_params=silence_hh)

grid_lfp = list()
for posx in np.arange(xmin, xmax, step):
    row_lfp = list()
    for posy in np.arange(ymin, ymax, step):
        row_lfp.append(_LFPElectrode((posx, posy, posz), sigma=sigma,
                                     method=method, cvode=_CVODE))
    grid_lfp.append(row_lfp)

# %% Stimulate cell and run simulation
syn_deep = l5p.synapses['basal_1_nmda']
syn_superf = l5p.synapses['apical_2_nmda']

stim_deep = h.NetStim()  # Make a new stimulator
stim_deep.number = 1
stim_deep.start = 49 + burnin  # ms
ncstim_deep = h.NetCon(stim_deep, syn_deep)
ncstim_deep.delay = 10
ncstim_deep.weight[0] = 0.02  # NetCon weight is a vector.

stim_superf = h.NetStim()  # Make a new stimulator
stim_superf.number = 2
stim_superf.start = 199 + burnin  # ms
ncstim_superf = h.NetCon(stim_superf, syn_superf)
ncstim_superf.delay = 1
ncstim_superf.weight[0] = 0.02  # NetCon weight is a vector.

t = h.Vector().record(h._ref_t)

h.finitialize()

for tt in range(0, int(h.tstop), 10):
    _CVODE.event(tt, lambda: print(f'Simulation time {h.t: .2f} ms ...'))

h.fcurrent()
print(f'Running simulation with {len(grid_lfp) * len(grid_lfp[0])} electrodes')
# h.run()
_PC.psolve(h.tstop)

# %% Plot
times_lfp = np.array(t.to_python())
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

# %% Colorbar
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
