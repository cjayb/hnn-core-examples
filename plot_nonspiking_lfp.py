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


def calc_monopole_multiplier(ele_pos, sec_mid, sigma=.3):
    ele_pos = np.array(ele_pos)  # electrode position

    # distance from compartment to electrode
    dis = np.linalg.norm(ele_pos - mid)

    phi = 1. / dis

    return 1000.0 * phi / (4.0 * np.pi * sigma)


_PC = h.ParallelContext(1)
_PC.set_maxstep(10)

_CVODE = h.CVode()
_CVODE.active(0)
_CVODE.use_fast_imem(1)

sigma = 0.3  # S / m
method = 'psa'

load_custom_mechanisms()

params = dict(tstop=300, dt=0.025, burnin=200)

h.load_file("stdrun.hoc")
h.tstop = params['tstop'] + params['burnin']
h.dt = params['dt']
h.celsius = 37

silence_hh = {'L5Pyr_soma_gkbar_hh2': 0.0,
              'L5Pyr_soma_gnabar_hh2': 0.0,
              'L5Pyr_dend_gkbar_hh2': 0.0,
              'L5Pyr_dend_gnabar_hh2': 0.0}
l5p = pyramidal(pos=(0, 0, 0), cell_name='L5Pyr',
                override_params=silence_hh)
# for sec in l5p.sections:
#     sec.insert('pas')
#     for seg in sec:
#         seg.pas.g = 4.26e-05
#         if 'soma' in sec.name():
#             seg.pas.e = -65.
#         else:
#             seg.pas.e = -71.


def laminar_array(ymin, ymax, ystep, posx=10, posz=10):
    el_array = list()
    for posy in np.arange(ymin, ymax, ystep):
        el_array.append(_LFPElectrode((posx, posy, posz), sigma=sigma, pc=None,
                                      cvode=_CVODE, method=method))
    return el_array


laminar_lfp = laminar_array(-200, 2000, 100)

# %%
ls = h.allsec()
allsecs = [s for s in ls]
imem_vecs = dict()
for ii, sec in enumerate(allsecs):
    key = '_'.join(sec.name().split('_')[1:])
    imem_vecs.update({key: list()})
    for seg in sec.allseg():
        imem_vecs[key].append(h.Vector().record(sec(seg.x)._ref_i_membrane_))


syn_deep = l5p.synapses['basal_1_nmda']
syn_superf = l5p.synapses['apical_2_nmda']
syn_deep_i = h.Vector().record(syn_deep._ref_i)
syn_superf_i = h.Vector().record(syn_superf._ref_i)

soma_v = l5p.rec_v.record(l5p.sections['soma'](0.5)._ref_v)
t = h.Vector().record(h._ref_t)

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


def simulation_time():
    print('Simulation time: {0} ms...'.format(round(h.t, 2)))


for tt in range(0, int(h.tstop), 10):
    _CVODE.event(tt, simulation_time)

h.fcurrent()
_PC.psolve(h.tstop)

times = np.array(t.to_python())
v_soma = np.array(soma_v.to_python())
i_deep = np.array(syn_deep_i.to_python())
i_superf = np.array(syn_superf_i.to_python())

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(times[times >= params['burnin']],
           v_soma[times >= params['burnin']])
ax[1].plot(times[times >= params['burnin']],
           i_deep[times >= params['burnin']],
           label='$I_{deep}$')
ax[1].plot(times[times >= params['burnin']],
           i_superf[times >= params['burnin']],
           label='$I_{superf}$')

ax[0].set_title('Somatic membrane potential (mV)')
ax[1].legend(loc='lower right')
ax[1].set_title('Synaptic currents (nA)')

# laminar LFP
efig, eax = plt.subplots(1, 1)
times_lfp = np.array(laminar_lfp[0].lfp_t.to_python())
tind = times_lfp >= params['burnin']
cmap = plt.get_cmap('inferno')
depths = np.arange(-200, 2000, 100)
for ii, lfp in enumerate(laminar_lfp):
    eax.plot(times_lfp[tind] + ii * 2,
             10 * ii + 10 * np.array(lfp.lfp_v)[tind],
             color=cmap(ii / len(laminar_lfp)))
    eax.text(200, 10 * ii, f'{int(depths[ii])}')
eax.set_ylim((-20, len(depths) * 10))
eax.set_yticks(())
eax.set_title('NMDA events at 250 ms (basal_1) and 400 ms (apical_2)')
eax.set_ylabel('Potential as function of distance from soma')
eax.set_xlabel('Time (ms)')


# %% plot NET membrane current in each section
# NB summing over segments
sec_order = ['basal_3', 'basal_2', 'basal_1', 'soma',
             'apical_trunk', 'apical_oblique',
             'apical_1', 'apical_2', 'apical_tuft']
sec_order.reverse()
soma_idx = 3
fig, ax = plt.subplots(len(sec_order), 1, figsize=(8, 12))
for ii_sec, key in enumerate(sec_order):
    for ii_seg, _h_seg_im in enumerate(imem_vecs[key]):
        seg_im = np.array(_h_seg_im.to_python())[:-1]
        ax[ii_sec].plot(times_lfp[tind] + ii_seg * 10,
                        ii_seg + 10 * seg_im[tind])
    # ax[ii_sec].text(150, im, key)
    ax[ii_sec].set_ylabel(key)
    ax[ii_sec].set_ylim((-0.2, (ii_seg + 1) + 0.2))
ax[ii_sec].set_xlabel('Time (ms)')
fig.suptitle('Transmembrane current (nA) in sub-segments')

# %%
# calculate LFP manually from imem_secs
laminar_lfp_man = list()
for ele_depth in depths:
    ele_pos = (10, ele_depth, 10)
    lfp = 0
    for ii, (seg_im_list, sec) in enumerate(zip(imem_vecs.values(), allsecs)):
        sec_start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        sec_end = np.array([sec.x3d(1), sec.y3d(1), sec.z3d(1)])
        mid = (sec_start + sec_end) / 2
        mult = calc_monopole_multiplier(ele_pos, mid, sigma=sigma)
        for _h_seg_im in seg_im_list[1:-1]:  # ignore endpoints (zero)
            seg_im = np.array(_h_seg_im.to_python())
            lfp += seg_im * mult

    laminar_lfp_man.append(lfp)
# plot manual LFP
efig, eax = plt.subplots(1, 1)
for ii, lfp in enumerate(laminar_lfp_man):
    eax.plot(times_lfp[tind] + ii * 2,
             10 * ii + 10 * np.array(lfp)[:-1][tind],
             color=cmap(ii / len(laminar_lfp_man)))
    eax.text(200, 10 * ii, f'{int(depths[ii])}')
eax.set_ylim((-20, len(depths) * 10))
eax.set_yticks(())
eax.set_title('NMDA events at 250 ms (basal_1) and 400 ms (apical_2)')
eax.set_ylabel('Potential as function of distance from soma')
eax.set_xlabel('Time (ms)')

# %% Check that all currents are accounted for!
fig, ax = plt.subplots(1, 1, figsize=(6, 3))

# synapses at midpoints
n_seg_deep = int((l5p.sections['basal_1'].nseg + 2) / 2)
n_seg_superf = int((l5p.sections['apical_2'].nseg + 2) / 2)

syn_current = i_deep + i_superf
leak_currents = np.zeros(syn_current.shape)
for key, segs in imem_vecs.items():
    for idx, seg_current in enumerate(segs):
        this_seg_net_current = np.array(seg_current.to_python())
        if key == 'basal_1' and idx == n_seg_deep:
            this_seg_net_current -= i_deep
        elif key == 'apical_2' and idx == n_seg_superf:
            this_seg_net_current -= i_superf
        leak_currents += np.array(this_seg_net_current)

ax.plot(times_lfp[tind], -syn_current[:-1][tind],
        ls='--', lw=4, label='$-I_{syn}$')
ax.plot(times_lfp[tind], leak_currents[:-1][tind],
        lw=2, label='$I_{leak}$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Net current (nA)')
ax.legend()
fig.suptitle('Synaptic currents must leave cell as leak current')

plt.show()
