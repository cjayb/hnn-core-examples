"""
=================================
Simulate somatosensory mu rhythms
=================================

This example reproduces qualitatively the continous 10/20 Hz somatomotor mu,
as published in Figure 5B of [1], and a single mu event, as in Figure 4B of
[2].

NB! At present, ``hnn_core`` cannot exactly reproduce [2], as changes were made
to the network that cannot be implemented using the current API.
"""

# Authors: Christopher Bailey <cjb@cfin.au.dk>

import os.path as op

###############################################################################
# Let us import hnn_core

import hnn_core
from hnn_core import simulate_dipole, read_params, Network
from hnn_core.viz import plot_dipole, plot_tfr_morlet
from hnn_core import MPIBackend

import numpy as np
import matplotlib.pyplot as plt

hnn_core_root = op.dirname(hnn_core.__file__)

###############################################################################
# Reproduce Sherman 2016, Fig 5
###############################################################################
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)
params['tstop'] = 1000.0

net = Network(params)

location = 'proximal'
burst_std = 20
weights_ampa_p = {'L2_basket': 4.e-5, 'L2_pyramidal': 2.e-5,
                  'L5_basket': 2e-5, 'L5_pyramidal': 2e-5}
syn_delays_p = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,
                'L5_basket': 1., 'L5_pyramidal': 1.}

net.add_bursty_drive(
    'beta_prox', tstart=0., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, repeats=10, location=location, weights_ampa=weights_ampa_p,
    synaptic_delays=syn_delays_p, seedcore=3)

location = 'distal'
burst_std = 15  # NB Sherman [1] and Law [2] use different values here
weights_ampa_d = {'L2_basket': 3.2e-4, 'L2_pyramidal': 8.e-5,
                  'L5_pyramidal': 4e-5}
syn_delays_d = {'L2_basket': 0.5, 'L2_pyramidal': 0.5,
                'L5_basket': 0.5, 'L5_pyramidal': 0.5}
net.add_bursty_drive(
    'beta_dist', tstart=0., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, repeats=10, location=location, weights_ampa=weights_ampa_d,
    synaptic_delays=syn_delays_d, seedcore=3)

with MPIBackend(n_procs=4):
    dpls_beta = simulate_dipole(net, n_trials=1)

trial_idx = 0
decim = [10, 10]  # decimate by a factor of 100
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 8),
                         gridspec_kw={'height_ratios': [1, 1, 2, 4]})
net.cell_response.plot_spikes_hist(ax=axes[0], spike_types=['beta_dist'],
                                   show=False)
net.cell_response.plot_spikes_hist(ax=axes[1], spike_types=['beta_prox'],
                                   show=False)
plot_dipole(dpls_beta[trial_idx], ax=axes[2], decim=decim, show=False)

freqs = np.arange(5., 40., 1.)
n_cycles = 7
plot_tfr_morlet(dpls_beta[trial_idx], freqs=freqs, n_cycles=n_cycles,
                decim=decim, ax=axes[3])
###############################################################################
# Reproduce Law 2019, Figure 4
###############################################################################
# Values to be changed relative to default parameters are given in
# Supplemental Table 1 [2]

# Cell parameters
params['L2Pyr_gabab_tau1'] = 45.
params['L2Pyr_gabab_tau2'] = 200.0
params['L5Pyr_gabab_tau1'] = 45.
params['L5Pyr_gabab_tau2'] = 200.0
params['L5Pyr_soma_gbar_ca'] = 0.0
# NB not all changes implemented in [2] are currently available in hnn-core
# XXX raise issue at hnn-core to implement
params['L5Pyr_dend_gbar_ca'] = 60.0

# Network connectivity
# NB L2b -> L5p connections are given separately for GABAA and GABAB in [2]
# XXX raise issue at hnn-core to implement
params['gbar_L2Basket_L5Pyr'] = 0.0002
params['gbar_L5Pyr_L5Pyr_nmda'] = 0.0004
params['gbar_L5Basket_L5Pyr_gabaa'] = 0.02
params['gbar_L5Basket_L5Pyr_gabab'] = 0.005

params['tstop'] = 300
eve = Network(params)

burst_std = 20
eve.add_bursty_drive(
    'prox_event', tstart=150., tstop=200., burst_rate=1, burst_std=burst_std,
    numspikes=2, spike_isi=10, repeats=10, location='proximal',
    weights_ampa=weights_ampa_p, synaptic_delays=syn_delays_p)

burst_std = 10  # NB Sherman [1] and Law [2] use different values here
eve.add_bursty_drive(
    'dist_event', tstart=150., tstop=200., burst_rate=1, burst_std=burst_std,
    numspikes=2, spike_isi=10, repeats=10, location='distal',
    weights_ampa=weights_ampa_d, synaptic_delays=syn_delays_d)

with MPIBackend(n_procs=4):
    dpls_eve = simulate_dipole(eve, n_trials=1)

# plot spikes, dipole waveform, and time-frequency representation
trial_idx = 0
decim = [3, 10]  # decimate by a factor of 30
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 8),
                         gridspec_kw={'height_ratios': [1, 1, 2, 4]})
eve.cell_response.plot_spikes_hist(ax=axes[0], spike_types=['dist_event'],
                                   show=False)
eve.cell_response.plot_spikes_hist(ax=axes[1], spike_types=['prox_event'],
                                   show=False)
plot_dipole(dpls_eve[trial_idx], ax=axes[2], decim=decim, show=False)

freqs = np.arange(5., 40., 1.)
n_cycles = 2  # short simulation window, only few cycles fit at lowest freq's
plot_tfr_morlet(dpls_eve[trial_idx], freqs=freqs, n_cycles=n_cycles,
                decim=decim, ax=axes[3])

# References
# ----------
# [1] Sherman, M. A. et al. Neural mechanisms of transient neocortical beta
# rhythms: Converging evidence from humans, computational modeling, monkeys, and
# mice. PNAS 113, E4885-94 (2016).
# [2] Law, R. G. et al. A supragranular nexus for the effects of neocortical beta
# events on human tactile perception. Biorxiv 750992 (2019) doi:10.1101/750992.
