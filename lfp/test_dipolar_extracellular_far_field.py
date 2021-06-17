import os.path as op
import numpy as np
from numpy.linalg import norm
import hnn_core
from hnn_core import Network, read_params, simulate_dipole
from hnn_core.parallel_backends import MPIBackend


hnn_core_root = op.dirname(hnn_core.__file__)
params_fname = op.join(hnn_core_root, 'param', 'default.json')
params = read_params(params_fname)


def _mathematical_dipole(e_pos, d_pos, d_Q):
    rr = e_pos - d_pos
    R = norm(rr)
    Q = norm(d_Q)
    cosT = np.dot(rr, d_Q) / (R * Q)
    return (Q * cosT) / (4 * np.pi * R ** 2)


# require MPI to speed up due to large number of extracellular electrodes
def test_dipolar_far_field():
    """Test that LFP in the far field is dipolar when expected."""
    params.update({'N_pyr_x': 3,
                   'N_pyr_y': 3,
                   'tstop': 25,
                   })
    # initialise an unconnected network
    net = Network(params)

    # issue _weak_ excitatory drive to distal apical dendrites
    # NB must not cause Na- or Ca-spiking, as these are not associated with
    # dipolar fields
    weights_nmda = {'L2_basket': .0, 'L2_pyramidal': .0005,
                    'L2_basket': .0, 'L5_pyramidal': .0005}
    net.add_evoked_drive('d', mu=10., sigma=0., numspikes=1, location='distal',
                         sync_within_trial=True, weights_nmda=weights_nmda)

    conductivity = 0.3

    # create far-field grid of LFP electrodes; note that cells are assumed
    # to lie in the XZ-plane
    xmin, xmax = -5e4, 5e4
    zmin, zmax = -5e4, 5e4
    step = 5e3
    posy = 1e2  # out-of-plane
    electrode_pos = list()
    for posx in np.arange(xmin, xmax, step):
        for posz in np.arange(zmin, zmax, step):
            electrode_pos.append((posx, posy, posz))
    net.add_electrode_array('grid_psa', electrode_pos,
                            conductivity=conductivity, method='psa')
    net.add_electrode_array('grid_lsa', electrode_pos,
                            conductivity=conductivity, method='lsa')

    with MPIBackend(n_procs=2):
        dpl = simulate_dipole(net, postproc=False)

    X_p = np.arange(xmin, xmax, step) / 1000
    Z_p = np.arange(zmin, zmax, step) / 1000
    Y_p = posy / 1000
    idt = np.argmin(np.abs(dpl[0].times - 15.))
    phi_p_psa = np.zeros((len(X_p), len(Z_p)))
    phi_p_lsa = np.zeros((len(X_p), len(Z_p)))
    phi_p_theory = np.zeros((len(X_p), len(Z_p)))

    # location of equivalent current dipole for this stimulation (manual)
    d_pos = np.array((0, 0, 800)) / 1000  # um -> mm
    # dipole orientation is along the apical dendrite, towards the soma
    # the amplitude is really irrelevant, only shape is compared
    d_Q = 5e2 * np.array((0, 0, -1))

    for ii, row in enumerate(X_p):
        for jj, col in enumerate(Z_p):

            e_pos = np.array((row, Y_p, col))

            # ignore 10 mm radius closest to dipole
            if norm(e_pos - d_pos) < 10:
                phi_p_psa[ii][jj] = 0
                phi_p_lsa[ii][jj] = 0
                phi_p_theory[ii][jj] = 0
                continue

            phi_p_psa[ii][jj] = net.rec_arrays['grid_psa']._data[0][
                ii * len(X_p) + jj][idt] * 1e3
            phi_p_lsa[ii][jj] = net.rec_arrays['grid_lsa']._data[0][
                ii * len(X_p) + jj][idt] * 1e3
            phi_p_theory[ii][jj] = \
                _mathematical_dipole(e_pos, d_pos, d_Q) / conductivity

    # compare the shape of the far fields
    for phi_p in [phi_p_psa, phi_p_lsa]:
        cosT = np.dot(phi_p.ravel(), phi_p_theory.ravel()) / (
            norm(phi_p.ravel()) * norm(phi_p_theory.ravel()))
        # the far field should be very close to dipolar, though threshold may
        # need adjusting when new mechanisms are included in the cells
        assert 1 - cosT < 1e-3

    # for diagnostic plots, uncomment the following:
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import SymLogNorm
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # for ax, phi in zip(axs, [phi_p, phi_p_theory]):
    #     ax.pcolormesh(X_p, Y_p, phi.T,
    #                   norm=SymLogNorm(linthresh=1e-2, linscale=1.,
    #                                   vmin=-5e0, vmax=5e0, base=10),
    #                   cmap='BrBG_r', shading='auto')
    # plt.show()
