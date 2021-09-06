from neuron import h
import numpy as np

from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms


_PC = h.ParallelContext()
# sets the default max solver step in ms (purposefully large)
_PC.set_maxstep(10)

_CVODE = h.CVode()
_CVODE.active(0)
_CVODE.use_fast_imem(1)


class MockExtracellularArray:
    def __init__(self, positions):
        self.positions = positions
        self.n_contacts = len(self.positions)

    def _build(self, cvode=None):
        """Assemble NEURON objects for calculating extracellular potentials.

        The handler is set up to maintain a vector of membrane currents at at
        every inner segment of every section of every cell on each CVODE
        integration step. In addition, it records a time vector of sample
        times.

        Parameters
        ----------
        cvode : instance of h.CVode
            Multi order variable time step integration method.
        """
        secs_on_rank = h.allsec()  # get all h.Sections known to this MPI rank

        segment_counts = [sec.nseg for sec in secs_on_rank]
        n_total_segments = np.sum(segment_counts)

        # pointers assigned to _ref_i_membrane_ at each EACH internal segment
        self._nrn_imem_ptrvec = h.PtrVector(n_total_segments)
        # placeholder into which pointer values are read on each sim time step
        self._nrn_imem_vec = h.Vector(n_total_segments)

        ptr_idx = 0
        for sec in secs_on_rank:
            for seg in sec:  # section end points (0, 1) not included
                # set Nth pointer to the net membrane current at this segment
                self._nrn_imem_ptrvec.pset(
                    ptr_idx, sec(seg.x)._ref_i_membrane_)
                ptr_idx += 1
        if ptr_idx != n_total_segments:
            raise RuntimeError(f'Expected {n_total_segments} imem pointers, '
                               f'got {ptr_idx}.')

        # transfer resistances for each segment (keep in Neuron Matrix object)
        self._nrn_r_transfer = h.Matrix(self.n_contacts, n_total_segments)

        for row in range(len(self.positions)):
            # for testing, make a matrix of ones
            self._nrn_r_transfer.setrow(row, h.Vector(n_total_segments, 1.))

        # record time for each array
        self._nrn_times = h.Vector().record(h._ref_t)

        # contributions of all segments on this rank to total calculated
        # potential at electrode (_PC.allreduce called in _simulate_dipole)
        self._nrn_voltages = h.Vector()

        if cvode is not None:
            # This line is needed!
            recording_callback = self._embedded_potentials_callback
            cvode.extra_scatter_gather(0, recording_callback)
            return recording_callback
            # This won't work
            # cvode.extra_scatter_gather(0, self._embedded_potentials_callback)
            # return self._embedded_potentials_callback

    def _embedded_potentials_callback(self):
        # keep all data in Neuron objects for efficiency
        # print(f'{h.t}')  # confirmed: repeated calls print multiple t's

        # 'gather' the values of seg.i_membrane_ into self.imem_vec
        self._nrn_imem_ptrvec.gather(self._nrn_imem_vec)

        # Calculate potentials by multiplying the _nrn_imem_vec by the matrix
        # _nrn_r_transfer. This is equivalent to a row-by-row dot-product:
        # V_i(t) = SUM_j ( R_i,j x I_j (t) )
        self._nrn_voltages.append(
            self._nrn_r_transfer.mulv(self._nrn_imem_vec))
        # NB all values appended to the h.Vector _nrn_voltages at current time
        # step. The vector will have size (n_contacts x n_samples, 1), which
        # will be reshaped later to (n_contacts, n_samples).


def separate_potentials_callback(nrn_arr):
    # keep all data in Neuron objects for efficiency

    # 'gather' the values of seg.i_membrane_ into self.imem_vec
    nrn_arr._nrn_imem_ptrvec.gather(nrn_arr._nrn_imem_vec)

    # Calculate potentials by multiplying the _nrn_imem_vec by the matrix
    # _nrn_r_transfer. This is equivalent to a row-by-row dot-product:
    # V_i(t) = SUM_j ( R_i,j x I_j (t) )
    nrn_arr._nrn_voltages.append(
        nrn_arr._nrn_r_transfer.mulv(nrn_arr._nrn_imem_vec))
    # NB all values appended to the h.Vector _nrn_voltages at current time
    # step. The vector will have size (n_contacts x n_samples, 1), which
    # will be reshaped later to (n_contacts, n_samples).


def test_extra_scatter_gather(loop=1, use_separate_callback=True):
    """Test default cell objects."""
    load_custom_mechanisms()

    l5p = pyramidal(cell_name='L5Pyr')
    l5p.build(sec_name_apical='apical_trunk')

    stim = h.IClamp(l5p.sections['soma'](0.5))
    stim.delay = 5
    stim.dur = 5.
    stim.amp = 2.

    h.load_file("stdrun.hoc")
    h.tstop = 40.
    h.dt = 0.025
    h.celsius = 37

    for _ in range(loop):

        nrn_arr = MockExtracellularArray([(10, 10, 10)])

        if use_separate_callback:
            nrn_arr._build(cvode=None)
            recording_callback = (separate_potentials_callback, nrn_arr)
            _CVODE.extra_scatter_gather(0, recording_callback)
        else:
            recording_callback = nrn_arr._build(cvode=_CVODE)

        h.finitialize()
        h.fcurrent()

        # initialization complete, but wait for all procs to start the solver
        _PC.barrier()

        # actual simulation - run the solver
        _PC.psolve(h.tstop)

        _PC.barrier()

        _PC.allreduce(nrn_arr._nrn_voltages, 1)

        times_ecell = nrn_arr._nrn_times.to_python()
        n_contacts = 1
        n_samples = len(times_ecell) - 1

        extmat = h.Matrix(n_contacts, n_samples)
        extmat.from_vector(nrn_arr._nrn_voltages)
        lfp = [extmat.getrow(ii).to_python() for ii in range(extmat.nrow())]

        _CVODE.extra_scatter_gather_remove(recording_callback)


if __name__ == '__main__':
    import time

    SEPARATE = False  # and True now equivalent in terms of calc time

    for tr in range(5):
        start_time = time.time()
        test_extra_scatter_gather(loop=10, use_separate_callback=SEPARATE)
        print(f'Loop of 10 simulations, trial {tr}:\t'
              f'{time.time() - start_time:.4f} seconds')
