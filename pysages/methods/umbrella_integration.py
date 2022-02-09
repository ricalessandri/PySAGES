# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Umbrella Integration.

Umbrella integration uses multiple replica placed along a pathway in the free energy
landscape by Harmonic Bias simulations.
From the statistics of the simulations, the thermodynamic forces along the path are
determine and integrated to obtain an approximation of the free-energy landscape.
This class implements the replica simulations and approximates the the free energy.
However, the method is not very and accurate it is preferred that more advanced methods
are used for the analysis of the simulations.
"""

from typing import Callable, Optional

from pysages.methods.core import SamplingMethod, Result, _run
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.utils import HistogramLogger, SerialExecutor
from pysages.utils import dispatch


class UmbrellaIntegration(SamplingMethod):
    """
    Umbrella Integration class.

    This class combines harmonic biasing with multiple replicas.
    It also collects histograms of the collective variables through out
    the simulations for later analysis.
    By default the class also estimates an approximation of the
    free energy landscape along the given path via umbrella integration.
    Note that this is not very accurate and ususally requires more sophisticated analysis on top.
    """

    def __init__(self, cvs, centers, ksprings, hist_periods, hist_offsets, **kwargs):
        """
        Initialization, mostly defining the collective variables and setting up
        the underlying Harmonic Bias.

        Arguments
        ---------
        centers: list[numbers.Real]
            CV centers along the path of integration.
            Its length defines the number replicas.

        ksprings: Union[float, list[float]]
            Spring constants of the harmonic biasing potential for each replica.

        hist_periods: Union[int, list[int]]
            Describes the period for the histrogram logging of each replica.

        hist_offsets: Union[int, list[int]]
            Offset applied before starting the histogram of each replica.
        """

        super().__init__(cvs, **kwargs)

        replicas = len(centers)
        ksprings = collect(ksprings, replicas, "ksprings", float)

        self.subsamplers = [
            HarmonicBias(cvs, k, c) for (k, c) in zip(ksprings, centers)
        ]
        self.hist_periods = collect(hist_periods, replicas, "hist_periods", int)
        self.hist_offsets = collect(hist_offsets, replicas, "hist_offsets", int)

    # We delegate the sampling work to HarmonicBias
    # (or possibly other methods in the future)
    def build(self):  # pylint: disable=arguments-differ
        pass


@dispatch
def run(  # pylint: disable=arguments-differ
    method: UmbrellaIntegration,
    context_generator: Callable,
    timesteps: int,
    context_args: Optional[dict] = None,
    executor=SerialExecutor(),
    **kwargs,
):
    """
    Implementation of the serial execution of umbrella integration with up to linear
    order (ignoring second order terms with covariance matrix) as described in
    J. Chem. Phys. 131, 034109 (2009); https://doi.org/10.1063/1.3175798 (equation 13).
    Higher order approximations can be implemented by the user using the provided
    covariance matrix.

    Arguments
    ---------
    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.conext.SimulationContext` for HOOMD-blue and
        `openmm.Context` for OpenMM.
        The function gets `context_args` unpacked for additional user args.
        For each replica along the path, the argument `replica_num` in [0, ..., N-1]
        is set in the `context_generator` to load the appropriate initial condition.

    timesteps: int
        Number of timesteps the simulation is running.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    kwargs:
        Passed to the backend run function as additional user arguments.

    * Note:
        This method does not accepts a user defined callback.
    """

    context_args = {} if context_args is None else context_args

    def submit_work(executor, method, callback):
        return executor.submit(
            _run, method, context_generator, timesteps, context_args, callback, **kwargs
        )

    callbacks_params = zip(method.hist_periods, method.hist_offsets)
    callbacks = [HistogramLogger(period, offset) for (period, offset) in callbacks_params]
    futures = []

    with executor as ex:
        futures_inputs = zip(method.subsamplers, callbacks)
        futures = [submit_work(ex, sampler, cb) for sampler, cb in futures_inputs]
        states = [future.result() for future in futures]

    return Result(method, states, callbacks)


@dispatch
def analyze(result: Result[UmbrellaIntegration]):
    """
    Computes the free energy from the result of an `UmbrellaIntegration` run.
    """
    subsamplers = result.method.subsamplers

    ksprings = [s.kspring for s in subsamplers]
    centers = [s.center for s in subsamplers]
    histogram_means = [cb.get_means() for cb in result.callbacks]
    mean_forces = []
    free_energy = [0.0]

    for i, center in enumerate(centers):
        mean_forces.append(free_energy_gradient(ksprings[i], histogram_means[i], center))
        if i > 0:
            free_energy.append(integrate(free_energy, mean_forces, centers, i))

    return dict(
        ksprings=ksprings,
        centers=centers,
        histograms=result.callbacks,
        histogram_means=histogram_means,
        mean_forces=mean_forces,
        free_energy=free_energy
    )


def collect(arg, replicas, name, dtype):
    """
    Returns a list of with lenght `replicas` of `arg` if `arg` is not a list,
    or `arg` if it is already a list of length `replicas`.
    """
    if isinstance(arg, list):
        if len(arg) != replicas:
            raise RuntimeError(
                f"Invalid length for argument {name} (got {len(arg)}, expected {replicas})"
            )
        return arg

    return [dtype(arg) for i in range(replicas)]


def free_energy_gradient(K, xi_mean, xi_ref):
    "Equation 13 from https://doi.org/10.1063/1.3175798"
    return -(K @ (xi_mean - xi_ref))


def integrate(A, nabla_A, centers, i):
    "Backward Riemann integration"
    return A[i-1] + nabla_A[i-1].T @ (centers[i] - centers[i-1])
