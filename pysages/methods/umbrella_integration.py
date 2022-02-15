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

from typing import Callable

from jax import numpy as np
from mpi4py.futures import MPIPoolExecutor

from pysages.backends import ContextWrapper
from pysages.methods.harmonic_bias import HarmonicBias
from pysages.methods.utils import HistogramLogger


def free_energy_gradient(K_spring, mean, center):
    "Equation 13 from https://doi.org/10.1063/1.3175798"
    return -(K_spring @ (mean - center))


def integrate(A, nabla_A, centers, i):
    return A[i-1] + nabla_A[i-1].T @ (centers[i] - centers[i-1])


def collect(arg, n_replica, name, dtype):
    if isinstance(arg, list):
        if len(arg) != n_replica:
            raise RuntimeError(
                f"Invalid length for argument {name} (got {len(arg)}, expected {n_replica})"
            )
    else:
        arg = [dtype(arg) for i in range(n_replica)]
    return arg


def run_umbrella_sampling(sampler_args, context_generator, context_args, center, kspring, hist_period, hist_offset, timesteps):
    sampler = UmbrellaIntegration(**sampler_args)
    sampler.center = center
    sampler.kspring = kspring
    callback = HistogramLogger(hist_period, hist_offset)

    context = context_generator(**context_args)
    wrapped_context = ContextWrapper(context, sampler, callback)

    with wrapped_context:
        wrapped_context.run(timesteps)

    mean = callback.get_means()
    nabla_A = free_energy_gradient(sampler.kspring, mean, sampler.center)

    return dict(kspring=kspring, center=sampler.center, histogram=callback, histogram_means=mean, nabla_A=nabla_A)


class UmbrellaIntegration(HarmonicBias):
    """
    Umbrella Integration class.

    This class combines harmonic biasing with multiple replicas.
    It also collects histograms of the collective variables through out the simulations
    for later analysis.
    By default the class also estimates an approximation of the free energy landscape
    along the given path via umbrella integration.
    Note that this is not very accurate and ususally requires more sophisticated analysis on top.
    """

    def __init__(self, cvs, *args, **kwargs):
        """
        Initialization, mostly defining the collective variables and setting up the
        underlying Harmonic Bias.
        """
        kspring = center = np.zeros(len(cvs))
        super().__init__(cvs, kspring, center, args, kwargs)

    def run(  # pylint: disable=arguments-differ
        self,
        context_generator: Callable,
        timesteps: int,
        centers,
        ksprings,
        hist_periods,
        hist_offsets=0,
        context_args=None,
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

        centers: list[numbers.Real]
            CV centers along the path of integration. The length defines the number replicas.

        ksprings: Union[float, list[float]]
            Spring constants of the harmonic biasing potential for each replica.

        hist_periods: Union[int, list[int]]
            Describes the period for the histrogram logging of each replica.

        hist_offsets: Union[int, list[int]]
            Offset applied before starting the histogram of each replica.

        kwargs:
            Passed to the backend run function as additional user arguments.

        * Note:
            This method does not accepts a user defined callback are not available.
        """

        if context_args is None:
            context_args = {}

        n_replica = len(centers)
        timesteps = collect(timesteps, n_replica, "timesteps", int)
        ksprings = collect(ksprings, n_replica, "kspring", float)
        hist_periods = collect(hist_periods, n_replica, "hist_periods", int)
        hist_offsets = collect(hist_offsets, n_replica, "hist_offsets", int)

        result = {}
        result["histogram"] = []
        result["histogram_means"] = []
        result["kspring"] = []
        result["center"] = []
        result["nabla_A"] = []
        result["A"] = []

        futures = []
        with MPIPoolExecutor() as executor:
            for rep in range(n_replica):
                context_args["replica_num"] = rep
                sampler_args = dict(cvs=self.cvs)
                futures.append(
                    executor.submit(
                        run_umbrella_sampling,
                        sampler_args,
                        context_generator,
                        context_args,
                        centers[rep],
                        ksprings[rep],
                        hist_periods[rep],
                        hist_offsets[rep],
                        timesteps[rep]
                    )
                )
            for future in futures:
                for key, val in future.result().items():
                    result[key].append(val)

        # Discrete forward integration of the free-energy
        result["A"].append(0)
        for i in range(1, n_replica):
            result["A"].append(
                integrate(result["A"], result["nabla_A"][:i+1], result['center'][:i+1], i)
            )

        return result
