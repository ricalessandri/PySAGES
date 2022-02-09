# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collection of helpful classes for methods.

This includes callback functors (callable classes).
"""

from concurrent.futures import Executor, Future

from jax import numpy as np
from plum import Dispatcher


# We use this to dispatch on the different `run` implementations
# for `SamplingMethod`s.
methods_dispatch = Dispatcher()


class SerialExecutor(Executor):
    """
    Subclass of `concurrent.futures.Executor` used as the default
    task manager. It will execute all tasks in serial.
    """

    def submit(self, fn, *args, **kwargs):  # pylint: disable=arguments-differ
        """
        Executes `fn(*args, **kwargs)` and returns a `Future` object wrapping the result.
        """
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class ReplicasConfiguration:
    """
    Stores the information necessary to execute multiple simulation runs,
    including the number of copies of the system and the task manager.
    """

    def __init__(self, copies: int = 1, executor=SerialExecutor()):
        """
        ReplicasConfiguration constructor.

        Arguments
        ---------
        copies: int
            Number of replicas of the simulation system to be generated.
            Defaults to `1`.

        executor:
            Task manager that satisfies the `concurrent.futures.Executor` interface.
            Defaults to `SerialExecutor()`.
        """
        self.copies = copies
        self.executor = executor


class HistogramLogger:
    """
    Implements a Callback functor for methods.
    Logs the state of the collective variable to generate histograms.
    """

    def __init__(self, period: int, offset: int = 0):
        """
        HistogramLogger constructor.

        Arguments
        ---------
        period:
            Timesteps between logging of collective variables.

        offset:
            Timesteps at the beginning of a run used for equilibration.
        """
        self.period = period
        self.counter = 0
        self.offset = offset
        self.data = []

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        self.counter += 1
        if self.counter > self.offset and self.counter % self.period == 0:
            self.data.append(state.xi[0])

    def get_histograms(self, **kwargs):
        """
        Helper function to generate histrograms from the collected CV data.
        `kwargs` are passed on to `numpy.histogramdd` function.
        """
        data = np.asarray(self.data)
        if "density" not in kwargs:
            kwargs["density"] = True
        return np.histogramdd(data, **kwargs)

    def get_means(self):
        """
        Returns mean values of the histogram data.
        """
        data = np.asarray(self.data)
        return np.mean(data, axis=0)

    def get_cov(self):
        """
        Returns covariance matrix of the histgram data.
        """
        data = np.asarray(self.data)
        return np.cov(data.T)

    def reset(self):
        """
        Reset internal state.
        """
        self.counter = 0
        self.data = []


def average_forces(state, n=1):
    """
    Given a `SamplingMethod` state with attributes `force_sum` and `hist`,
    computes the mean force with appropriate dimensions.
    """
    force_sum = state.force_sum
    shape = (*force_sum.shape[:-1], 1)
    return force_sum / np.maximum(state.hist.reshape(shape), n)
