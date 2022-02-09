# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021: PySAGES contributors
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import reduce
from inspect import getfullargspec
from operator import or_
from typing import Callable, Optional

from jax import jit, numpy as np
from plum import parametric

from pysages.backends import ContextWrapper
from pysages.collective_variables.core import build
from pysages.grids import build_grid, get_info
from pysages.methods.utils import ReplicasConfiguration, methods_dispatch
from pysages.utils import JaxArray, dispatch, identity


# #  Base Classes  # #

class SamplingMethod(ABC):
    """
    Abstract base class for all sampling methods.

    Defines the constructor that expects the collective variables,
    the build method to initialize the GPU execution for the biasing,
    and the run method that executes the simulation run.
    All these are intended be enhanced/overwritten by inheriting classes.
    """

    __special_args__ = set()

    snapshot_flags = set()

    def __init__(self, cvs, **kwargs):
        self.cvs = cvs
        self.cv = build(*cvs)
        self.requires_box_unwrapping = reduce(
            or_, (cv.requires_box_unwrapping for cv in cvs), False
        )
        self.kwargs = kwargs

    def __getstate__(self):
        return default_getstate(self)

    def __setstate__(self, state):
        default_setstate(self, state)

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        """
        Returns the snapshot, and two functions: `initialize` and `update`.
        `initialize` is intended to allocate any runtime information required
        by `update`, while `update` is intended to be called after each call to
        the wrapped context's `run` method.
        """
        pass


class GriddedSamplingMethod(SamplingMethod):
    __special_args__ = {"grid"}

    def __init__(self, cvs, grid, **kwargs):
        check_dims(cvs, grid)
        super().__init__(cvs, **kwargs)
        self.grid = grid

    def __getstate__(self):
        return (get_info(self.grid), *default_getstate(self))

    def __setstate__(self, state):
        grid_args, args, kwargs = state
        args["grid"] = build_grid(*grid_args)
        default_setstate(self, (args, kwargs))

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


class NNSamplingMethod(GriddedSamplingMethod):
    def __init__(self, cvs, grid, topology, **kwargs):
        super().__init__(cvs, grid, **kwargs)
        self.topology = topology

    @abstractmethod
    def build(self, snapshot, helpers, *args, **kwargs):
        pass


@parametric
class Result:
    @classmethod
    def __infer_type_parameter__(self, method, *args):
        return type(method)

    @dispatch
    def __init__(self, method: SamplingMethod, states, callbacks):
        self.method = method
        self.states = states
        self.callbacks = callbacks


# #  Main methods  # #

# This is the user facing method for running simulations
@dispatch
def run(
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: int,
    context_args: Optional[dict] = None,
    config: ReplicasConfiguration = ReplicasConfiguration(),
    callback: Optional[Callable] = None,
    **kwargs
):
    """
    Base implementation for running one or more simulation replicas with the
    specified `SamplingMethod`.

    Arguments
    ---------
    method: SamplingMethod

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of timesteps the simulation is running.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    config: ReplicasConfiguration = ReplicasConfiguration()
        Specifies the number of replicas of the simulation to generate.
        It also contains an `executor` which will manage different process
        or threads in case the multiple simulation are to be run in parallel.
        Defaults to `ReplicasConfiguration(1, SerialExecutor())`,
        which means only one simulation is run.

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function. Default value is `None`.
    """

    context_args = {} if context_args is None else context_args

    def submit_work(executor, method, callback):
        return executor.submit(
            _run, method, context_generator, timesteps, context_args, callback, **kwargs
        )

    callbacks = [deepcopy(callback) for _ in range(config.copies)]
    futures = []

    with config.executor as ex:
        futures = [submit_work(ex, method, cb) for cb in callbacks]
        states = [future.result() for future in futures]

    return Result(method, states, None if callback is None else callbacks)


def _run(method, *args, **kwargs):
    run = methods_dispatch._functions["run"]
    return run(method, *args, **kwargs)


# We use `methods_dispatch` instead of the global dispatcher `dispatch` to
# separate the definitions for a single run vs multiple replica simulations.
@methods_dispatch
def run(
    method: SamplingMethod,
    context_generator: Callable,
    timesteps: int,
    context_args: Optional[dict] = None,
    callback: Optional[Callable] = None,
    **kwargs
):
    """
    Base implementation for running a single simulation with the specified `SamplingMethod`.

    Arguments
    ---------
    method: SamplingMethod

    context_generator: Callable
        User defined function that sets up a simulation context with the backend.
        Must return an instance of `hoomd.context.SimulationContext` for HOOMD-blue
        and `openmm.Simulation` for OpenMM. The function gets `context_args`
        unpacked for additional user arguments.

    timesteps: int
        Number of timesteps the simulation is running.

    context_args: Optional[dict] = None
        Arguments to pass down to `context_generator` to setup the simulation context.

    callback: Optional[Callable] = None
        Allows for user defined actions into the simulation workflow of the method.
        `kwargs` gets passed to the backend `run` function. Default value is `None`.

    *Note*: All arguments must be pickable.
    """

    context_args = {} if context_args is None else context_args

    context = context_generator(**context_args)
    wrapped_context = ContextWrapper(context, method, callback)
    with wrapped_context:
        wrapped_context.run(timesteps, **kwargs)

    return wrapped_context.sampler.state


@dispatch.abstract
def analyze(result: Result):
    pass


# #  Utils  # #

def default_getstate(method: SamplingMethod):
    init_args = set(getfullargspec(method.__init__).args[1:]) - method.__special_args__
    return {key: method.__dict__[key] for key in init_args}, method.kwargs


def default_setstate(method, state):
    args, kwargs = state
    method.__init__(**args, **kwargs)


def check_dims(cvs, grid):
    if len(cvs) != grid.shape.size:
        raise ValueError("Grid and Collective Variable dimensions must match.")


def generalize(concrete_update, helpers, jit_compile=True):
    if jit_compile:
        _jit = jit
    else:
        _jit = identity

    _update = _jit(concrete_update)

    def update(snapshot, state):
        return _update(state, helpers.query(snapshot))

    return _jit(update)
