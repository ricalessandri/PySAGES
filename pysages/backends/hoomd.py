# SPDX-License-Identifier: MIT
# This file is part of `hoomd-dlext`, see LICENSE.md


import importlib

from functools import partial

from hoomd.dlext import (
    AccessLocation, AccessMode, HalfStepHook, SystemView,
    net_forces, positions_types, rtags, tags, velocities_masses,
)

from jax.dlpack import from_dlpack as wrap


if hasattr(AccessLocation, 'OnDevice'):
    DEFAULT_DEVICE = AccessLocation.OnDevice
else:
    DEFAULT_DEVICE = AccessLocation.OnHost


def is_on_gpu(context):
    return context.on_gpu()


def view(context):
    #
    dt = context.integrator.dt
    system_view = SystemView(context.system_definition)
    #
    positions = wrap(positions_types(system_view, DEFAULT_DEVICE, AccessMode.Read))
    momenta = wrap(velocities_masses(system_view, DEFAULT_DEVICE, AccessMode.Read))
    forces = wrap(net_forces(system_view, DEFAULT_DEVICE, AccessMode.ReadWrite))
    ids = wrap(rtags(system_view, DEFAULT_DEVICE, AccessMode.Read))
    #
    box = system_view.particle_data().getGlobalBox()
    L  = box.getL()
    xy = box.getTiltFactorXY()
    xz = box.getTiltFactorXZ()
    yz = box.getTiltFactorYZ()
    lo = box.getLo()
    H = (
        (L.x, xy * L.y, xz * L.z, 0.0),  # Last column is a hack until
        (0.0,      L.y, yz * L.z, 0.0),  # https://github.com/google/jax/issues/4196
        (0.0,      0.0,      L.z, 0.0)   # gets fixed
    )
    origin = (lo.x, lo.y, lo.z)
    #
    return (positions, momenta, forces, ids, H, origin, dt)


class Hook(HalfStepHook):
    def initialize_from(self, sampler, bias):
        snapshot, initialize, update = sampler
        self.snapshot = snapshot
        self.state = initialize()
        self.update_from = update
        self.bias = bias
        return None
    #
    def update(self, timestep):
        self.state = self.update_from(self.snapshot, self.state)
        self.bias(self.snapshot, self.state)
        return None


def bind(context, sampler):
    # Depending on the device being used we need to use either cupy or numpy
    # (or numba) to generate a view of jax's DeviceArrays
    if is_on_gpu(context):
        cupy = importlib.import_module("cupy")
        wrap = cupy.asarray
    else:
        utils = importlib.import_module(".utils", package = "pysages.backends")
        wrap = utils.view
    #
    def bias(snapshot, state, sync):
        """Adds the computed bias to the forces."""
        # Forces may be computed asynchronously on the GPU, so we need to
        # synchronize them before applying the bias.
        sync()
        # TODO: Factor out the views so we can eliminate two function calls here.
        # Also, check if this can be JIT compiled with numba.
        forces = wrap(snapshot.forces)
        biases = wrap(state.bias.block_until_ready())
        forces += biases
        return None
    #
    system_view = SystemView(context.system_definition)
    sync_and_bias = partial(bias, sync = system_view.synchronize)
    #
    hook = Hook()
    hook.initialize_from(sampler, sync_and_bias)
    context.integrator.cpp_integrator.setHalfStepHook(hook)
    #
    # Return the hook to ensure it doesn't get garbage collected within the scope
    # of this function (another option is to store it in a global).
    return hook
