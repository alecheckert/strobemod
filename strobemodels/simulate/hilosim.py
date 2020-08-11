#!/usr/bin/env python
"""
hilosim.py -- simulated regular and anomalous diffusion processes in 
stroboscopic highly inclined laminated optical sheet (HiLo) geometries.

Generally, this means that a diffusion process in 3D space is only
observable (1) at regular frame intervals and (2) across a finite range
of positions in *z*. Additionally, the 3D trajectory is projected onto
the 2D plane of the camera.

This module has methods to simulate trajectories acquired in HiLo 
geometry.

"""
import os
import numpy as np 
from strobemod.simulate.fbm import FractionalBrownianMotion
from strobemod.simulate.levy import LevyFlight3D

DIFFUSION_MODELS = {
    "brownian": FractionalBrownianMotion,
    "fbm": FractionalBrownianMotion,
    "levy": LevyFlight3D
}

def strobe_one_state_infinite_plane(
    n_tracks=10000,
    track_len=10,
    model="brownian",
    model_obj=None,
    dz=0.7,
    dt=0.01,
    **model_kwargs
):
    """

    """
    # Generate the modeler
    if model_obj is None:
        model_obj = DIFFUSION_MODELS[model](dt=dt, track_len=track_len, **model_kwargs)

    # Simulate a bunch of 3D trajectories
    tracks = model_obj(n_tracks)

    # Format as dataframe







