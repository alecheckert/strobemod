#!/usr/bin/env python
"""
test_hilosim.py -- unit tests for strobemodels.simulate.hilosim

"""
import os
import sys
import numpy as np 
import pandas as pd 

# Testing utilities
import unittest
from numpy import testing 

# Functions to test
from strobemodels.simulate.fbm import FractionalBrownianMotion3D
from strobemodels.simulate import hilosim 

class TestStrobeInfinitePlane(unittest.TestCase):
    """
    Tests for the function hilosim.strobe_infinite_plane, which takes a diffusion
    modeler and simulates 3D trajectories in a HiLo-like slice.

    This function has several wrappers that also generate the diffusion model 
    in advance of running the simulation. The tests for these functions are 
    mostly "does it run?"-type tests. The detailed testing for the math behind
    the simulations should be reserved for the tests in test_fbm.py and test_levy.py.

    tests:
        strobemodels.simulate.hilosim.strobe_infinite_plane
        strobemodels.simulate.hilosim.strobe_one_state_infinite_plane

    """
    def test_strobe_infinite_plane(self):
        print("\ntesting strobemodels.simulate.hilosim.strobe_infinite_plane...")

        # Brownian motion
        D = 3.5
        dt = 0.01
        n_tracks = 1000
        dz = 0.7
        hz = dz / 2.0  
        loc_error = 0.0

        BM = FractionalBrownianMotion3D(track_len=10, hurst=0.5, D=D, dt=dt)

        # Generate the trajectories
        print("\tdoes it even run?")
        tracks = hilosim.strobe_infinite_plane(BM, n_tracks, dz=dz, loc_error=loc_error,
            exclude_outside=True, return_dataframe=True)
        assert isinstance(tracks, pd.DataFrame)

        # Check that none of the z-positions lie outside the focal slice
        print("\tchecking that defocalized particles are removed...")
        assert (np.abs(tracks["z"]) <= hz).all()

        # Check that all particles start at the origin in the XY plane
        assert (np.asarray(tracks.groupby('trajectory')[['y', 'x']].first()) == 0.0).all()

        # Check that we can return trajectories as an ndarray
        print("\tchecking the ability to return as ndarray...")
        tracks = hilosim.strobe_infinite_plane(BM, n_tracks, dz=dz, loc_error=loc_error,
            exclude_outside=True, return_dataframe=False)       
        assert isinstance(tracks, np.ndarray)
        nonnan = ~np.isnan(tracks[:,:,0])
        assert (np.abs(tracks[:,:,0][nonnan]) <= hz).all()

        # Check that we can remove the defocalization feature
        print("\tchecking the exclude_outside keyword...")
        tracks = hilosim.strobe_infinite_plane(BM, n_tracks, dz=dz, loc_error=loc_error,
            exclude_outside=False, return_dataframe=True)
        assert len(tracks) == (n_tracks * 10)

    def test_strobe_one_state_infinite_plane(self):
        print("\ntesting strobemodels.simulate.hilosim.strobe_one_state_infinite_plane...")

        n_tracks = 10000
        track_len = 10
        dz = 0.7
        dt = 0.01
        loc_error = 0.0

        hz = dz / 2.0

        D = 3.5

        # Try generating a Brownian motion
        print("\tdoes it run? (test case: Brownian motion)")
        tracks = hilosim.strobe_one_state_infinite_plane("brownian", n_tracks, track_len=track_len,
            dz=dz, dt=dt, loc_error=loc_error, exclude_outside=True, return_dataframe=True, D=D)

        assert isinstance(tracks, pd.DataFrame)
        assert (np.abs(tracks['z']) <= hz).all()

    def test_strobe_two_state_infinite_plane(self):
        print("\ntesting strobemodels.simulate.hilosim.strobe_two_state_infinite_plane...")

        # Two-state Brownian motion
        print("\tdoes it even run?")
        model_0_kwargs = {"D": 0.02}
        model_1_kwargs = {"D": 3.5}
        tracks = hilosim.strobe_two_state_infinite_plane("brownian", 1000, track_len=10,
            f0=0.5, dz=0.7, dt=0.01, loc_error=0.0, model_0_kwargs=model_0_kwargs,
            model_1_kwargs=model_1_kwargs, return_dataframe=True, exclude_outside=True)
        assert isinstance(tracks, pd.DataFrame)
        assert (np.abs(tracks['z']) <= 0.7/2).all()       

    def test_strobe_three_state_infinite_plane(self):
        print("\ntesting strobemodels.simulate.hilosim.strobe_three_state_infinite_plane...")

        # Three-state Brownian motion
        print("\tdoes it even run?")
        model_0_kwargs = {"D": 0.02}
        model_1_kwargs = {"D": 3.5}
        model_2_kwargs = {"D": 8.0}
        tracks = hilosim.strobe_three_state_infinite_plane("brownian", 1000, track_len=10,
            f0=0.5, f1=0.25, dz=0.7, dt=0.01, loc_error=0.0, model_0_kwargs=model_0_kwargs,
            model_1_kwargs=model_1_kwargs, model_2_kwargs=model_2_kwargs,
            return_dataframe=True, exclude_outside=True)
        assert isinstance(tracks, pd.DataFrame)
        assert (np.abs(tracks['z']) <= 0.7/2).all()       
