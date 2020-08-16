#!/usr/bin/env python
"""
test_simutils.py -- unit tests for the module
strobemodels.simulate.simutils

"""
import sys
import os
import numpy as np 
import pandas as pd 

import unittest
from numpy import testing 

# Functions to test
from strobemodels.simulate import simutils 

# Numeric tolerance
tolerance = 1.0e-10

def assert_allclose(a, b, rtol=tolerance):
    testing.assert_allclose(a, b, rtol=rtol)

class TestSimUtils(unittest.TestCase):
    """
    Unit tests for strobemodels.simulate.simutils.

    """
    def test_sample_sphere(self):
        """
        Sample Cartesian points uniformly from the surface of the unit sphere. 

        tests:
            - strobemodels.simulate.simutils.sample_sphere

        """
        print("\ntesting strobemodels.simulate.simutils.sample_sphere...")

        # Sample 100 points from a sphere in R3
        print("\tsampling from the surface of a sphere in R3...")
        points = simutils.sample_sphere(100, d=3)
        assert points.shape == (100, 3)
        assert_allclose(np.sqrt((points**2).sum(axis=1)), np.ones(100))

        # Sample 100 points from the edge of a circle
        print("\tsampling from the surface of a circle in R2...")
        points = simutils.sample_sphere(100, d=2)
        assert points.shape == (100, 2)
        assert_allclose(np.sqrt((points**2).sum(axis=1)), np.ones(100))

        # Stability testing: try sampling 0 points
        print("\tstability test: sampling 0 points...")
        points = simutils.sample_sphere(0, d=3)
        assert points.shape == (0, 3)

    def test_tracks_to_dataframe(self):
        """
        Convert trajectories in a 3D ndarray with shape (track_index, timepoint, dimension)
        into a pandas.DataFrame format.

        tests:
            - strobemodels.simulate.simutils.tracks_to_dataframe

        """
        print("\ntesting strobemodels.simulate.simutils.tracks_to_dataframe...")

        n_tracks = 5
        track_len = 4
        n_dim = 3
        positions = np.random.normal(size=(5, 4, 3))
        positions = np.cumsum(positions, axis=1)

        # Test for correct incorporation of the ZYX positions
        print("\ttesting correctness...")
        tracks = simutils.tracks_to_dataframe(positions, kill_nan=True)
        for t, track in tracks.groupby("trajectory"):
            assert_allclose(
                np.asarray(track['z']),
                positions[t,:,0]
            )
            assert_allclose(
                np.asarray(track['y']),
                positions[t,:,1]
            )
            assert_allclose(
                np.asarray(track['x']),
                positions[t,:,2]
            )
            assert_allclose(
                np.asarray(track['frame']),
                np.arange(track_len)
            )

        # Test that, when NaNs are present, they can be excluded from the result
        print("\ttesting that exclusion of points with z = NaN works...")
        positions[0,2,0] = np.nan 
        positions[4,0,0] = np.nan 
        tracks = simutils.tracks_to_dataframe(positions, kill_nan=True)
        assert not (2 in tracks.loc[tracks['trajectory'] == 0, 'frame'].tolist())
        for j in [0, 1, 3]:
            assert (j in tracks.loc[tracks['trajectory'] == 0, 'frame'].tolist())
        assert not (0 in tracks.loc[tracks['trajectory'] == 4, 'frame'].tolist())
        for j in [1, 2, 3]:
            assert (j in tracks.loc[tracks['trajectory'] == 4, 'frame'].tolist())









