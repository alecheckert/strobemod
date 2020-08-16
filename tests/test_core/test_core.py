#!/usr/bin/env python
"""
test_core.py -- unit tests for functions in strobemodels.core

"""
import os
import sys
import numpy as np 
import pandas as pd 

# Testing functions
import unittest
from numpy import testing 

# Functions to test 
from strobemodels import core 

# Helper
from strobemodels.utils import FIXTURE_DIR

class TestFitModelCDF(unittest.TestCase):
    """
    A variety of tests for the core least-squares CDF fitting function,
    strobemodels.core.fit_model_cdf.

    """
    def setUp(self):
        """
        Load some testing data.

        """
        test_file = os.path.join(FIXTURE_DIR, "one_state_infinite_plane_brownian_10000tracks_dt-0.01_D-2.0_locerror-0.035.csv")
        self.tracks = pd.read_csv(test_file)

    def test_fit_model_cdf(self):
        print("\ntesting strobemodels.core.fit_model_cdf...")

        # Check that it runs with simple settings
        print("\tdoes it even run?")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)
        assert isinstance(fit_pars, dict)
        assert isinstance(bin_edges, np.ndarray)
        assert isinstance(cdfs, np.ndarray)
        assert isinstance(pmfs, np.ndarray)
        assert isinstance(model_cdfs, np.ndarray)
        assert isinstance(model_pmfs, np.ndarray)
        assert bin_edges.shape == (5001,)
        assert cdfs.shape == (4, 5000)
        assert pmfs.shape == (4, 5000)
        assert model_cdfs.shape == (4, 5000)
        assert model_pmfs.shape == (4, 5000)

        # Correct number of jumps corresponding to each frame interval
        print("\tchecking for numerical correctness...")
        assert abs(fit_pars["D"] - 1.9973534968154916) <= 1.0e-8
        assert abs(fit_pars["loc_error"] - 0.039059778137488566) <= 1.0e-8

        # Run with a more complicated model that requires specialized keyword
        # arguments, such as "dz"
        print("\ttesting on a model that takes extra keyword arguments...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0, dz=1.0)

        # Check weighting by the number of observations in each temporal bin
        print("\tchecking the ability to weight each frame interval by its number of jumps...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=True, weight_short_disps=False, max_jump=5.0)
        assert abs(fit_pars["D"] - 1.99523250338949) <= 1.0e-8
        assert abs(fit_pars["loc_error"] - 0.03949055055433056) <= 1.0e-8

        # Check weighting toward the short displacements, a common request from users
        print("\tchecking the ability to weight the fits toward the lower end " \
            "of the jump length histogram...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=True, weight_short_disps=False, max_jump=5.0)

        # Check the ability to give it an initial guess
        print("\tchecking ability to issue initial guesses...")
        guess = np.array([10.0, 0.05])
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=guess, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)

        # Multiple initial guesses
        guesses = [np.array([10.0, 0.05]), np.array([5.0, 0.04]), np.array([3.0, 0.02])]
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=guesses, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)

        # Check the ability to constrain the fit parameters
        print("\ttesting the ability to constrain fit parameters...")
        bounds = (np.array([0.0, 0.0]), np.array([np.inf, 1.0e-10]))
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=bounds, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)
        assert abs(fit_pars["loc_error"]) <= 1.0e-9

        # Constrain the localization error to ~0.05
        bounds = (np.array([0.0, 0.05]), np.array([np.inf, 0.05]))
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=bounds, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)
        assert abs(0.05 - fit_pars['loc_error']) <= 1.0e-9

        # Test that it runs on pathological input: empty dataframe
        print("\tstability test: giving it empty input...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks[:0], model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)
        for k in fit_pars.keys():
            assert np.isnan(fit_pars[k])

        # Test that it runs when we give it trajectories that do not have any 
        # displacements in a given frame interval. The function should discard
        # frame intervals for which there are no displacements.
        print("\tstability test: giving it data with missing frame intervals...")
        T = self.tracks[self.tracks["frame"] <= 2]
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            T, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)
        for k in fit_pars.keys():
            assert not np.isnan(fit_pars[k])
        assert cdfs.shape == (2, 5000)
        assert pmfs.shape == (2, 5000)
        assert model_cdfs.shape == (2, 5000)
        assert model_pmfs.shape == (2, 5000)

        # Test that it runs on pathological input: a set of trajectories with
        # no displacements in the first frame interval
        print("\tstability test: giving it trajectories with no displacements in " \
            "the first frame interval...")
        T = self.tracks[self.tracks["frame"].isin([0, 2, 3, 4, 5, 6, 7])]
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            T, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, save_png=None, 
            weight_timesteps=False, weight_short_disps=False, max_jump=5.0)
        for k in fit_pars.keys():
            assert np.isnan(fit_pars[k])

        # Testing plotting
        print("\ttesting plotting capabilities...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=True, show_plot=False, 
            save_png=None, weight_timesteps=False, weight_short_disps=False,
            max_jump=5.0)

        print("\ttesting plotting with saving...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=True, show_plot=False, 
            save_png="_test_out.png", weight_timesteps=False, weight_short_disps=False,
            max_jump=5.0)
        os.remove("_test_out_pmf.png")
        os.remove("_test_out_cdf.png")


