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

class TestFitModels(unittest.TestCase):
    """
    A variety of tests for the core least-squares CDF fitting functions:
        - strobemodels.core.fit_model_cdf
        - strobemodels.core.fit_ml

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
        print("\tchecking plotting option...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=True, show_plot=False, 
            save_png=None, weight_timesteps=False, weight_short_disps=False,
            max_jump=5.0)

        print("\tchecking ability to save plots...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_model_cdf(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=True, show_plot=False, 
            save_png="_test_out.png", weight_timesteps=False, weight_short_disps=False,
            max_jump=5.0)
        os.remove("_test_out_pmf.png")
        os.remove("_test_out_cdf.png")

    def test_fit_ml(self):
        print("\ntesting strobemodels.core.fit_ml...")

        # Test that it runs with simple settings
        print("\tdoes it even run?")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, show_plot=False,
            save_png=None)
        assert isinstance(fit_pars, dict)
        assert bin_edges.shape == (5001,)
        assert cdfs.shape == (4, 5000)
        assert pmfs.shape == (4, 5000)
        assert model_cdfs.shape == (4, 5000)
        assert model_pmfs.shape == (4, 5000)

        # Check for numerical correctness
        print("\tchecking numerical correctness...")
        assert abs(fit_pars["D"] - 1.9957911194268367) <= 1.0e-6
        assert abs(fit_pars["loc_error"] - 0.03925184705124344) <= 1.0e-6

        # Run with a more complicated model that accepts extra keyword arguments
        # passed to the model function
        print("\trunning on a more complicated model that accepts keyword arguments...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, show_plot=False,
            save_png=None, dz=0.7)

        # Issue the initial guess
        print("\tchecking ability to issue initial guesses...")
        guesses = [
            np.array([0.25, 0.0025, 5.0, 0.035]),
            np.array([0.50, 0.0025, 5.0, 0.035]),
            np.array([0.75, 0.0025, 5.0, 0.035]),
            np.array([1.00, 0.0025, 5.0, 0.035]),
        ]

        # Try with multiple initial guesses
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=guesses, plot=False, show_plot=False,
            save_png=None, dz=0.7)

        # Try with a single initial guess
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=guesses[0], plot=False, show_plot=False,
            save_png=None, dz=0.7)

        # Check the ability to constrain the estimate with parameter bounds
        print("\tchecking ability to constrain parameters...")
        bounds = (np.array([0.0, 0.0]), np.array([np.inf, 0.0]))
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=bounds, guess=None, plot=False, show_plot=False,
            save_png=None, dz=0.7)
        assert abs(fit_pars['loc_error']) <= 1.0e-10

        # Constrain the diffusion coefficient (may throw some warnings here)
        bounds = (np.array([1.5, 0.0]), np.array([1.75, 0.1]))
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=bounds, guess=None, plot=False, show_plot=False,
            save_png=None, dz=0.7)
        assert (fit_pars["D"] >= 1.5) and (fit_pars["D"] <= 1.75)
        assert (fit_pars["loc_error"] >= 0.0) and (fit_pars["loc_error"] <= 0.1)

        # Give it a guess that lies outside of the parameter bounds. The function
        # should discard this guess and default to the middle of the bound interval
        # for each parameter.
        print("\tstability test: issuing a guess that lies outside the parameter bounds...")
        bounds = (np.array([1.0, 0.0]), np.array([3.0, 0.1]))
        guess = np.array([5.0, 0.0])
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="one_state_brownian", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=bounds, guess=guess, plot=False, show_plot=False,
            save_png=None) 
        assert abs(fit_pars["D"] - 1.9957903290177341) <= 1.0e-6
        assert abs(fit_pars["loc_error"] - 0.03925210137100024) <= 1.0e-6

        # See how it deals with pathological input: empty dataframe
        print("\tstability test: issuing empty input...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks[:0], model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, show_plot=False,
            save_png=None) 
        for k in fit_pars.keys():
            assert np.isnan(fit_pars[k])

        # For the maximum likelihood method, it should not matter if the data completely
        # lacks displacements in a given frame interval. Here, exclude all displacements
        # beyond the first frame interval.
        print("\tstability test: giving input with all data after the first " \
            "frame interval erased...")
        T = self.tracks[self.tracks["frame"] <= 1]

        # Run once, calculating jumps up to 4 frames
        fit_pars_4_frames, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            T, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, show_plot=False,
            save_png=None) 

        # Run again, only calculating the first jump
        fit_pars_1_frame, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            T, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=False, show_plot=False,
            save_png=None)
        for k in fit_pars_1_frame.keys():
            assert abs(fit_pars_1_frame[k] - fit_pars_4_frames[k]) <= 1.0e-10

        # Check that we can do plotting
        print("\tchecking plotting option...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=True, show_plot=False,
            save_png=None)

        # Check that we can save plots
        print("\tchecking the ability to save plots...")
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = core.fit_ml(
            self.tracks, model="two_state_brownian_zcorr", n_frames=4, frame_interval=0.01,
            pixel_size_um=1.0, bounds=None, guess=None, plot=True, show_plot=False,
            save_png="_test_out.png")
        for path in ["_test_out_pmf.png", "_test_out_cdf.png"]:
            assert os.path.isfile(path)
            os.remove(path)


