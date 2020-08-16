#!/usr/bin/env python
"""
test_plot.py -- unit tests for strobemodels.plot

"""
import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Testing utilities
import unittest
from numpy import testing 

# Directory with testing data
from strobemodels.utils import FIXTURE_DIR

# Core fitting utility
from strobemodels.core import fit_model_cdf 

# Functions to test 
from strobemodels import plot 

# Some help utilities
from strobemodels.utils import (
    coarsen_histogram
)

class TestPMFCDFPlots(unittest.TestCase):
    """
    tests:
        - strobemodels.plot.plot_jump_length_pmf
        - strobemodels.plot.plot_jump_length_cdf

    """
    def test_plot_jump_length_pmf(self):
        print("\ntesting strobemodels.plot.plot_jump_length_pmf...")

        # Load some testing data
        test_path = os.path.join(FIXTURE_DIR, "one_state_infinite_plane_brownian_10000tracks_dt-0.01_D-2.0_locerror-0.035.csv")
        tracks = pd.read_csv(test_path)

        # Fit to a one-state Brownian model
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = fit_model_cdf(
            tracks, model="one_state_brownian", frame_interval=0.01,
            pixel_size_um=1.0)

        # Aggregate the jumps for the histogram
        coarse_pmfs, coarse_bin_edges = coarsen_histogram(pmfs, bin_edges, 20)

        # Some kwargs to pass to the function
        kwargs = {'frame_interval': 0.01, 'max_jump': 2.0, 'cmap': 'gray', 
            'figsize_mod': 1.0}

        # Make sure it runs
        print("\ttesting with only experimental PMFs...")
        fig, axes = plot.plot_jump_length_pmf(coarse_bin_edges, coarse_pmfs, model_pmfs=None,
            model_bin_edges=None, out_png=None, **kwargs)
        plt.close('all')

        # Try to run with model overlay
        print("\ttesting with model overlay...")
        fig, axes = plot.plot_jump_length_pmf(coarse_bin_edges, coarse_pmfs, model_pmfs=model_pmfs,
            model_bin_edges=bin_edges, out_png=None, **kwargs)
        plt.close('all')

        # Try to save 
        print("\ttesting with plot saving...")
        fig, axes = plot.plot_jump_length_pmf(coarse_bin_edges, coarse_pmfs, model_pmfs=model_pmfs,
            model_bin_edges=bin_edges, out_png="_test_out.png", **kwargs)
        plt.close('all')
        os.remove("_test_out.png")

    def test_plot_jump_length_cdf(self):
        print("\ntesting strobemodels.plot.plot_jump_length_cdf...")

        # Load some testing data
        test_path = os.path.join(FIXTURE_DIR, "one_state_infinite_plane_brownian_10000tracks_dt-0.01_D-2.0_locerror-0.035.csv")
        tracks = pd.read_csv(test_path)

        # Fit to a one-state Brownian model
        fit_pars, bin_edges, cdfs, pmfs, model_cdfs, model_pmfs = fit_model_cdf(
            tracks, model="one_state_brownian", frame_interval=0.01,
            pixel_size_um=1.0)

        # Some kwargs
        kwargs = {'frame_interval': 0.01, 'max_jump': 5.0, 'cmap': 'gray', 'figsize_mod': 1.0,
            'fontsize': 8}

        # Make sure it runs
        print("\ttesting with only experimental CDFs...")
        fig, axes = plot.plot_jump_length_cdf(bin_edges, cdfs, model_cdfs=None,
            model_bin_edges=None, out_png=None, **kwargs)
        plt.close('all')

        # Try to run with model overlay
        print("\ttesting with model overlay...")
        fig, axes = plot.plot_jump_length_cdf(bin_edges, cdfs, model_cdfs=model_cdfs,
            model_bin_edges=bin_edges, out_png=None, **kwargs)
        plt.show(); plt.close('all')

        # Try to save
        print("\ttesting with plot saving...")
        fig, axes = plot.plot_jump_length_cdf(bin_edges, cdfs, model_cdfs=model_cdfs,
            model_bin_edges=bin_edges, out_png="_test_out_cdf.png", **kwargs)
        plt.close('all')
        os.remove("_test_out_cdf.png")
