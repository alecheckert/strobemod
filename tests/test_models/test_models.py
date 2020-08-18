#!/usr/bin/env python
"""
test_models.py -- unit tests for diffusion models. These tests are mostly
for the numerical accuracy of these models - they compare the result of 
the model on a past run and look for equivalence. They also test a few
edge cases - when the diffusion coefficient is zero and so on.

"""
import os
import sys

# Numeric
import numpy as np 
import pandas as pd 

# Testing utilities
import unittest
from numpy import testing 

# Functions to test 
from strobemodels import models 

# Utilities
from strobemodels import utils 
FIXTURE_DIR = utils.FIXTURE_DIR 

# Models to test
from strobemodels.models import (
    CDF_MODELS,
    PDF_MODELS
)

# The parameter sets to use for each model
MODEL_PARAMETERS = {
    "one_state_brownian": (3.5, 0.035),
    "one_state_fbm": (0.4, 3.5, 0.035, 0.01),
    "two_state_brownian": (0.3, 0.001, 2.5, 0.035),
    "two_state_brownian_zcorr": (0.3, 0.001, 2.5, 0.035),
    "two_state_fbm": (0.4, 0.3, 0.001, 3.0, 0.035),
    "two_state_fbm_zcorr": (0.4, 0.3, 0.001, 3.0, 0.035),
    "three_state_brownian": (0.3, 0.3, 0.001, 0.5, 8.0, 0.035),
    "three_state_brownian_zcorr": (0.3, 0.3, 0.001, 0.5, 8.0, 0.035),
}

# Testing absolute numerical tolerance
tolerance = 1.0e-10

class TestAllModels(unittest.TestCase):
    """
    Compare the numerical output for each of the diffusion models against 
    a past run, and also run on some edge cases to check for stability.

    tests:
        all CDF and PDF model functions in strobemodels.models

    """
    def test_numeric_output(self):
        print("\ntesting all diffusion models for numerical correctness...")

        # The set of (r, dt) tuples to evaluate the models on 
        jump_bins = np.linspace(0.0, 5.0, 5001)
        n_frames = 8
        frame_interval = 0.01
        rt_tuples = utils.generate_support(jump_bins, n_frames, frame_interval)

        # The models to test
        models = list(MODEL_PARAMETERS.keys())

        for model in models:
            print("\tchecking {}...".format(model))

            # Evaluate the model
            cdf = CDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            pdf = PDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])

            # Load the correct result
            data_file = os.path.join(FIXTURE_DIR, "test_numeric_output_{}.csv".format(model))
            expected = pd.read_csv(data_file)
            exp_cdf = np.asarray(expected["cdf"])
            exp_pdf = np.asarray(expected["pdf"])

            # # To rebenchmark
            # df = pd.DataFrame(index=np.arange(len(cdf)), columns=["cdf", "pdf", "r", "dt"])
            # df["cdf"] = cdf 
            # df["pdf"] = pdf 
            # df["r"] = rt_tuples[:,0]
            # df["dt"] = rt_tuples[:,1]
            # df.to_csv(data_file, index=False)

            # Compare results
            testing.assert_allclose(cdf, exp_cdf, atol=tolerance)
            testing.assert_allclose(pdf, exp_pdf, atol=tolerance)

    def test_edge_cases(self):
        print("\nstability testing on all diffusion models with some edge cases...")

        # The set of models to test
        models = list(MODEL_PARAMETERS.keys())

        def is_sanitary(arraylike):
            A = np.asarray(arraylike)
            return np.logical_and((~np.isinf(A)), (~np.isnan(A))).all()

        # Make sure that the models run with only a single frame interval
        print("\tchecking that the models run with only a single frame interval...")
        jump_bins = np.linspace(0.0, 5.0, 5001)
        n_frames = 1
        frame_interval = 0.01
        rt_tuples = utils.generate_support(jump_bins, n_frames, frame_interval)
        for model in models:
            cdf = CDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            pdf = PDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            assert is_sanitary(cdf)
            assert is_sanitary(pdf)

        # See what happens when all of the displacements passed are zero
        print("\tpassing pathological input: all zero displacements...")
        rt_tuples = np.zeros((5000, 2), dtype=np.float64)
        rt_tuples[:2500,1] = 0.005
        rt_tuples[2500:,1] = 0.010
        for model in models:
            cdf = CDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            pdf = PDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            assert is_sanitary(cdf)
            assert is_sanitary(pdf)

        # See what happens when all of the timepoints are zero
        print("\tpassing pathological input: all zero timepoints...")
        rt_tuples = np.zeros((5000, 2), dtype=np.float64)
        rt_tuples[:,0] = np.arange(0.0, 5.0, 0.001)
        for model in models:
            cdf = CDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            pdf = PDF_MODELS[model](rt_tuples, *MODEL_PARAMETERS[model])
            assert is_sanitary(cdf)
            assert is_sanitary(pdf)       

