#!/usr/bin/env python
"""
test_utils.py -- unit tests for functions in strobemodels.utils

"""
import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import unittest
from numpy import testing 

# Subjects of tests
from strobemodels import utils

class TestSupportGeneration(unittest.TestCase):
    """
    Generate a support for bivariate fitting routines.

    tests:
        - strobemodels.utils.generate_support

    """
    def test_generate_support(self):
        print("\ntesting strobemodels.utils.generate_support...")

        # Basic test
        dt = 0.01
        n_frames = 4
        jump_bins = np.arange(0.0, 0.6, 0.1)
        support = utils.generate_support(jump_bins, n_frames, dt)

        print("\tchecking shape...")
        assert support.shape == (20, 2)

        print("\tchecking numerical correctness...")
        for t in range(4):
            assert (support[t*5:(t+1)*5, 1] == (t+1) * dt).all()
            assert (np.abs(support[t*5:(t+1)*5, 0] - np.array([0.1, 0.2, 0.3, 0.4, 0.5])) <= 1.0e-10).all()

        # Test an edge case: only one frame interval
        print("\tmaking sure it runs with only one frame interval...")
        support = utils.generate_support(jump_bins, 1, dt)
        assert support.shape == (5, 2)
        assert (support[:,1] == dt).all()

        # Test an edge case: zero frame intervals
        print("\tstability test: running with zero frame intervals...")
        support = utils.generate_support(jump_bins, 0, dt)
        assert support.shape == (0, 2)

class TestBrownianDefocTools(unittest.TestCase):
    """
    Test utilities that enable the calculation of the fraction
    of Brownian molecules that defocalize in a HiLo setup.

    tests:
        - strobemodels.utils.generate_brownian_transfer_function
        - strobemodels.utils.defoc_prob_brownian

    """
    def test_generate_brownian_transfer_function(self):
        print("\nttesting strobemodels.utils.generate_brownian_transfer_function...")

        D = 0.01
        dt = 0.01
        support = np.arange(-2.5, 2.501, 0.001)
        transfer_func = utils.generate_brownian_transfer_function(support, D, dt)
        assert transfer_func.shape[0] == (support.shape[0]//2 + 1)

        # Compare to the analytical characteristic function for the 1D displacements
        # of a Brownian motion
        k = np.fft.rfftfreq(support.shape[0], d=(support[1]-support[0]))
        cf = np.exp(-D * dt * ((k*np.pi*2)**2))

        print("\tcomparing to the analytical C.F...")
        testing.assert_allclose(np.abs(transfer_func), np.abs(cf), atol=1.0e-10)

    def test_defoc_prob_brownian(self):
        print("\ntesting strobemodels.utils.defoc_prob_brownian...")

        D = 3.5
        dt = 0.01
        n_frames = 9
        dz = 0.7

        # Test that it runs at all
        print("\tdoes it even run?")
        out = utils.defoc_prob_brownian(D, n_frames, dt, dz)
        assert out.shape[0] == n_frames 

        # Check for correctness (below, *sim_result* is from simulated data)
        print("\tchecking for numerical correctness...")
        sim_result = np.array([
            0.699356,
            0.498955,
            0.356289,
            0.254413,
            0.181720,
            0.129852,
            0.092796,
            0.066345,
            0.047383
        ])
        testing.assert_allclose(out, sim_result, atol=1.0e-3)

        # Make sure it runs with just one frame interval
        print("\tstability test: running with just one frame interval...")
        out2 = utils.defoc_prob_brownian(D, 1, dt, dz)
        assert out2.shape[0] == 1
        assert np.abs(out2[0] - out[0]) <= 1.0e-10

        # Make sure it runs with zero frame intervals
        print("\tstability test: running with zero frame intervals...")
        out = utils.defoc_prob_brownian(D, 0, dt, dz)
        assert out.shape[0] == 0

class TestNormalizationMethods(unittest.TestCase):
    """
    Test methods that normalize PDFs.

    tests:
        - strobemodels.utils.normalize_pmf
        - strobemodels.utils.normalize_flat_cdf 
        - strobemodels.utils.radnorm

    """
    def test_normalize_pmf(self):
        print("\ntesting strobemodels.utils.normalize_pmf...")

        r = np.linspace(0.0, 5.0, 5001)
        D = 3.5
        dt = 0.01

        # Normalize a 1D PMF
        print("\ttesting 1D case...")
        pmf = r * np.exp(-(r**2) / (4 * D * dt))
        pmf_norm = utils.normalize_pmf(pmf)
        assert abs(pmf_norm.sum() - 1.0) <= 1.0e-8

        # If all zeros, return zeros
        print("\ttesting empty distribution...")
        pmf_norm = utils.normalize_pmf(np.zeros(r.shape[0]))
        assert (pmf_norm == 0.0).all()

        # Normalize a function of both space and time on each timepoint
        print("\ttesting 2D case...")
        result = np.empty((4, r.shape[0]))
        for t in range(4):
            result[t,:] = r * np.exp(-(r**2) / (4 * D * dt * (t + 1)))
        result_norm = utils.normalize_pmf(result)
        assert np.abs((result_norm.sum(axis=1) - 1.0)<1.0e-8).all()

    def test_normalize_flat_cdf(self):
        print("\ttesting strobemodels.utils.normalize_flat_cdf...")

        # A hypothetical CDF and a finite support
        rt_tuples = np.array([
            [0.1, 0.01],
            [0.2, 0.01],
            [0.3, 0.01],
            [0.1, 0.02],
            [0.2, 0.02],
            [0.3, 0.02]
        ])
        cdf = np.array([0.3, 0.6, 0.9, 0.2, 0.4, 0.6])

        result = utils.normalize_flat_cdf(rt_tuples, cdf)
        expected = np.array([1.0/3, 2.0/3, 1.0])
        assert (np.abs(result[:3] - expected) < 1.0e-10).all()
        assert (np.abs(result[3:] - expected) < 1.0e-10).all()

    def test_radnorm(self):
        print("\ntesting strobemodels.utils.radnorm...")

        # Test case: normalizing a Gaussian random variable into a Rayleigh
        # distribution
        D = 3.5
        dt = 0.01
        r = np.linspace(0.0, 5.0, 5001)

        # 1D Gaussian PDF 
        g = np.exp(-(r**2) / (4 * D * dt))
        g /= g.sum()

        # Expected corresponding Rayleigh distribution
        ray_exp = r * np.exp(-(r**2) / (4 * D * dt))
        ray_exp /= ray_exp.sum()

        # Normalize in 2D
        print('\ttest case: Rayleigh distribution...')
        ray_obs = utils.radnorm(r, g, d=2)
        testing.assert_allclose(ray_obs, ray_exp, atol=1.0e-8)

        # Normalize in 3D
        print("\ttest case: Maxwell-Boltzmann distribution...")

        max_exp = (r**2) * np.exp(-(r**2) / (4 * D * dt))
        max_exp /= max_exp.sum()
        max_obs = utils.radnorm(r, g, d=3)
        testing.assert_allclose(max_exp, max_obs, atol=1.0e-8)

class TestCharacteristicFunctionUtilities(unittest.TestCase):
    """
    Test methods that estimate the PDF, given the characteristic 
    function for a random variable.

    tests:
        - strobemodels.utils.pdf_from_cf
        - strobemodels.utils.pdf_from_cf_rad

    """
    def test_pdf_from_cf(self):
        print("\ntesting strobemodels.utils.pdf_from_cf...")

        # Characteristic function for a 1D Gaussian random
        # variable
        print("\ttest case: 1D Gaussian random variable...")
        r = np.linspace(-2.5, 2.5, 5001)
        D = 3.5
        dt = 0.01
        func_cf = lambda x: np.exp(-D * dt * (x**2))

        # The expected result for the PDF
        sig2 = 2 * D * dt 
        pdf_real = np.exp(-(r**2) / (2 * sig2))
        pdf_real = pdf_real / pdf_real.sum()

        # Generate the PDF for a Gaussian random variable in 1D
        # from its characteristic function
        pdf_gen = utils.pdf_from_cf(func_cf, r)

        # Test for closeness
        testing.assert_allclose(pdf_real, pdf_gen, atol=1.0e-4)

        # Characteristic function for a 1D Cauchy random variable
        print("\ttest case: 1D Cauchy random variable...")
        func_cf = lambda x: np.exp(-D * dt * np.abs(x))
        pdf_gen = utils.pdf_from_cf(func_cf, r)

        pdf_real = 1.0 / (D * dt * np.pi * (1 + (r/(D*dt))**2))
        pdf_real = pdf_real / pdf_real.sum()

        testing.assert_allclose(pdf_real, pdf_gen, atol=1.0e-4)
        

    def test_pdf_from_cf_rad(self):
        print("\ntesting strobemodels.utils.pdf_from_cf_rad...")

        # Radial characteristic function for a symmetric 3D Gaussian
        # random vector
        print("\ttest case: symmetric 3D Gaussian random variable...")

        D = 3.5
        dt = 0.01

        # Real-space radial support
        r = np.linspace(0.0, 5.0, 5001)

        # Characteristic function
        func_cf = lambda rho: np.exp(-D * dt * (rho**2))

        # Analytical PDF (Maxwell-Boltzmann distribution)
        pdf_real = (r**2) * np.exp(-(r**2) / (4 * D * dt))
        pdf_real = pdf_real / pdf_real.sum()

        # PDF from the CF
        pdf_gen = utils.pdf_from_cf_rad(func_cf, r)
        pdf_gen = utils.radnorm(r, pdf_gen, d=3)

        testing.assert_allclose(pdf_real, pdf_gen, atol=1.0e-5)





