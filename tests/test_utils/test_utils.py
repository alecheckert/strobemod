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

class TestNormalizationMethods(unittest.TestCase):
    """
    Test methods that normalize PDFs.

    tests:
        - strobemodels.utils.normalize_pmf
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
        print("\ttest case: Maxwell-Boltmann distribution...")

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





