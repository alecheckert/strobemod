#!/usr/bin/env python
"""
test_levy.py -- unit tests for strobemodels.simulate.levy

"""
import os
import sys
import matplotlib.pyplot as plt 

# Numeric
import numpy as np
from scipy.special import erf 

# Spline interpolation
from scipy.interpolate import InterpolatedUnivariateSpline as spline 

# Testing 
import unittest
from numpy import testing 

# Functions to test
from strobemodels.simulate import levy 

# Numeric tolerance
tolerance = 1.0e-10

def assert_close(a, b):
    return abs(a-b) <= tolerance

def assert_allclose(a, b, rtol=tolerance):
    testing.assert_allclose(a, b, rtol=rtol)

class TestLevyFlight3D(unittest.TestCase):
    """
    Unit tests for the class
        strobemodels.simulate.levy.LevyFlight3D

    """
    def setUp(self):

        self.scale = 1.0
        self.dt = 0.01

        # Equivalent to 3D Brownian motion with diffusion coefficient 1.0 um^2 s^-1
        self.BM = levy.LevyFlight3D(
            2.0,
            self.scale,
            dt=self.dt,
            track_len=4,
            interpolation_range=(0.0, 10.0, 10001)
        )

        # Equivalent to 3D Cauchy walk
        self.CW = levy.LevyFlight3D(
            1.0,
            self.scale,
            dt=self.dt,
            track_len=4,
            interpolation_range=(0.0, 10.0, 10001)
        )

    def test_bm(self):
        print("\ntesting strobemodels.simulate.levy.LevyFlight3D with a Brownian motion model...")

        # Check that a characteristic function corresponding to Brownian motion
        # is actually generated
        print("\tchecking characteristic function...")
        k = np.logspace(-3.0, 3.0, 1001)
        expected_cf = np.exp(-((self.scale * k)**2) * self.dt)
        assert_allclose(
            expected_cf,
            self.BM.cf(k)
        )

        # Check the CDF for correctness (using the Maxwell-Boltzmann CDF)
        print("\tchecking CDF for correctness...")
        r = np.linspace(0.0, 5.0, 5001)
        expected_cdf = erf(r/np.sqrt(4*self.dt)) - np.sqrt(2/np.pi) * \
            r * np.exp(-(r**2)/(4*self.dt)) / np.sqrt(2*self.dt)
        expected_cdf = expected_cdf / expected_cdf[-1]
        obs_cdf = self.BM.cdf_rad(r)
        obs_cdf = obs_cdf / obs_cdf[-1]
        testing.assert_allclose(
            expected_cdf,
            obs_cdf,
            atol=1.0e-2   # max difference 0.002; this is rather high
        )

        # Check the PDF for correctness (should be a Maxwell-Boltzmann 
        # distribution)
        print("\tchecking PDF for correctness....")
        r = np.linspace(0.0, 5.0, 5001)
        expected_pdf = (r**2) * np.exp(-(r**2) / (4 * self.scale * self.dt))
        expected_pdf = expected_pdf / expected_pdf.sum()
        obs_pdf = self.BM.pdf_rad(r)
        obs_pdf = obs_pdf / obs_pdf.sum()
        testing.assert_allclose(
            expected_pdf,
            obs_pdf,
            atol=1.0e-5  # max difference ~1.0e-6
        )

        # Check that we can return the CDF first derivative (no numerical checks
        # here, just make sure it runs)
        print("\tchecking that we can get the CDF first derivative...")
        cdf_dev = self.BM.cdf_rad_dev(r)
        assert cdf_dev.shape == r.shape 

        # Check that the inverse CDF works. This mostly checks that this function
        # runs at all.
        print("\tchecking inverse CDF sampling...")
        p = np.arange(0.0, 1.001, 0.001)
        inv_cdf = self.BM.inverse_cdf(p, n_iter=20)

        # Check this against the "real" CDF by spline interpolation
        sp_0 = spline(r, obs_cdf)
        sp_1 = spline(inv_cdf, p)
        A0 = sp_0(inv_cdf)
        A1 = sp_1(inv_cdf)
        testing.assert_allclose(
            A0,
            A1,
            atol=1.0e-2
        )

        # Check that we can simulate Levy flights with this method.
        print("\ttesting simulation capabilities...")

        # Simulate 100 trajectories
        tracks = self.BM(100)
        assert tracks.shape == (100, 4, 3)


    def test_cauchy(self):
        print("\ntesting strobemodels.simulate.levy.LevyFlight3D with a Cauchy motion model...")

        # Check that a characteristic function corresponding to a Cauchy motion
        # is actually generated
        print("\tchecking characteristic function...")
        k = np.logspace(-3.0, 3.0, 1001)
        expected_cf = np.exp(-np.abs(self.scale * k) * self.dt)
        assert_allclose(
            expected_cf,
            self.CW.cf(k)
        )

        # Check that PDF/CDF generation runs (no numerical tests here; the distribution
        # does not have a closed form)
        print("\tchecking generation of PDFs/CDFs...")
        r = np.linspace(0.0, 5.0, 5001)
        cdf = self.CW.cdf_rad(r)

        # Check that we can generate sample trajectories
        print("\tchecking random trajectory sampling...")
        tracks = self.CW(100)
        assert tracks.shape == (100, 4, 3)




