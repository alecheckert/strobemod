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

    The most important tests here are for the Brownian motion and Cauchy motion cases
    (corresponding to alpha = 2 and alpha = 1 respectively), for which we have analytical
    expressions for the 2D and 3D radial jump length distributions. 

    These are:

        Brownian motion, 2D radial jumps (Rayleigh distribution)

            R ~ r exp(-r^2 / (4 * D * t))

        Brownian motion, 3D radial jumps (Maxwell-Boltzmann distribution):

            R ~ r^2 exp(-r^2 / (4 * D * t))

        Cauchy motion, 2D radial jumps

            f(r) = D * t * r / ( (D * t)^2 + r^2)^{3/2}

        Cauchy motion, 3D radial jumps

            f(r) = 4 * D * t * r^2 / ( pi * ((Dt)^2 + r^2)^2 )

    """
    def setUp(self):

        self.max_R = 10.0
        self.bin_size = 0.001
        self.scale = 1.0
        self.dt = 0.01
        self.track_len = 4

    def test_bm(self):
        print("\ntesting strobemodels.simulate.levy.LevyFlight3D with a Brownian motion model...")

        # The three different simulation methods for levy.LevyFlight3D
        methods = ["hankel", "radon", "radon_alt"]

        for method in methods:
            print("\trunning tests for simulation method '%s'..." % method)

            # Generate the simulator
            BM = levy.LevyFlight3D(self.max_R, bin_size=self.bin_size, alpha=2.0,
                D=self.scale, dt=self.dt, method=method, track_len=self.track_len)

            # Check that a characteristic function corresponding to Brownian motion
            # is actually generated
            print("\t\tchecking characteristic function...")
            k = np.logspace(-3.0, 3.0, 1001)
            expected_cf = np.exp(-((self.scale * k)**2) * self.dt)
            assert_allclose(
                expected_cf,
                BM.cf(k)
            )

            # Check the CDF for correctness (using the Maxwell-Boltzmann CDF)
            print("\t\tchecking CDF for correctness...")
            r = np.linspace(0.0, 5.0, 5001)
            expected_cdf = erf(r/np.sqrt(4*self.dt)) - np.sqrt(2/np.pi) * \
                r * np.exp(-(r**2)/(4*self.dt)) / np.sqrt(2*self.dt)
            expected_cdf = expected_cdf / expected_cdf[-1]
            # obs_cdf = BM.cdf(r-0.0005)
            obs_cdf = BM.cdf(r)
            print("\t\tmaximum CDF deviation from analytical case: %f" % np.abs(obs_cdf-expected_cdf).max())
            testing.assert_allclose(
                expected_cdf,
                obs_cdf,
                atol=1.0e-5
            )

            # Check the PDF for correctness (should be a Maxwell-Boltzmann 
            # distribution)
            print("\t\tchecking PDF for correctness....")
            r = np.linspace(0.0, 5.0, 5001)
            expected_pdf = (r**2) * np.exp(-(r**2) / (4 * self.scale * self.dt))
            expected_pdf = expected_pdf / expected_pdf.sum()
            obs_pdf = BM.pdf(r)
            obs_pdf = obs_pdf / obs_pdf.sum()
            testing.assert_allclose(
                expected_pdf,
                obs_pdf,
                atol=1.0e-5  # max difference ~1.0e-6
            )

            # Check that we can return the CDF first derivative (no numerical checks
            # here, just make sure it runs)
            print("\t\tchecking that we can get the CDF first derivative...")
            cdf_dev = BM.cdf_dev(r)
            assert cdf_dev.shape == r.shape 

            # Check that the inverse CDF works. This mostly checks that this function
            # runs at all.
            print("\t\tchecking inverse CDF sampling...")
            p = np.arange(0.0, 1.001, 0.001)
            inv_cdf = BM.inverse_cdf(p, n_iter=20)

            # Check this against the "real" CDF by spline interpolation
            sp_0 = spline(r, obs_cdf)
            sp_1 = spline(inv_cdf, p)
            A0 = sp_0(inv_cdf)
            A1 = sp_1(inv_cdf)
            testing.assert_allclose(
                A0,
                A1,
                atol=1.0e-4
            )

            # Check that we can simulate Levy flights with this method.
            print("\t\ttesting simulation capabilities...")

            # Simulate 100 trajectories
            tracks = BM(100)
            assert tracks.shape == (100, 4, 3)

    def test_cauchy(self):
        print("\ntesting strobemodels.simulate.levy.LevyFlight3D with a Cauchy motion model...")

        # Each of the three simulation methods in levy.LevyFlight3D
        methods = ["hankel", "radon", "radon_alt"]

        for method in methods:
            print("\trunning tests for simulation method %s..." % method)

            # Generate the simulator object
            CW = levy.LevyFlight3D(self.max_R, bin_size=self.bin_size, alpha=1.0,
                 D=self.scale, dt=self.dt, method=method, track_len=self.track_len)

            # Check that a characteristic function corresponding to a Cauchy motion
            # is actually generated
            print("\t\tchecking characteristic function...")
            k = np.logspace(-3.0, 3.0, 1001)
            expected_cf = np.exp(-np.abs(self.scale * k) * self.dt)
            assert_allclose(
                expected_cf,
                CW.cf(k)
            )

            # Check that PDF/CDF generation runs (no numerical tests here; the distribution
            # does not have a closed form)
            print("\t\tchecking generation of PDFs/CDFs...")
            r = np.linspace(0.0, 5.0, 5001)
            cdf = CW.cdf(r)

            # Check the PDF against the analytical case
            print("\t\tchecking PDF for numerical correctness...")
            pdf = CW.pdf(r)
            pdf_ana = (r**2) / ((self.scale * self.dt)**2 + r**2)**2
            pdf_ana /= pdf_ana.sum()
            print("\t\tmaximum PDF deviation from analytical case: %f" % np.abs(pdf-pdf_ana).max())
            testing.assert_allclose(
                pdf,
                pdf_ana,
                atol=1.0e-4
            )

            # Check that we can generate sample trajectories
            print("\t\tchecking random trajectory sampling...")
            tracks = CW(100)
            assert tracks.shape == (100, 4, 3)




