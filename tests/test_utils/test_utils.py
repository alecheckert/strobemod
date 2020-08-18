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

class TestCoarsenHistogram(unittest.TestCase):
    """
    Given a histogram of jump lengths, aggregate so that the result is a 
    histogram with fewer bins.

    tests:
        - strobemodels.utils.coarsen_histogram

    """
    def test_coarsen_histogram(self):
        print("\nttesting strobemodels.utils.coarsen_histogram...")

        # Get some testing data
        test_file = os.path.join(utils.FIXTURE_DIR, "one_state_infinite_plane_brownian_10000tracks_dt-0.01_D-2.0_locerror-0.035.csv")
        tracks = pd.read_csv(test_file)

        # Accumulate a jump length histogram
        H, bin_edges = utils.rad_disp_histogram_2d(tracks, n_frames=4,
            bin_size=0.001, max_jump=5.0, pixel_size_um=1.0, first_only=True)

        # Run aggregation, with 20 bins in the old histogram to 1 bin in the 
        # new one
        print("\ttest case: the aggregation factor goes evenly into the number of bins...")
        H_new, bin_edges_new = utils.coarsen_histogram(H, bin_edges, 20)
        assert H_new.shape == (4, 250)
        assert bin_edges_new.shape == (251,)

        # The first 20 bins of the original histogram should add up to the first
        # bin of the new histogram
        assert (H[:,:20].sum(axis=1) == H_new[:,0]).all()

        # Do this for a few more test cases
        for j in range(1, 250):
            assert (H[:,20*j:20*(j+1)].sum(axis=1) == H_new[:,j]).all()

        # Run aggregation, with 30 bins in the old histogram to 1 bin in the
        # new histogram. This time, we'll need to truncate the last part of 
        # the old histogram, since 30 doesn't go evenly into 5000
        print("\ttest case: the aggregation factor does not go evenly into the number of bins...")
        H_new, bin_edges_new = utils.coarsen_histogram(H, bin_edges, 30)
        assert H_new.shape == (4, 166)
        assert bin_edges_new.shape == (167,)
        for j in range(166):
            assert (H[:,30*j:30*(j+1)].sum(axis=1) == H_new[:,j]).all()

class TestTrackTools(unittest.TestCase):
    """
    Test utilities that calculate the lengths and distributions of 
    displacements for trajectories in a pandas.DataFrame.

    tests:
        - strobemodels.utils.track_length
        - strobemodels.utils.rad_disp_histogram_2d

    """
    def test_track_length(self):
        print("\ntesting strobemodels.utils.track_length...")

        # A hypothetical trajectory dataframe
        print("\ttesting for numerical correctness...")
        tracks = pd.DataFrame(index=np.arange(10), columns=["trajectory"])
        tracks["trajectory"] = np.array([0, 0, 0, 0, 1, 1, 1, 2, 3, 3])
        tracks = utils.track_length(tracks)
        testing.assert_allclose(
            np.asarray(tracks["track_length"]),
            np.array([4, 4, 4, 4, 3, 3, 3, 1, 2, 2])
        )

        # Make sure that this can be run, even if the dataframe already has
        # a "track_length" column
        print("\tchecking that track lengths are recalculated if they already exist...")
        tracks.loc[6, "trajectory"] = 2
        tracks = utils.track_length(tracks)
        testing.assert_allclose(
            np.asarray(tracks["track_length"]),
            np.array([4, 4, 4, 4, 2, 2, 2, 2, 2, 2])
        )

        # Make sure that this can be run on an empty dataframe without errors
        print("\tchecking that it doesn't fail with empty dataframes...")
        tracks = pd.DataFrame(index=[], columns=["trajectory"])
        tracks = utils.track_length(tracks)
        assert tracks.empty 

    def test_rad_disp_2d(self):
        print("\ntesting strobemodels.utils.rad_disp_2d...")

        # Some testing data
        tracks = pd.DataFrame(index=np.arange(10), columns=["y", "x", "trajectory", "frame"])
        tracks["trajectory"] = np.array([0,     0,    0,    0,   1,   1,   1,   2,   3,      3])
        tracks["frame"] =      np.array([0,     1,    2,    3,   0,   1,   2,   5,   6,      7])
        tracks["y"] =          np.array([0.0, 1.0,  1.0,  1.5, 0.0, 0.2, 0.0, 1.0, 1.5, 1.5000])
        tracks["x"] =          np.array([0.0, 0.0, -1.0, -1.0, 0.2, 0.0, 0.2, 2.7, 1.5, 1.4995])

        # Check that we get the right displacements
        print("\tchecking numerical correctness...")
        result = utils.rad_disp_2d(tracks, n_frames=4, frame_interval=0.01, pixel_size_um=1.0,
            first_only=True)
        assert (result[:,1] == 0.01).sum() == 3
        assert (result[:,1] == 0.02).sum() == 2
        assert (result[:,1] == 0.03).sum() == 1
        assert (result[:,1] == 0.04).sum() == 0
        single_disps = result[result[:,1] == 0.01, 0]
        testing.assert_allclose(
            result[result[:,1] == 0.01, 0],
            np.array([1.0, np.sqrt(2 * 0.2**2), 0.0005]),
            atol=1.0e-6
        )
        testing.assert_allclose(
            result[result[:,1] == 0.02, 0],
            np.array([np.sqrt(2), 0.0]),
            atol=1.0e-6
        )
        testing.assert_allclose(
            result[result[:,1] == 0.03, 0],
            np.array([np.sqrt(1.5**2 + 1.0)]),
            atol=1.0e-6
        )       

    def test_rad_disp_histogram_2d(self):
        print("\ntesting strobemodels.utils.rad_disp_histogram_2d...")

        # Some testing data
        tracks = pd.DataFrame(index=np.arange(10), columns=["y", "x", "trajectory", "frame"])
        tracks["trajectory"] = np.array([0,     0,    0,    0,   1,   1,   1,   2,   3,      3])
        tracks["frame"] =      np.array([0,     1,    2,    3,   0,   1,   2,   5,   6,      7])
        tracks["y"] =          np.array([0.0, 1.0,  1.0,  1.5, 0.0, 0.2, 0.0, 1.0, 1.5, 1.5000])
        tracks["x"] =          np.array([0.0, 0.0, -1.0, -1.0, 0.2, 0.0, 0.2, 2.7, 1.5, 1.4995])

        # Check for numerical correctness
        print("\tchecking numerical correctness...")
        H, bin_edges = utils.rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, max_jump=5.0,
            pixel_size_um=1.0, first_only=True)

        # Check the first frame interval
        assert H[0,:].sum() == 3
        H[0,0] == 1
        H[0,282] == 1
        H[0,1000] == 1
        assert (H[0,~(np.isin(np.arange(5000), np.array([0, 282, 1000])))] == 0).all()

        # Check the second frame interval 
        assert H[1,:].sum() == 2 
        assert H[1, 1414] == 1
        assert H[1, 0] == 1
        assert (H[1,~(np.isin(np.arange(5000), np.array([1414, 0])))] == 0).all()

        # Check the third frame interval
        assert H[2, :].sum() == 1
        assert H[2, 1802] == 1
        assert (H[2,~(np.isin(np.arange(5000), np.array([1802])))] == 0).all()

        # Check the fourth frame interval
        assert H[3,:].sum() == 0

        # Check that when using all displacements we get the right answer
        print("\tchecking numerical correctness when all displacements from each track are included...")
        H, bin_edges = utils.rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, max_jump=5.0,
            pixel_size_um=1.0, first_only=False)
        assert H[0,0] == 1
        assert H[0,282] == 2
        assert H[0,1000] == 2 
        assert H[0,500] == 1
        assert (H[0,~(np.isin(np.arange(5000), np.array([0, 282, 1000, 500])))] == 0).all()

        # Check that it acts sensibly when handed empty dataframes
        print("\tstability test: handing it empty dataframes...")
        H, bin_edges = utils.rad_disp_histogram_2d(tracks[:0], n_frames=4, bin_size=0.001, max_jump=5.0,
            pixel_size_um=1.0, first_only=False)
        assert H.shape == (4, 5000)
        assert H.sum() == 0

        # Checks that it acts sensibly when handed jump lengths that exceed the maximum
        # jump length bin
        print("\tstability test: handing it tracks with very large jumps...")
        tracks = pd.DataFrame(index=np.arange(10), columns=["y", "x", "trajectory", "frame"])
        tracks["trajectory"] = np.array([0,     0,    0,    0,   1,   1,   1,   2,   3,      3])
        tracks["frame"] =      np.array([0,     1,    2,    3,   0,   1,   2,   5,   6,      7])
        tracks["y"] =          np.array([0.0, 1.0e3,  1.0,  1.5, 0.0, 0.2, 0.0, 1.0, 1.5, 1.5000])
        tracks["x"] =          np.array([0.0, 0.0, -1.0, -1.0, 0.2, 0.0, 0.2, 2.7, 1.5, 1.4995])
        H, bin_edges = utils.rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, max_jump=5.0,
            pixel_size_um=1.0, first_only=True)
        assert H.shape == (4, 5000)
        assert H.sum() == 5
        assert H[0,:].sum() == 2
        H[0,0] == 1
        H[0,282] == 1
        assert (H[0,~(np.isin(np.arange(5000), np.array([0, 282])))] == 0).all()

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

    def test_defoc_prob_fbm(self):
        print("\ntesting strobmodels.utils.defoc_prob_fbm...")

        D = 3.5
        dt = 0.01
        hurst = 0.4
        n_frames = 8
        dz = 0.7

        # Test that the function runs at all
        print("\tdoes it even run?")
        out = utils.defoc_prob_fbm(D, hurst, n_frames, dt, dz)
        assert out.shape[0] == n_frames 

        # Check for correctness (below, *sim_result* is the result from
        # simulated data)
        print("\tchecking for numerical correctness...")
        sim_result = np.array([
            0.6656603, 0.4592361, 0.3193609, 0.2224852, 0.1551785,
            0.1082383, 0.0755337, 0.0527409
        ])
        testing.assert_allclose(out, sim_result, atol=1.0e-3)

        # Make sure it can run with just one frame interval
        print("\tstability test: running with just one frame interval...")
        out2 = utils.defoc_prob_fbm(D, hurst, 1, dt, dz)
        assert out2.shape[0] == 1
        assert np.abs(out2[0] - out[0]) <= 1.0e-10

        # Make sure it runs with zero frame intervals
        print("\tstability test: running with zero frame intervals...")
        out2 = utils.defoc_prob_fbm(D, hurst, 0, dt, dz)
        assert out2.shape[0] == 0

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
        print("\ntesting strobemodels.utils.normalize_flat_cdf...")

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
        print("\tis it correct?")
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

class TestSplineUtilities(unittest.TestCase):
    """
    Test spline utilities associated with interpolation of the fractional Brownian
    motion defocalization function.

    tests:
        - strobemodels.utils.load_fbm_defoc_spline
        - strobemodels.utils.eval_spline
        - strobemodels.utils.load_spline_coefs  # Not yet implemented
        - strobemodels.utils.load_spline_coefs_multiple_frame_interval
        - strobemodels.utils.save_spline_coefs  # Not yet implemented 
        - strobemodels.utils.save_spline_coefs_multiple_frame_interval  # Not yet implemented

    """
    def test_load_fbm_defoc_spline(self):
        """
        Note that this is also essentially a test for 
        load_spline_coefs_multiple_frame_interval.

        """
        print("\ntesting strobemodels.utils.load_fbm_defoc_spline...")

        # Mostly this is a "does it run?"-type test
        tcks = utils.load_fbm_defoc_spline(dz=0.7)

        # Output is the expected shape
        assert len(tcks) == 8
        for j in range(8):
            assert isinstance(tcks[j], tuple)
            assert len(tcks[j]) == 5

            # Orders of the cubic spline in x and y
            assert tcks[j][3] == 3
            assert tcks[j][4] == 3

        # Output is numerically correct for a few test values
        assert abs(tcks[0][2][100] - 0.010231155665098638) <= 1.0e-10
        assert abs(tcks[0][2][300] - 0.39828100158110774) <= 1.0e-10
        assert abs(tcks[0][2][500] - 0.9827226762660545) <= 1.0e-10

    def test_eval_spline(self):
        print("\ntesting strobemodels.utils.eval_spline...")

        tcks = utils.load_fbm_defoc_spline(dz=0.7)
        test_c = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
        expected = np.array([0.1934370862308236, 0.1932513816495122, 0.1926027052582681,
            0.19076036328736865, 0.18443935968392583, 0.1674346387463391])
        for i, c in enumerate(test_c):
            assert abs(expected[i] - utils.eval_spline(0.4, c, tcks[0])) <= 1.0e-8

class TestMiscUtilities(unittest.TestCase):
    """
    Unit tests for other, miscellaneous utilities in the strobemodels.utils module.

    tests:
        - strobemodels.utils.bounds_center

    """
    def test_bounds_center(self):
        print("\ntesting strobemodels.utils.bounds_center...")

        # Simple test for correctness
        print("\tchecking numerical correctness...")
        bounds = (
            np.array([0.0, 2.0]),
            np.array([1.0, 10.0])
        )
        result = utils.bounds_center(bounds, replace_inf=10.0)
        assert result.shape == (2,)
        assert result[0] == 0.5
        assert result[1] == 6.0

        # Check that it behaves correctly when the upper bound is inf 
        print("\tchecking that it behaves correctly when the upper bound is not finite...")
        bounds = (
            np.array([0.0, 2.0, 10.0]),
            np.array([1.0, 10.0, np.inf])
        )       
        result = utils.bounds_center(bounds, replace_inf=10.0)
        assert result.shape == (3,)
        assert result[0] == 0.5 
        assert result[1] == 6.0
        assert result[2] == 15.0

        # Check that it behaves correctly when run with no parameters
        print("\tstability test: does it accept empty input?")
        bounds = (np.array([]), np.array([]))
        result = utils.bounds_center(bounds, replace_inf=10.0)
        assert result.shape == (0,)

    def test_concat_tracks(self):
        print("\ntesting strobemodels.utils.concat_tracks...")

        # Some sample data
        tracks_0 = pd.DataFrame(index=np.arange(5), columns=["frame", "trajectory", "x", "y"])
        tracks_1 = pd.DataFrame(index=np.arange(7), columns=["frame", "trajectory", "x", "y"])

        tracks_0["frame"]      = [0, 1, 2, 0, 1]
        tracks_0["trajectory"] = [0, 0, 0, 1, -1]
        tracks_0["y"]          = [0.5, 1.5, 4.5, 6.5, 10.5]
        tracks_0["x"]          = [0.5, 1.5, 4.5, 6.5, 10.5]

        tracks_1["frame"]      = [0, 1, 2, 0, 1, 0, 1]
        tracks_1["trajectory"] = [0, 0, 0, 1, 1, -1, 2]
        tracks_1["y"]          = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        tracks_1["x"]          = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        tracks = utils.concat_tracks(tracks_0, tracks_1)
        for t in [-1, 0, 1, 2, 3, 4]:
            assert t in tracks["trajectory"].unique().tolist()
        assert len(tracks) == len(tracks_0) + len(tracks_1)
        testing.assert_allclose(
            np.asarray(tracks.loc[tracks["dataframe_index"] == 1, ["frame", "x", "y"]]),
            np.asarray(tracks_0[["frame", "x", "y"]]),
            rtol=1.0e-10
        )
        testing.assert_allclose(
            np.asarray(tracks.loc[tracks["dataframe_index"] == 0, ["frame", "x", "y"]]),
            np.asarray(tracks_1[["frame", "x", "y"]]),
            rtol=1.0e-10
        )

