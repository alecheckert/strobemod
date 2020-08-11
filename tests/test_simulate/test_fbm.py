#!/usr/bin/env python
"""
test_fbm.py -- unit tests for strobemodels.simulate.fbm

"""
import os
import sys

# Numeric
import numpy as np

# Testing
import unittest
from numpy import testing 

# Functions to test
from strobemodels.simulate import fbm 

# Numerical tolerance
tolerance = 1.0e-10

def assert_close(a, b):
    return abs(a-b) <= tolerance

def assert_allclose(a, b, rtol=tolerance):
    testing.assert_allclose(a, b, rtol=rtol)

class TestMultivariateNormalRandomVector(unittest.TestCase):
    """
    Unit tests for the class
        strobemodels.simulate.fbm.MultivariateNormalRandomVector

    """
    def setUp(self):
        # Covariance matrix for a simple two-step Wiener process
        self.C = np.array([[2.0, 2], [2, 4]])

        # Covariance matrix for a three-step Wiener process
        self.C3 = np.array([[1.0, 1, 1], [1, 2, 2], [1, 2, 3]])

        # Generate the random vector objects
        self.X = fbm.MultivariateNormalRandomVector(self.C)
        self.X3 = fbm.MultivariateNormalRandomVector(self.C3)

    def test_MultivariateNormalRandomVector_methods(self):
        print("\ntesting strobemodels.simulate.fbm.MulivariateNormalRandomVector...")

        # For convenience; testing data
        C = self.C 
        C3 = self.C3 
        X = self.X 
        X3 = self.X3 

        # Rank check
        print("\tchecking rank...")
        assert X.N == 2

        # Check that we can simulate the random vector
        print("\tchecking random vector generation...")
        vecs = X(10)
        assert vecs.shape == (10, 2)

        # Inverse check
        print("\tchecking inverses...")
        assert_allclose(
            X.inv_covariance @ C,
            np.identity(2)
        )

        # Determinant check
        print("\tchecking determinants...")
        assert_close(X.det_covariance, 4.0)

        # Cholesky decomposition check
        print("\tchecking Cholesky decomposition...")
        assert_allclose(
            C,
            X.cholesky @ X.cholesky.T 
        )
        assert_close(X.cholesky[0,0], np.sqrt(2))
        assert_close(X.cholesky[1,0], np.sqrt(2))
        assert_close(X.cholesky[1,1], np.sqrt(2))
        assert_close(X.cholesky[0,1], 0)

        # PDF check -- make sure we can evaluate the PDF
        # on any set of random vectors
        # print("\tchecking PDF...")

        # Check linear transformations
        print("\tchecking linear transformations...")
        A = np.array([[2, 0], [0, 1]])
        Y = X.linear_transformation(A)
        assert_allclose(Y.C, A.T @ X.C @ A)
        assert_allclose(Y.mean, np.array([0, 0]))

        # Check addition of random variables
        Z = X.add(Y)
        assert_allclose(Z.C, X.C + Y.C)
        assert_allclose(Z.mean, np.array([0, 0]))

        # Check marginalization
        print("\tchecking marginalization...")

        # Marginalize on the second position
        Y = X.marginalize(1)
        assert_allclose(Y.C, np.array([[2.]]))

        # Marginalize on the first position
        Y = X.marginalize(0)
        assert_allclose(Y.C, np.array([[4.]]))

        # For a 3-step Wiener process, marginalize on the third position
        Y = X3.marginalize(2)
        assert_allclose(Y.C, C3[:2,:2])

        # For a 3-step Wiener process, marginalize on both the 
        # second and third positions
        Y = X3.marginalize([1,2])
        assert_allclose(Y.C, np.array([[1.]]))

        # Check conditioning
        print("\tchecking conditioning...")
        
        # For a 2-step Wiener process, condition the first
        # position on the second position, holding the latter
        # to be 0.0
        Y = X.condition(1, 0.0)
        assert_allclose(Y.C, np.array([[1.]]))
        assert_allclose(Y.mean, np.array([0.]))

        # If we condition on a value other than 0, the mean 
        # should in general not be equal to 0
        Y = X.condition(1, 1.0)
        assert_allclose(Y.mean, np.array([0.5]))
        assert_allclose(Y.C, np.array([[1.]]))

        # Condition a 3-step Wiener process on the second step
        Y = X3.condition(1, 1.0)
        assert_allclose(Y.mean, np.array([0.5, 1.]))
        assert_allclose(Y.C, np.array([[0.5, 0], [0, 1.]]))
        
class TestFractionalBrownianMotion(unittest.TestCase):
    """
    Unit tests for the class
        strobemodels.simulate.fbm.FractionalBrownianMotion

    """
    def setUp(self):

        # Regular Brownian motion
        self.BM = fbm.FractionalBrownianMotion(track_len=2, 
            hurst=0.5, D=1.0, dt=0.01, D_type=1)

        # Subdiffusive FBM of type I
        self.FBM = fbm.FractionalBrownianMotion(track_len=2,
            hurst=0.4, D=1.0, dt=0.01, D_type=1)

        # Subdiffusive FBM of type II
        self.FBM2 = fbm.FractionalBrownianMotion(track_len=2,
            hurst=0.4, D=1.0, dt=0.01, D_type=2)

    def test_FractionalBrownianMotion(self):
        print("\ntesting strobemodels.simulate.fbm.FractionalBrownianMotion...")

        # Make sure that the covariance matrices are correct
        print("\tchecking covariance matrices are properly generated...")
        assert_allclose(
            np.array([[1, 1], [1, 2]]) * self.BM.D * 2 * self.BM.dt,
            self.BM.C
        )
        assert_allclose(
            np.array([
                [0.05023773, 0.04373448],
                [0.04373448, 0.08746897]
            ]),
            self.FBM.C,
            rtol=1.0e-6
        )
        assert_allclose(
            np.array([[0.05023773, 0.04373448],
                      [0.04373448, 0.08746897]]),
            self.FBM2.C,
            rtol=1.0e-6
        )

        # Simulate some trajectories
        print("\tchecking ability to simulate trajectories...")
        tracks = self.FBM(20)
        assert tracks.shape == (20, self.FBM.track_len)

        # Check the method to return times
        print("\tchecking get_time()...")
        times = self.BM.get_time()
        assert len(times) == self.BM.track_len
        assert_allclose(
            np.arange(self.BM.track_len) * self.BM.dt,
            times
        )

        # Should not depend on type of motion
        assert_allclose(
            self.BM.get_time(),
            self.FBM.get_time()
        )

class TestBrownianMotion(unittest.TestCase):
    """
    Unit tests for the class
        strobemodels.simulate.fbm.BrownianMotion

    This is mostly a wrapper on FractionalBrownianMotion,
    so the tests are minimal.

    """
    def setUp(self):
        self.BM = fbm.BrownianMotion(track_len=10, D=3.5, dt=0.01)
    
    def test_BrownianMotion(self):
        print("\ntesting strobemodels.simulate.fbm.BrownianMotion...")

        # Check covariance matrix
        print("\tchecking covariance matrix is properly generated...")
        C = np.minimum(*(np.indices((self.BM.track_len, self.BM.track_len))+1)) * self.BM.dt * 2 * self.BM.D 
        assert_allclose(
            C,
            self.BM.C
        )

class TestFractionalBrownianMotion3D(unittest.TestCase):
    """
    Unit tests for the class
        strobemodels.simulate.fbm.FractionalBrownianMotion3D

    This is mostly a wrapper on FractionalBrownianMotion,
    so tests are minimal.

    """
    def setUp(self):
        self.FBM3D = fbm.FractionalBrownianMotion3D(
            track_len=10,
            hurst=0.5,
            D=3.5,
            dt=0.01,
        )

    def test_FractionalBrownianMotion3D(self):
        print("\ntesting strobemodels.simulate.fbm.FractionalBrownianMotion3D...")
 
        # Make sure that the underlying FBM object is good
        print("\tchecking covariance matrix...")
        C = np.minimum(*(np.indices((10, 10))+1)) * 0.01 * 3.5 * 2
        assert_allclose(
            C,
            self.FBM3D.fbm.C
        )

        # Make sure that we can simulate 3D trajectories
        print("\tchecking simulation of 3D trajectories...")
        tracks = self.FBM3D(100)
        assert tracks.shape == (100, 10, 3)
        


        
    
