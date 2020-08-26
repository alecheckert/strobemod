#!/usr/bin/env python
"""
sim_abel.py -- generate a numerical projection matrix that accomplishes the 
Abel transform on a predefined support.

The basic idea is this: we have a probability mass function f_{R3} defined on 
some set of bins R. Given that the PMF defines the 3D radial displacements
of a radially symmetric random vector R3, we'd like to get the corresponding 
distribution of radial displacements of R3 when it is projected onto a plane
that intersects the origin. Call this f_{R2}. 

Then there exists a matrix A such that the PMF of the 2D radial displacements,
defined on the same support R, is 

    f_{R2} = A . f_{R3}

    (where "." is the matrix dot product)

The purpose of this module is to approximate the matrix *A* using a Monte 
Carlo method.

Approach. We start with *R*, defined as a random variable such that (1) R lies
between B0 and B1 and (2) it is the modulus of a 3D random vector with 
uniform probability density in Euclidean space, provided (1) is satisfied. Then
the probability density of R goes as 

    f_{R}(r) ~ r^2  if  r in [B0, B1], 0 otherwise

which generates the inverse CDF sampling method

    r = F_{R}^{-1} = ( B0^3 + (B1^3 - B0^3) * p )^{1/3}

such that 

    p ~ Uniform(0, 1)

For each bin [B0, B1], we choose R according to this inverse CDF method and 
then choose an angular displacement by sampling uniformly on the surface of 
the corresponding sphere. The result is then projected onto the XY plane,
and its radius in that plane is accumulated in a new histogram.

This process is repeated for each bin. The result is the matrix *A* described
above. 

"""
import os
import sys 
import numpy as np 
import dask 
from dask.diagnostics import ProgressBar 
import pandas as pd 
import matplotlib.pyplot as plt 

# The global bin scheme for this module
r_edges = np.linspace(0.0, 10.0, 5001)
n_bins = r_edges.shape[0] - 1
bin_size = r_edges[1] - r_edges[0]
r_c = r_edges[:-1] + bin_size * 0.5

def radprojsim_bin(B0, B1, n_samples=1000000, n_iter=1, dz=None):
    """
    Given a 3D radial displacement R such that R in [B0, B1] and 
    R is uniformly distributed in the available 3D Euclidean space,
    return the distribution of 2D radial displacements R2 produced
    by projecting the vector R onto a plane intersecting the origin.

    args
    ----
        B0          :   float, the lower bin limit
        B1          :   float, the upper bin limit
        n_samples   :   int, the number of samples to accumulate per
                        iteration
        n_iter      :   int, the number of iterations to use 
        dz          :   float, thickness of the observation slice in z.
                        If specified, then particles start out with 
                        a random position in the interval [-dz/2, dz/2]
                        and are not counted if their final z position lies
                        outside of this interval. If left *None*, then no
                        such constraint is imposed. Note that the total 
                        probability across all bins will generally not 
                        sum to 1 if *dz* is set.

    returns
    -------
        1D ndarray of shape (5000,), the distribution of corresponding
            2D radial displacements according to the global bin scheme

    """
    global r_edges 
    global n_bins

    result = np.zeros(n_bins, dtype=np.float64)

    for iter_idx in range(n_iter):

        # Simulate the radial displacements using inverse CDF sampling
        r = np.cbrt(B0**3 + (B1**3 - B0**3) * np.random.random(size=n_samples))

        # Simulate angular displacements by sampling on the surface of 
        # the unit sphere
        a = np.random.normal(size=(n_samples, 3))
        a = (a.T / np.sqrt((a**2).sum(axis=1))).T 

        # Combine radial and angular parts 
        a = (a.T * r).T 

        # If desired, simulate a finite range of observation in z
        if not dz is None:
            hz = dz * 0.5
            a[:,0] = a[:,0] + np.random.uniform(-hz, hz, size=n_samples)
            a = a[np.abs(a[:,0])<=hz, :]

        # Take the XY displacements 
        r = np.sqrt((a[:,1:]**2).sum(axis=1))
        H = np.histogram(r, bins=r_edges)[0].astype(np.float64)
        result += H 

    result /= (n_iter * n_samples)
    return result 

def radprojsim(n_samples=10000000, n_iter=10, num_workers=8, out_csv=None,
    dz=None):
    """
    Combine the full numerical approximative Abel transform. For each 
    bin in the global bin scheme, compute the corresponding distribution
    of 2D radial displacements.

    args
    ----
        n_samples   :   int, the number of samples per iteration per bin
        n_iter      :   int, the number of iterations per bin
        num_workers :   int, number of dask workers to use in this job
        out_csv     :   str, file to save the projection to
        dz          :   float, axial containment range

    returns
    -------
        2D ndarray of shape (5000, 5000), the matrix for performing 
            a numerical Abel transform

    """
    global r_edges 
    global n_bins
    result = np.zeros((n_bins, n_bins), dtype=np.float64)

    c = 0

    @dask.delayed 
    def simulate_bin(i):
        result = radprojsim_bin(r_edges[i], r_edges[i+1], 
            n_samples=n_samples, n_iter=n_iter, dz=dz)
        return result 

    results = [simulate_bin(i) for i in range(n_bins)]

    if num_workers > 1:
        scheduler = "processes"
    else:
        scheduler = "single-threaded"

    with ProgressBar():
        results = dask.compute(*results, scheduler=scheduler, num_workers=num_workers)

    # Accumulate into a single ndarray
    A = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i, r in enumerate(results):
        A[:,i] = r 

    # Format as a dataframe and save 
    if not out_csv is None:
        cols = ["%.3f" % i for i in r_edges[1:]]
        df = pd.DataFrame(A, columns=cols)
        df["r_right_edge_um"] = r_edges[1:]
        df.to_csv(out_csv, index=False)

    return A

if __name__ == '__main__':
    result = radprojsim(n_samples=100000, n_iter=10, num_workers=4, dz=None,
        out_csv="free_abel_transform_10um.csv")
    plt.imshow(result, vmax=result.max()*0.05, cmap='gray')
    plt.show(); plt.close()

