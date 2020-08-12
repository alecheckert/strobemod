#!/usr/bin/env python
"""
utils.py

"""
import os
import numpy as np

# 3D Hankel transform
from hankel import SymmetricFourierTransform 
HankelTrans3D = SymmetricFourierTransform(ndim=3, N=10000, h=0.005)

# Univariate spline interpolation
from scipy.interpolate import InterpolatedUnivariateSpline as spline 

# hilonom package directory
PACKAGE_DIR = os.path.split(os.path.abspath(__file__))[0]

# Data directory with immutable config stuff
DATA_DIR = os.path.join(PACKAGE_DIR, "data")

########################
## ARRAY MANIPULATION ##
########################

def generate_support(jump_bins, n_frames, frame_interval):
    """
    For fitting CDFs with multiple timepoints, generate a bivariate
    support for the fitting routine.

    This is a set of (r, t) tuples, where *r* is the radial displacement
    in the 2D plane of the camera, expressed in um*, and *t* is the
    interval between the first and second observations in seconds.

    args
    ----
        jump_bins       :   1D ndarray, the edges of each jump length
                            bin in um. Must be at least length 2. 
        n_frames        :   int, the maximum number of frame intervals
                            to consider
        frame_interval  :   float, the length of a single frame interval
                            in seconds

    returns
    -------
        2D ndarray of shape (n_frames * n_jump_bins), the support

    """
    n_jump_bins = jump_bins.shape[0] - 1
    M = n_frames * n_jump_bins
    rt_tuples = np.zeros((M, 2), dtype=np.float64)
    for t in range(n_frames):
        rt_tuples[t*n_jump_bins : (t+1)*n_jump_bins, 0] = jump_bins[1:]
        rt_tuples[t*n_jump_bins : (t+1)*n_jump_bins, 1] = (t+1) * frame_interval 
    return rt_tuples 

###############################
## DIFFUSION MODEL UTILITIES ##
###############################

def generate_brownian_transfer_function(support, D, frame_interval):
    """
    Generate a transfer function for Brownian diffusion.

    args
    ----
        support         :   1D ndarray, the points in the support,
                            in um
        D               :   float, diffusion coefficient in um^2 s^-1
        frame_interval  :   float, the time between frames in seconds

    returns
    -------
        1D ndarray of dtype complex128, the RFFT for the Green's
            function of this Brownian motion

    """
    g = np.exp(-(support**2) / (4*D*frame_interval))
    g /= g.sum()
    return np.fft.rfft(g)

def defoc_prob_brownian(D, n_frames, frame_interval, dz):
    """
    Calculate the fraction of Brownian molecules remaining in the focal
    volume at a few timepoints.

    Specifically:

    A Brownian motion is generated (photoactivated) with uniform probability
    across the focal depth *dz*, and is then observed at regular intervals.
    If the particle is outside the focal volume at any one frame interval,
    it is counted as "lost" and is not observed for any subsequent frame, 
    even if it diffuses back into the focal volume.

    This function returns the probability that such a particle is observed
    at each frame.

    args
    ----
        D           :   float, diffusion coefficient in um^2 s^-1
        n_frames    :   int, the number of frame intervals to consider
        frame_interval: float, in seconds
        dz          :   float, focal volume depth in um

    returns
    -------
        1D ndarray, shape (n_frames), the probability of defocalization 
            at each frame

    """
    # Define the initial probability mass 
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 
    pmf = inside.astype("float64")
    pmf /= pmf.sum()

    # Define the transfer function for this BM
    g_rft = generate_brownian_transfer_function(support, D, frame_interval)

    # Propagate over subsequent frame intervals
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(pmf) * g_rft))
        pmf[outside] = 0.0
        result[t] = pmf.sum()

    return result 

###################
## NORMALIZATION ##
###################

def normalize_pmf(pmfs):
    """
    Normalize a 1D or 2D histogram to a probability mass function,
    avoiding divide-by-zero errors for axes with zero observations.

    The typical format for a PMF in strobemodels is a 2D ndarray where
    the first dimension corresponds to time and the second dimension
    corresponds to the spatial (2D radial) displacement. Here, each
    timestep is normalized independently. If there are zero 
    displacements corresponding to a given timestep, then the PMF for 
    that timestep is defined to be 0.

    args
    ----
    	pmfs	:	2D ndarray of shape (n_timesteps, n_spatial_bins),
    				or 1D ndarray of shape (n_spatial_bins)

    returns
    -------
    	ndarray of shape pmfs.shape, the normalized PMFs

    """
    if len(pmfs.shape) == 1:
        s = pmfs.sum()
        if s == 0.0:
            return np.zeros(pmfs.shape, dtype=np.float64)
        else:
            return pmfs / s 
    elif len(pmfs.shape) == 2:
        nonzero = pmfs.sum(axis=1) != 0
        result = np.zeros(pmfs.shape, dtype=np.float64)
        result[nonzero,:] = (pmfs[nonzero,:].T / pmfs[nonzero,:].sum(axis=1)).T 
        return result 

def normalize_flat_cdf(rt_tuples, cdf):
    """
    Normalize a flat CDF separately for each timestep. These CDFs, which are 
    used by MINPACK for fitting, are typically an end-to-end concatenation of
    CDFs corresponding to each timepoint.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the set of
                            independent variables used for fitting (generated
                            by *generate_support*)
        cdf             :   1D ndarray of shape (n_points), the CDF

    returns
    -------
        1D ndarray, normalized CDF

    """
    unique_dt = np.unique(rt_tuples[:,1])
    for t in unique_dt:

        # Find the set of support points corresponding to this timepoint
        match = rt_tuples[:,1]==t 

        # Get the last point in the CDF corresponding to this timepoint
        c = np.argmax(rt_tuples[match,0]) + match.nonzero()[0][0]

        # Normalize
        cdf[match] = cdf[match] / cdf[c]

    return cdf 

def radnorm(r, pdf, d=2):
    """
    Given a PDF with radial symmetry, return the PDF for the radial distance
    from the origin. This is equivalent to taking the PDF, expressed in 
    hyperspherical coordinates, and marginalizing on all angular components.

    For instance, in 2D, we would do

    	pdf_rad(r) = 2 pi int_{0}^{infty} r pdf(r) dr 

    Normalizing the Gaussian density with d = 2 would give a Rayleight
    distribution, with d = 3 would give a Maxwell-Boltzmann distribution,
    and so on.

    This method approximates the PDF at a discrete set of radial points. For
    it to be accurate, the spacing of the support *r* must be small relative
    to local changes in the PDF.

    args
    ----
    	r 			:	1D ndarray of shape (n_r), the radial support
    	pdf         :   1D ndarray of shape (n_r), the PDF as a function of 
    					the support
    	d           :   int, the number of spatial dimensions

    returns
    -------
    	1D ndarray of shape (n_r), the PDF for the radial distance from the
    		origin

    """
    result = pdf * np.power(r, d-1)
    result /= result.sum()
    return result 

###################################################
## GENERATING PDFs FROM CHARACTERISTIC FUNCTIONS ##
###################################################

def pdf_from_cf(func_cf, x, **kwargs):
    """
    Evaluate the PDF of a 1-dimensional real-valued random variable given
    its characteristic function. 

    args
    ----
    	func_cf 		:	function with signature (1D np.ndarray, **kwargs),
    						the characteristic function
    	x				:	1D ndarray, the set of points at which to evaluate
    						the PDF 
    	kwargs			:	to *func_cf*

    returns
    -------
    	1D ndarray of shape x.shape, the PDF

    """
    # Spectrum
    k = np.fft.rfftfreq(x.shape[0], d=(x[1]-x[0]))

    # Evaluate characteristic function
    cf = func_cf(2.0 * np.pi * k, **kwargs)

    # Inverse transform
    return np.fft.fftshift(np.fft.irfft(cf, x.shape[0]))

def pdf_from_cf_rad(func_cf, x, **kwargs):
    """
    Evaluate a radially symmetric PDF defined on 3D space, given the spectral
    radial profile of its characteristic function.

    important
    ---------
    	The output of this function is not necessarily normalized. If the user
    		wants to normalize it on 3D real space - for instance, to get the 
    		PDF for the radial displacement from the origin of a trajectory in 
    		3D space - they should use radnorm() with d = 3.

    args
    ----
    	func_cf 	:	function with signature (1D ndarray, **kwargs), the
    					characteristic function
    	x			:	1D ndarray, the real-space points at which to evaluate
    					the PDF
    	kwargs      :	to *func_cf*

    returns
    -------
    	1D ndarray of shape x.shape, the PDF

    """
    F = lambda j: func_cf(j, **kwargs)
    return HankelTrans3D.transform(F, x, ret_err=False, inverse=True)









