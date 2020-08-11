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









