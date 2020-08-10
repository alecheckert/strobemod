#!/usr/bin/env python
"""
utils.py

"""
import os
import numpy as np

# hilonom package directory
PACKAGE_DIR = os.path.split(os.path.abspath(__file__))[0]

# Data directory with immutable config stuff
DATA_DIR = os.path.join(PACKAGE_DIR, "data")

def normalize_pmf(pmfs):
	"""
	Normalize a 1D or 2D histogram to a probability mass function,
	avoiding divide-by-zero errors for axes with zero observations.

	The typical format for a PMF in hilonom is a 2D ndarray where
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
