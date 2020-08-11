#!/usr/bin/env python
"""
radproj.py -- numerically project 3D radial displacement displacement
distributions into 2D

"""
import os 

# To cache the projection matrices
from functools import lru_cache 

# Data handling
import numpy as np 
import pandas as pd 

# Custom utilities and paths
from .utils import DATA_DIR, normalize_pmf 

def radproj(pmf, delta_z, renorm=False):
    """
    Project a 3D radial displacement distribution into 2D, only taking 
    displacements that fall in a thin 2D slab with thickness *delta_z*.

    args
    ----
        pmf 			:	1D ndarray, the distribution of radial 
        					displacements in 3D. Modified in place.
        delta_z 		:	float

    returns
    -------
        1D ndarray of shape dist.shape, the distribution of 2D 
        	radial displacements

    """
    # Get the projection matrix for this HiLo slice thickness
    proj_matrix = get_proj_dist(delta_z)

    # Compute the projection
    if len(pmf.shape) == 1:
        result = (proj_matrix.T * dist).sum(axis=1)
    elif len(pmf.shape) == 2:
        result = np.empty(pmf.shape, dtype=np.float64)
        for t in range(result.shape[0]):
            result[t,:] = (proj_matrix.T * pmf[t,:]).sum(axis=1)

    # Renormalize
    if renorm:
        result = normalize_pmf(result)

    return result 

@lru_cache(maxsize=1)
def get_proj_dist(delta_z):
    """
    Return a matrix suitable for transforming a 3D radial displacement
    distribution into a 2D radial displacement distribution in HiLo 
    data.

    args
    ----
    	delta_z 		:	float, the thickness of the observation slice
    						in um. The closest matching data will be 
    						returned.

    returns
    -------
    	2D ndarray of shape (n_bins, n_bins), dtype float64, suitable
    		as input to the *dist* argument of radproj()

    """
    # Get the closest matching file
    avail_delta_z = np.array([0.7])
    m_idx = np.argmin(np.abs(avail_delta_z - delta_z))
    filepath = "radial_proj_dz-%.1f.csv" % avail_delta_z[m_idx]
    filepath = os.path.join(DATA_DIR, filepath)

    # The set of columns corresponding to each radial displacement bin
    bin_seq = [str(j) for j in np.linspace(0.0, 5.0, 5001)[:-1]]

    # 2D ndarray with the projection factors
    f = pd.read_csv(filepath)[bin_seq]
    f = f.fillna(0.0)
    return np.asarray(f)



