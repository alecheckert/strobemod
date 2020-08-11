#!/usr/bin/env python
"""
simutils.py -- useful utilities for the simulation module in strobemod

"""
import sys
import os 
import numpy as np 
import pandas as pd 

def sample_sphere(N, d=3):
    """
    Sample *N* points from the surface of the unit sphere, returning
    the result as a Cartesian set of points.

    args
    ----
        N       :   int, the number of trajectories to simulate. Alternatively,
                    N may be a tuple of integers, in which case the return
                    array has shape (*N, d)
        d       :   int, number of dimensions for the (hyper)sphere

    returns
    -------
        2D ndarray of shape (N, d), the positions of the points 

    """
    # 
    if isinstance(N, int):
        N = [N]
    p = np.random.normal(size=(np.prod(N), d))
    p = (p.T / np.sqrt((p**2).sum(axis=1)))
    return p.reshape(tuple(list(N) + [d]))

def tracks_to_dataframe(positions, kill_nan=True):
    """
    Given a set of trajectories as a 3D ndarray indexed by (trajectory,
    timepoint, spatial dimension), convert to a pandas.DataFrame.

    args 
    ----
    	positions		:	3D ndarray of shape (n_trajs, n_frames, 3),
    						the ZYX positions of a 3D trajectory
    	kill_nan		:	bool, remove NaNs after making the dataframe.
    						These are used internally to indicate defocalized
    						or otherwise lost particles.

    returns
    -------
    	pandas.DataFrame with the columns "trajectory", "frame", "z0",
    		"y0", "x0"

    """
    n_trajs, n_frames, n_dim = positions.shape
    assert n_dim == 3

    # Extract ZYX positions from the ndarray
    Z, Y, X = positions.T

    # Size of output dataframe
    M = n_trajs * n_frames

    # Format output dataframe
    df = pd.DataFrame(index=np.arange(M), columns=["frame", "trajectory", "z0", "y0", "x0"])
    df["trajectory"] = df.index // n_frames
    df["frame"] = df.index % n_frames
    df["z0"] = Z.T.ravel()
    df["y0"] = Y.T.ravel()
    df["x0"] = X.T.ravel()

    # Remove NaN particles if desired
    if kill_nan:
        df = df[~pd.isnull(df["z0"])]

    return df