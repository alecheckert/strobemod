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
    p = (p.T / np.sqrt((p**2).sum(axis=1))).T 
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
    	pandas.DataFrame with the columns "trajectory", "frame", "z",
    		"y", "x"

    """
    n_trajs, n_frames, n_dim = positions.shape
    assert n_dim == 3

    # Extract ZYX positions from the ndarray
    Z, Y, X = positions.T

    # Size of output dataframe
    M = n_trajs * n_frames

    # Format output dataframe
    df = pd.DataFrame(index=np.arange(M), columns=["frame", "trajectory", "z", "y", "x"])
    df["trajectory"] = df.index // n_frames
    df["frame"] = df.index % n_frames
    df["z"] = Z.T.ravel()
    df["y"] = Y.T.ravel()
    df["x"] = X.T.ravel()

    # Remove NaN particles if desired
    if kill_nan:
        df = df[~pd.isnull(df["z"])]

    return df

def tracks_to_dataframe_gapped(positions, n_gaps=0):
    """
    Given a set of trajectories as a 3D ndarray indexed by (trajectory,
    frame, dimension), convert to a pandas.DataFrame format.

    This set of trajectories may be broken up by unobserved localizations,
    which are indicated in the ndaray as np.nan values. 

    If this is the case, then we only count each trajectory as contiguous
    if the number of unobserved localizations in each gap is less than
    or equal to *n_gaps*.

    So, for instance, a trajectory that looks like this:

        [[1.0, 2.0, 3.0],
         [1.5, 2.5, 3.5],
         [nan, nan, nan],
         [4.0, 4.5, 5.0]]

    would be considered as two separate trajectories if n_gaps = 0, or 
    a single trajectory if n_gaps = 1.

    args
    ----
        positions       :   3D ndarray of shape (n_tracks, n_frames, 3),
                            the ZYX positions of each localization in each
                            trajectory
        n_gaps          :   int, the number of gap frames tolerated

    returns
    -------
        pandas.DataFrame, the trajectories in dataframe format. This includes
            the "trajectory", "frame", "z", "y", and "x"

    """
    # Work with a copy of the original set of positions
    positions = positions.copy()

    # Shape of the problem
    n_tracks, n_frames, n_dim = positions.shape 
    assert n_dim == 3

    # Finished tracks, in chunks
    all_tracks = []

    # Exclude trajectories that are never observed
    never_obs = np.isnan(positions[:,:,0]).all(axis=1)
    positions = positions[~never_obs, :, :]

    _round = 0

    while (~np.isnan(positions)).any():

        P = positions.copy()
        track_indices = np.arange(positions.shape[0])

        # Find the first observed localization in each trajectory
        first_obs_frame = np.argmin(np.isnan(positions[:,:,0]), axis=1)

        # Find the last observed localization in each trajectory, 
        # allowing for gaps
        last_obs_frame = first_obs_frame.copy()
        active = np.ones(positions.shape[0], dtype=np.bool)
        gap_count = np.zeros(positions.shape[0], dtype=np.int64)

        while active.any():

            # Remove points from future consideration
            P[track_indices, last_obs_frame, :] = np.nan 

            # Extend all active trajectories
            last_obs_frame[active] += 1

            # If we've reached the end of the available frames in the
            # simulation, terminate
            at_end = last_obs_frame >= positions.shape[1]
            last_obs_frame[at_end] = positions.shape[1] - 1
            active[at_end] = False

            # Drop trajectories if they haven't been observed for more 
            # than the maximum number of tolerated gap frames
            not_obs = np.isnan(positions[track_indices, last_obs_frame, 0])
            # last_obs_frame[at_end] = positions.shape[1] # test
            gap_count[not_obs] += 1
            gap_count[~not_obs] = 0
            dropped = gap_count > n_gaps 
            active = np.logical_and(active, ~dropped)

        # Aggregate finished trajectories
        for i, t in enumerate(range(positions.shape[0])):
            all_tracks.append(positions[t,first_obs_frame[i]:last_obs_frame[i]+1,:])

        # Remove trajectories that have been completely exhausted
        never_obs = np.isnan(P[:,:,0]).all(axis=1)
        positions = P[~never_obs, :, :].copy()

        _round += 1

    # Format the result as a dataframe
    n_points = sum([t.shape[0] for t in all_tracks])
    track_ids = np.zeros(n_points, dtype=np.int64)
    c = 0
    for i, t in enumerate(all_tracks):
        L = t.shape[0]
        track_ids[c:c+L] = i 
        c += L 
    tracks_concat = np.concatenate(all_tracks, axis=0)
    result = pd.DataFrame(tracks_concat, columns=["z", "y", "x"])
    result["trajectory"] = track_ids
    result["one"] = 1
    result["frame"] = result.groupby("trajectory")["one"].cumsum() - 1
    result = result.drop("one", axis=1)
    result = result[~pd.isnull(result["z"])]
    return result

def photobleach(tracks, k, frame_interval=None):
    """
    Simulate photobleaching for a set of trajectories by removing localizations
    after the bleach.

    Note that the 

    args
    ----
        tracks          :   pandas.DataFrame
        k               :   float, the photobleaching rate parameter
        frame_interval  :   float, acquisition frame interval. If None, then 
                            *k* is assumed to be the bleach probability per frame.
                            If *frame_interval* is specified, then *k* is assumed
                            to be in Hz.

    note
    ----
        If *frame_interval* is specified, then we assume that the product 
        k * frame_interval << 1 (bleaching is in the Poisson process regime).

    returns
    -------
        pandas.DataFrame, with trajectories truncated as they photobleach

    """
    T = np.asarray(tracks["trajectory"])
    keep = np.ones(T.shape, dtype='bool')
    k_frame = k * frame_interval 
    bleach = np.random.random(T.shape) < k_frame 
    for t in np.unique(T):
        w = T == t
        if bleach[w].any():
            start, stop = np.nonzero(w)[0][[0,-1]]
            bleach_frame = np.argmax(bleach[w]) + start 
            keep[bleach_frame:stop+1] = False 
    return tracks[keep].copy()

