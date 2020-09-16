#!/usr/bin/env python
"""
specdiffuse.py -- decompose a jump length distribution into a spectrum of 
underlying diffusivities using expectation maximization

"""
import os
import sys

# Numeric
import numpy as np 

# Dataframes
import pandas as pd 

# Defocalization probability of a Brownian motion 
# in a thin plane
from .utils import defoc_prob_brownian

# Model for regular one-state Brownian motion
from .models import pdf_1state_brownian, cdf_1state_brownian

def expect_max(data, likelihood, D_values, n_iter=1000, **kwargs):
    """
    Use expectation-maximization algorithm to estimate the state occupations
    for a mixture model, given a set of observed jumps.

    args
    ----
        data        :   ndarray of shape (n_jumps, ...), a set of observed 
                        data points
        likelihood  :   function with signature (data, D, **kwargs), the 
                        likelihood function for jumps produced by the diffusion
                        model
        D_values    :   1D ndarray, the set of diffusivities to consider
        n_iter      :   int, the number of iterations
        kwargs      :   additional keyword arguments to the likelihood

    returns
    -------
        1D ndarray of shape D_values.shape, the estimated occupations for 
            each diffusive state

    """
    M = data.shape[0]
    nD = len(D_values)

    # Current guess for the state occupations
    p = np.ones(nD, dtype=np.float64) / nD 

    # The likelihood of each observation given each diffusion coefficient
    L = np.zeros((M, nD), dtype=np.float64)
    for i, D in enumerate(D_values):
        L[:,i] = likelihood(data, D, **kwargs)

    # The probabilities of each diffusive state, given each observation
    # and the current parameter estimate
    T = np.zeros((M, nD), dtype=np.float64)

    for iter_idx in range(n_iter):

        # Evaluate the probability of each diffusive state for each data point
        T = L * p  
        T = (T.T / T.sum(axis=1)).T 

        # Find the value of p that maximizes the expected log-likelihood
        # under these parameters
        p[:] = T.sum(axis=0) / M 

        sys.stdout.write("finished with iteration %d/%d...\r" % (iter_idx+1, n_iter))
        sys.stdout.flush()

    print("")
    return p

def expect_max_defoc(tracks, D_values, n_iter=10000, n_frames=4, frame_interval=0.01,
    dz=0.7, loc_error=0.0):
    """
    Estimate the fraction of trajectories in each of a spectrum of diffusive states,
    accounting for defocalization over the experimental frame interval.

    The result is a vector with the predicted occupations of each state.

    note on choice of diffusivities
    -------------------------------
        This algorithm estimates the occupation of each diffusive state rather than
        the corresponding diffusivity, so the vector of diffusivities (*D_values*) 
        should be chosen with values that encompasses the range of diffusivities that
        are reasonably expected for this experiment. For instance, a biological 
        molecule might merit diffusivites ranging from 0.0 to 10.0 um^2 s^-1, with 
        0.1 or 0.2 um^2 s^-1 increments.

    note on convergence
    -------------------
        Especially when dealing with populations with many diffusive states, it is 
        recommended to allow >=100000 iterations for the algorithm to converge.

    note on gaps
    ------------
        This function assumes that tracking was done without gaps. Gaps present in 
        the data could lead to errors in inference.

    args
    ----
        tracks          :   pandas.DataFrame, the trajectories. Must contain "y", "x",
                            "frame" (int), and "trajectory" columns
        D_values        :   1D ndarray, the set of diffusivities to consider in um^2 
                            s^-1
        n_iter          :   int, the number of iterations
        n_frames        :   int, the maximum number of frame intervals to consider
        frame_interval  :   float, the experimental frame interval in seconds
        dz              :   float, the thickness of the focal slice in um
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape D_values.shape, the estimated occupations of each 
            diffusive state

    """
    nD = len(D_values)

    # Work with a copy of the trajectories rather than the original
    tracks = tracks.copy()

    # Throw out all points in the trajectories after the first *n_frames*
    tracks["one"] = 1
    tracks["index_in_track"] = tracks.groupby("trajectory")["one"].cumsum() - 1
    tracks = tracks.drop("one", axis=1)
    tracks = tracks[tracks["index_in_track"] <= n_frames]

    # Calculate trajectory length, then exclude singlets and non-trajectories from
    # the analysis
    tracks = tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )
    tracks = tracks[np.logical_and(tracks["track_length"]>1, tracks["trajectory"]>=0)]
    tracks = tracks.sort_values(by=["trajectory", "frame"])
    n_tracks = tracks['trajectory'].nunique()

    # Work with an ndarray copy, for speed. Note that the last two columns
    # are placeholders.
    _T = np.asarray(tracks[['track_length', 'trajectory', 'y', 'x', 'track_length', 'track_length']])

    # Calculate the YX displacement vectors
    vecs = _T[1:,:] - _T[:-1,:]

    # Map the corresponding trajectory indices back to each displacement. 
    # This is defined as the trajectory index corresponding to the first 
    # point that makes up each displacement
    vecs[:,4] = _T[:-1,1]

    # Map the corresponding track lengths back to each displacement
    vecs[:,0] = _T[:-1,0]

    # Only consider vectors between points originating from the same track
    vecs = vecs[vecs[:,1] == 0.0, :]

    # Calculate the corresponding 2D radial displacements
    vecs[:,5] = np.sqrt(vecs[:,2]**2 + vecs[:,3]**2)

    # Get the probability of remaining in the focal volume for each 
    # diffusive state at each of the frame intervals under consideration
    F_remain = np.zeros((n_frames, nD), dtype=np.float64)
    for j, D in enumerate(D_values):
        F_remain[:,j] = defoc_prob_brownian(D, n_frames, frame_interval=frame_interval,
            dz=dz, n_gaps=n_gaps)
    f_remain_one_interval = F_remain[0,:].copy()

    # Get the probability that a trajectory with a given diffusion coefficient
    # remains in focus for *exactly* so many frame intervals
    for frame_idx in range(1, n_frames):
        F_remain[frame_idx-1,:] -= F_remain[frame_idx,:]

    # Normalize (is this necessary?)
    F_remain = F_remain / F_remain.sum(axis=0)

    # Evaluate the probability density at each jump, given each diffusion
    # coefficient. Each element of this is like the PDF for the displacement
    # conditioned on knowledge of the underlying state
    L_cond = np.zeros((vecs.shape[0], nD), dtype=np.float64)
    for j, D in enumerate(D_values):
        sig2 = 2 * (D * frame_interval + loc_error**2)
        L_cond[:,j] = vecs[:,5] * np.exp(-(vecs[:,5]**2) / (2 * sig2)) / sig2

    # Evaluate the conditional probability of each trajectory under each of the diffusion
    # models. The following nonsense, which seems like a rather roundabout way of doing 
    # things, is really just to speed up this computation by relying on the fast numpy/pandas
    # routines. The essential idea is: for every trajectory, evaluate its probability under
    # a given diffusion coef. by multiplying (probability of observing a trajectory of this length)
    # by (probability to observe this specific trajectory, given that length). The latter 
    # conditional probability is just the product of the probability of each of the jump lengths
    # under the diffusion model, since Brownian diffusion is Markovian.
    L_cond = pd.DataFrame(L_cond, columns=D_values)
    L_cond["trajectory"] = vecs[:,4]
    L_cond["track_length"] = vecs[:,0].astype(np.int64)

    L = np.zeros((n_tracks, nD), dtype=np.float64)
    for j, D in enumerate(D_values):
        L_cond["f_remain"] = L_cond["track_length"].map({i+2: F_remain[i,j] for i in range(n_frames)})
        L[:,j] = np.asarray(L_cond.groupby("trajectory")[D].prod() * L_cond.groupby("trajectory")["f_remain"].first())

    # The probabilities that each observation belongs to each diffusive state,
    # given the set of observations, their accompanying complete likelihood
    # functions, and their likelihoods under the current set of model parameters
    T = np.zeros((n_tracks, nD), dtype=np.float64)

    # The current guess at state occupations
    p = np.ones(nD, dtype=np.float64) / nD 

    # Expectation-maximization loop
    for iter_idx in range(n_iter):

        # Evaluate the probability that each observation belongs to each 
        # diffusive state
        T = L * p 
        T = (T.T / T.sum(axis=1)).T 

        # Find the value of p that maximizes the expected log-likelihood
        # under the current value of T 
        p[:] = T.sum(axis=0) / n_tracks 

        sys.stdout.write("Finished with %d/%d iterations...\r" % (iter_idx+1, n_iter))
        sys.stdout.flush()

    # Correct for the probability of defocalization at one frame interval
    p = p / f_remain_one_interval
    p /= p.sum()

    print("")
    return p 

def pdf_specdiffuse_model(rt_tuples, D_values, D_occs, frame_interval=0.01, dz=0.7,
    loc_error=0.0):
    """
    Evaluate the probability density function for a mixed Brownian model with
    any number of non-interconverting states, accounting for defocalization.

    args
    ----
        rt_tuples           :   2D ndarray of shape (N, 2), the set of (radial displacement
                                in um, delay in seconds) tuples at which to evaluate the PDF
        D_values            :   1D ndarray of shape (M,), the set of diffusivities
        D_occs              :   1D ndarray of shape (M,), the set of state occupations
        frame_interval      :   float, time between frames in seconds
        dz                  :   float, the thickness of the observation slice in um
        loc_error           :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (N,), the PDF evaluated at each point of the support

    """
    N = rt_tuples.shape[0]
    M = len(D_values)

    # Get the unique time intervals present in this set of data
    unique_dt = np.unique(rt_tuples[:,1])
    n_frames = len(unique_dt)

    # Evaluate the defocalization probabilities for each diffusive state
    F_remain = np.zeros((M, n_frames), dtype=np.float64)
    for i, D in enumerate(D_values):
        F_remain[i,:] = defoc_prob_brownian(D, n_frames, frame_interval,
            dz, n_gaps=0)

    # Multiply by the state occupation estimates
    F_remain = (F_remain.T * D_occs).T 

    # Normalize on each frame interval
    F_remain = F_remain / F_remain.sum(axis=0)

    # Evaluate the PDFs for each state
    pdfs = np.zeros(N, dtype=np.float64)
    r2 = rt_tuples[:,0] ** 2
    for i, D in enumerate(D_values):

        pdf_D = np.zeros(N, dtype=np.float64)

        for j, dt in enumerate(unique_dt):

            # Evaluate the naive PDF for this diffusivity
            match = rt_tuples[:,1] == dt 
            sig2 = 2 * (D * dt + loc_error**2)
            pdf_D[match] = rt_tuples[match, 0] * np.exp(-r2[match] / (2 * sig2)) / sig2 
            pdf_D[match] = pdf_D[match] * F_remain[i,j]

        pdfs += pdf_D 

    return pdfs 

def cdf_specdiffuse_model(rt_tuples, D_values, D_occs, frame_interval=0.01, dz=0.7,
    loc_error=0.0):
    """
    Evaluate the cumulative distribution function for a mixed Brownian model
    with any number of non-interconverting states, accounting for defocalization.

    args
    ----
        rt_tuples           :   2D ndarray of shape (N, 2), the set of (radial displacement
                                in um, delay in seconds) tuples at which to evaluate the PDF
        D_values            :   1D ndarray of shape (M,), the set of diffusivities
        D_occs              :   1D ndarray of shape (M,), the set of state occupations
        frame_interval      :   float, time between frames in seconds
        dz                  :   float, the thickness of the observation slice in um
        loc_error           :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (N,), the PDF evaluated at each point of the support

    """
    N = rt_tuples.shape[0]
    M = len(D_values)

    # Get the unique time intervals present in this set of data
    unique_dt = np.unique(rt_tuples[:,1])
    n_frames = len(unique_dt)

    # Evaluate the defocalization probabilities for each diffusive state
    F_remain = np.zeros((M, n_frames), dtype=np.float64)
    for i, D in enumerate(D_values):
        F_remain[i,:] = defoc_prob_brownian(D, n_frames, frame_interval,
            dz, n_gaps=0)

    # Multiply by the state occupation estimates
    F_remain = (F_remain.T * D_occs).T 

    # Normalize on each frame interval
    F_remain = F_remain / F_remain.sum(axis=0)

    # Evaluate the CDFs for each state
    cdfs = np.zeros(N, dtype=np.float64)
    r2 = rt_tuples[:,0] ** 2
    for i, D in enumerate(D_values):

        cdf_D = np.zeros(N, dtype=np.float64)

        for j, dt in enumerate(unique_dt):
            sig2 = 2 * (D * dt + loc_error**2)
            match = rt_tuples[:,1] == dt 
            cdf_D[match] = 1.0 - np.exp(-r2[match] / (2 * sig2))
            cdf_D[match] = cdf_D[match] * F_remain[i,j]

        cdfs += cdf_D 

    return cdfs 
