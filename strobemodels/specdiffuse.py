#!/usr/bin/env python
"""
specdiffuse.py -- decompose a jump length distribution into a spectrum of 
underlying diffusivities using expectation maximization

"""
import os
import sys
import warnings 

# Numeric
import numpy as np 
from scipy import ndimage as ndi 

import matplotlib.pyplot as plt 

# Special functions for likelihoods
from scipy.special import expi, gammainc, gamma 

# Dataframes
import pandas as pd 

# Parallelization
import dask 

from time import time

# Defocalization probability of a Brownian motion 
# in a thin plane
from .utils import defoc_prob_brownian

# Model for regular one-state Brownian motion
from .models import pdf_1state_brownian, cdf_1state_brownian

# Custom utilities
from .utils import (
    assign_index_in_track,
    track_length
)

def expect_max(data, likelihood, diffusivities, n_iter=1000, **kwargs):
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
        diffusivities:   1D ndarray, the set of diffusivities to consider
        n_iter      :   int, the number of iterations
        kwargs      :   additional keyword arguments to the likelihood

    returns
    -------
        1D ndarray of shape diffusivities.shape, the estimated occupations for 
            each diffusive state

    """
    M = data.shape[0]
    nD = len(diffusivities)

    # Current guess for the state occupations
    p = np.ones(nD, dtype=np.float64) / nD 

    # The likelihood of each observation given each diffusion coefficient
    L = np.zeros((M, nD), dtype=np.float64)
    for i, D in enumerate(diffusivities):
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

def rad_disp_squared(tracks, start_frame=None, n_frames=4, pixel_size_um=1.0):
    """
    Helper function for expectation-maximization and Gibbs sampling routines, used
    to preprocess a set of trajectories for evaluation of likelihoods.

    Given a set of trajectories, return the squared displacements of those trajectories
    as an ndarray.

    args
    ----
        tracks          :   pandas.DataFrame
        start_frame     :   int, disregard trajectories before this frame
        n_frames        :   int
        pixel_size_um   :   float

    returns
    -------
        (
            vecs, a 2D ndarray of shape (N, 6), where *N* is the total number of 
                displacements;
            int, the number of trajectories considered
        )

        The columns of *vecs* have the following meaning:

            vecs[:,0] -> length of the the corresponding trajectory
            vecs[:,1] -> difference in the trajectory index between the first and
                         second points of each displacement. Should always be 0.
            vecs[:,2] -> y-displacement in um
            vecs[:,3] -> x-displacement in um
            vecs[:,4] -> squared 2D radial displacement in um^2
            vecs[:,5] -> index of the corresponding trajectory

    """
    # Do not modify the original dataframe
    tracks = tracks.copy()

    # Only consider trajectories after some start frame
    if not start_frame is None:
        tracks = tracks[tracks['frame'] >= start_frame]

    # Convert from pixels to um
    tracks[['y', 'x']] *= pixel_size_um 

    # Throw out all points in each trajectory after the first *n_frames+1*,
    # so that we have a maximum of *n_frames* displacements in the resulting
    # set of trajectories
    tracks = assign_index_in_track(tracks)
    tracks = tracks[tracks["index_in_track"] <= n_frames].copy()

    # Calculate trajectory length
    tracks = track_length(tracks)

    # Exclude singlets and non-trajectories from the analysis
    tracks = tracks[np.logical_and(tracks["track_length"] > 1, tracks["trajectory"] >= 0)]
    tracks = tracks.sort_values(by=["trajectory", "frame"])
    n_tracks = tracks["trajectory"].nunique()

    # Work with an ndarray copy, for speed. Note that the last two columns
    # are placeholders, replaced by subsequent steps.
    _T = np.asarray(tracks[['track_length', 'trajectory', 'y', 'x', 'track_length', 'track_length']])

    # Calculate YX displacement vectors
    vecs = _T[1:,:] - _T[:-1,:]

    # Map the corresponding trajectory indices back to each displacement. 
    # The trajectory index for a given displacement is defined as the trajectory
    # corresponding to the first point that makes up the two-point displacement.
    vecs[:,5] = _T[:-1,1]

    # Map the corresponding track lengths back to each displacement
    vecs[:,0] = _T[:-1,0]

    # Only consider vectors between points originating from the same track
    vecs = vecs[vecs[:,1] == 0.0, :]

    # Calculate the corresponding 2D squared radial displacement
    vecs[:,4] = vecs[:,2]**2 + vecs[:,3]**2

    return vecs, n_tracks 

def evaluate_diffusivity_likelihoods_on_tracks(tracks, diffusivities, occupations,
    frame_interval=0.01, loc_error=0.0, n_frames=4, pixel_size_um=0.16, dz=0.7):
    """
    Given a set of trajectories and a particular mixture model of 
    diffusivities, evaluate the probability of each trajectory given 
    each separate diffusivity.

    args
    ----
        tracks          :   pandas.DataFrame, the set of trajectories
        diffusivities   :   1D ndarray of dtype float64, the set of 
                            diffusivities in um^2 s^-1
        occupations     :   1D ndarray of dtype float64, the occupations
                            of each diffusivity. Must sum to 1.
        frame_interval  :   float, the frame interval in seconds
        loc_error       :   float, the localization error in um
        n_frames        :   int, the number of displacements to consider
                            from each trajectory
        pixel_size_um   :   float, the number of um per unit pixel
        dz              :   float, focal depth in um

    returns
    -------
        pandas.DataFrame. Each row corresponds to one trajectory (whose
            index in the original *tracks* dataframe is given by the 
            "trajectory" column), and each column (apart from "trajectory")
            corresponds to one of the diffusivities. Then the element

            result.loc[track_idx, diffusivity]

            corresponds to the likelihood of *diffusivity* given the 
            observed trajectory with index *track_idx*.

    """
    # The number of distinct diffusivities
    K = len(diffusivities)

    # Calculate squared radial displacements
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=None, n_frames=n_frames,
        pixel_size_um=pixel_size_um)

    # Get the probability of remaining in the focal volume for each 
    # diffusive state at each of the frame intervals under consideration
    F_remain = np.zeros((n_frames, nD), dtype=np.float64)
    for j, D in enumerate(diffusivities):
        F_remain[:,j] = defoc_prob_brownian(D, n_frames, frame_interval=frame_interval,
            dz=dz, n_gaps=0)
    f_remain_one_interval = F_remain[0,:].copy()

    # Get the probability that a trajectory with a given diffusion coefficient
    # remains in focus for *exactly* so many frame intervals
    for frame_idx in range(1, n_frames):
        F_remain[frame_idx-1,:] -= F_remain[frame_idx,:]

    # Normalize
    F_remain = F_remain / F_remain.sum(axis=0)

    # Evaluate the likelihood of each diffusivity, given each trajectory. The
    # result, *L[i,j]*, gives the likelihood to observe trajectory i under 
    # diffusive state j
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(vecs, diffusivities, state_biases=F_remain,
        frame_interval=frame_interval, loc_error=loc_error, n_frames=n_frames)

    # Format the result as a dataframe
    columns = ["%.5f" % d for d in diffusivities]
    L = pd.DataFrame(L, columns=columns)
    L["trajectory"] = track_indices 

    return L 

def evaluate_diffusivity_likelihood_new(tracks, diffusivities, state_biases=None,
    frame_interval=0.01, loc_error=0.0, use_entire_track=True, max_jumps_per_track=10,
    pixel_size_um=1.0, start_frame=None, likelihood_mode="binned"):
    """
    Create a matrix that gives the likelihood of each of a set of diffusivities,
    given each of a set of trajectories.

    *likelihood_mode* specifies the type of likelihood to compute:

        "point":    evaluate the likelihood at the specific point diffusivities
                    in the array *diffusivities*. The resulting likelihood matrix
                    has shape (n_tracks, len(diffusivities))

        "binned":   evaluate the likelihood integrated between each consecutive
                    pair of points in the array *diffusivities*. The resulting
                    likelihood matrix has shape (n_tracks, len(diffusivities)-1)

        "binned_reg":   an experimental mode similar to binned, but which 
                    attempts to regularize the likelihood (particularly for 
                    doublets). The resulting likelihood matrix has shape 
                    (n_tracks, len(diffusivities)-1)

    args
    ----
        tracks          :   pandas.DataFrame, trajectories
        diffusivities   :   1D ndarray of shape (M,), the set of diffusivities
                                or edges of diffusivity bins, depending on the
                                value of *likelihood_mode*
        state_biases    :   1D ndarray of shape (M,), inherent experimental 
                                biases for or against each of the diffusivities
        frame_interval  :   float, time between frames in seconds
        loc_error       :   float, 1D localization error in um
        use_entire_track:   bool, use every displacement from every trajectory
        max_jumps_per_track :   int, the maximum number of displacements to 
                                consider from each trajectory, if *use_entire_track*
                                is False
        pixel_size_um   :   float, the size of each pixel in um
        start_frame     :   int, disregard trajectories before this frame
        likelihood_mode :   str, the type of likelihood to calculate. Must be 
                                "point", "binned", or "binned_reg"

    returns
    -------
        (
            2D ndarray of shape (n_tracks, K), the likelihood matrix;
            1D ndarray of shape (n_tracks,), the indices of each 
                trajectory;
            1D ndarray of shape (n_tracks,), the lengths of each 
                trajectory
        )

    """
    diffusivities = np.asarray(diffusivities)
    K = diffusivities.shape[0]
    le2 = loc_error ** 2
    assert likelihood_mode in ["point", "binned", "binned_reg"]

    # If using the entire track, set n_frames to the longest trajectory
    # in the sample. Otherwise set it to *max_frames*
    max_track_len = tracks.groupby("trajectory").size().max()
    n_frames = max_track_len if use_entire_track else \
        min(max_track_len, max_jumps_per_track)

    # Calculate squared radial displacements for all trajectories
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=start_frame,
        n_frames=n_frames, pixel_size_um=pixel_size_um)

    # Get the sum of squared displacements for each trajectory
    df = pd.DataFrame(vecs, columns=["track_length", "track_index_diff", 
        "dy", "dx", "squared_disp", "trajectory"])
    L_cond = pd.DataFrame(index=range(n_tracks), 
        columns=["sum_squared_disp", "trajectory", "track_length"])
    L_cond["trajectory"] = np.asarray(df.groupby("trajectory").apply(lambda i: i.name))
    L_cond["track_length"] = np.asarray(df.groupby("trajectory")["track_length"].first())
    L_cond["sum_squared_disp"] = np.asarray(df.groupby("trajectory")["squared_disp"].sum())
    del df 

    # Evaluate the diffusivity likelihoods at each of the points specified by
    # the diffusivity array
    if likelihood_mode == "point":

        # Likelihood of each of the point diffusivities in *diffusivities*
        L = np.zeros((n_tracks, K), dtype=np.float64)

        # Number of displacements per trajectory
        n_disps = np.asarray(L_cond["track_length"]) - 1

        # Other terms necessary to evaluate the log likelihood
        log_gamma_n_disps = np.log(gamma(n_disps))
        log_ss = (n_disps - 1) * np.log(L_cond["sum_squared_disp"])

        # Evaluate the log likelihood of each diffusivity, given each trajectory
        for j, D in enumerate(diffusivities):
            L[:,j] = np.asarray(log_ss - L_cond["sum_squared_disp"] / (4*D*frame_interval) - \
                    n_disps * np.log(4*D*frame_interval) - log_gamma_n_disps)

        # Regularize the problem by subtracting the largest log likelihood 
        # across all diffusivities (for each trajectory), which ensures that 
        # at least one diffusivity has a nonzero likelihood after exponentiation
        L = (L.T - L.max(axis=1)).T 

        # Convert from log likelihoods to likelihoods and normalize
        L = np.exp(L)
        L = (L.T / L.sum(axis=1)).T 

    # Integrate the diffusivity likelihoods over each bin
    elif likelihood_mode == "binned":

        # Likelihood of each of the diffusivity bins defined by the bin edges
        # in *diffusivities*
        L = np.zeros((n_tracks, K-1), dtype=np.float64)

        # Divide the trajectories into doublets and everything else, then take the 
        # sum of squared displacements ("ss") for each category
        doublets = L_cond["track_length"] == 2
        ss_doublets = np.asarray(L_cond.loc[doublets, "sum_squared_disp"])
        ss_nondoublets = np.asarray(L_cond.loc[~doublets, "sum_squared_disp"])
        tl_nondoublets = np.asarray(L_cond.loc[~doublets, "track_length"]) - 1
        doublets = np.asarray(doublets).astype(np.bool)

        for j in range(K-1):

            # Deal with doublets
            L[doublets,j] = expi(-ss_doublets/(4*(diffusivities[j]*frame_interval+le2))) - \
                expi(-ss_doublets/(4*(diffusivities[j+1]*frame_interval+le2)))

            # Deal with everything else
            L[~doublets, j] = gammainc(tl_nondoublets-1, ss_nondoublets/(4*(diffusivities[j]*frame_interval+le2))) - \
                gammainc(tl_nondoublets-1, ss_nondoublets/(4*(diffusivities[j+1]*frame_interval+le2)))

    # Integrate the diffusivity likelihoods over bin, using a regularizing term
    elif likelihood_mode == "binned_reg":

        # Likelihood of each of the diffusivity bins defined by the bin edges
        # in *diffusivities*
        L = np.zeros((n_tracks, K-1), dtype=np.float64)

        # Number of displacements in each trajectory
        tl = np.asarray(L_cond["track_length"]) - 1

        # Sum of squared displacements for each trajectory
        ss = np.asarray(L_cond["sum_squared_disp"])

        for j in range(K-1):
            L[:,j] = gammainc(tl, ss/(4*(diffusivities[j]*frame_interval+le2))) - \
                gammainc(tl, ss/(4*(diffusivities[j+1]*frame_interval+le2)))

    # Incorporate state biases inherent to the measurement, if any
    if not state_biases is None:
        L = L * state_biases 

    # Floating point errors
    L[L<0] = 0

    # Normalize over diffusivities for each trajectory
    L = (L.T / L.sum(axis=1)).T 

    # Format track lengths
    track_lengths = np.asarray(L_cond["track_length"]).astype(np.int64)

    # Format trajectory indices
    track_indices = np.asarray(L_cond["trajectory"]).astype(np.int64)

    return L, track_indices, track_lengths


def evaluate_diffusivity_likelihood_alt(tracks, diffusivities, state_biases=None, 
    frame_interval=0.01, loc_error=0.0, n_frames=200, pixel_size_um=1.0,
    start_frame=None):
    """
    args
    ----
        tracks          :   pd.DataFrame, trajectories. Must have "trajectory", 
                            "frame", "x", and "y" columns
        diffusivities   :   1D ndarray of shape (K+1,), the edges of the 
                            *K* diffusivity bins in um^2 s^-1
        state_biases    :   1D ndarray of shape (K,), the state occupation
                            biases evaluated at the midpoint of each diffusivity
                            bin
        frame_interval  :   float, the frame interval in seconds
        loc_error       :   float, the 1D localization error in um
        n_frames        :   int, the maximum number of frames to consider
                            from each trajectory
        pixel_size_um   :   float, the size of individual pixels in um. The "y"
                            and "x" coordinates of the DataFrame input are 
                            assumed to be given in pixels.

    returns
    -------
        (
            2D ndarray of shape (n_tracks, K), the likelihood of each diffusivity
                given each trajectory (normalized over the second axis);
            1D ndarray of shape (n_tracks,), the trajectory indices;
            1D ndarray of shape (n_tracks,), the number of displacements 
                per trajectory (the trajectory length minus 1)
        )

    """
    diffusivities = np.asarray(diffusivities)
    K = diffusivities.shape[0]
    le2 = loc_error ** 2

    # Calculate squared radial displacements for all trajectories
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=start_frame,
        n_frames=n_frames, pixel_size_um=pixel_size_um)

    # Get the sum of squared displacements for each trajectory
    df = pd.DataFrame(vecs, columns=["track_length", "traj_index_diff", "dy", "dx",
        "squared_disp", "trajectory"])
    L_cond = pd.DataFrame(index=list(range(n_tracks)),
        columns=["sum_squared_disp", "trajectory", "track_length"])
    L_cond["trajectory"] = np.asarray(df.groupby("trajectory").apply(lambda i: i.name))
    L_cond["track_length"] = np.asarray(df.groupby("trajectory")["track_length"].first())
    L_cond["sum_squared_disp"] = np.asarray(df.groupby("trajectory")["squared_disp"].sum())
    del df 

    # We have to deal with doublets (trajectories with a single displacement)
    # differently than non-doublets, due to the lack of convergence of the
    # likelihood of a diffusivity given a single displacement
    doublets = L_cond["track_length"] == 2
    ss_doublets = np.asarray(L_cond.loc[doublets, "sum_squared_disp"])
    ss_nondoublets = np.asarray(L_cond.loc[~doublets, "sum_squared_disp"])
    tl_nondoublets = np.asarray(L_cond.loc[~doublets, "track_length"]) - 1
    doublets = np.asarray(doublets).astype(np.bool)

    ## FOR TESTS
    tl = np.asarray(L_cond["track_length"]) - 1
    ss = np.asarray(L_cond["sum_squared_disp"])
    ##

    # The (relative) likelihood of each diffusivity, given each trajectory.
    # Here, the element L[i,j] is the likelihood of the diffusivity falling
    # into the bin (diffusivities[j], diffusivities[j+1]) given the observation
    # of trajectory i.
    L = np.zeros((n_tracks, K-1), dtype=np.float64)
    for j in range(K-1):

        # # Deal with doublets
        # L[doublets,j] = expi(-ss_doublets/(4*(diffusivities[j]*frame_interval+le2))) - \
        #     expi(-ss_doublets/(4*(diffusivities[j+1]*frame_interval+le2)))

        # # Deal with everything else
        # L[~doublets, j] = gammainc(tl_nondoublets-1, ss_nondoublets/(4*(diffusivities[j]*frame_interval+le2))) - \
        #     gammainc(tl_nondoublets-1, ss_nondoublets/(4*(diffusivities[j+1]*frame_interval+le2)))

        L[:,j] = gammainc(tl, ss/(4*(diffusivities[j]*frame_interval+le2))) - \
            gammainc(tl, ss/(4*(diffusivities[j+1]*frame_interval+le2)))

    print("n doublets: %d" % doublets.sum())
    print("n nondoublets: %d" % (~doublets).sum())

    # Multiply by state biases, if any
    if not state_biases is None:
        L = L * state_biases 

    # Floating point errors
    L[L<0] = 0

    # Normalize over diffusivities for each trajectory
    L = (L.T / L.sum(axis=1)).T

    # Format track lengths
    track_lengths = np.asarray(L_cond["track_length"]).astype(np.int64)

    # Format trajectory indices
    track_indices = np.asarray(L_cond["trajectory"]).astype(np.int64)

    return L, track_indices, track_lengths

def evaluate_diffusivity_likelihood_alt_alt(tracks, diffusivities, state_biases=None, 
    frame_interval=0.01, loc_error=0.0, n_frames=200, pixel_size_um=1.0,
    start_frame=None):
    """
    Only two displacements or more.

    args
    ----
        tracks          :   pd.DataFrame, trajectories. Must have "trajectory", 
                            "frame", "x", and "y" columns
        diffusivities   :   1D ndarray of shape (K+1,), the edges of the 
                            *K* diffusivity bins in um^2 s^-1
        state_biases    :   1D ndarray of shape (K,), the state occupation
                            biases evaluated at the midpoint of each diffusivity
                            bin
        frame_interval  :   float, the frame interval in seconds
        loc_error       :   float, the 1D localization error in um
        n_frames        :   int, the maximum number of frames to consider
                            from each trajectory
        pixel_size_um   :   float, the size of individual pixels in um. The "y"
                            and "x" coordinates of the DataFrame input are 
                            assumed to be given in pixels.

    returns
    -------
        (
            2D ndarray of shape (n_tracks, K), the likelihood of each diffusivity
                given each trajectory (normalized over the second axis);
            1D ndarray of shape (n_tracks,), the trajectory indices;
            1D ndarray of shape (n_tracks,), the number of displacements 
                per trajectory (the trajectory length minus 1)
        )

    """
    diffusivities = np.asarray(diffusivities)
    K = diffusivities.shape[0]
    le2 = loc_error ** 2

    # Calculate squared radial displacements for all trajectories
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=start_frame,
        n_frames=n_frames, pixel_size_um=pixel_size_um)

    # Exclude doublets
    vecs = vecs[vecs[:,0] != 2, :]
    n_tracks = len(np.unique(vecs[:,5]))

    # Get the sum of squared displacements for each trajectory
    df = pd.DataFrame(vecs, columns=["track_length", "traj_index_diff", "dy", "dx",
        "squared_disp", "trajectory"])
    L_cond = pd.DataFrame(index=list(range(n_tracks)),
        columns=["sum_squared_disp", "trajectory", "track_length"])
    L_cond["trajectory"] = np.asarray(df.groupby("trajectory").apply(lambda i: i.name))
    L_cond["track_length"] = np.asarray(df.groupby("trajectory")["track_length"].first())
    L_cond["sum_squared_disp"] = np.asarray(df.groupby("trajectory")["squared_disp"].sum())
    del df 

    tl = np.asarray(L_cond["track_length"]) - 1
    ss = np.asarray(L_cond["sum_squared_disp"])

    # Evaluate the likelihood
    L = np.zeros((n_tracks, K-1), dtype=np.float64)
    for j in range(K-1):
        L[:, j] = gammainc(tl-1, ss/(4*(diffusivities[j]*frame_interval+le2))) - \
            gammainc(tl-1, ss/(4*(diffusivities[j+1]*frame_interval+le2)))

    # Multiply by state biases, if any
    if not state_biases is None:
        L = L * state_biases 

    # Floating point errors
    L[L<0] = 0

    # Normalize over diffusivities for each trajectory
    L = (L.T / L.sum(axis=1)).T

    # Format track lengths
    track_lengths = np.asarray(L_cond["track_length"]).astype(np.int64)

    # Format trajectory indices
    track_indices = np.asarray(L_cond["trajectory"]).astype(np.int64)

    return L, track_indices, track_lengths

def evaluate_diffusivity_likelihood(vecs, diffusivities, state_biases=None,
    frame_interval=0.01, loc_error=0.0, n_frames=200):
    """
    Given a set of trajectories, evaluate the likelihood of each of a set
    of diffusivities, given each trajectory.

    args
    ----
        vecs            :   2D ndarray of shape (n_disps, 6), the output
                            of *rad_disp_squared* or similar. See the 
                            docstring for that function for the specifications
                            of this argument.
        diffusivities   :   1D ndarray of shape (n_states,), the diffusivities
                            corresponding to each state in um^2 s^-1
        state_biases    :   2D ndarray of shape (n_frames, n_states), a bias
                            term. The value of *state_biases[i,j]* indicates
                            the probability that a trajectory of length *i+2*
                            with diffusivity *diffusivities[j]* is observed 
                            without defocalizing.
        frame_interval  :   float, the time between frames in seconds
        loc_error       :   float, 1D localization error in um
        n_frames        :   int, the number of displacements from each 
                            trajectory to consider

    returns
    -------
        2D ndarray of shape (n_tracks, n_states), the likelihood of each 
            given each trajectory

    """
    n_states = diffusivities.shape[0]
    n_tracks = len(np.unique(vecs[:,5]))

    if state_biases is None:
        state_biases = np.ones((n_frames, n_states), dtype=np.float64)

    # Evaluate the naive likelihood of each observed 2D squared jump,
    # given each of the distinct diffusivities. This initial likelihood is
    # "naive" in the sense that it does not incorporate any biases resulting
    # from defocalization
    L_cond = np.zeros((vecs.shape[0], n_states), dtype=np.float64)
    for j, D in enumerate(diffusivities):
        sig2 = 2 * (D * frame_interval + loc_error ** 2)
        L_cond[:,j] = np.exp(-((vecs[:,2]**2 + vecs[:,3]**2))/(2*sig2))/(np.pi*2*sig2)

    # Formulate the naive likelihoods as a pandas.DataFrame. The following
    # few steps are kind of an obtuse but very fast way to calculate the 
    # likelihoods of each trajectory, given each diffusivity. The likelihood
    # of a trajectory is the product of the likelihoods of each jump, for a
    # Markov process
    L_cond = pd.DataFrame(L_cond, columns=diffusivities)
    L_cond["trajectory"] = vecs[:,5]
    L_cond["track_length"] = vecs[:,0].astype(np.int64)

    # Only consider the first few jumps to calculate the diffusivity likelihoods,
    # to prevent overflow errors when multiplying large numbers of likelihoods.
    # Note that this generally reduces the amount of information available for
    # inferring the diffusivities.
    L_cond["ones"] = 1 
    L_cond["index_in_track"] = L_cond.groupby("trajectory")["ones"].cumsum()
    L_cond = L_cond[L_cond["index_in_track"] <= n_frames]

    # Evaluate the likelihood of each trajectory given each diffusivity
    L = np.zeros((n_tracks, n_states), dtype=np.float64)
    for j, D in enumerate(diffusivities):
        L_cond["f_remain"] = L_cond["track_length"].map({i+2: state_biases[i,j] \
            for i in range(n_frames)})

        ## VERSION 1
        # L[:,j] = np.asarray(L_cond.groupby("trajectory")[D].prod() * \
        #     L_cond.groupby("trajectory")["f_remain"].first())

        ## VERSION 2 (stabilizes long trajectories against floating-point
        # errors). We'll get -np.inf for numbers that are too low for log
        # (close to 0), so catch the warnings that ensue. These -np.inf
        # values are set to 0 in the exponentiation step below.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            L_cond[D] = np.log(L_cond[D])
            L_cond["f_remain"] = np.log(L_cond["f_remain"])
            L[:,j] = np.asarray(L_cond.groupby("trajectory")[D].sum() + \
                L_cond.groupby("trajectory")["f_remain"].first())

    ## VERSION 2, CONT'D. This normalizes the likelihood across
    # diffusivities for each trajectory. The key here is subtracting
    # the maximum diffusivity log likelihood for each trajectory, which
    # means that at least one diffusivity in each trajectory has a nonzero
    # likelihood after normalization.
    L = (L.T - L.max(axis=1)).T
    L = np.exp(L)
    L = (L.T / L.sum(axis=1)).T 

    print("np.isnan(L).any(): ", np.isnan(L).any())
    print("np.isinf(L).any(): ", np.isinf(L).any())
    print("(L.sum(axis=1) == 0).any(): ", (L.sum(axis=1) == 0).any())

    # Record the trajectory indices as well
    track_indices = np.asarray(L_cond.groupby("trajectory").apply(lambda i: i.name)).astype(np.int64)

    # Record the trajectory lengths as well
    track_lengths = np.asarray(L_cond.groupby("trajectory")["track_length"].first())

    print("track_lengths.mean(): ", track_lengths.mean())
    print("track_lengths.max(): ", track_lengths.max())
    print("track_lengths.min(): ", track_lengths.min())
    print("track_lengths.std(): ", track_lengths.std())

    return L, track_indices, track_lengths 

def emdiff(tracks, diffusivities, n_iter=10000, n_frames=4, frame_interval=0.01,
    dz=0.7, loc_error=0.0, pixel_size_um=1.0, verbose=True,
    track_diffusivities_out_csv=None, mode="by_displacement", 
    disable_track_length_weighting=False, pseudoprob=0.001):
    """
    Estimate the fraction of trajectories in each of a spectrum of diffusive states,
    accounting for defocalization over the experimental frame interval, using an
    expectation-maximization algorithm.

    The result is a vector with the predicted occupations of each state.

    note on choice of diffusivities
    -------------------------------
        This algorithm estimates the occupation of each diffusive state rather than
        the corresponding diffusivity, so the vector of diffusivities (*diffusivities*) 
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
        diffusivities   :   1D ndarray, the set of diffusivities to consider in um^2 
                            s^-1
        n_iter          :   int, the number of iterations
        n_frames        :   int, the maximum number of frame intervals to consider
        frame_interval  :   float, the experimental frame interval in seconds
        dz              :   float, the thickness of the focal slice in um
        loc_error       :   float, 1D localization error in um
        pixel_size_um   :   float, the size of camera pixels in um. The "y" and "x"
                            columns of the input dataframe are assumed to be in 
                            units of pixels
        verbose         :   bool
        track_diffusivities_out_csv :   str, file to save the individual trajectory
                            diffusivities to. The result is indexed by trajectory
                            and each column corresponds to the likelihood of one
                            of the diffusivities given that trajectory, under the
                            posterior model.
        mode            :   str, either "by_displacement" or "by_trajectory", indicates
                            the basic statistical unit underpinning the EM routine.
                            Unless you have a very good reason, this should generally
                            by "by_displacement". 
        disable_track_length_weighting  :   bool, do not incorporate trajectory 
                            length when evaluating the likelihood of each diffusivity,
                            given each trajectory. It is preferable to set this 
                            to True when it is likely that trajectories will be 
                            dropped for reasons other than defocalization in the
                            experiment.

    returns
    -------
        1D ndarray of shape diffusivities.shape, the estimated occupations of each 
            diffusive state

    """
    diffusivities = np.asarray(diffusivities)
    nD = diffusivities.shape[0]

    # Calculate the squared 2D radial displacements
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=None, n_frames=n_frames,
        pixel_size_um=pixel_size_um)

    print("n_tracks = ", n_tracks)

    if mode == "by_displacement":

        # Get the probability of remaining in the focal volume for each 
        # diffusive state at each of the frame intervals under consideration
        F_remain = np.zeros((n_frames, nD), dtype=np.float64)
        for j, D in enumerate(diffusivities):
            F_remain[:,j] = defoc_prob_brownian(D, n_frames, frame_interval=frame_interval,
                dz=dz, n_gaps=0)
        f_remain_one_interval = F_remain[0,:].copy()

        # Get the probability that a trajectory with a given diffusion coefficient
        # remains in focus for *exactly* so many frame intervals
        for frame_idx in range(1, n_frames):
            F_remain[frame_idx-1,:] -= F_remain[frame_idx,:]

        # Normalize
        F_remain = F_remain / F_remain.sum(axis=0)

        # If desired, do not incorporate track lengths into the per-trajectory
        # likelihoods (for instance, if it is likely that trajectories will be
        # dropped for reasons other than defocalization)
        if disable_track_length_weighting:
            F_remain[:] = 1

        # ## EXPERIMENTAL
        # for j, D in enumerate(diffusivities):
        #     F_remain[:,j] = f_remain_one_interval[j]

    elif mode == "by_trajectory":

        # The fraction of molecules corresponding to each of the diffusivities
        # that, after being first observed at some point uniformly distributed
        # inside the focal volume, are found inside the focal volume after 1 frame.
        # This is equal to the number of trajectories sampled within the focal 
        # volume that actually contribute displacements
        f_remain_one_interval = np.zeros(nD, dtype=np.float64)
        for j, D in enumerate(diffusivities):
            f_remain_one_interval[j] = defoc_prob_brownian(D, 1,
                frame_interval=frame_interval, dz=dz, n_gaps=0)[0]

        # Correction factor all ones, if doing analysis by trajectory
        F_remain = np.ones((n_frames, nD), dtype=np.float64)

    # Evaluate the likelihood of each diffusivity, given each trajectory. The
    # result, *L[i,j]*, gives the likelihood to observe trajectory i under 
    # diffusive state j
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(vecs, diffusivities,
        state_biases=F_remain, frame_interval=frame_interval, loc_error=loc_error,
        n_frames=n_frames)

    # The number of displacements corresponding to each trajectory
    track_n_disps = track_lengths - 1
    n_disps_tot = track_n_disps.sum()

    # The probabilities that each observation belongs to each diffusive state,
    # given the set of observations, their accompanying complete likelihood
    # functions, and their likelihoods under the current set of model parameters
    T = np.zeros((n_tracks, nD), dtype=np.float64)

    # The current guess at state occupations
    p = np.ones(nD, dtype=np.float64) / nD 

    # Expectation-maximization loop
    for iter_idx in range(n_iter):

        # Add pseudocounts, if desired
        p = p + pseudoprob
        p /= p.sum()

        # Evaluate the probability that each observation belongs to each 
        # diffusive state. Optionally, weight each trajectory's influence
        # in the result by the number of displacements.
        T = L * p 
        T = (T.T / T.sum(axis=1)).T 

        # Find the value of p that maximizes the expected log-likelihood
        # under the current value of T. We can treat either trajectories or
        # displacements as the "observation unit" here. If using displacements,
        # note that the likelihood of each displacement is assumed to be equal
        # to the likelihood of its whole corresponding trajectory under a 
        # given diffusivity.
        if mode == "by_displacement":
            T = (T.T * track_n_disps).T 
            p[:] = T.sum(axis=0) / n_disps_tot 
        elif mode == "by_trajectory":
            p[:] = T.sum(axis=0) / n_tracks 

        if verbose and iter_idx % 100 == 0:
            print("Finished with %d/%d iterations..." % (iter_idx, n_iter))

    # Correct for the probability of defocalization at one frame interval
    if not dz is np.inf:
        p = p / f_remain_one_interval
    p /= p.sum()

    # Save the diffusivity probability vectors for each trajectory
    if not track_diffusivities_out_csv is None:
        columns = ["%.5f" % d for d in diffusivities]
        out = pd.DataFrame(T, columns=columns)
        out["trajectory"] = track_indices 
        out.to_csv(track_diffusivities_out_csv, index=False)

    if verbose: print("")
    return p 

def gsdiff(tracks, diffusivities, prior=None, n_iter=1000, burnin=500,
    use_entire_track=True, max_jumps_per_track=50, frame_interval=0.01,
    loc_error=0.0, pixel_size_um=1.0, dz=np.inf, verbose=True, pseudocounts=1,
    n_threads=1, track_diffusivities_out_csv=None, mode="by_displacement",
    defoc_corr="first_only"):
    """
    Use Gibbs sampling to estimate the posterior distribution of the 
    state occupations for a multistate Brownian motion with no state
    transitions, given an observed set of trajectories.

    args
    ----
        tracks          :   pandas.DataFrame, the observed trajectories.
                            Must contain the "y" and "x" columns with
                            the positions of the localizations in pixels,
                            a "frame" column with the frame index of 
                            each localization, and a "trajectory" column
                            with the index of the corresponding trajectory
        diffusivities   :   1D ndarray of shape (K,), the set of diffusion
                            coefficients, one for each state
        prior           :   1D ndarray of shape (K,), the prior over the 
                            state occupations. If *None*, we use a uniform
                            prior.
        n_iter          :   int, the number of iterations to do
        burnin          :   int, the number of iterations to ignore at the
                            beginning
        use_entire_track:   bool, use every displacement from every trajectory
                            for state occupation estimation
        max_jumps_per_track :   int, the maximum number of displacements to 
                            consider from each trajectory if *use_entire_track*
                            is False.
        frame_interval  :   float, the frame interval in seconds
        loc_error       :   float, the localization error in um
        pixel_size_um   :   float, the size of pixels in um
        dz              :   float, thickness of the focal plane in um. The 
                            default, *np.inf*, indicates an effectively 
                            infinite focal depth and perfect recovery of 
                            all displacements.
        verbose         :   bool
        pseudocounts    :   int, the weight of the prior. Only relevant if 
                            *prior* is *None* (that is, using a uniform prior)
        n_threads       :   int, the number of parallel threads to use. Each
                            thread executes a complete Gibbs sampling routine
                            with *n_iter* iterations, and the posterior mean
                            estimates are averaged at the end
        track_diffusivities_out_csv     :   str, file to save the individual
                            trajectory diffusivities to. The result is indexed
                            by trajectory and each column corresponds to the 
                            mean posterior weight of a given diffusivity
                            for that trajectory.
        defoc_corr      :   str, the type of defocalization correction to use
                            when weighting the diffusivity likelihoods. Either
                            "first_only", "by_length", or "none".

                            If "first_only", then the likelihood of a particular
                            diffusive state given some trajectory T is assumed 
                            to be proportional only to the probability of seeing
                            the first displacement from that trajectory.

                            If "by_length", then the likelihood of a particular
                            diffusive state given some trajectory T is assumed
                            to be proportional to the probability of seeing a 
                            trajectory of that length, given that diffusive state.

                            If "none", then the likelihood of each diffusive state
                            is assumed to depend only on the jumps themselves, not
                            on the trajectory length.

                            Note that we still correct for state occupation corrections
                            due to underobservation bias at the end, regardless of 
                            the value of *defoc_corr*. *defoc_corr* only affects the
                            estimation during the actual Gibbs sampling routine.

    returns
    -------
        1D ndarray of shape (K,), the mean of the posterior distribution
            over state occupations

    """
    diffusivities = np.asarray(diffusivities)

    # The number of diffusing states
    K = diffusivities.shape[0]

    # If no prior is specified, generate a uniform prior
    if prior is None:
        prior = np.ones(K, dtype=np.float64) * pseudocounts
    else:
        prior = np.asarray(prior)
        assert prior.shape == diffusivities.shape, "prior must have the same " \
            "number of diffusing states as diffusivities"

    # If using the entire track, set the number of frame intervals to 
    # consider to the maximum observed in the dataset
    if use_entire_track:
        tracks = track_length(tracks)
        n_frames = tracks["track_length"].max()
    else:
        n_frames = max_jumps_per_track 

    # Calculate squared radial displacements
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=None, n_frames=n_frames,
        pixel_size_um=pixel_size_um)

    # Probability to observe a single jump from each diffusive state in 
    # this focal volume, assuming the jump starts from a point that is 
    # chosen randomly from inside the focal volume
    f_remain_one_interval = np.empty(K, dtype=np.float64)
    for j, D in enumerate(diffusivities):
        f_remain_one_interval[j] = defoc_prob_brownian(D, 1, 
            frame_interval=frame_interval, dz=dz, n_gaps=0)[0]

    # Adjust prior for defocalization
    if not dz is np.inf:
        prior = prior * f_remain_one_interval

    # State bias terms to use during Gibbs sampling. These reflect the 
    # idea that, if we have a trajectory under which two different diffusivities
    # have the same naive likelihood (considering only the magnitudes of each
    # displacement), then the one with the lower diffusivity is actually more
    # likely because there is a higher chance that it is observed before
    # defocalization.
    F_remain = np.empty((n_frames, K), dtype=np.float64)
    if defoc_corr == "none" or (dz is np.inf):

        # The likelihood of each diffusive state given a particular trajectory
        # is not assumed to depend at all on the trajectory length
        F_remain[:] = 1.0 / K 

    elif defoc_corr == "first_only":

        # Here, F_remain[i,j] is the probability that we observe a displacement
        # from diffusivity j, regardless of the identity of the trajectory i
        for j in range(K):
            F_remain[:,j] = f_remain_one_interval[j] / f_remain_one_interval.sum()

    elif defoc_corr == "by_length":

        # Here, element F[i,j] is the relative probability that at least 
        # *i+1* frames are observed for diffusivity *j*
        for j, D in enumerate(diffusivities):
            F_remain[:,j] = defoc_prob_brownian(D, n_frames,
                frame_interval=frame_interval, dz=dz, n_gaps=0)

        # Normalize over diffusivities. As a result, if we had an equal
        # fraction of trajectories that start with each diffusivity at 
        # random points in the focal volume, then F[i,j] gives the fraction
        # of trajectories with diffusivity j after i+1 frames
        F_remain = (F_remain.T / F_remain.sum(axis=1)).T 

    ## EXPERIMENTAL

    # F_remain = np.zeros((n_frames, K), dtype=np.float64)
    # for j, D in enumerate(diffusivities):

    ##


    # Evaluate the likelihood of each trajectory given each diffusive state.
    # The result, *L[i,j]*, gives the likelihood of diffusivity j given 
    # trajectory i, and is proportional to the probability to observe trajectory
    # i given diffusivity j
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(vecs,
        diffusivities, state_biases=F_remain, frame_interval=frame_interval,
        loc_error=loc_error, n_frames=n_frames)

    # The number of displacements corresponding to each trajectory, which is 
    # the statistical weight of each trajectory toward state vector estimation
    n_disps = track_lengths - 1

    assert len(track_indices) == n_tracks
    print("Total trajectory count: {}".format(n_tracks))

    @dask.delayed
    def _gibbs_sample(thread_idx, verbose=False):
        """
        Run one instance of Gibbs sampling.

        args
        ----
            thread_idx      :   int, the index of the thread
            verbose         :   bool, show the current iteration

        returns
        -------
            1D ndarray of shape (K,), the estimated posterior mean
                over state occupations.

        """
        # Draw the initial estimate for the state occupations from the prior
        p = np.random.dirichlet(prior)

        # The accumulating posterior mean
        mean_p = np.zeros(K, dtype=np.float64)

        # For sampling from the random state occupation vector
        cdf = np.zeros(n_tracks, dtype=np.float64)
        unassigned = np.ones(n_tracks, dtype=np.bool)
        viable = np.zeros(n_tracks, dtype=np.bool)

        # The total counts of each state in the state occupation vector
        # n = np.zeros(K, dtype=np.int64)
        n = np.zeros(K, dtype=np.float64)

        # Sampling loop
        for iter_idx in range(n_iter):

            # Calculate the probability of each diffusive state, given each 
            # trajectory and the current parameter values
            T = L * p 
            T = (T.T / T.sum(axis=1)).T 

            # Draw a state occupation vector from the current set of probabilities
            # for each diffusive state, then count the number of instances of each
            # state among all trajectories
            u = np.random.random(size=n_tracks)
            unassigned[:] = True
            cdf[:] = 0
            n[:] = 0
            for j in range(K):
                cdf += T[:,j]
                viable = np.logical_and(u<=cdf, unassigned)

                # Accumulate the weight toward the jth state. This is either 
                # proportional to the number of displacements corresponding to 
                # trajectories that have been assigned this state (if weight_by_number_of_disps
                # is True), or simply to the number of trajectories corresponding
                # to this state (if weight_by_number_of_disps is False).
                if mode == "by_displacement":
                    n[j] = n_disps[viable].sum()
                else:
                    n[j] = viable.sum()
                unassigned[viable] = False

            # Whatever trajectories remain (perhaps due to floating point error), throw 
            # into the last bin
            if mode == "by_displacement":
                n[-1] += n_disps[unassigned].sum()
            else:
                n[-1] += unassigned.sum()

            # Determine the posterior distribution over the state occupations
            posterior = prior + n 

            # Draw a new state occupation vector
            p = np.random.dirichlet(posterior)

            # If after the burnin period, accumulate this estimate into the posterior
            # mean estimate
            if iter_idx > burnin:
                mean_p += p

            if verbose and iter_idx % 10 == 0:
                sys.stdout.write("Finished with %d/%d iterations...\r" % (iter_idx, n_iter))
                sys.stdout.flush()

        mean_p /= (n_iter - burnin)
        return mean_p 

    # Run multi-threaded if the specified number of threads is greater than 1
    if n_threads == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"
    jobs = [_gibbs_sample(j, verbose=(j==0)) for j in range(n_threads)]
    posterior_means = dask.compute(*jobs, scheduler=scheduler, num_workers=n_threads)

    # Accumulate the posterior means across threads
    posterior_means = np.asarray(posterior_means)
    posterior_means = posterior_means.sum(axis=0)
    posterior_means /= posterior_means.sum()

    # Correct for the probability of defocalization at one frame interval
    # (resulting in no trajectory)
    posterior_means = posterior_means / f_remain_one_interval
    posterior_means /= posterior_means.sum()

    # If desired, evaluate the likelihoods of each diffusivity for each trajectory
    # under the model specified by the posterior mean. Then save these to a csv
    if not track_diffusivities_out_csv is None:

        # Calculate the probability of each diffusive state, given each trajectory
        # and the posterior model distribution of diffusivities
        T = L * posterior_means 
        T = (T.T / T.sum(axis=1)).T 

        # Format as a pandas.DataFrame
        columns = ["%.5f" % d for d in diffusivities]
        out_df = pd.DataFrame(T, columns=columns)
        out_df["trajectory"] = track_indices

        # Save 
        out_df.to_csv(track_diffusivities_out_csv, index=False)

    return posterior_means

def gsdiff_alt_alt(tracks, diffusivities, prior=None, n_iter=1000, burnin=500,
    use_entire_track=True, max_jumps_per_track=np.inf, frame_interval=0.01,
    loc_error=0.0, pixel_size_um=1.0, dz=np.inf, verbose=True, pseudocounts=1,
    n_threads=1, track_diffusivities_out_csv=None, mode="by_displacement",
    defoc_corr="first_only", damp=0.1, diagnostic=True,
    likelihood_mode="binned", start_frame=0):

    diffusivities = np.asarray(diffusivities)

    # Treat the diffusivities array as bin edges, with the center of 
    # each bin defined as the logistic mean of the edges
    if likelihood_mode in ["binned", "binned_reg"]:
        K = diffusivities.shape[0] - 1
        diffusivities_mid = np.sqrt(diffusivities[1:] * diffusivities[:-1])

    # Treat the diffusivities array as points, with likelihoods 
    # representative of the entire neighborhood of diffusivities
    elif likelihood_mode == "point":
        K = diffusivities.shape[0]
        diffusivities_mid = diffusivities 

    # Calculate defocalization probabilities for each state after one 
    # frame interval
    f_remain_one_interval = np.empty(K, dtype=np.float64)
    for j, D in enumerate(diffusivities_mid):
        f_remain_one_interval[j] = defoc_prob_brownian(D, 1, 
            frame_interval=frame_interval, dz=dz, n_gaps=0)[0]

    # Choose the prior
    if dz is np.inf:
        prior = np.ones(K, dtype=np.float64) * pseudocounts 
    else:
        prior = f_remain_one_interval * pseudocounts / f_remain_one_interval.max()

    # Evaluate the likelihood of each trajectory given each diffusive state
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood_new(
        tracks, diffusivities, state_biases=f_remain_one_interval,
        frame_interval=frame_interval, loc_error=loc_error,
        use_entire_track=use_entire_track, max_jumps_per_track=max_jumps_per_track,
        pixel_size_um=pixel_size_um, start_frame=start_frame,
        likelihood_mode=likelihood_mode)

    # # If using the entire track, set the number of frame intevals
    # # to consider to the maximum observed in the dataset
    # if use_entire_track:
    #     tracks = track_length(tracks)
    #     n_frames = tracks["track_length"].max()
    # else:
    #     n_frames = max_jumps_per_track 

    # # Evaluate the likelihood of each trajectory given each diffusive state
    # if min_n_disps == 1:
    #     L, track_indices, track_lengths = evaluate_diffusivity_likelihood_alt(
    #         tracks, diffusivities, state_biases=f_remain_one_interval,
    #         frame_interval=frame_interval, loc_error=loc_error, n_frames=n_frames,
    #         pixel_size_um=pixel_size_um, start_frame=None)
    # elif min_n_disps == 2:
    #     L, track_indices, track_lengths = evaluate_diffusivity_likelihood_alt_alt(
    #         tracks, diffusivities, state_biases=f_remain_one_interval,
    #         frame_interval=frame_interval, loc_error=loc_error, n_frames=n_frames,
    #         pixel_size_um=pixel_size_um, start_frame=None)       
    n_tracks = len(track_indices)
    print("Total trajectory count: {}".format(n_tracks))

    # The number of displacements corresponding to each trajectory, which is 
    # the statistical weight of each trajectory toward state vector estimation
    n_disps = track_lengths - 1

    if diagnostic:

        # Show the distribution of diffusivities across all trajectories
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax[0].plot(diffusivities_mid, L.mean(axis=0), color="k")
        ax[1].plot(diffusivities_mid, (L.T * track_lengths).sum(axis=1), color="k")
        for j in range(2):
            ax[j].set_xscale("log")
            ax[j].set_ylabel("likelihood")
            ax[j].set_xlabel("Diffusivity ($\mu$m$^{2}$ s$^{-1}$")
        ax[0].set_title("Summed across trajectories")
        ax[1].set_title("Summed across trajectories and weighted by # disps")
        plt.tight_layout(); plt.show(); plt.close()

        # Show some sample trajectory likelihoods
        for i in range(20):
            print("Trajectory %d:" % i)
            plt.plot(diffusivities_mid, L[i,:], color="k")
            plt.title("Sample trajectory %d" % i)
            plt.xscale("log")
            plt.xlabel("Diffusivity")
            plt.ylabel("Likelihood")
            plt.show(); plt.close()

    @dask.delayed
    def _gibbs_sample(thread_idx, verbose=False):
        """
        Run one instance of Gibbs sampling.

        args
        ----
            thread_idx      :   int, the index of the thread
            verbose         :   bool, show the current iteration

        returns
        -------
            1D ndarray of shape (K,), the estimated posterior mean
                over state occupations.

        """
        # Draw the initial estimate for the state occupations from the prior
        p = np.random.dirichlet(prior)

        # The accumulating posterior mean
        # mean_p = np.zeros(K, dtype=np.float64)
        mean_n = np.zeros(K, dtype=np.float64)

        # For sampling from the random state occupation vector
        cdf = np.zeros(n_tracks, dtype=np.float64)
        unassigned = np.ones(n_tracks, dtype=np.bool)
        viable = np.zeros(n_tracks, dtype=np.bool)

        # The total counts of each state in the state occupation vector
        n = np.zeros(K, dtype=np.float64)

        # Sampling loop
        for iter_idx in range(n_iter):

            # Calculate the probability of each diffusive state, given each 
            # trajectory and the current parameter values
            T = L * p 

            # Normalize over diffusivities
            T = (T.T / T.sum(axis=1)).T 

            # Draw a state occupation vector from the current set of probabilities
            # for each diffusive state, then count the number of instances of each
            # state among all trajectories
            u = np.random.random(size=n_tracks)
            unassigned[:] = True
            cdf[:] = 0
            n[:] = 0
            for j in range(K):
                cdf += T[:,j]
                viable = np.logical_and(u<=cdf, unassigned)

                # Accumulate the weight toward the jth state. This is either 
                # proportional to the number of displacements corresponding to 
                # trajectories that have been assigned this state (if weight_by_number_of_disps
                # is True), or simply to the number of trajectories corresponding
                # to this state (if weight_by_number_of_disps is False).
                if mode == "by_displacement":
                    n[j] = n_disps[viable].sum()
                else:
                    n[j] = viable.sum()
                unassigned[viable] = False

            # Whatever trajectories remain (perhaps due to floating point error), throw into the last bin
            if mode == "by_displacement":
                n[-1] += n_disps[unassigned].sum()
            else:
                n[-1] += unassigned.sum()

            # Determine the posterior distribution over the state occupations
            posterior = prior + n * damp

            # Draw a new state occupation vector
            p = np.random.dirichlet(posterior)

            if diagnostic and verbose and iter_idx % 200 == 0:
                fig, ax = plt.subplots(6, 1, figsize=(8, 10))
                ax[0].plot(diffusivities_mid, L.sum(axis=0), color="k")
                ax[0].set_xscale("log")
                ax[0].set_xlabel("diffusivity")
                ax[0].set_ylabel("naive likelihood")

                ax[1].plot(diffusivities_mid, T.sum(axis=0), color="k")
                ax[1].set_xscale("log")
                ax[1].set_ylabel("summed likelihood")
                ax[1].set_xlabel("Diffusivity")

                ax[2].plot(diffusivities_mid, n, color="r")
                ax[2].set_title("n")
                ax[2].set_xlabel("Diffusivity")
                ax[2].set_ylabel("count")
                ax[2].set_xscale("log")

                ax[3].plot(diffusivities_mid, p, color="b")
                ax[3].set_xscale("log")
                ax[3].set_xlabel("Diffusivity")
                ax[3].set_ylabel("p")

                ax[4].plot(diffusivities_mid, mean_p, color="k")
                ax[4].set_xscale("log")
                ax[4].set_xlabel("Diffusivity")
                ax[4].set_ylabel("mean_p")

                ax[5].plot(diffusivities_mid, mean_n, color="k")
                ax[5].set_xscale("log")
                ax[5].set_xlabel("Diffusivity")
                ax[5].set_ylabel("mean_n")

                plt.tight_layout()
                plt.show(); plt.close()           

            # If after the burnin period, accumulate this estimate into the posterior
            # mean estimate
            if iter_idx > burnin:
                # mean_p += p
                mean_n += n

            if verbose and iter_idx % 10 == 0:
                sys.stdout.write("Finished with %d/%d iterations...\r" % (iter_idx, n_iter))
                sys.stdout.flush()

        # mean_p /= (n_iter - burnin)
        # return mean_p 

        mean_n /= mean_n.sum()
        return mean_n

    # Run multi-threaded if the specified number of threads is greater than 1
    if n_threads == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"
    jobs = [_gibbs_sample(j, verbose=(j==0)) for j in range(n_threads)]
    posterior_means = dask.compute(*jobs, scheduler=scheduler, num_workers=n_threads)

    # Accumulate the posterior means across threads
    posterior_means = np.asarray(posterior_means)
    posterior_means = posterior_means.sum(axis=0)
    posterior_means /= posterior_means.sum()

    # Correct for the probability of defocalization at one frame interval
    # (resulting in no trajectory)
    posterior_means = posterior_means / f_remain_one_interval
    posterior_means /= posterior_means.sum()

    # If desired, evaluate the likelihoods of each diffusivity for each trajectory
    # under the model specified by the posterior mean. Then save these to a csv
    if not track_diffusivities_out_csv is None:

        # Calculate the probability of each diffusive state, given each trajectory
        # and the posterior model distribution of diffusivities
        T = L * posterior_means 
        T = (T.T / T.sum(axis=1)).T 

        # Format as a pandas.DataFrame
        columns = ["%.5f" % d for d in diffusivities]
        out_df = pd.DataFrame(T, columns=columns)
        out_df["trajectory"] = track_indices

        # Save 
        out_df.to_csv(track_diffusivities_out_csv, index=False)

    return posterior_means, diffusivities_mid

def gsdiff_alt(tracks, n_iter=1000, burnin=500,
    use_entire_track=True, max_jumps_per_track=50, frame_interval=0.01,
    loc_error=0.0, pixel_size_um=1.0, dz=np.inf, verbose=True, pseudocounts=1,
    n_threads=1, track_diffusivities_out_csv=None, mode="by_displacement",
    defoc_corr="first_only"):
    """
    Use Gibbs sampling to estimate the posterior distribution of the 
    state occupations for a multistate Brownian motion with no state
    transitions, given an observed set of trajectories.

    args
    ----
        tracks          :   pandas.DataFrame, the observed trajectories.
                            Must contain the "y" and "x" columns with
                            the positions of the localizations in pixels,
                            a "frame" column with the frame index of 
                            each localization, and a "trajectory" column
                            with the index of the corresponding trajectory
        n_iter          :   int, the number of iterations to do
        burnin          :   int, the number of iterations to ignore at the
                            beginning
        use_entire_track:   bool, use every displacement from every trajectory
                            for state occupation estimation
        max_jumps_per_track :   int, the maximum number of displacements to 
                            consider from each trajectory if *use_entire_track*
                            is False.
        frame_interval  :   float, the frame interval in seconds
        loc_error       :   float, the localization error in um
        pixel_size_um   :   float, the size of pixels in um
        dz              :   float, thickness of the focal plane in um. The 
                            default, *np.inf*, indicates an effectively 
                            infinite focal depth and perfect recovery of 
                            all displacements.
        verbose         :   bool
        pseudocounts    :   int, the weight of the prior. Only relevant if 
                            *prior* is *None* (that is, using a uniform prior)
        n_threads       :   int, the number of parallel threads to use. Each
                            thread executes a complete Gibbs sampling routine
                            with *n_iter* iterations, and the posterior mean
                            estimates are averaged at the end
        track_diffusivities_out_csv     :   str, file to save the individual
                            trajectory diffusivities to. The result is indexed
                            by trajectory and each column corresponds to the 
                            mean posterior weight of a given diffusivity
                            for that trajectory.
        defoc_corr      :   str, the type of defocalization correction to use
                            when weighting the diffusivity likelihoods. Either
                            "first_only", "by_length", or "none".

                            If "first_only", then the likelihood of a particular
                            diffusive state given some trajectory T is assumed 
                            to be proportional only to the probability of seeing
                            the first displacement from that trajectory.

                            If "by_length", then the likelihood of a particular
                            diffusive state given some trajectory T is assumed
                            to be proportional to the probability of seeing a 
                            trajectory of that length, given that diffusive state.

                            If "none", then the likelihood of each diffusive state
                            is assumed to depend only on the jumps themselves, not
                            on the trajectory length.

                            Note that we still correct for state occupation corrections
                            due to underobservation bias at the end, regardless of 
                            the value of *defoc_corr*. *defoc_corr* only affects the
                            estimation during the actual Gibbs sampling routine.

    returns
    -------
        1D ndarray of shape (K,), the mean of the posterior distribution
            over state occupations

    """
    subdiffusivities = np.logspace(-2.0, 2.0, 8000)
    # diffusivities = subdiffusivities.reshape((100, 80)).mean(axis=1)
    diffusivities = np.median(subdiffusivities.reshape((100, 80)), axis=1)

    # The number of diffusing states
    K = diffusivities.shape[0]
    sub_K = subdiffusivities.shape[0]

    # If no prior is specified, generate a uniform prior
    subprior = np.ones(sub_K, dtype=np.float64) * pseudocounts

    # If using the entire track, set the number of frame intervals to 
    # consider to the maximum observed in the dataset
    if use_entire_track:
        tracks = track_length(tracks)
        n_frames = tracks["track_length"].max()
    else:
        n_frames = max_jumps_per_track 

    # Calculate squared radial displacements
    if verbose: print("Calculating trajectory displacements...")
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=None, n_frames=n_frames,
        pixel_size_um=pixel_size_um)

    # Probability to observe a single jump from each diffusive state in 
    # this focal volume, assuming the jump starts from a point that is 
    # chosen randomly from inside the focal volume
    if verbose: print("Evaluating defocalization probabilities...")
    f_remain_one_interval = np.empty(sub_K, dtype=np.float64)
    for j, D in enumerate(subdiffusivities):
        f_remain_one_interval[j] = defoc_prob_brownian(D, 1, 
            frame_interval=frame_interval, dz=dz, n_gaps=0)[0]

    # Adjust prior for defocalization
    if not dz is np.inf:
        subprior = subprior * f_remain_one_interval
    prior = subprior.reshape((100, 80)).mean(axis=1)

    # Mean defocalization probability
    f_remain_one_interval_diff = f_remain_one_interval.reshape((100, 80)).mean(axis=1)

    # State bias terms to use during Gibbs sampling. These reflect the 
    # idea that, if we have a trajectory under which two different diffusivities
    # have the same naive likelihood (considering only the magnitudes of each
    # displacement), then the one with the lower diffusivity is actually more
    # likely because there is a higher chance that it is observed before
    # defocalization.
    F_remain = np.empty((n_frames, sub_K), dtype=np.float64)
    if defoc_corr == "none" or (dz is np.inf):

        # The likelihood of each diffusive state given a particular trajectory
        # is not assumed to depend at all on the trajectory length
        F_remain[:] = 1.0 / sub_K 

    elif defoc_corr == "first_only":

        # Here, F_remain[i,j] is the probability that we observe a displacement
        # from diffusivity j, regardless of the identity of the trajectory i
        for j in range(sub_K):
            F_remain[:,j] = f_remain_one_interval[j] / f_remain_one_interval.sum()

    elif defoc_corr == "by_length":

        # Here, element F[i,j] is the relative probability that at least 
        # *i+1* frames are observed for diffusivity *j*
        for j, D in enumerate(subdiffusivities):
            F_remain[:,j] = defoc_prob_brownian(D, n_frames,
                frame_interval=frame_interval, dz=dz, n_gaps=0)

        # Normalize over diffusivities. As a result, if we had an equal
        # fraction of trajectories that start with each diffusivity at 
        # random points in the focal volume, then F[i,j] gives the fraction
        # of trajectories with diffusivity j after i+1 frames
        F_remain = (F_remain.T / F_remain.sum(axis=1)).T 

    # Evaluate the likelihood of each trajectory given each diffusive state.
    # The result, *L[i,j]*, gives the likelihood of diffusivity j given 
    # trajectory i, and is proportional to the probability to observe trajectory
    # i given diffusivity j
    if verbose: print("Evaluating diffusivity likelihoods on an upsampled grid (can take a few minutes)...")
    sub_L, track_indices, track_lengths = evaluate_diffusivity_likelihood(vecs,
        subdiffusivities, state_biases=F_remain, frame_interval=frame_interval,
        loc_error=loc_error, n_frames=n_frames)

    # Aggregate the diffusivities
    L = np.zeros((n_tracks, K), dtype=np.float64)
    for j in range(K):

        # # VERSION 1
        # L[:,j] = sub_L[:,j*80:(j+1)*80].mean(axis=1)

        # # VERSION 2
        # weights = np.log10(subdiffusivities)[j*80:(j+1)*80]
        # weights /= weights.sum()
        # L[:,j] = (sub_L[:,j*80:(j+1)*80] * weights).sum(axis=1)

        # # VERSION 3
        # weights = subdiffusivities[j*80:(j+1)*80]
        # weights /= weights.sum()
        # L[:,j] = (sub_L[:,j*80:(j+1)*80] * weights).sum(axis=1)

        # # VERSION 4
        # weights = (1.0/subdiffusivities)[j*80:(j+1)*80]
        # weights /= weights.sum()
        # L[:,j] = (sub_L[:,j*80:(j+1)*80] * weights).sum(axis=1)

        # # VERSION 5
        # weights = np.power(10.0, subdiffusivities)[j*80:(j+1)*80]
        # weights /= weights.sum()
        # L[:,j] = (sub_L[:,j*80:(j+1)*80] * weights).sum(axis=1)

        # # VERSION 6
        # weights = 1.0/np.power(10.0, subdiffusivities)[j*80:(j+1)*80]
        # weights /= weights.sum()
        # L[:,j] = (sub_L[:,j*80:(j+1)*80] * weights).sum(axis=1)

        # # VERSION 7
        # L[:,j] = np.median(sub_L[:,j*80:(j+1)*80], axis=1)

        # # VERSION 8
        # L[:,j] = sub_L[:,j*80:(j+1)*80].max(axis=1)

        # # VERSION 9
        # L[:,j] = sub_L[:,j*80:(j+1)*80].sum(axis=1)

        # VERSION 10 (first apply antialiasing filter, then take midpoint)
        pass 

    # VERSION 10 (first apply antialiasing filter, then take midpoint)
    sub_L = np.apply_along_axis(ndi.gaussian_filter, 1, sub_L, 320)
    L = sub_L[:,40::80]

    del sub_L

    # The number of displacements corresponding to each trajectory, which is 
    # the statistical weight of each trajectory toward state vector estimation
    n_disps = track_lengths - 1

    assert len(track_indices) == n_tracks
    print("Total trajectory count: {}".format(n_tracks))

    @dask.delayed
    def _gibbs_sample(thread_idx, verbose=False):
        """
        Run one instance of Gibbs sampling.

        args
        ----
            thread_idx      :   int, the index of the thread
            verbose         :   bool, show the current iteration

        returns
        -------
            1D ndarray of shape (K,), the estimated posterior mean
                over state occupations.

        """
        # Draw the initial estimate for the state occupations from the prior
        p = np.random.dirichlet(prior)

        # The accumulating posterior mean
        mean_p = np.zeros(K, dtype=np.float64)

        # For sampling from the random state occupation vector
        cdf = np.zeros(n_tracks, dtype=np.float64)
        unassigned = np.ones(n_tracks, dtype=np.bool)
        viable = np.zeros(n_tracks, dtype=np.bool)

        # The total counts of each state in the state occupation vector
        # n = np.zeros(K, dtype=np.int64)
        n = np.zeros(K, dtype=np.float64)

        # Sampling loop
        for iter_idx in range(n_iter):

            # Calculate the probability of each diffusive state, given each 
            # trajectory and the current parameter values
            T = L * p 
            T = (T.T / T.sum(axis=1)).T 

            # Draw a state occupation vector from the current set of probabilities
            # for each diffusive state, then count the number of instances of each
            # state among all trajectories
            u = np.random.random(size=n_tracks)
            unassigned[:] = True
            cdf[:] = 0
            n[:] = 0
            for j in range(K):
                cdf += T[:,j]
                viable = np.logical_and(u<=cdf, unassigned)

                # Accumulate the weight toward the jth state. This is either 
                # proportional to the number of displacements corresponding to 
                # trajectories that have been assigned this state (if weight_by_number_of_disps
                # is True), or simply to the number of trajectories corresponding
                # to this state (if weight_by_number_of_disps is False).
                if mode == "by_displacement":
                    n[j] = n_disps[viable].sum()
                else:
                    n[j] = viable.sum()
                unassigned[viable] = False

            # Whatever trajectories remain (perhaps due to floating point error), throw into the last bin
            if mode == "by_displacement":
                n[-1] += n_disps[unassigned].sum()
            else:
                n[-1] += unassigned.sum()

            # Determine the posterior distribution over the state occupations
            posterior = prior + n 

            # Draw a new state occupation vector
            p = np.random.dirichlet(posterior)

            # If after the burnin period, accumulate this estimate into the posterior
            # mean estimate
            if iter_idx > burnin:
                mean_p += p

            if verbose and iter_idx % 10 == 0:
                sys.stdout.write("Finished with %d/%d iterations...\r" % (iter_idx, n_iter))
                sys.stdout.flush()

        mean_p /= (n_iter - burnin)
        return mean_p 

    # Run multi-threaded if the specified number of threads is greater than 1
    if verbose: print("Running Gibbs sampling...")
    if n_threads == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"
    jobs = [_gibbs_sample(j, verbose=(j==0)) for j in range(n_threads)]
    posterior_means = dask.compute(*jobs, scheduler=scheduler, num_workers=n_threads)

    # Accumulate the posterior means across threads
    posterior_means = np.asarray(posterior_means)
    posterior_means = posterior_means.sum(axis=0)
    posterior_means /= posterior_means.sum()

    # Correct for the probability of defocalization at one frame interval
    # (resulting in no trajectory)
    posterior_means = posterior_means / f_remain_one_interval_diff
    posterior_means /= posterior_means.sum()

    # If desired, evaluate the likelihoods of each diffusivity for each trajectory
    # under the model specified by the posterior mean. Then save these to a csv
    if not track_diffusivities_out_csv is None:

        # Calculate the probability of each diffusive state, given each trajectory
        # and the posterior model distribution of diffusivities
        T = L * posterior_means 
        T = (T.T / T.sum(axis=1)).T 

        # Format as a pandas.DataFrame
        columns = ["%.5f" % d for d in diffusivities]
        out_df = pd.DataFrame(T, columns=columns)
        out_df["trajectory"] = track_indices

        # Save 
        out_df.to_csv(track_diffusivities_out_csv, index=False)

    return posterior_means, diffusivities

def associate_diffusivity(tracks, track_diffusivities, diffusivity):
    """
    Map diffusivity likelihoods back to their respective trajectories.

    The purpose of this function is to take the output of 
    evaluate_diffusivity_likelihoods_on_tracks and map one of the 
    components of the diffusivity likelihood vector back onto the 
    original trajectories.

    args
    ----
        tracks      :   pandas.DataFrame, a set of trajectories
        track_diffusivities     :   pandas.DataFrame, the probability
                       of each diffusivity for each trajectory
        diffusivity     :   float or str, the specific diffusivity
                        to associate into the track CSV

    returns
    -------
        pandas.DataFrame, the tracks CSV with the new column

    """
    if isinstance(diffusivity, float):
        diffusivity = str(diffusivity)

    col = "diffusivity_{}".format(diffusivity)
    tracks[col] = tracks["trajectory"].map(track_diffusivities[diffusivity])

    return tracks 

