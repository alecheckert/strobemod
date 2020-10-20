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

# Default support on which to evaluate diffusivity occupations
DEFAULT_DIFFUSIVITIES = np.logspace(-2.0, 2.0, 301)


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

def rad_disp_squared(tracks, start_frame=None, n_frames=4, pixel_size_um=1.0, 
    min_track_length=2):
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
        min_track_length:   int, the minimum trajectory length to consider in frames

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

    # Exclude trajectories that are too short, if desired
    tracks = track_length(tracks)
    if min_track_length > 2:
        tracks = tracks[tracks["track_length"] >= min_track_length]

    # Only consider trajectories after some start frame
    if not start_frame is None:
        tracks = tracks[tracks['frame'] >= start_frame]

    # Convert from pixels to um
    tracks[['y', 'x']] = tracks[['y', 'x']] * pixel_size_um 

    # Throw out all points in each trajectory after the first *n_frames+1*,
    # so that we have a maximum of *n_frames* displacements in the resulting
    # set of trajectories
    tracks = assign_index_in_track(tracks)
    tracks = tracks[tracks["index_in_track"] <= n_frames].copy()

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
    frame_interval=0.01, loc_error=0.0, pixel_size_um=1.0, dz=0.7,
    use_entire_track=True, max_jumps_per_track=np.inf, likelihood_mode="binned"):
    """
    Given a set of trajectories and a particular mixture model of 
    diffusivities, evaluate the probability of each trajectory given 
    each separate diffusivity.

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
    diffusivities = np.asarray(diffusivities)
    if likelihood_mode in ["binned", "binned_reg"]:
        K = diffusivities.shape[0]-1
        diffusivities_mid = np.sqrt(diffusivities[1:] * diffusivities[:-1])

    elif likelihood_mode == "point":
        K = diffusivities.shape[0]
        diffusivities_mid = diffusivities

    # Calculate defocalization probabilities for each state after one 
    # frame interval
    f_remain_one_interval = np.empty(K, dtype=np.float64)
    for j, D in enumerate(diffusivities_mid):
        f_remain_one_interval[j] = defoc_prob_brownian(D, 1, 
            frame_interval=frame_interval, dz=dz, n_gaps=0)[0]

    # Evaluate the likelihood of each diffusivity, given each trajectory. The
    # result, *L[i,j]*, gives the likelihood to observe trajectory i under 
    # diffusive state j
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(
        tracks, diffusivities, state_biases=f_remain_one_interval,
        frame_interval=frame_interval, loc_error=loc_error, 
        use_entire_track=use_entire_track, max_jumps_per_track=max_jumps_per_track,
        pixel_size_um=pixel_size_um, likelihood_mode=likelihood_mode)

    # Format the result as a dataframe
    columns = ["%.5f" % d for d in diffusivities_mid]
    L = pd.DataFrame(L, columns=columns)
    L["trajectory"] = track_indices 

    return L 

def evaluate_diffusivity_likelihood(tracks, diffusivities, state_biases=None,
    frame_interval=0.01, loc_error=0.0, use_entire_track=True, max_jumps_per_track=10,
    min_jumps_per_track=1, pixel_size_um=1.0, start_frame=None, likelihood_mode="binned"):
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
    if max_jumps_per_track is None:
        max_jumps_per_track = np.inf 
    assert likelihood_mode in ["point", "binned", "binned_reg"]

    # If using the entire track, set n_frames to the longest trajectory
    # in the sample. Otherwise set it to *max_frames*
    max_track_len = tracks.groupby("trajectory").size().max()
    n_frames = max_track_len if use_entire_track else \
        min(max_track_len, max_jumps_per_track)

    # Calculate squared radial displacements for all trajectories
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=start_frame,
        n_frames=n_frames, pixel_size_um=pixel_size_um,
        min_track_length=min_jumps_per_track+1)

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
            L[:,j] = np.asarray(log_ss - L_cond["sum_squared_disp"] / (4*(D*frame_interval+le2)) - \
                    n_disps * np.log(4*(D*frame_interval+le2)) - log_gamma_n_disps)

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

def emdiff(tracks, diffusivities, n_iter=10000, frame_interval=0.01,
    use_entire_track=True, max_jumps_per_track=np.inf, 
    dz=0.7, loc_error=0.0, pixel_size_um=1.0, verbose=True,
    track_diffusivities_out_csv=None, mode="by_displacement", 
    pseudoprob=0.001, likelihood_mode="point", start_frame=0):
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

    returns
    -------
        1D ndarray of shape diffusivities.shape, the estimated occupations of each 
            diffusive state

    """
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

    # Evaluate the likelihood of each diffusivity, given each trajectory
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(
        tracks, diffusivities, state_biases=f_remain_one_interval,
        frame_interval=frame_interval, loc_error=loc_error,
        use_entire_track=use_entire_track, max_jumps_per_track=max_jumps_per_track,
        pixel_size_um=pixel_size_um, start_frame=start_frame,
        likelihood_mode=likelihood_mode)
    n_tracks = len(track_indices)
    print("Number of trajectories: %d" % n_tracks)

    # The number of displacements corresponding to each trajectory
    track_n_disps = track_lengths - 1
    n_disps_tot = track_n_disps.sum()

    # The probabilities that each observation belongs to each diffusive state,
    # given the set of observations, their accompanying complete likelihood
    # functions, and their likelihoods under the current set of model parameters
    T = np.zeros((n_tracks, K), dtype=np.float64)

    # The current guess at state occupations
    p = np.ones(K, dtype=np.float64) / K 

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
    return p, diffusivities_mid 

def gsdiff_subsample(tracks, diffusivities, n_partitions=6, subsample_size=0.3,
    subsample_type="fraction", n_threads=6, **kwargs):
    """
    Given a set of trajectories, subsample the trajectories and run gsdiff
    to estimate the state occupations of each subsample. Then average the 
    results at the end.

    args
    ----
        tracks              :   pd.DataFrame
        diffusivities       :   1D ndarray
        n_partitions        :   int, the number of subsamples
        subsample_size      :   float or int, the size of each subsample.
                                If *subsample_type* is "fraction", then this
                                is interpreted as the fraction of the original
                                dataset to use for subasmpling. If *subsample_type*
                                is "number", then this is interpreted as the absolute
                                size of each subsample in trajectories
        subsample_type      :   str, either "fraction" or "number"
        **kwargs            :   to gsdiff()

    returns
    -------
        (
            1D ndarray of shape K, the average of each subsample
            1D ndarray of shape K, the diffusivity bin midpoints
        )

    """
    assert subsample_type in ["fraction", "number"]

    # Exclude singlets
    tracks = track_length(tracks)
    tracks = tracks[tracks["track_length"] > 1].copy()
    track_indices = tracks["trajectory"].unique()
    n_tracks = len(track_indices)

    # Choose the subsamples
    if subsample_type == "fraction":
        subsample_size = int(subsample_size * n_tracks)

    elif subsample_type == "number":
        if subsample_size > n_tracks:
            raise RuntimeError("specdiffuse.gsdiff_subsample: subsample_size (%d) " \
                "cannot be greater than the number of unique trajectories (%d)" % (
                    subsample_size, n_tracks))

    subsamples = [np.random.choice(track_indices, size=subsample_size, replace=False) \
        for j in range(n_partitions)]

    # Run gsdiff on a single subsample
    @dask.delayed
    def driver(i):
        subtracks = tracks.loc[tracks["trajectory"].isin(subsamples[i])]
        return gsdiff(subtracks, diffusivities, n_threads=1, **kwargs)

    # Results array
    jobs = [driver(i) for i in range(n_partitions)]
    results = dask.compute(*jobs, scheduler="processes", num_workers=n_threads)
    posterior_means = np.asarray([i[0] for i in results])
    diffusivities_mid = results[0][1]
    print(posterior_means.shape)
    posterior_means = posterior_means.mean(axis=0)
    posterior_means /= posterior_means.sum()
    return posterior_means, diffusivities_mid 

def gsdiff(tracks, diffusivities=None, prior=None, n_iter=1000, burnin=20,
    n_threads=1, frame_interval=0.01, loc_error=0.0, pixel_size_um=0.16,
    dz=np.inf, pseudocounts=2, max_weight=1, damp=1.0, use_entire_track=True,
    max_jumps_per_track=np.inf, min_jumps_per_track=1, verbose=True, 
    mode="by_displacement", likelihood_mode="binned", start_frame=0,
    diagnostic=False, track_diffusivities_out_csv=None):
    """
    Estimate a distribution of diffusivities from a set of trajectories 
    using Gibbs sampling.

    args
    ----
        tracks          :   pandas.DataFrame, trajectories. Must contain
                            the "frame", "trajectory", "y", and "x" columns
        diffusivities   :   1D ndarray of shape K+1, the edges of the diffusivity bins
                            in um^2 s^-1
        prior           :   1D ndarray of shape (K), the prior distribution
                            over the diffusivity bins 
        n_iter          :   int, number of iterations to run
        burnin          :   int, the minimum number of iterations to run 
                            before starting to record the results
        use_entire_track:   bool, use every jump from every trajectory
        max_jumps_per_track: int, if not *use_entire_track*, the max number of 
                            jumps to consider per track
        min_jumps_per_track: int, the minimum number of jumps to consider per
                            track
        frame_interval  :   float, time between frames in seconds
        loc_error       :   float, estimated 1D localization error in um
        pixel_size_um   :   float, size of pixels in um
        dz              :   float, depth of field
        verbose         :   bool
        pseudocounts    :   int, the relative weight of the prior. 1 is a noninformative
                            prior.
        n_threads       :   int, the number of parallel threads to run
        track_diffusivities_out_csv     :   str, path to a CSV at which to
                            save the likelihood of each diffusivity bin for
                            each trajectory in the dataset. Potentially large.
        mode            :   str, either "by_displacement" or "by_trajectory", 
                            the meaning of the counts in the posterior distribution.
                            Unless you have a good reason, use "by_displacement".
        damp            :   float, the relative statistical weight of each 
                            displacement/trajectory. Lower is more conservative.
        diagnostic      :   bool, make some diagnostic reports for debugging
        likelihood_mode :   str, either "binned", "point", or "binned_reg", the
                            type of likelihood function for the diffusivity
                            bins. "binned" is a good default.
        start_frame     :   int, disregard displacements before this frame
        max_weight      :   int, the maximum weight to give trajectories 
                            when building the intermediate sampling distributions
                            over the diffusivity bins. While the "correct" approach
                            is to give each trajectory a weight proportional to the
                            observed number of displacements in that trajectory,
                            in practice this can be a little too confident and a 
                            few trajectories can end up dominating the data. On the
                            other hand, max_weight = 1 is very conservative, only 
                            calling diffusivity peaks when they are extremely well
                            supported by many trajectories.

    returns
    -------
        (
            2D ndarray of shape (n_threads, K), the posterior means over
                the diffusivity bins for each thread;
            1D ndarray of shape (K), the geometric means of each 
                diffusivity bin
        )

    """
    if diffusivities is None:
        diffusivities = DEFAULT_DIFFUSIVITIES
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

    # Calculate defocalization probabilities for each state after the
    # minimum observation time (defined as the minimum number of frame
    # intervals required to include a trajectory)
    f_remain_one_interval = np.empty(K, dtype=np.float64)
    for j, D in enumerate(diffusivities_mid):
        f_remain_one_interval[j] = defoc_prob_brownian(D, 1, 
            frame_interval=frame_interval, dz=dz, n_gaps=0)[-1]

    # Choose the prior
    if dz is np.inf or dz is None:
        prior = np.ones(K, dtype=np.float64) * pseudocounts 
    else:
        prior = f_remain_one_interval * pseudocounts / f_remain_one_interval.max()

    # Evaluate the likelihood of each trajectory given each diffusive state
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(
        tracks, diffusivities, state_biases=f_remain_one_interval,
        frame_interval=frame_interval, loc_error=loc_error,
        use_entire_track=use_entire_track, max_jumps_per_track=max_jumps_per_track,
        pixel_size_um=pixel_size_um, start_frame=start_frame,
        likelihood_mode=likelihood_mode, min_jumps_per_track=min_jumps_per_track)

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
            ax[j].set_xlabel("Diffusivity ($\mu$m$^{2}$ s$^{-1}$)")
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

        # For sampling from the random state occupation vector
        cdf = np.zeros(n_tracks, dtype=np.float64)
        unassigned = np.ones(n_tracks, dtype=np.bool)
        viable = np.zeros(n_tracks, dtype=np.bool)

        # The total counts of each state in the state occupation vector
        n = np.zeros(K, dtype=np.float64)

        # The effective counts of each state in the state occupation vector,
        # for the purposes of damping the contribution of any one trajectory
        n_red = np.zeros(K, dtype=np.float64)

        # The sequence of samples produced by the Gibbs sampler
        samples = np.zeros((n_iter-burnin, K), dtype=np.float64)

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
                    n_red[j] = np.minimum(n_disps[viable], max_weight).sum()
                else:
                    n[j] = viable.sum()
                unassigned[viable] = False

            # Whatever trajectories remain (perhaps due to floating point error), throw into the last bin
            if mode == "by_displacement":
                n[-1] += n_disps[unassigned].sum()
                n_red[-1] += np.minimum(n_disps[unassigned], max_weight).sum()
            else:
                n[-1] += unassigned.sum()

            # Determine the posterior distribution over the state occupations
            posterior = prior + n_red * damp 
            # posterior = prior + np.minimum(n*damp, max_weight) # EXPERIMENTAL

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
            if iter_idx >= burnin:
                samples[iter_idx-burnin,:] = n / f_remain_one_interval 

            if verbose and iter_idx % 10 == 0:
                sys.stdout.write("Finished with %d/%d iterations...\r" % (iter_idx, n_iter))
                sys.stdout.flush()

        return samples 

    # Run multi-threaded if the specified number of threads is greater than 1
    if n_threads == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"
    jobs = [_gibbs_sample(j, verbose=(j==0)) for j in range(n_threads)]
    samples = dask.compute(*jobs, scheduler=scheduler, num_workers=n_threads)
    samples = np.concatenate(samples, axis=0)

    # Normalize over diffusivities for each sample
    samples = (samples.T / samples.sum(axis=1)).T 

    # If desired, evaluate the likelihoods of each diffusivity for each trajectory
    # under the model specified by the posterior mean. Then save these to a csv
    if not track_diffusivities_out_csv is None:

        # Accumulate the posterior means across threads
        posterior_means = samples.mean(axis=0)

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

    return samples, diffusivities_mid

def gsdiff_median(tracks, diffusivities, prior=None, n_iter=1000, burnin=500,
    use_entire_track=True, max_jumps_per_track=np.inf, min_jumps_per_track=1,
    frame_interval=0.01, loc_error=0.0, pixel_size_um=1.0, dz=np.inf,
    verbose=True, pseudocounts=1, n_threads=1, track_diffusivities_out_csv=None,
    mode="by_displacement", defoc_corr="first_only", damp=0.1, diagnostic=True,
    likelihood_mode="binned", start_frame=0):
    """
    Estimate a distribution of diffusivities from a set of trajectories 
    using Gibbs sampling.

    This time, use the posterior geometric median rather than the posterior mean.

    """
    diffusivities = np.asarray(diffusivities)

    # Weiszfeld's algorithm for computing the L2 geometric median
    def weiszfeld(X, convergence=1.0e-8, max_iter=100, reg=1.0e-6):
        """
        args
        ----
            X   :   2D ndarray of shape (n_samples, dimension)

        returns
        -------
            1D ndarray of shape (dimension,)

        """
        prev = X.mean(axis=0)
        for iter_idx in range(max_iter):
            dist = np.abs(X - prev) + reg
            curr = (X/dist).sum(axis=0) / (1.0/dist).sum(axis=0)
            if (np.abs(curr - prev) < convergence).all():
                break 
            else:
                prev = curr 
        return curr 

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

    # Calculate defocalization probabilities for each state after the
    # minimum observation time (defined as the minimum number of frame
    # intervals required to include a trajectory)
    f_remain_one_interval = np.empty(K, dtype=np.float64)
    for j, D in enumerate(diffusivities_mid):
        f_remain_one_interval[j] = defoc_prob_brownian(D, 1, 
            frame_interval=frame_interval, dz=dz, n_gaps=0)[-1]

    # Choose the prior
    if dz is np.inf:
        prior = np.ones(K, dtype=np.float64) * pseudocounts 
    else:
        prior = f_remain_one_interval * pseudocounts / f_remain_one_interval.max()

    # Evaluate the likelihood of each trajectory given each diffusive state
    L, track_indices, track_lengths = evaluate_diffusivity_likelihood(
        tracks, diffusivities, state_biases=f_remain_one_interval,
        frame_interval=frame_interval, loc_error=loc_error,
        use_entire_track=use_entire_track, max_jumps_per_track=max_jumps_per_track,
        pixel_size_um=pixel_size_um, start_frame=start_frame,
        likelihood_mode=likelihood_mode, min_jumps_per_track=min_jumps_per_track)

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

        # The sequence of samples produced by the Gibbs sampler
        samples = np.zeros((n_iter-burnin, K), dtype=np.float64)

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
            if iter_idx >= burnin:
                # mean_p += p
                # mean_n += n
                samples[iter_idx-burnin, :] = n 

            if verbose and iter_idx % 10 == 0:
                sys.stdout.write("Finished with %d/%d iterations...\r" % (iter_idx, n_iter))
                sys.stdout.flush()

        # mean_p /= (n_iter - burnin)
        # return mean_p 

        # mean_n /= mean_n.sum()
        return samples 

    # Run multi-threaded if the specified number of threads is greater than 1
    if n_threads == 1:
        scheduler = "single-threaded"
    else:
        scheduler = "processes"
    jobs = [_gibbs_sample(j, verbose=(j==0)) for j in range(n_threads)]
    sample_sets = dask.compute(*jobs, scheduler=scheduler, num_workers=n_threads)
    sample_sets = np.concatenate(sample_sets, axis=0).astype(np.float64)

    sample_sets = sample_sets / f_remain_one_interval
    return sample_sets, diffusivities_mid 

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

