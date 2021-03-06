#!/usr/bin/env python
"""
dirichlet_process.py -- use a Dirichlet process mixture model to 
estimate the posterior distribution of diffusivities given trajectories
sampled from a Brownian mixture.

Suppose we have *n* trajectories that are labeled i = 1, ..., n. Let
X[i] be the summed 2D radial squared displacement of the i^th trajectory,
let L[i] be the length of the i^th trajectory in frames, and assume that
the diffusivity of the i^th trajectory is some random variable D[i].

Then our model is

    X[i] | D[i] ~ Gamma(L[i] - 1, 1 / (4 * D[i] * dt))
    D[i] | G ~ G
    G ~ DP(alpha, H)

where dt is the frame interval, G is a candidate diffusivity
distribution drawn from the Dirichlet process DP(alpha, H), and 
H is the prior on that distribution. In most of the cases here,
we assume that H is a uniform distribution on the range of 
diffusivities of interest, unless otherwise specified.

Our goal is to evaluate the posterior distribution D | X, 
marginalizing on G. We do this using algorithm 8 from Neal 2000,
which is a hybrid Gibbs sampling/Metropolis scheme based on the
basic conditional/marginal likelihoods from the Blackwell-MacQueen
1973 urn model. See

        Radford M. Neal. Markov Chain Sampling Methods for Dirichlet
        Process Mixture Models. Journal of Computational and Graphical
        Statistics, 9:2, 249-265 (2000).

for more details.

"""
import os
import sys
import time
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import dask 

from strobemodels.utils import defoc_prob_brownian, track_length
from strobemodels.specdiffuse import rad_disp_squared
from .specdiffuse import (
    evaluate_diff_coef_likelihoods_on_tracks
)

# Default diffusivity binning scheme
DEFAULT_DIFFUSIVITY_BIN_EDGES = np.logspace(-2.0, 2.0, 301)

def gs_dp_log_diff(tracks, diffusivity_bin_edges, alpha=10.0, m=10,
    m0=30, n_iter=1000, burnin=20, frame_interval=0.01,
    pixel_size_um=1.0, max_jumps_per_track=20, min_jumps_per_track=1,
    B=10000, metropolis_sigma=0.1):
    raise NotImplementedError

def gs_dp_log_diff_par(tracks, diffusivity_bin_edges=None, alpha=10.0, m=10,
    m0=30, n_iter=200, burnin=20, frame_interval=0.00748, pixel_size_um=0.16, 
    max_jumps_per_track=None, min_jumps_per_track=1, B=10000, 
    metropolis_sigma=0.2, num_workers=6, max_occ_weight=1, loc_error=0.0,
    dz=None, incorp_defoc_likelihoods=False):
    """
    Given a set of trajectories and a log-uniform prior, estimate the
    posterior distribution of diffusivities.

    Our Gibbs sampling algorithm to do this is based on algorithm 8 from 
    Radford Neal's paper:

        Radford M. Neal. Markov Chain Sampling Methods for Dirichlet Process
        Mixture Models. Journal of Computational and Graphical Statistics,
        9:2, 249-265 (2000).

    The idea is to represent the (continuous) spectrum of all possible
    diffusivities as a Dirichlet process mixture model, and to specify
    the state of a Markov chain sampler as the set of components generated
    by one draw from the corresponding Dirichlet process. Each component
    has an associated probability and an associated diffusivity.

    For inference, our observations are trajectories that are assumed to have
    been produced by some component of the mixture. If these observations are
    indexed by i so that observation X[i] was produced by the component D[i],
    then our model can be written

        X[i] | D[i] ~ Gamma(L[i] - 1, 1 / (4 * D[i] * dt))
        D[i] | G ~ G 
        G ~ DP(alpha, prior)

    where X[i] is the summed squared 2D radial displacement of the i^th 
    observation, L[i] is the length of trajectory i in frames, 
    dt is the frame interval, and DP(alpha, prior) is a Dirichlet process.

    The user passes a set of trajectories (*tracks*) and a particular set of
    diffusivities (*diffusivity_bin_edges*) on which to discretize the 
    posterior. The prior is taken to be a log-uniform distribution between 
    the minimum and maximum values of *diffusivity_bin_edges*. The Markov
    chains are not allowed to venture outside of this support.

    important
    ---------
        This function relies on a compiled version of the C++ program 
        "gs_dp_diff" in $PATH.

    args
    ----
        tracks                  :   pandas.DataFrame, trajectories

        diffusivity_bin_edges   :   1D ndarray of shape (n_bins+1,), the
                                    edges of each diffusivity bin in
                                    um^2 s^-1

        alpha                   :   float, the concentration parameter
                                    for the Dirichlet process

        m                       :   int, the number of auxiliary Markov
                                    chains to use at each step. The
                                    higher this number is, the better
                                    the sampling of the diffusivity 
                                    distribution conditional on some
                                    trajectories, but also the algorithm
                                    is slower.

        m0                      :   int, the number of Markov chains to
                                    start with

        n_iter                  :   int, the number of iterations to do

        burnin                  :   int, the number of iterations to do
                                    before recording any results

        frame_interval          :   float, time between frames in seconds

        pixel_size_um           :   float, the size of each pixel in um

        max_jumps_per_track     :   int, the maximum number of jumps to
                                    consider per track. If *None*, use all
                                    jumps from all trajectories.

        min_jumps_per_track     :   int, the minimum number of jumps to
                                    consider per track 

        B                       :   int, the buffer size defining the
                                    maximum number of components active at
                                    any given time

        metropolis_sigma        :   float, the standard deviation of steps
                                    in log diffusivity for the parameter
                                    update step

        num_workers             :   int, the number of threads to use. Each
                                    thread runs an independent replicate.

        max_occ_weight          :   int, the maximum number of displacements 
                                    to use for weighting the current occupation
                                    of any given component.

        loc_error               :   float, the 1D localization error in um

        dz                      :   float, the depth of field in um. If *np.inf*
                                    or *None*, then the microscope is assumed
                                    to gather trajectories without any 
                                    defocalization bias.

    returns
    -------
        1D ndarray of shape (n_bins,), the integrated density of Markov
            chains on each bin of *diffusivity_bin_edgse*. This is not
            normalized.

    """
    # Check that the gs_dp_diff executable exists
    assert_gs_dp_diff_exists(incorp_defoc_likelihoods)

    # If not given a binning scheme for the diffusivity, use the 
    # default scheme
    if diffusivity_bin_edges is None:
        diffusivity_bin_edges = DEFAULT_DIFFUSIVITY_BIN_EDGES

    # The minimum and maximum log (diffusivity) to consider
    min_log_D = np.log(diffusivity_bin_edges.min())
    max_log_D = np.log(diffusivity_bin_edges.max())
    n_bins = diffusivity_bin_edges.shape[0] - 1

    # Whether to correct for defocalization
    defoc_corr = (not dz is None) and (not dz is np.inf)


    ## PREPROCESSING

    # Calculate the maximum trajectory length present among
    # these trajectories
    tracks = track_length(tracks)

    # If not using a cap on the number of jumps per trajectory, use
    # every jump from every trajectory
    if (max_jumps_per_track is None) or (max_jumps_per_track is np.inf):
        n_frames = int(tracks["track_length"].max())
    else:
        n_frames = min(tracks["track_length"].max(), max_jumps_per_track+1)

    # Calculate the 2D radial squared displacements corresponding to every
    # jump in the dataset
    vecs, n_tracks = rad_disp_squared(tracks, start_frame=0,
        n_frames=n_frames, pixel_size_um=pixel_size_um, 
        min_track_length=min_jumps_per_track+1)
    vecs = pd.DataFrame(
        vecs,
        columns=[
            "track_length", "track_index_diff",
            "dy", "dx", "squared_disp", "trajectory"
        ]
    )

    # Calculate the sum of squared displacements for each trajectory
    track_data = pd.DataFrame(index=range(n_tracks),
        columns=["sum_squared_disp", "trajectory", "track_length"])
    track_data["sum_squared_disp"] = np.asarray(
        vecs.groupby("trajectory")["squared_disp"].sum()
    )
    track_data["track_length"] = np.asarray(
        vecs.groupby("trajectory")["track_length"].first()
    )
    track_data["trajectory"] = np.asarray(
        vecs.groupby("trajectory").apply(lambda i: i.name)
    )
    track_data["n_disps"] = track_data["track_length"] - 1
    del vecs 

    # Save the sum of squared displacements and the displacement number
    # for each trajectory to a CSV
    track_csv = "_TEMP.csv"
    track_data["n_disps"] = track_data["n_disps"].astype(np.int64)
    track_data[["sum_squared_disp", "n_disps"]].to_csv(
        track_csv, index=False, header=None)

    # Save the defocalization corrections to a CSV to be passed to the 
    # Gibbs sampler. These are a set of modifiers for the state occupations
    # that account for the effect of defocalization on Brownian motion. 
    bias_csv = "_TEMP_CORR.csv"
    if incorp_defoc_likelihoods:

        # The user must pass an argument for the focal depth
        assert (not dz is None)

        # Evaluate the fraction of particles that remain inside the focal
        # volume after one frame interval
        f_remain = np.zeros(n_bins, dtype=np.float64)

        # Use the geometric mean of each diffusivity bin in the user-supplied
        # support
        Ds = np.sqrt(diffusivity_bin_edges[1:] * diffusivity_bin_edges[:-1])
        for i, D in enumerate(Ds):
            f_remain[i] = defoc_prob_brownian(D, min_jumps_per_track,
                frame_interval, dz, n_gaps=0)[-1]

        # Save to a CSV that can subsequently be passed to gs_dp_diff_defoc
        _out = pd.DataFrame(index=range(len(Ds)), columns=["D", "f_remain"])
        _out["D"] = Ds 
        _out["f_remain"] = f_remain 
        _out.to_csv(bias_csv, index=False, header=None)
        del _out 


    ## FORMAT CALLS TO THE GIBBS SAMPLER
    kwargs = {
        "alpha": alpha,
        "frame_interval": frame_interval,
        "m": m,
        "metropolis_sigma": metropolis_sigma,
        "B": B,
        "n_iter": n_iter,
        "burnin": burnin,
        "min_log_D": min_log_D,
        "max_log_D": max_log_D,
        "max_occ_weight": max_occ_weight,
        "loc_error": loc_error,
        "incorp_defoc_likelihoods": incorp_defoc_likelihoods
    }   
    if incorp_defoc_likelihoods:
        kwargs["bias_csv"] = bias_csv 

    commands = []
    for i in range(num_workers):
        commands.append(format_cl_args(
            track_csv, 
            "_TEMP_OUT_{}.csv".format(i),
            verbose=(i==0),
            seed=int((time.perf_counter()+i)*1777)%373,
            **kwargs
        ))

    # Run Gibbs sampling independently for each thread
    @dask.delayed
    def gb(i):

        # Execute Gibbs sampling with gs_dp_diff
        os.system(commands[i])

        # Read the output and discretize the posterior density of
        # Markov chains into a histogram
        df = pd.read_csv("_TEMP_OUT_{}.csv".format(i), header=None)
        df["D"] = (np.exp(df[0]) - 4 * (loc_error**2)) / (4 * frame_interval)
        H = np.histogram(df["D"], bins=diffusivity_bin_edges,
            weights=df[1])[0].astype(np.float64)

        return H 

    # Manage threads with dask
    scheduler = "processes" if num_workers > 1 else "single-threaded"
    jobs = [gb(i) for i in range(num_workers)]
    results = dask.compute(
        *jobs,
        scheduler=scheduler,
        num_workers=num_workers
    )

    # Aggregate results
    record = np.asarray(results).sum(axis=0)

    # Correct for defocalization biases
    if (not dz is None) and (not dz is np.inf):
        f_remain = np.zeros(n_bins, dtype=np.float64)
        Ds = np.sqrt(diffusivity_bin_edges[1:] * diffusivity_bin_edges[:-1])
        for i, D in enumerate(Ds):
            f_remain[i] = defoc_prob_brownian(D, min_jumps_per_track,
                frame_interval, dz, n_gaps=0)[-1]
        nonzero = record > 0
        record[nonzero] = record[nonzero] / f_remain[nonzero]

    # Clean up
    for fn in ([track_csv, bias_csv] +
        ["_TEMP_OUT_{}.csv".format(i) for i in range(num_workers)]):

        if os.path.isfile(fn):
            os.remove(fn)

    # Take the geometric mean of each diffusivity bin
    diffusivities_mid = np.sqrt(diffusivity_bin_edges[1:] * \
        diffusivity_bin_edges[:-1])

    # Return the posterior sum of Markov chain densities across
    # all threads
    return record, diffusivities_mid

def format_cl_args(in_csv, out_csv, verbose=False, **kwargs):
    """
    Format a set of arguments for a call to gs_dp_diff. This essentially
    translates the keyword arguments from a call to gs_dp_log_diff_par
    to a command-line call to gs_dp_diff.

    args
    ----
        in_csv          :   str, path to a CSV with the summed squared 
                            displacements and the number of displacements
                            for each trajectory out_csv         :   str, path to the output CSV
        verbose         :   str, use the verbose option
        kwargs          :   keyword arguments passed to gs_dp_log_diff_par
                            relevant to the gs_dp_diff call

    returns
    -------
        str, a command line call to gs_dp_diff

    """
    keymap = {
        "alpha": "a",
        "frame_interval": "t",
        "m": "m",
        "metropolis_sigma": "s",
        "B": "z",
        "n_iter": "n",
        "burnin": "b",
        "min_log_D": "c",
        "max_log_D": "d",
        "seed": "e",
        "max_occ_weight": "x",
        "loc_error": "l",
        "bias_csv": "i"
    }
    executable = "gs_dp_diff_defoc" if kwargs.get( \
        "incorp_defoc_likelihoods", False) else "gs_dp_diff"
    optstr = " ".join(["-{} {}".format(str(keymap.get(k)), str(kwargs.get(k))) \
        for k in kwargs.keys() if k in keymap.keys()])
    if verbose:
        return "{} {} -v {} {}".format(executable, optstr, in_csv, out_csv)
    else:
        return "{} {} {} {}".format(executable, optstr, in_csv, out_csv)

def assert_gs_dp_diff_exists(incorp_defoc_likelihoods=False):
    """
    Determine whether the gs_dp_diff or gs_dp_diff_defoc executables
    currently exist in the user's $PATH. These are the guts of the 
    Gibbs samplers used by gs_dp_log_diff_par:

    Throws a RuntimeError if the appropriate executable is not found.

    args
    ----
        incorp_defoc_likelihoods    :   bool, whether to use
                                        defocalization in the 
                                        likelihoods of the squared
                                        jumps

    """
    if incorp_defoc_likelihoods:
        if (not os.path.isfile("gs_dp_diff_defoc")) and \
                (os.access("gs_dp_diff_defoc", os.X_OK)):
                raise RuntimeError("gs_dp_log_diff_par: must have a compiled " \
                    "version of gs_dp_diff in PATH")
    else:
        if (not os.path.isfile("gs_dp_diff")) and \
            (os.access("gs_dp_diff", os.X_OK)):
            raise RuntimeError("gs_dp_log_diff_par: must have a compiled " \
                "version of gs_dp_diff in PATH")

def evaluate_model_on_nuclei(csv_files, diffusivity_bin_edges,
    model_posterior_mean=None, frame_interval=0.00748, loc_error=0.0,
    pixel_size_um=0.16, dz=None, use_entire_track=True,
    max_jumps_per_track=np.inf, start_frame=0, out_plot=None,
    vmax=None):
    """
    Evaluate a diffusivity model on several files from a dataset,
    optionally producing a plot that shows the likelihood of each 
    diffusive state as a function of the specific files.

    args
    ----
        csv_files               :   list of str, paths to trajectory files
        diffusivity_bin_edges   :   1D ndarray of shape (K+1,), the 
                                    edges of each diffusivity bin
        model_posterior_mean    :   1D ndarray of shape (K), the occupations
                                    of each diffusivity bin. If *None*,
                                    then each bin is given equal weight.
        frame_interval          :   float, time between frames in seconds
        loc_error               :   float, localization error in um
        pixel_size_um           :   float, size of pixels in um
        dz                      :   float, focal depth in um
        use_entire_track        :   bool, use every displacement from 
                                    every track
        max_jumps_per_track     :   int, the maximum number of displacements
                                    to use from each track if *use_entire_track*
                                    is False
        start_frame             :   int, only consider trajectories after
                                    this frame
        out_plot                :   str, path to a PNG to save a summary
                                    plot of the result

    returns
    -------
        2D ndarray of shape (n_files, K), the summed likelihood
            of each diffusivity bin for each file

    """
    n = len(csv_files)
    K = diffusivity_bin_edges.shape[0] - 1
    L_by_file = np.zeros((n, K), dtype=np.float64)
    if model_posterior_mean is None:
        model_posterior_mean = np.ones(K)

    for i, csv_file in enumerate(csv_files):
        tracks = pd.read_csv(csv_file)
        tracks = tracks[tracks["frame"] >= start_frame]
        tracks = track_length(tracks)
        tracks = tracks[tracks["track_length"] > 1]
        tracks["source_file"] = csv_file 
        L, diff_cols = evaluate_diff_coef_likelihoods_on_tracks(
            tracks, diffusivity_bin_edges, model_posterior_mean, 
            frame_interval=frame_interval, loc_error=loc_error,
            pixel_size_um=pixel_size_um, dz=dz, use_entire_track=use_entire_track,
            max_jumps_per_track=max_jumps_per_track, likelihood_mode="binned",
            norm=True, map_columns=["source_file"])

        L["n_disps"] = L["track_length"] - 1
        for j, c in enumerate(diff_cols):
            L[c] = L[c] * L["n_disps"]
            L_by_file[i,j] = L[c].mean()

        print("Finished with {}".format(csv_file))

    L_by_file = (L_by_file.T / L_by_file.sum(axis=1)).T

    if not out_plot is None:

        fig, ax = plt.subplots(figsize=(4, 2))
        y_ext = 0.2 * n 
        x_ext = 5.0
        fontsize = 8
        if vmax is None:
            vmax = L_by_file.max()
        s = ax.imshow(L_by_file, vmin=0, vmax=vmax, extent=(0, x_ext, 0, y_ext))
        cbar = plt.colorbar(s, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=fontsize)
        ax.set_yticks([])
        ax.set_ylabel("Nucleus", fontsize=fontsize)
        diff_mid = np.sqrt(diffusivity_bin_edges[:-1] * diffusivity_bin_edges[1:])
        spacer = 40
        tick_locs = np.arange(K)[::spacer] * x_ext / K
        tick_labels = ["%.2f" % j for j in diff_mid[::spacer]]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=fontsize)
        ax.set_xlabel("Diffusivity ($\mu$m$^{2}$ s$^{-1}$)", fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=600)
        plt.close()
        if sys.platform == "darwin":
            os.system("open {}".format(out_plot))

    return L_by_file 


