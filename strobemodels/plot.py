#!/usr/bin/env python
"""
plot.py -- visualizations for stroboscopic SPT model fits

"""
import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib.colors import Normalize
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

from .utils import covariance_matrix

def wrapup(out_png, dpi=600):
    """
    Save a plot to a file.

    args
    ----
        out_png         :   str, the output file

    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    # if sys.platform == 'darwin':
    #     os.system("open {}".format(out_png))

def plot_jump_length_pmf(bin_edges, pmfs, model_pmfs=None, model_bin_edges=None, 
    frame_interval=0.01, max_jump=2.0, cmap="gray", figsize_mod=1.0, out_png=None):
    """
    Plot jump length histograms at different frame intervals, possibly with a model
    overlay.

    args
    ----
        bin_edges       :   1D ndarray of shape (n_bins+1), the edges of each jump
                            length bin in um
        pmfs            :   2D ndarray of shape (n_frames, n_bins), the jump length
                            histogram. This is normalized, if not already normalized.
        model_pmfs      :   2D ndarray of shape (n_frames, n_bins_model), the
                            model PMFs for each frame interval in um
        model_bin_edges :   1D ndarray of shape (n_bins_model+1), the edges of each 
                            jump length bin for the model PMFs in um. If not given,
                            this function defaults to *bin_edges*.
        frame_interval  :   float, the time between frames in seconds
        max_jump        :   float, the maximum jump length to show in um
        cmap            :   str, color palette to use for each jump length. If a hex color
                            (for instance, "#A1A1A1"), then each frame interval is 
                            colored the same.
        figsize_mod     :   float, modifier for the default figure size
        out_png         :   str, a file to save this plot to. If not specified, the plot
                            is not saved.

    returns
    -------
        (
            matplotlib.pyplot.Figure,
            1D ndarray of matplotlib.axes.Axes
        )

    """
    # Check user inputs and get the number of bins and bin size
    assert len(pmfs.shape) == 2
    n_frames, n_bins = pmfs.shape 
    exp_bin_size = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + 0.5 * exp_bin_size    
    assert bin_edges.shape[0] == n_bins + 1
    if not model_pmfs is None:
        assert model_pmfs.shape[0] == n_frames 
        _, n_bins_model = model_pmfs.shape 
        if not model_bin_edges is None:
            assert n_bins_model == model_bin_edges.shape[0] - 1
            model_bin_size = model_bin_edges[1] - model_bin_edges[0]
            model_bin_centers = model_bin_edges[:-1] + model_bin_size * 0.5
        else:
            assert n_bins_model == n_bins
            model_bin_centers = bin_centers
            model_bin_size = exp_bin_size

        # PMF scaling, accounting for disparate bin sizes
        scale_factor = exp_bin_size / model_bin_size 


    # Bar width for bar plot
    width = exp_bin_size * 0.8

    # Generate the plot axes
    fig, axes = plt.subplots(n_frames, 1, figsize=(4.2*figsize_mod, 0.75*n_frames*figsize_mod),
        sharex=True)
    if n_frames == 1:
        axes = np.array([axes])

    # Make colors for each frame interval
    assert isinstance(cmap, str)
    if cmap[0] == "#":
        palette = [cmap for j in range(n_frames)]
    else:
        palette = sns.color_palette(cmap, n_frames)

    # Plot the PMF for each frame interval
    for t in range(n_frames):

        # Plot the experimental data
        if pmfs[t,:].sum() == 0:
            exp_pmf = np.zeros(pmfs[t,:].shape, dtype=np.float64)
        else:
            exp_pmf = pmfs[t,:].astype(np.float64) / pmfs[t,:].sum()
        axes[t].bar(bin_centers, exp_pmf, color=palette[t], edgecolor="k", linewidth=1, 
            width=width, label=None)

        # Plot the model 
        if not model_pmfs is None:
            axes[t].plot(model_bin_centers, model_pmfs[t,:]*scale_factor, linestyle='-',
                linewidth=1.5, color='k', label=None)

        # For labels
        axes[t].plot([], [], linestyle="", marker=None, color="w",
            label="$\Delta t = ${:.4f} sec".format((t+1)*frame_interval))

        axes[t].legend(frameon=False, prop={"size": 6}, loc="upper right")
        axes[t].set_yticks([])

        # Kill some of the plot spines
        for s in ["top", "right", "left"]:
            axes[t].spines[s].set_visible(False)

    # Only show jumps up to the max jump length 
    if not max_jump is None:
        axes[0].set_xlim((0, max_jump))
    axes[-1].set_xlabel("2D radial displacement ($\mu$m)", fontsize=10)

    # Save to a file, if desired
    if not out_png is None:
        wrapup(out_png)

    return fig, axes 

def plot_jump_length_cdf(bin_edges, cdfs, model_cdfs=None, model_bin_edges=None,
    frame_interval=0.01, max_jump=5.0, cmap='gray', figsize_mod=1.0, out_png=None,
    fontsize=8):
    """
    Plot jump length cumulative distribution functions at different frame intervals,
    potentially with a model overlay.

    args
    ----
        bin_edges       :   1D ndarray of shape (n_bins+1), the edges of each jump
                            length bin in um
        cdfs            :   2D ndarray of shape (n_frames, n_bins), the jump length
                            CDFs
        model_cdfs      :   2D ndarray of shape (n_frames, n_bins_model), the
                            model CDFs for each frame interval in um
        model_bin_edges :   1D ndarray of shape (n_bins_model+1), the edges of each 
                            jump length bin for the model CDFs in um. If not given,
                            this function defaults to *bin_edges*.
        frame_interval  :   float, the time between frames in seconds
        max_jump        :   float, the maximum jump length to show in um
        cmap            :   str, color palette to use for each jump length. If a hex color
                            (for instance, "#A1A1A1"), then each frame interval is 
                            colored the same.
        figsize_mod     :   float, modifier for the default figure size
        out_png         :   str, a file to save this plot to. If not specified, the plot
                            is not saved.

    returns
    -------
        (
            matplotlib.pyplot.Figure,
            list of matplotlib.axes.Axes
        )

    """
    # Check user inputs and figure out what kind of plot to make. 
    # plot_case == 0: plot the experimental CDFs, model overlay, and model residuals
    # plot_case == 1: plot the experimental CDFs and model overlay, but no residuals
    # plot_case == 2: plot only the experimental CDFs
    n_frames, n_bins = cdfs.shape 
    assert bin_edges.shape[0] == n_bins + 1
    bins_right = bin_edges[1:]
    bin_size = bin_edges[1] - bin_edges[0]
    if not model_cdfs is None:
        n_frames_model, n_bins_model = model_cdfs.shape
        if not model_bin_edges is None:
            assert model_bin_edges.shape[0] == n_bins_model + 1
            model_bin_size = model_bin_edges[1] - model_bin_edges[0]
            model_bins_right = model_bin_edges[1:]
        else:
            assert model_cdfs.shape == cdfs.shape 
            model_bins_right = bins_right 

        # Choose whether or not to plot the residuals
        if model_bins_right.shape == bins_right.shape:
            plot_case = 0
        else:
            plot_case = 1
    else:
        plot_case = 2

    # Configure the colors to use during plotting
    assert isinstance(cmap, str)
    if cmap[0] == "#":
        palette = [cmap for j in range(n_frames)]
    else:
        palette = sns.color_palette(cmap, n_frames)

    # Plot the experimental CDFs with a model overlay and residuals below
    if plot_case == 0:
        fig, ax = plt.subplots(2, 1, figsize=(3*figsize_mod, 3*figsize_mod),
            gridspec_kw={'height_ratios': [3,1]}, sharex=True)

    # Plot the experimental CDFs, potentially with a model overlay, and no residuals
    else:
        fig, ax = plt.subplots(figsize=(3*figsize_mod, 2*figsize_mod))
        ax = [ax]

    # Plot the experimental CDFs
    for t in range(n_frames):
        ax[0].plot(bins_right, cdfs[t,:], color=palette[t], linestyle='-',
            label="{:.4f} sec".format((t+1)*frame_interval))

    # Plot the model CDFs
    if plot_case == 0 or plot_case == 1:
        for t in range(n_frames):
            ax[0].plot(model_bins_right, model_cdfs[t,:], color="k", 
                linestyle="--", label=None)
        ax[0].plot([], [], color="k", linestyle="--", label="Model")

    # Plot the model residuals
    if plot_case == 0:
        residuals = cdfs - model_cdfs 
        for t in range(n_frames):
            ax[1].plot(bins_right, residuals[t,:], color=palette[t], linestyle='-',
                label="{:.4f} sec".format((t+1)*frame_interval), linewidth=1)
        ax[1].set_xlabel("Jump length ($\mu$m)", fontsize=fontsize)
        ax[1].set_ylabel("Residuals", fontsize=fontsize)

        # Center the residuals on zero
        ax1_ylim = np.abs(residuals).max() * 1.5
        ax[1].set_ylim((-ax1_ylim, ax1_ylim))
        ax[1].set_xlim((0, max_jump))
        ax[1].tick_params(labelsize=fontsize)

    # Axis labels and legend
    ax[0].set_ylabel("CDF", fontsize=fontsize)
    ax[0].set_xlim((0, max_jump))
    ax[0].legend(frameon=False, prop={'size': fontsize}, loc="lower right")
    ax[0].tick_params(labelsize=fontsize)

    # Save to a file, if desired
    if not out_png is None:
        wrapup(out_png)

    return fig, ax 

###################################
## VISUALIZE COVARIANCE MATRICES ##
###################################

def plot_covariance_matrix(tracks, n=7, covtype="position",
    immobile_thresh=0.1, loc_error=0.035, frame_interval=0.00748,
    pixel_size_um=0.16):
    """
    Plot the sample covariance matrix for a set of trajectories.

    args
    ----
        tracks          :   pandas.DataFrame
        n               :   int, rank of the covariance to compute,
                            the minimum number of points/displacements
                            to consider
        covtype         :   str, "position", "jump", or "jump_cross_dimension",
                            the type of covariance matrix to calculate
        immobile_thresh :   float, the minimum diffusion coefficient
                            to consider in um^2 s^-1
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, frame interval in seconds
        pixel_size_um   :   float, size of pixels in um

    returns
    -------
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes,
            2D ndarray (covariance matrix)
        )

    """
    # Calculate the covariance matrix
    C = covariance_matrix(
        tracks,
        n=n,
        covtype=covtype,
        immobile_thresh=immobile_thresh,
        loc_error=loc_error,
        frame_interval=frame_interval,
        pixel_size_um=pixel_size_um
    )

    # Plot the covariance matrix
    if covtype in ["jump", "jump_cross_dimension"]:

        fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
        x = np.abs(C.min())
        norm = Normalize(vmin=-x, vmax=x)
        c = ax[0].imshow(C, cmap="RdBu", origin="lower", norm=norm)
        plt.colorbar(c, ax=ax[0], shrink=0.75)

        x = np.abs(C).max()
        norm1 = Normalize(vmin=-x, vmax=x)
        c1 = ax[1].imshow(C, cmap="RdBu", origin="lower", norm=norm1)
        plt.colorbar(c1, ax=ax[1], shrink=0.75)

        # Ticks
        for j in range(2):
            ax[j].set_xticks(np.arange(n))
            ax[j].set_xticklabels(np.arange(1, n+1))
            ax[j].set_yticks(np.arange(n))
            ax[j].set_yticklabels(np.arange(1, n+1))

        ax[0].set_title("Scaled to min covariance")
        ax[1].set_title("Scaled to max covariance")

    elif covtype == "position":
        fig, ax = plt.subplots(figsize=(3, 3))
        c = ax.imshow(C, cmap="viridis", origin="lower")
        plt.colorbar(c, ax=ax, shrink=0.75)

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(np.arange(1, n+1))
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(np.arange(1, n+1))

    return fig, ax, C

#########################################
## VISUALIZE DIFFUSIVITY DISTRIBUTIONS ##
#########################################

def fbme_dist(occupations, hurst_pars, diff_coefs, axes=None, cmap="viridis",
    do_colorbar=True, diff_coef_log_scale=True, log_scale_range=(-2, 2),
    extent=(0, 4, 0, 1.5), vmax=None, xlabel=None):
    """
    args
    ----
        occupations         :   2D ndarray of shape (n, m), the occupations
                                of each diffusive state, where occupations[i,j]
                                is the occupation of the state with Hurst
                                parameter hurst_pars[i] and diffusion coefficient
                                diff_coefs[j]
        hurst_pars          :   1D ndarray of shape (n,), the set of Hurst
                                parameters for each column
        diff_coefs          :   1D ndarray of shape (m,), the set of diffusion
                                coefficients for each row
        axes                :   matplotlib.axes.Axes
        cmap                :   str
        do_colorbar         :   bool
        diff_coef_log_scale :   bool
        log_scale_range     :   (int, int), log upper and lower limits for the 
                                x-axis
        extent              :   (float, float, float, float), the plot shape
        vmax                :   float, upper color map limit
        xlabel              :   str

    returns
    -------
        matplotlib.axes.Axes

    """
    n, m = occupations.shape
    assert hurst_pars.shape[0] == n 
    assert diff_coefs.shape[0] == m 

    # Generate the axes if it does not exist
    if axes is None:
        fig, axes = plt.subplots(figsize=(4, 2))

    # Upper color scaling
    if vmax is None:
        vmax = occupations.max()

    # Make the main plot
    im_obj = axes.imshow(
        occupations,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        extent=extent,
        origin='lower'
    )

    # Make a colorbar
    if do_colorbar:
        plt.colorbar(im_obj, ax=axes, shrink=0.5)

    # Make the x-axis labels
    if diff_coef_log_scale:
        space = 5
        delta = int(log_scale_range[1] - log_scale_range[0])
        xmajorticklocs = np.arange(delta+1) * extent[1] / delta 
        xminorticklocs = []
        kernel = np.log10(np.arange(1, 11)) * (xmajorticklocs[1]-xmajorticklocs[0])
        for j in range(delta):
            xminorticklocs += list(kernel+xmajorticklocs[j])
        xticklabels = ["%.2f" % j for j in diff_coefs[::space]]
        xticklabels = ["$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$10^{1}$", "$10^{2}$"]
        axes.set_xticks(xmajorticklocs, minor=False)
        axes.set_xticks(xminorticklocs, minor=True)
        axes.set_xticklabels(xticklabels)
    else:
        space = 5
        xticklocs = np.arange(m)[::space] * extent[1] / (m-1)
        xticklabels = ["%.2f" % j for j in diff_coefs[::space]]
        axes.set_xticks(xticklocs)
        axes.set_xticklabels(xticklabels)
    if xlabel is None:
        xlabel = "Modified diffusion coefficient ($\mu$m$^{2}$ s)"
    axes.set_xlabel(xlabel)

    # Make the y-axis labels
    space = 4
    yticklocs = np.arange(n)[1::space] * extent[3] / (n-1)
    yticklabels = ['%.2f' % j for j in hurst_pars[1::space]]
    axes.set_yticks(yticklocs)
    axes.set_yticklabels(yticklabels)
    axes.set_ylabel("Hurst parameter")

    return axes 

