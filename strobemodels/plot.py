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
    axes[-1].set_xlabel("Jump length ($\mu$m)", fontsize=10)

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

