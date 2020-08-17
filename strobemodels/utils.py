#!/usr/bin/env python
"""
utils.py

"""
import os
import numpy as np
import pandas as pd 

# 3D Hankel transform
from hankel import SymmetricFourierTransform 
HankelTrans3D = SymmetricFourierTransform(ndim=3, N=10000, h=0.005)

# Univariate spline interpolation
from scipy.interpolate import InterpolatedUnivariateSpline as spline 

# strobemodels package directory
PACKAGE_DIR = os.path.split(os.path.abspath(__file__))[0]

# strobemodels repo top-level directory
REPO_DIR = os.path.split(PACKAGE_DIR)[0]

# strobemodels testing directory
TEST_DIR = os.path.join(REPO_DIR, "tests")

# strobemodels fixtures for tests
FIXTURE_DIR = os.path.join(TEST_DIR, "fixtures")

# data directory with config stuff
DATA_DIR = os.path.join(PACKAGE_DIR, "data")

########################
## ARRAY MANIPULATION ##
########################

def generate_support(jump_bins, n_frames, frame_interval):
    """
    For fitting CDFs with multiple timepoints, generate a bivariate
    support for the fitting routine.

    This is a set of (r, t) tuples, where *r* is the radial displacement
    in the 2D plane of the camera, expressed in um*, and *t* is the
    interval between the first and second observations in seconds.

    args
    ----
        jump_bins       :   1D ndarray, the edges of each jump length
                            bin in um. Must be at least length 2. 
        n_frames        :   int, the maximum number of frame intervals
                            to consider
        frame_interval  :   float, the length of a single frame interval
                            in seconds

    returns
    -------
        2D ndarray of shape (n_frames * n_jump_bins), the support

    """
    n_jump_bins = jump_bins.shape[0] - 1
    M = n_frames * n_jump_bins
    rt_tuples = np.zeros((M, 2), dtype=np.float64)
    for t in range(n_frames):
        rt_tuples[t*n_jump_bins : (t+1)*n_jump_bins, 0] = jump_bins[1:]
        rt_tuples[t*n_jump_bins : (t+1)*n_jump_bins, 1] = (t+1) * frame_interval 
    return rt_tuples 

def coarsen_histogram(jump_length_histo, bin_edges, factor):
    """
    Given a jump length histogram with many small bins, aggregate into a 
    histogram with a small number of larger bins.

    This is useful for visualization.

    args
    ----
        jump_length_histo       :   2D ndarray, the jump length histograms
                                    indexed by (frame interval, jump length bin)
        bin_edges               :   1D ndarray, the edges of each jump length
                                    bin in *jump_length_histo*
        factor                  :   int, the number of bins in the old histogram
                                    to aggregate for each bin of the new histogram

    returns
    -------
        (
            2D ndarray, the aggregated histogram,
            1D ndarray, the edges of each jump length bin the aggregated histogram
        )

    """
    # Get the new set of bin edges
    n_frames, n_bins_orig = jump_length_histo.shape 
    bin_edges_new = bin_edges[::factor]
    n_bins_new = bin_edges_new.shape[0] - 1

    # May need to truncate the histogram at the very end, if *factor* doesn't
    # go cleanly into the number of bins in the original histogram
    H_old = jump_length_histo[:, (bin_edges<bin_edges_new[-1])[:-1]]

    # Aggregate the histogram
    H = np.zeros((n_frames, n_bins_new), dtype=jump_length_histo.dtype)
    for j in range(factor):
        H = H + H_old[:, j::factor]

    return H, bin_edges_new 

##########################################
## DATAFRAME-LEVEL TRAJECTORY UTILITIES ##
##########################################

def concat_tracks(*tracks):
    """
    Join some trajectory dataframes together into a larger dataframe,
    while preserving uniqe trajectory indices.

    args
    ----
        tracks      :   pandas.DataFrame with the "trajectory" column

    returns
    -------
        pandas.DataFrame, the concatenated trajectories

    """
    n = len(tracks)

    # Iteratively concatenate each dataframe to the first while 
    # incrementing the trajectory index as necessary
    out = tracks[0].assign(dataframe_index=0)
    c_idx = out["trajectory"].max() + 1

    for t in range(1, n):

        # Get the next set of trajectories and keep track of the origin
        # dataframe
        new = tracks[t].assign(dataframe_index=t)

        # Ignore negative trajectory indices (facilitating a user filter)
        new.loc[new["trajectory"]>=0, "trajectory"] += c_idx 

        # Increment the total number of trajectories
        c_idx = new["trajectory"].max() + 1

        # Concatenate
        out = pd.concat([out, new], ignore_index=True, sort=False)

    return out 

#############################################
## GENERATE RADIAL DISPLACEMENT HISTOGRAMS ##
#############################################

def track_length(tracks):
    """
    Compute the length of each trajectory.

    args
    ----
        tracks          :   pandas.DataFrame, the raw trajectories

    returns
    -------
        pandas.DataFrame, input with the *track_length* column

    example
    -------
        tracks["track_length"] = track_length(tracks)

    """
    if "track_length" in tracks.columns:
        tracks = tracks.drop("track_length", axis=1)
    return tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )

def rad_disp_2d(tracks, n_frames=4, frame_interval=0.01, pixel_size_um=0.16, first_only=True):
    """
    Calculate all of the radial displacements in the XY plane for a set of
    trajectories, returning the raw displacements as an ndarray.

    important note
    --------------
        This function is only designed to deal with gapless trajectories.

    args
    ----
        tracks          :   pandas.DataFrame, trajectories
        n_frames        :   int, the maximum number of frame delays to
                            consider
        frame_interval  :   float, the interval in seconds between frames
        pixel_size_um   :   float, the size of individual pixels in um
        first_only      :   bool, only consider displacements relative to the 
                            first localization of each track   

    returns
    -------
        2D ndarray, the radial displacements and the corresponding frame
            intervals. For instance, 

            result[10,0]

        is the radial displacement of the 10th jump in um, while 

            result[10,1]

        is the frame interval in seconds corresponding to the same jump.

    """
    # Assign track lengths
    if "track_length" not in tracks.columns:
        tracks = track_length(tracks)

    # Filter out unassigned localizations and singlets
    T = tracks[
        np.logical_and(tracks["trajectory"]>=0, tracks["track_length"]>1)
    ][["frame", "trajectory", "y", "x"]]

    # Assign each localization an index in the respective track 
    T["ones"] = np.ones(len(T), dtype=np.int64)
    T["index_in_track"] = T.groupby("trajectory")["ones"].cumsum()
    T = T.drop("ones", axis=1)

    # Put a handle on the first localization of each track 
    T["first_in_track"] = np.zeros(len(T), dtype="uint8")
    T.loc[T["index_in_track"]==1, "first_in_track"] = 1

    # Convert to ndarray for speed
    T = np.asarray(T[["frame", "trajectory", "y", "x", "first_in_track"]])

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,2:4] = T[:,2:4] * pixel_size_um 

    result = []

    # For each frame interval and each track, calculate the vector change in position
    for t in range(1, n_frames+1):
        diff = T[t:,:] - T[:-t,:]

        # Only consider vectors between points originating in the same track
        diff = diff[diff[:,1] == 0.0, :]

        # Only consider vectorss that match the delay being considered
        diff = diff[diff[:,0] == t, :]

        # Only consider vectors relative to the first localization in that track
        if first_only:
            diff = diff[diff[:,4] == -1, :]

        # Calculate radial displacements
        result_t = np.empty((diff.shape[0], 2), dtype=np.float64)
        result_t[:,0] = np.sqrt((diff[:,2:4]**2).sum(axis=1))
        result_t[:,1] = t * frame_interval 

        result.append(result_t)

    return np.concatenate(result, axis=0)

def rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, 
    max_jump=5.0, pixel_size_um=0.160, first_only=True):
    """
    Compile a histogram of radial displacements in the XY plane for 
    a set of trajectories ("tracks").

    args
    ----
        tracks          :   pandas.DataFrame
        n_frames        :   int, the number of frame delays to consider.
                            A separate histogram is compiled for each
                            frame delay.
        bin_size        :   float, the size of the bins in um. For typical
                            experiments, this should not be changed because
                            some diffusion models (e.g. Levy flights) are 
                            contingent on the default binning parameters.
        max_jump        :   float, the max radial displacement to consider in 
                            um
        pixel_size_um   :   float, the size of individual pixels in um
        first_only      :   bool, only consider displacements relative to the 
                            first localization of each track

    returns
    -------
        (
            2D ndarray of shape (n_frames, n_bins), the distribution of 
                displacements at each time point;
            1D ndarray of shape (n_bins+1), the edges of each bin in um
        )

    """
    # Assign track lengths
    if "track_length" not in tracks.columns:
        tracks = track_length(tracks)

    # Filter out unassigned localizations and singlets
    T = tracks[
        np.logical_and(tracks["trajectory"]>=0, tracks["track_length"]>1)
    ][["frame", "trajectory", "y", "x"]]

    # Assign each localization an index in the respective track 
    T["ones"] = np.ones(len(T), dtype=np.int64)
    T["index_in_track"] = T.groupby("trajectory")["ones"].cumsum()
    T = T.drop("ones", axis=1)

    # Put a handle on the first localization of each track 
    T["first_in_track"] = np.zeros(len(T), dtype="uint8")
    T.loc[T["index_in_track"]==1, "first_in_track"] = 1

    # Convert to ndarray for speed
    T = np.asarray(T[["frame", "trajectory", "y", "x", "first_in_track"]])

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,2:4] = T[:,2:4] * pixel_size_um 

    # Format output histogram
    bin_edges = np.arange(0.0, max_jump+bin_size, bin_size)
    n_bins = bin_edges.shape[0]-1
    H = np.zeros((n_frames, n_bins), dtype=np.int64)

    # For each frame interval and each track, calculate the vector change in position
    for t in range(1, n_frames+1):
        diff = T[t:,:] - T[:-t,:]

        # Only consider vectors between points originating in the same track
        diff = diff[diff[:,1] == 0.0, :]

        # Only consider vectorss that match the delay being considered
        diff = diff[diff[:,0] == t, :]

        # Only consider vectors relative to the first localization in that track
        if first_only:
            diff = diff[diff[:,4] == -1, :]

        # Calculate radial displacements
        r_disps = np.sqrt((diff[:,2:4]**2).sum(axis=1))
        H[t-1,:] = np.histogram(r_disps, bins=bin_edges)[0]

    return H, bin_edges 

###############################
## DIFFUSION MODEL UTILITIES ##
###############################

def generate_brownian_transfer_function(support, D, frame_interval):
    """
    Generate a transfer function for Brownian diffusion.

    args
    ----
        support         :   1D ndarray, the points in the support,
                            in um
        D               :   float, diffusion coefficient in um^2 s^-1
        frame_interval  :   float, the time between frames in seconds

    returns
    -------
        1D ndarray of dtype complex128, the RFFT for the Green's
            function of this Brownian motion

    """
    g = np.exp(-(support**2) / (4*D*frame_interval))
    g /= g.sum()
    return np.fft.rfft(g)

def defoc_prob_brownian(D, n_frames, frame_interval, dz):
    """
    Calculate the fraction of Brownian molecules remaining in the focal
    volume at a few timepoints.

    Specifically:

    A Brownian motion is generated (photoactivated) with uniform probability
    across the focal depth *dz*, and is then observed at regular intervals.
    If the particle is outside the focal volume at any one frame interval,
    it is counted as "lost" and is not observed for any subsequent frame, 
    even if it diffuses back into the focal volume.

    This function returns the probability that such a particle is observed
    at each frame.

    args
    ----
        D           :   float, diffusion coefficient in um^2 s^-1
        n_frames    :   int, the number of frame intervals to consider
        frame_interval: float, in seconds
        dz          :   float, focal volume depth in um

    returns
    -------
        1D ndarray, shape (n_frames), the probability of defocalization 
            at each frame

    """
    # Define the initial probability mass 
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 
    pmf = inside.astype("float64")
    pmf /= pmf.sum()

    # Define the transfer function for this BM
    g_rft = generate_brownian_transfer_function(support, D, frame_interval)

    # Propagate over subsequent frame intervals
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(pmf) * g_rft))
        pmf[outside] = 0.0
        result[t] = pmf.sum()

    return result 

###################
## NORMALIZATION ##
###################

def normalize_pmf(pmfs):
    """
    Normalize a 1D or 2D histogram to a probability mass function,
    avoiding divide-by-zero errors for axes with zero observations.

    The typical format for a PMF in strobemodels is a 2D ndarray where
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

def normalize_flat_cdf(rt_tuples, cdf):
    """
    Normalize a flat CDF separately for each timestep. These CDFs, which are 
    used by MINPACK for fitting, are typically an end-to-end concatenation of
    CDFs corresponding to each timepoint.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the set of
                            independent variables used for fitting (generated
                            by *generate_support*)
        cdf             :   1D ndarray of shape (n_points), the CDF

    returns
    -------
        1D ndarray, normalized CDF

    """
    unique_dt = np.unique(rt_tuples[:,1])
    for t in unique_dt:

        # Find the set of support points corresponding to this timepoint
        match = rt_tuples[:,1]==t 

        # Get the last point in the CDF corresponding to this timepoint
        c = np.argmax(rt_tuples[match,0]) + match.nonzero()[0][0]

        # Normalize
        cdf[match] = cdf[match] / cdf[c]

    return cdf 

def radnorm(r, pdf, d=2):
    """
    Given a PDF with radial symmetry, return the PDF for the radial distance
    from the origin. This is equivalent to taking the PDF, expressed in 
    hyperspherical coordinates, and marginalizing on all angular components.

    For instance, in 2D, we would do

    	pdf_rad(r) = 2 pi int_{0}^{infty} r pdf(r) dr 

    Normalizing the Gaussian density with d = 2 would give a Rayleight
    distribution, with d = 3 would give a Maxwell-Boltzmann distribution,
    and so on.

    This method approximates the PDF at a discrete set of radial points. For
    it to be accurate, the spacing of the support *r* must be small relative
    to local changes in the PDF.

    args
    ----
    	r 			:	1D ndarray of shape (n_r), the radial support
    	pdf         :   1D ndarray of shape (n_r), the PDF as a function of 
    					the support
    	d           :   int, the number of spatial dimensions

    returns
    -------
    	1D ndarray of shape (n_r), the PDF for the radial distance from the
    		origin

    """
    result = pdf * np.power(r, d-1)
    result /= result.sum()
    return result 

###################################################
## GENERATING PDFs FROM CHARACTERISTIC FUNCTIONS ##
###################################################

def pdf_from_cf(func_cf, x, **kwargs):
    """
    Evaluate the PDF of a 1-dimensional real-valued random variable given
    its characteristic function. 

    args
    ----
    	func_cf 		:	function with signature (1D np.ndarray, **kwargs),
    						the characteristic function
    	x				:	1D ndarray, the set of points at which to evaluate
    						the PDF 
    	kwargs			:	to *func_cf*

    returns
    -------
    	1D ndarray of shape x.shape, the PDF

    """
    # Spectrum
    k = np.fft.rfftfreq(x.shape[0], d=(x[1]-x[0]))

    # Evaluate characteristic function
    cf = func_cf(2.0 * np.pi * k, **kwargs)

    # Inverse transform
    return np.fft.fftshift(np.fft.irfft(cf, x.shape[0]))

def pdf_from_cf_rad(func_cf, x, **kwargs):
    """
    Evaluate a radially symmetric PDF defined on 3D space, given the spectral
    radial profile of its characteristic function.

    important
    ---------
    	The output of this function is not necessarily normalized. If the user
    		wants to normalize it on 3D real space - for instance, to get the 
    		PDF for the radial displacement from the origin of a trajectory in 
    		3D space - they should use radnorm() with d = 3.

    args
    ----
    	func_cf 	:	function with signature (1D ndarray, **kwargs), the
    					characteristic function
    	x			:	1D ndarray, the real-space points at which to evaluate
    					the PDF
    	kwargs      :	to *func_cf*

    returns
    -------
    	1D ndarray of shape x.shape, the PDF

    """
    F = lambda j: func_cf(j, **kwargs)
    return HankelTrans3D.transform(F, x, ret_err=False, inverse=True)

#########################################################
## OTHER UTILITIES USEFUL IN THE CORE FITTING ROUTINES ##
#########################################################

def bounds_center(bounds, replace_inf=10.0):
    """
    Given a set of bounds on fit parameters - for instance,

        bounds = (
            np.array([0.0, 2.0]),
            np.array([1.0, 10.0])
        ),

    return the vector of values centered between the upper and lower bounds. In the 
    above example, that would be 

        np.array([0.5, 6.0])

    If the upper bound is np.inf, replace it with a value equal to the lower bound
    plus *replace_inf*.

    args
    ----
        bounds          :   2-tuple of 1D ndarray, the lower and upper bounds on 
                            each parameter, suitable to passing to the *bounds*
                            argument for scipy.optimize.curve_fit
        replace_inf     :   float, treatment for np.inf values in the upper bounds

    returns
    -------
        1D ndarray, the vector of bound centers

    """
    assert len(bounds[0]) == len(bounds[1])
    n = len(bounds[0])
    result = np.empty(n, dtype=np.float64)
    for j in range(n):
        lower = bounds[0][j]
        upper = bounds[1][j] if not np.isinf(bounds[1][j]) else lower + replace_inf 
        result[j] = (lower + upper) * 0.5
    return result 

def bounds_transpose(bounds):
    """
    Transpose a set of parameter bounds from the format expected by scipy.optimize.curve_fit
    to the format expected by scipy.optimize.minimize.

    example
    -------
    Input:
        bounds = (np.array([0.0, 0.1]), np.array([np.inf, 0.2]))

    Output:
        ((0.0, np.inf), (0.1, 0.2))

    args
    ----
        bounds      :   2-tuple of 1D ndarray, the lower and upper bounds on each parameter

    returns
    -------
        list of (min, max) bound pairs for each parameter

    """
    return [(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]

def bounds_antitranspose(bounds):
    """
    Inverse to *bounds_transpose*.

    args
    ----
        bounds      :   list of (min, max) bound pairs for each fit parameter

    returns
    -------
        2-tuple of 1D ndarray, the lower and upper bounds on each parameter

    """
    return (
        np.array([bounds[i][0] for i in range(len(bounds))]),
        np.array([bounds[i][1] for i in range(len(bounds))])
    )





