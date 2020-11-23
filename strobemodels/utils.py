#!/usr/bin/env python
"""
utils.py

"""
import os
import numpy as np
import pandas as pd 
from functools import lru_cache 
from scipy import interpolate 

# 3D Hankel transform
from hankel import SymmetricFourierTransform 
HankelTrans3D = SymmetricFourierTransform(ndim=3, N=10000, h=0.005)
# HankelTrans2D = SymmetricFourierTransform(ndim=2, N=10000, h=0.005)
HankelTrans2D = SymmetricFourierTransform(ndim=2, N=250, h=0.01)

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

    # Sort the tracks dataframes by their size. The only important thing
    # here is that if at least one of the tracks dataframes is nonempty,
    # we need to put that one first.
    df_lens = [len(t) for t in tracks]
    try:
        tracks = [t for _, t in sorted(zip(df_lens, tracks))][::-1]
    except ValueError:
        pass

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

def concat_tracks_files(*csv_paths, out_csv=None, start_frame=0,
    drop_singlets=False):
    """
    Given a set of trajectories stored as CSVs, concatenate all
    of them, storing the paths to the original CSVs in the resulting
    dataframe, and optionally save the result to another CSV.

    args
    ----
        csv_paths       :   list of str, a set of trajectory CSVs.
                            Each must contain the "y", "x", "trajectory",
                            and "frame" columns
        out_csv         :   str, path to save to 
        start_frame     :   int, exclude any trajectories that begin before
                            this frame
        drop_singlets   :   bool, drop singlet localizations before
                            concatenating

    returns
    -------
        pandas.DataFrame, the concatenated result

    """
    n = len(csv_paths)

    def drop_before_start_frame(tracks, start_frame):
        """
        Drop all trajectories that start before a specific frame.

        """
        tracks = tracks.join(
            (tracks.groupby("trajectory")["frame"].first() >= start_frame).rename("_take"),
            on="trajectory"
        )
        tracks = tracks[tracks["_take"]]
        tracks = tracks.drop("_take", axis=1)
        return tracks

    def drop_singlets_dataframe(tracks):
        """
        Drop all singlets and unassigned localizations from a 
        pandas.DataFrame with trajectory information.

        """
        if start_frame != 0:
            tracks = drop_before_start_frame(tracks, start_frame)

        tracks = track_length(tracks)
        tracks = tracks[np.logical_and(tracks["track_length"]>1,
            tracks["trajectory"]>=0)]
        return tracks 

    # Load the trajectories into memory
    tracks = []
    for path in csv_paths:
        if drop_singlets:
            tracks.append(drop_singlets_dataframe(pd.read_csv(path)))
        else:
            tracks.append(pd.read_csv(path))

    # Concatenate 
    tracks = concat_tracks(*tracks)

    # Map the original path back to each file
    for i, path in enumerate(csv_paths):
        tracks.loc[tracks["dataframe_index"]==i, "source_file"] = \
            os.path.abspath(path)

    # Optionally save concatenated trajectories to a new CSV
    if not out_csv is None:
        tracks.to_csv(out_csv, index=False)

    return tracks 

#################################
## COVARIANCE MATRIX UTILITIES ##
#################################

def covariance_matrix(tracks, n=7, covtype="position", immobile_thresh=0.1,
    loc_error=0.035, frame_interval=0.00748, pixel_size_um=0.16):
    """
    Calculate the empirical covariance matrix for a set of trajectories.
    This is an *n* by *n* matrix such that element (i, j) is the covariance
    of the particle's 1D position at time points i and j:

        C[i,j] := E[ X[i] X[j] ]

    where X is path of the particle, centered such that X[0] = 0.

    args
    ----
        tracks          :   pandas.DataFrame or 3D ndarray
        n               :   int, rank of the covariance matrix to calculate.
                            All trajectories with fewer than *n* displacements
                            are excluded.
        covtype         :   str (either "position" or "jump"), the 
                            type of covariance matrix to compute. If "position",
                            the covariance of the position relative to the first 
                            displacement is used. If "jump", the covariance
                            of the subsequent displacements is used. 
        immobile_thresh :   float, exclude trajectories with an estimated
                            diffusion coefficient below this threshold, to
                            mitigate the effects of localization error.
                            This threshold is specified in um^2 s^-1.
        loc_error       :   float, 1D localization error in um. Only used
                            if immobile_thresh is not None.
        frame_interval  :   float, frame interval in seconds. Only used if
                            immobile_thresh is not None.
        pixel_size_um   :   float, the size of pixels in um

    returns
    -------
        2D ndarray of shape (n, n), the empirical covariance matrix

    """
    # Convert from ndarray to pandas.DataFrame if necessary
    if isinstance(tracks, np.ndarray):
        assert len(tracks.shape) == 3
        n_tracks, n, n_dim = tracks.shape 
        M = n_tracks * n 
        df = pd.DataFrame(index=np.arange(M),
            columns=["trajectory", "frame", "y", "x"])
        df["trajectory"] = np.arange(M) // n 
        df["frame"] = np.arange(M) % n 
        df["y"] = tracks[:,:,0].ravel()
        df["x"] = tracks[:,:,1].ravel()
        tracks = df 
    else:

        # Consider one more point than we would otherwise, if 
        # we're calculating an (n-by-n) covariance matrix on the
        # displacements
        if covtype in ["jump", "jump_cross_dimension"]:
            n += 1

    # Only consider trajectories of requisite length
    tracks = track_length(tracks)
    tracks = tracks[tracks["track_length"] >= (n+1)].copy()
    tracks[['y', 'x']] = tracks[['y', 'x']] * pixel_size_um 

    # Sort first by trajectory, then by frame
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Subtract the first point of each trajectory
    if "first_y" in tracks.columns:
        tracks = tracks.drop('first_y', axis=1)
    if "first_x" in tracks.columns:
        tracks = tracks.drop('first_x', axis=1)

    tracks = tracks.join(
        tracks.groupby("trajectory")["y"].first().rename("first_y"),
        on="trajectory"
    )
    tracks = tracks.join(
        tracks.groupby("trajectory")["x"].first().rename("first_x"),
        on="trajectory"
    )
    tracks["y"] -= tracks["first_y"]
    tracks["x"] -= tracks["first_x"]
    tracks = tracks.drop("first_y", axis=1)
    tracks = tracks.drop("first_x", axis=1)

    # Only take the position from the first displacement onward
    tracks = assign_index_in_track(tracks)
    tracks = tracks[tracks["index_in_track"] > 0]

    # Exclude "immobile" trajectories
    if not immobile_thresh is None:

        # Maximum likelihood estimation of diffusion coefficient
        n_disps = np.asarray(tracks.groupby("trajectory").size())
        tracks['y2'] = tracks['y']**2
        tracks['x2'] = tracks['x']**2
        ssy = np.asarray(tracks.groupby("trajectory")["y2"].sum())
        ssx = np.asarray(tracks.groupby("trajectory")["x2"].sum())
        ss = ssy + ssx 
        D_ml = (ss / (4 * n_disps) - (loc_error**2)) / frame_interval 
        track_indices = np.asarray(tracks.groupby("trajectory").apply(lambda i: i.name))
        include_indices = track_indices[D_ml > immobile_thresh]
        tracks = tracks[tracks["trajectory"].isin(include_indices)]
        n_tracks = tracks["trajectory"].nunique()

    # Take the first *n* points of each trajectory
    tracks = tracks[tracks["index_in_track"] <= n]

    # Format the result as an ndarray
    tracks = np.asarray(tracks[['y', 'x']]).reshape((n_tracks, n, 2))

    if covtype == "position":

        # Calculate the covariance matrix on the positions of 
        # the trajectory at each time point
        C = ((tracks[:,:,0].T @ tracks[:,:,0]) + \
            (tracks[:,:,1].T @ tracks[:,:,1])) / \
            (2 * n_tracks - 1)

    elif covtype == "jump":

        # Calculate the vectorial displacements
        tracks = tracks[:,1:,:] - tracks[:,:-1,:]
        n_disps = tracks.shape[0] * tracks.shape[1]

        # Calculate the covariance matrix on the displacements
        # of the trajectory at each time point
        C = ((tracks[:,:,0].T @ tracks[:,:,0]) + \
            (tracks[:,:,1].T @ tracks[:,:,1])) / \
            (2 * n_disps - 1)       

    elif covtype == "jump_cross_dimension":

        # Claculate the vectorial displacement
        tracks = tracks[:,1:,:] - tracks[:,:-1,:]
        n_disps = tracks.shape[0] * tracks.shape[1]

        # Calculate the covariance matrix between the X and Y 
        # displacements at each time point
        C = (tracks[:,:,0].T @ tracks[:,:,1]) / (n_disps - 1)

    return C 

def fbme_jump_covariance(n, hurst=0.5, D=1.0, dt=0.01,
    D_type=4, loc_error=0.0):

    h2 = hurst * 2

    T, S = np.indices((n, n)) + 1

    if D_type == 1:
        C = D * np.power(dt, 2*hurst) * ( \
                np.power(np.abs(T-S+1), h2) + \
                np.power(np.abs(T-S-1), h2) - \
                2*np.power(np.abs(T-S), h2)
            )
    elif D_type == 2:
        raise NotImplementedError
    elif D_type == 3:
        D_mod = D * dt / (2 * hurst)
        C = D_mod * (np.power(np.abs(T-S+1), h2) + \
            np.power(np.abs(T-S-1), h2) - \
            2*np.power(np.abs(T-S), h2))       
    elif D_type == 4:
        D_mod = D * dt 
        C = D_mod * (np.power(np.abs(T-S+1), h2) + \
            np.power(np.abs(T-S-1), h2) - \
            2*np.power(np.abs(T-S), h2))

    if loc_error > 0.0:
        le2 = loc_error ** 2
        C = C + np.diag(np.ones(n) * le2)
        C += le2 

    return C 

###############################
## SEPARABILITY IN X/Y TESTS ##
###############################

def jump_magnitude_dependence(tracks, n_frames=1, max_jump=1.0, bin_size=0.01,
    normalize=False, frame_interval=0.01, pixel_size_um=0.16,
    use_entire_track=True, max_jumps_per_track=10, n_gaps=0):
    """
    Determine how the magnitude of the jumps in the *x* dimension
    depends on their magnitude in the *y* dimension. 

    args
    ----
        tracks          :   pandas.DataFrame
        n_frames        :   int, number of frame intervals over
                            which to compute the jump
        max_jump        :   float, maximum jump to consider in um
        bin_size        :   float, jump length bin size in um
        normalize       :   bool
        frame_interval  :   float, seconds
        pixel_size_um   :   float, um 
        use_entire_track:   bool
        max_jumps_per_track:    int
        n_gaps              :   int

    returns
    -------
        (
            2D ndarray of shape (n_bins, n_bins), the histogram;
            1D ndarray of shape (n_bins+1,), the bin edges in um 
        )

    """
    n_bins = int(max_jump / bin_size)
    max_jump = n_bins * bin_size 

    # Jump bin edges in um
    jump_bin_edges =  np.linspace(0, max_jump, n_bins+1)

    # Compute the jumps
    xy_jumps = np.abs(xy_disp_2d(tracks, n_frames=n_frames, frame_interval=frame_interval,
        pixel_size_um=pixel_size_um, use_entire_track=use_entire_track,
        max_jumps_per_track=max_jumps_per_track, n_gaps=n_gaps))

    # Matrix of displacement densities
    M = np.zeros((n_bins, n_bins), dtype=np.float64)

    for i in range(n_bins):

        # Select only displacements in the desired range
        x_jumps = xy_jumps[np.logical_and(
            xy_jumps[:,1]>=jump_bin_edges[i],
            xy_jumps[:,1]<jump_bin_edges[i+1]
        ), 0]

        # Make a histogram
        M[i,:] = np.histogram(x_jumps, bins=jump_bin_edges)[0]

    return M, jump_bin_edges 

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

def assign_index_in_track(tracks):
    """
    Given a set of trajectories, determine the index of each localization in the
    context of its respective trajectory.

    args
    ----
        tracks      :   pandas.DataFrame, containing the "trajectory" and "frame"
                        columns

    returns
    -------
        pandas.DataFrame, the same dataframe with a new column, "index_in_track"

    """
    tracks["one"] =  1
    tracks["index_in_track"] = tracks.groupby("trajectory")["one"].cumsum() - 1
    tracks = tracks.drop("one", axis=1)
    return tracks 

def track_array(tracks, n_frames=4, frame_interval=0.01, pixel_size_um=0.16):
    """
    Convert a set of trajectories from pandas.DataFrame format into a specific
    list format [T1, T2, ...] where each Ti is a trajectory, represented as a 
    2D ndarray of shape (traj_length, 3) with the column identities (y, x, t).

    As a result, the result of this function can be indexed the following way:

        result[track_index][0, 0] -> y-coordinate of the first point in this
                                     trajectory
        result[track_index][1, 2] -> timepoint (in seconds) of the second point
                                     in this trajectory
        result[track_index][0, 1] -> x-coordinate of the first point in this
                                     trajectory

    and so on.

    args
    ----
        tracks          :   pandas.DataFrame, trajectories in standard format
        n_frames        :   int, the maximum number of frame intervals to 
                            consider
        frame_interval  :   float, the time between frames in seconds
        pixel_size_um   :   float, the size of individual pixels in um

    returns
    -------
        list of 2D ndarray, the trajectories

    """
    raise NotImplementedError

def xy_disp_2d(tracks, n_frames=1, frame_interval=0.01, pixel_size_um=0.16,
    use_entire_track=True, max_jumps_per_track=10, n_gaps=0):
    """
    Gather every vectorial (x, y) displacement in the dataset. If
    *use_entire_track* is *True*, then every displacement from every
    trajectory is used. Otherwise, we only use the first few jumps
    from every trajectory, with the jump determined by *max_jumps_per_track*.

    args
    ----
        tracks          :   pandas.DataFrame
        n_frames        :   int, the number of frame intervals over 
                            which to compute the jump 
        frame_interval  :   float, seconds
        pixel_size_um   :   float, um 
        use_entire_track:   bool
        max_jumps_per_track :   int

    returns
    -------
        2D ndarray, shape (n_jumps, 2), where index [i, 0] 
            corresponds to the x displacement of the i^th 
            displacement and [i, 1] corresponds to the y 
            displacement of the same jump

    """
    tracks = track_length(tracks)

    # Exclude singlets
    T = tracks[tracks["track_length"] > 1].copy()

    if not use_entire_track:
        T = assign_index_in_track(T)
        T = T[T['index_in_track'] >= max_jumps_per_track]

    # Sort first by trajectory then by frame
    T = T.sort_values(by=["trajectory", "frame"])

    # Format as ndarray
    T = np.asarray(T[['trajectory', 'frame', 'x', 'y']])

    # Convert to um
    T[:,2:4] = T[:,2:4] * pixel_size_um 

    # Collect the target jumps
    all_jumps = []

    for i in range(1, (n_gaps+1) * n_frames + 1):

        # Determine the vectorial displacements
        disps = T[i:,:] - T[:-i,:]

        # Only include displacements originating from the same
        # trajectory
        disps = disps[disps[:,0] == 0, :]

        # Only include displacements originating from the target
        # frame interval
        disps = disps[disps[:,1] == n_frames, :]

        if disps.shape[0] > 0:
            all_jumps.append(disps.copy())

    # Concatenate
    return np.concatenate(all_jumps, axis=0)[:,2:4]

def rad_disp_2d(tracks, n_frames=4, frame_interval=0.01, pixel_size_um=0.16,
    use_entire_track=False, max_jumps_per_track=10):
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
        use_entire_track:   bool. Use all displacements from every trajectory.
        max_jumps_per_track :   int, the maximum number of displacements to 
                                include from each trajectory, if *use_entire_track*
                                is *False*

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

    # Convert to ndarray for speed
    T = np.asarray(T[["frame", "trajectory", "y", "x", "trajectory"]])

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,2:4] = T[:,2:4] * pixel_size_um 

    result = []

    # For each frame interval and each track, calculate the vector change in position
    for t in range(1, n_frames+1):
        diff = T[t:,:] - T[:-t,:]

        # Map trajectory indices back to the first localization of each trajectory
        diff[:,4] = T[t:,1]

        # Only consider vectors between points originating in the same track
        diff = diff[diff[:,1] == 0.0, :]

        # Only consider vectors that match the delay being considered
        diff = diff[diff[:,0] == t, :]

        # Only consider a finite number of displacements from each trajectory
        if not use_entire_track:
            _df = pd.DataFrame(diff[:,4], columns=["traj"])
            _df["ones"] = 1
            _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum()
            diff = diff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

        # Calculate radial displacements
        result_t = np.empty((diff.shape[0], 2), dtype=np.float64)
        result_t[:,0] = np.sqrt((diff[:,2:4]**2).sum(axis=1))
        result_t[:,1] = t * frame_interval 

        result.append(result_t)

    return np.concatenate(result, axis=0)

def rad_disp_histogram_2d(tracks, n_frames=4, bin_size=0.001, 
    max_jump=5.0, pixel_size_um=0.160, n_gaps=0, use_entire_track=False,
    max_jumps_per_track=10):
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
        n_gaps          :   int, the number of gaps allowed during tracking
        use_entire_track:   bool, use every displacement in the dataset
        max_jumps_per_track:   int, the maximum number of displacements
                            to consider per trajectory. Ignored if 
                            *use_entire_track* is *True*.

    returns
    -------
        (
            2D ndarray of shape (n_frames, n_bins), the distribution of 
                displacements at each time point;
            1D ndarray of shape (n_bins+1), the edges of each bin in um
        )

    """
    # Sort by trajectory, then frame
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Assign track lengths
    if "track_length" not in tracks.columns:
        tracks = track_length(tracks)

    # Filter out unassigned localizations and singlets
    T = tracks[
        np.logical_and(tracks["trajectory"]>=0, tracks["track_length"]>1)
    ][["frame", "trajectory", "y", "x"]]

    # Convert to ndarray for speed
    T = np.asarray(T[["frame", "trajectory", "y", "x", "trajectory"]]).astype(np.float64)

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,2:4] = T[:,2:4] * pixel_size_um 

    # Format output histogram
    bin_edges = np.arange(0.0, max_jump+bin_size, bin_size)
    n_bins = bin_edges.shape[0]-1
    H = np.zeros((n_frames, n_bins), dtype=np.int64)

    # Consider gap frames
    if n_gaps > 0:

        # The maximum trajectory length, including gap frames
        max_len = (n_gaps + 1) * n_frames + 1

        # Consider every shift up to the maximum trajectory length
        for l in range(1, max_len+1):

            # Compute the displacement for all possible jumps
            diff = T[l:,:] - T[:-l,:]

            # Map the trajectory index corresponding to the first point in 
            # each trajectory
            diff[:,4] = T[l:,1]

            # Only consider vectors between points originating from the same track
            diff = diff[diff[:,1] == 0.0, :]

            # Look for jumps corresponding to each frame interval being considered
            for t in range(1, n_frames+1):

                # Find vectors that match the delay being considered
                subdiff = diff[diff[:,0] == t, :]

                # Only consider a finite number of displacements from each trajectory
                if not use_entire_track:
                    _df = pd.DataFrame(subdiff[:,4], columns=["traj"])
                    _df["ones"] = 1
                    _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum() 
                    subdiff = subdiff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

                # Calculate radial displacements
                r_disps = np.sqrt((subdiff[:,2:4]**2).sum(axis=1))
                H[t-1,:] = H[t-1,:] + np.histogram(r_disps, bins=bin_edges)[0]

    # No gap frames
    else:

        # For each frame interval and each track, calculate the vector change in position
        for t in range(1, n_frames+1):
            diff = T[t:,:] - T[:-t,:]

            # Map trajectory indices back to the first localization of each trajectory
            diff[:,4] = T[t:,1]

            # Only consider vectors between points originating in the same track
            diff = diff[diff[:,1] == 0.0, :]

            # Only consider vectors that match the delay being considered
            diff = diff[diff[:,0] == t, :]

            # Only consider a finite number of displacements from each trajectory
            if not use_entire_track:
                _df = pd.DataFrame(diff[:,4], columns=["traj"])
                _df["ones"] = 1
                _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum()
                diff = diff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

            # Calculate radial displacements
            r_disps = np.sqrt((diff[:,2:4]**2).sum(axis=1))
            H[t-1,:] = np.histogram(r_disps, bins=bin_edges)[0]

    return H, bin_edges 

def rad_disp_histogram_3d(tracks, n_frames=4, bin_size=0.001, 
    max_jump=5.0, pixel_size_um=0.160, use_entire_track=True, 
    max_jumps_per_track=10):
    """
    Compile a histogram of 3D radial displacements for a set of
    trajectories ("tracks").

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
        use_entire_track:   bool, use every displacement in every trajectory
                            in the dataset
        max_jumps_per_track:   int, the maximum number of displacements
                            to consider per trajectory if *use_entire_track*
                            is *False*.

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
    ][["frame", "trajectory", "z", "y", "x"]]

    # Convert to ndarray for speed
    T = np.asarray(T[["frame", "trajectory", "z", "y", "x", "trajectory"]])

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,2:5] = T[:,2:5] * pixel_size_um 

    # Format output histogram
    bin_edges = np.arange(0.0, max_jump+bin_size, bin_size)
    n_bins = bin_edges.shape[0]-1
    H = np.zeros((n_frames, n_bins), dtype=np.int64)

    # For each frame interval and each track, calculate the vector change in position
    for t in range(1, n_frames+1):
        diff = T[t:,:] - T[:-t,:]

        # Map trajectory indices back to the first localization of each displacement
        diff[:,5] = T[:-t,1]

        # Only consider vectors between points originating in the same track
        diff = diff[diff[:,1] == 0.0, :]

        # Only consider vectors that match the delay being considered
        diff = diff[diff[:,0] == t, :]

        # Only consider a finite number of displacements from each trajectory
        if not use_entire_track:

            _df = pd.DataFrame(diff[:,5], columns=["traj"])
            _df["ones"] = 1
            _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum()
            diff = diff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

        # Calculate radial displacements
        r_disps = np.sqrt((diff[:,2:5]**2).sum(axis=1))
        H[t-1,:] = np.histogram(r_disps, bins=bin_edges)[0]

    return H, bin_edges 

###########################
## ANGULAR DISTRIBUTIONS ##
###########################

def bond_angles(tracks, min_disp=0.2):
    """
    Return the angles between subsequent displacements for a 
    set of trajectories. Angles between pi and 2 * pi are 
    reflected onto the interval 0, pi.

    args
    ----
        tracks      :   pandas.DataFrame
        min_disp    :   float, discard displacements less than
                        this displacement. This prevents us from
                        being biased by localization error.

    returns
    -------
        1D ndarray of shape (n_angles,), the observed
            angles in radians (from 0 to pi)

    """
    tracks = track_length(tracks)
    T = np.asarray(
        tracks[(tracks["trajectory"] >= 0) & (tracks["track_length"] > 2)][
            ["trajectory", "frame", "y", "x"]
        ]
    )
    if T.shape[0] == 0:
        return np.nan

    traj_indices = np.unique(T[:, 0])
    n_angles = T.shape[0] - 2 * len(traj_indices)
    angles = np.zeros(n_angles, dtype="float64")

    c = 0
    for i, j in enumerate(traj_indices):
        traj = T[T[:, 0] == j, 2:]
        disps = traj[1:, :] - traj[:-1, :]
        mags = np.sqrt((disps ** 2).sum(axis=1))
        traj_angles = (disps[1:, :] * disps[:-1, :]).sum(axis=1) / (mags[1:] * mags[:-1])

        # Only take angles above a given displacement, if desired
        traj_angles = traj_angles[(mags[1:] >= min_disp) & (mags[:-1] >= min_disp)]

        # Aggregate
        n_traj_angles = traj_angles.shape[0]
        angles[c : c + n_traj_angles] = traj_angles
        c += n_traj_angles

    # We'll lose some angles because of the min_disp cutoff
    angles = angles[:c]

    # Some floatint point errors occur here - values slightly
    # greater than 1.0 or less than -1.0
    angles[angles > 1.0] = 1.0 
    angles[angles < -1.0] = -1.0 

    return np.arccos(angles[~pd.isnull(angles)])

def ang_dist(tracks, min_disp=0.2, n_bins=50):
    """
    Calculate the angular distribution of some trajectories. The 
    support of the histogram is produced by dividing the angular
    range 0 to pi into some number of bins.

    args
    ----
        tracks      :   pandas.DataFrame
        min_disp    :   float, minimum displacement to consider in um
        n_bins      :   int, the number of bins in the histogram

    returns
    -------
        (
            1D ndarray of shape (n_bins,), the histogram;
            1D ndarray of shape (n_bins+1,), the edges of the 
                bins in radians
        )

    """
    # Calculate angles
    angles = bond_angles(tracks, min_disp=min_disp)

    # Make the histogram
    bin_edges = np.linspace(0, np.pi, n_bins+1)
    H = np.histogram(angles, bins=bin_edges)[0]
    
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

def defoc_prob_brownian(D, n_frames, frame_interval, dz, n_gaps=0,
    start_outside=False):
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
        n_gaps      :   int, the number of gap frames allowed during tracking

    returns
    -------
        1D ndarray of shape (n_frames,), the probability that the particle
            remains in the focal volume at the end of each frame

    """
    if dz is np.inf:
        return np.ones(n_frames, dtype=np.float64)

    if n_gaps > 0:
        return defoc_prob_brownian_gapped(D, n_frames, frame_interval, dz, n_gaps=n_gaps)

    # Real-space support
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 

    # Define the transfer function for this BM
    g_rft = generate_brownian_transfer_function(support, D, frame_interval)

    # Define the initial probability mass
    if start_outside:
        pmf = outside.astype(np.float64)
        pmf /= pmf.sum()
        pmf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(pmf) * g_rft, n=pmf.shape[0]))
        pmf[outside] = 0.0
        pmf /= pmf.sum()
    else:
        pmf = inside.astype("float64")
        pmf /= pmf.sum()

    # Propagate over subsequent frame intervals
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.fftshift(np.fft.irfft(np.fft.rfft(pmf) * g_rft, n=pmf.shape[0]))
        pmf[outside] = 0.0
        result[t] = pmf.sum()

    return result 

def defoc_prob_brownian_gapped(D, n_frames, frame_interval, dz, n_gaps=0):
    """
    Calculate the fraction of Brownian molecules remaining in the 
    focal volume at a few timepoints, allowing gap frames during which
    the molecule is not present in the focal volum.

    args
    ----
        D           :   float, diffusion coefficient in um^2 s^-1
        n_frames    :   int, the number of frame intervals to consider
        frame_interval: float, in seconds
        dz          :   float, focal volume depth in um
        n_gaps      :   int, the number of gap frames allowed during tracking

    returns
    -------
        1D ndarray of shape (n_frames,), the probability that the particle
            remains in the focal volume at the end of each frame       

    """
    if dz is np.inf:
        return np.ones(n_frames, dtype=np.float64)

    # z-axis support
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    n = support.shape[0]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 

    # Buffer for probability mass
    pmf = np.zeros((n_gaps+1, n), dtype=np.float64)

    # Initially all probability mass starts out uniformly distributed 
    # in the focal volume
    pmf[0,:] = inside.astype(np.float64)
    pmf[0,:] = pmf[0,:] / pmf[0,:].sum()

    # A buffer for the probability density that lands inside the focal 
    # volume at each frame
    spillover = np.zeros(n, dtype=np.float64)

    # Define the transfer function for this BM
    g_rft = generate_brownian_transfer_function(support, D, frame_interval)

    # The frame gaps to consider, in order of evaluation
    gaps = list(range(n_gaps+1))[::-1]

    # Fraction of molecules remaining at each frame
    result = np.zeros(n_frames, dtype=np.float64)

    # Propagate
    for t in range(n_frames):

        for g in gaps:
            prop = np.fft.fftshift(np.fft.irfft(np.fft.rfft(pmf[g,:]) * g_rft, n=n))
            spillover[inside] += prop[inside]
            prop[inside] = 0
            if g < n_gaps:
                pmf[g+1,:] = prop

        pmf[0,:] = spillover
        result[t] = spillover.sum()
        spillover[:] = 0
    return result 

def defoc_prob_levy(D, alpha, n_frames, frame_interval, dz, n_gaps=0):
    """
    Calculate the fraction of Levy flights remaining in the focal volume at 
    a few frame intervals.

    Specifically:

    A Levy flight is generated ("photoactivated") with uniform probability 
    across the focal depth *dz*, and then observed at regular intervals. If
    the particle is outside the focal volume at any one interval, it is counted
    as "lost" and is not observed for any subsequent frame, even if it diffuses
    back into the focal volume.

    This function returns the probability that such a particle is observed at 
    each frame.

    args
    ----
        D               :   float, diffusion coefficient in um^2 s^-1
        alpha           :   float, stability parameter for the Levy flight
        n_frames        :   int, the number of frame intervals
        frame_interval  :   float, the frame interval in seconds
        dz              :   float, focal depth in um

    returns
    -------
        1D ndarray of shape (n_frames,), the probability that the particle
            remains in the focal volume at the end of each frame

    """
    if dz is np.inf:
        return np.ones(n_frames, dtype=np.float64)

    if n_gaps > 0:
        return defoc_prob_levy_gapped(D, alpha, n_frames, frame_interval,
            dz, n_gaps=n_gaps)

    # Generate the initial profile in *z* on a support ranging from -5 to +5 um,
    # with 1 nm bins
    support = np.linspace(-5.0, 5.0, 10001)
    hz = dz * 0.5 
    inside = np.abs(support) <= hz 
    outside = ~inside 
    pmf = inside.astype(np.float64)
    pmf /= pmf.sum()

    # Generate the transfer function for PDF evolution
    k = 2 * np.pi * np.fft.rfftfreq(support.shape[0], d=0.001)
    cf = np.exp(-D * frame_interval * np.power(np.abs(k), alpha))

    # Evolve the distribution, annihilating all probability density outside 
    # the focal plane at each frame interval
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.irfft(np.fft.rfft(pmf) * cf, n=pmf.shape[0])
        pmf[outside] = 0
        result[t] = pmf[inside].sum()

    return result 

def defoc_prob_levy_gapped(D, alpha, n_frames, frame_interval, dz, n_gaps=0):
    """
    Calculate the fraction of Levy flights remaining in the focal volume at 
    a few frame intervals, allowing gap frames where the trajectories can 
    be outside the focal volume before returning.

    args
    ----
        D               :   float, diffusion coefficient in um^2 s^-1
        alpha           :   float, stability parameter for the Levy flight
        n_frames        :   int, the number of frame intervals
        frame_interval  :   float, the frame interval in seconds
        dz              :   float, focal depth in um
        n_gaps          :   int, the number of gap frames to tolerate 
                            before assuming the trajectory is dropped

    returns
    -------
        1D ndarray of shape (n_frames,), the probability that the particle
            remains in the focal volume at the end of each frame

    """
    if dz is np.inf:
        return np.ones(n_frames, dtype=np.float64)

    # z-axis support
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    n = support.shape[0]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 

    # Buffer for probability mass
    pmf = np.zeros((n_gaps+1, n), dtype=np.float64)

    # Initially all probability mass starts out uniformly distributed 
    # in the focal volume
    pmf[0,:] = inside.astype(np.float64)
    pmf[0,:] = pmf[0,:] / pmf[0,:].sum()

    # A buffer for the probability density that lands inside the focal 
    # volume at each frame
    spillover = np.zeros(n, dtype=np.float64)

    # Generate the transfer function for PDF evolution
    k = 2 * np.pi * np.fft.rfftfreq(support.shape[0], d=0.001)
    cf = np.exp(-D * frame_interval * np.power(np.abs(k), alpha))

    # The frame gaps to consider, in order of evaluation
    gaps = list(range(n_gaps+1))[::-1]

    # Fraction of molecules remaining at each frame
    result = np.zeros(n_frames, dtype=np.float64)

    # Propagate
    for t in range(n_frames):

        for g in gaps:
            prop = np.fft.irfft(np.fft.rfft(pmf[g,:]) * cf, n=n)
            spillover[inside] += prop[inside]
            prop[inside] = 0
            if g < n_gaps:
                pmf[g+1,:] = prop

        pmf[0,:] = spillover
        result[t] = spillover.sum()
        spillover[:] = 0
    return result 

def defoc_prob_fbm(D, hurst, n_frames, frame_interval, dz, D_type=4):
    """
    Return a vector representing the defocalization probability of a
    fractional Brownian motion with a given Hurst parameter and 
    diffusion coefficient at each frame up to *n_frames*.

    args
    ----
        D           :   float, the diffusion coefficient of the FBM. This
                        is assumed to the "normalized" diffusion coefficient
                        with units of um^2 s^-1 (D_type = 3), the same that
                        is internally used in the fitting routines
        hurst       :   float between 0.0 and 1.0, Hurst parameter
        n_frames    :   int, the number of frame intervals
        frame_interval  :   float, the frame interval in seconds
        dz          :   float, the depth of the focal plane in um
        D_type      :   int, the convention for the diffusion coefficient.
                        Unless you know what you're doing, don't touch this.

    returns
    -------
        1D ndarray of dtype float64 and length *n_frames*, the fraction
            of FBMs remaining in the slice at each of the frame intervals

    """
    if dz is np.inf:
        return np.ones(n_frames, dtype=np.float64)
        
    if n_frames > 8:
        raise RuntimeError("strobemodels.utils.defoc_prob_fbm: no more than " \
            "8 frame intervals supported for FBM fitting")

    # Get the dispersion parameter
    if D_type == 1:
        c = np.log(D * np.power(frame_interval, 2*hurst))
    elif D_type == 2:
        c = 2 * hurst * np.log(D * frame_interval)
    elif D_type == 3:
        c = np.log(D * frame_interval / (2 * hurst))
    elif D_type == 4:
        c = np.log(D * frame_interval)

    # If the dispersion lies outside the interpolated range
    # if c < -23.0:
    #     return np.ones(n_frames, dtype=np.float64)
    # if c > 6.9:
    #     return np.zeros(n_frames, dtype=np.float64)

    # Load the spline coefficients
    tcks = load_fbm_defoc_spline(dz=dz)

    # Evaluate the probability of defocalization at each frame interval 
    return np.asarray([eval_spline(hurst, c, tck) for tck in tcks[:n_frames]])

#################################
## SPLINE EVALUATION UTILITIES ##
#################################

@lru_cache(maxsize=1)
def load_fbm_defoc_spline(dz=0.7):
    """
    Given a focal depth, get a spline interpolator that enables calculation
    of the fraction of FBMs that defocalize at various frame intervals.

    args
    ----
        dz      :   float, the focal depth in um

    returns
    -------
        5-tuple, the *tck* argument expected by scipy.interpolate's spline
            evaluators -- specifically scipy.interpolate.bisplev

    """
    # Available frame intervals
    avail_dz = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

    # Get the closest available focal depth
    m = np.argmin(np.abs(avail_dz - dz))
    sel_dz = avail_dz[m]

    # Path to this file
    path = os.path.join(DATA_DIR, "fbm_defoc_splines_dz-%.1f.csv" % sel_dz)

    # Load the spline coefficients
    tcks = load_spline_coefs_multiple_frame_interval(path)
    return tcks

def eval_spline(x, y, tck):
    """
    Evaluate a bivariate spline at each combination of a set of X and Y 
    points.

    args
    ----
        x       :   1D ndarray, the array of unique x points
        y       :   1D ndarray, the array of unique y points
        tck     :   5-tuple, bivariate spline coefficients of the type
                    read by *load_spline_coefs*

    returns
    -------
        2D ndarray of shape (y.shape[0], x.shape[0]), the evaluated
            bivariate spline at each combination of the input points

    """
    return interpolate.bisplev(x, y, tck).T 

def load_spline_coefs(path):
    """
    Load a single set of bivariate spline coefficients from a file. These
    are in the format required by scipy.interpolate for evaluation of bivariate
    splines.

    args
    ----
        path        :   str, path to a file of the type written by 
                        save_spline_coefs()

    returns
    -------
        5-tuple, the *tck* argument expected by scipy.interpolate.bisplev

    """
    with open(coefs_path, "r") as f:
        lines = f.readlines()
    lines = [l.replace("\n", "") for l in lines]
    x = np.asarray([float(j) for j in lines[0].split(",")])
    y = np.asarray([float(j) for j in lines[1].split(",")])
    coefs = np.asarray([float(j) for j in lines[2].split(",")])
    kx = int(lines[3])
    ky = int(lines[4])
    return (x, y, coefs, kx, ky)

def load_spline_coefs_multiple_frame_interval(path):
    """
    Load multiple sets of bivariate spline coefficients from a file.
    These are in the format required by scipy.interpolate for 
    evaluation of bivariate splines.

    The individual sets of spline coefficients are ;-delimited, while
    the different parts of the coefficient 5-tuple are newline-delimited and
    the individual numbers are ,-delimited.

    args
    ----
        path        :   str, path to a file of the type written by
                        save_spline_coefs_multiple()

    returns
    -------
        list of 5-tuple, the bivariate spline coefficients for each
            frame interval

    """
    with open(path, "r") as f:
        S = f.read().split(";")
    S = [j.split("\n") for j in S]
    result = []
    for lines in S:
        x = np.asarray([float(j) for j in lines[0].split(",")])
        y = np.asarray([float(j) for j in lines[1].split(",")])
        coefs = np.array([float(j) for j in lines[2].split(",")])
        kx = int(lines[3])
        ky = int(lines[4])
        result.append((x, y, coefs, kx, ky))
    return result 

def save_spline_coefs(path, tck):
    """
    Save a set of bivariate spline coefficients to a file that can be 
    later read by load_spline_coefs.

    args
    ----
        path        :   str, out path 
        tck         :   5-tuple, the spline coefficients produced by 
                        running scipy.interpolate.interp2d,
                        scipy.interpolate.bisplrep, or similar

    """
    def str_concat(arraylike):
        return ",".join([str(j) for j in arraylike])

    with open(coefs_path, "w") as o:
        o.write("\n".join([str_concat(tck[i]) for i in range(3)]))
        o.write("\n%d\n%d" % (tck[3], tck[4]))

def save_spline_coefs_multiple_frame_interval(path, interpolators):
    """
    Similar to save_spline_coefs(), but save spline coefficients
    originating from multiple frame intervals.

    The result can be read by load_spline_coefs_multiple_frame_intervals().

    args
    ----
        path            :   str, out path
        interpolators   :   list of scipy.interpolate.interpolate.interp2d,
                            spline interpolators.

    """
    def str_concat(arraylike):
        return ",".join([str(j) for j in arraylike])   

    def tck_rep(tck):
        S = "\n".join([str_concat(tck[i]) for i in range(3)])
        S = "%s\n%d\n%d\n" % (S, tck[3], tck[4])
        return S 

    with open(path, "w") as o:
        o.write(";".join([tck_rep(interpolator.tck) for interpolator in interpolators]))

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

###########################################################
## PROJECTION OF 3D RADIALLY SYMMETRIC FUNCTIONS INTO 2D ##
###########################################################

@lru_cache(maxsize=1)
def get_proj_matrix(dz=None):
    """
    Get a matrix operator for numerical projection of the distribution of 3D radial
    displacements into 2D. This can be done either assuming free space, or assuming
    that the displacements can only be observed in a thin focal plane.

    args
    ----
        dz          :   float, focal depth in um. If not set, then we assume that 
                        the focal depth is effectively infinite and all displacements
                        are observed. Otherwise, the closest presimulated projection
                        matrix is loaded.

    returns
    -------
        2D ndarray of shape (5000, 5000), the projection matrix

    """
    if dz is None:
        proj_file = os.path.join(DATA_DIR, "free_abel_transform_range-20um.csv")
    else:
        options = np.array([0.7])
        dz_close = options[np.argmin(np.abs(options - dz))]
        proj_file = os.path.join(DATA_DIR, "abel_transform_dz-%.1fum_range-20um_uniform.csv" % dz_close)
    P = np.array(pd.read_csv(proj_file).drop("r_right_edge_um", axis=1))
    return P 

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

def pdf_from_cf_rad(func_cf, x, d=3, hankel_trans_2d=HankelTrans2D,
    hankel_trans_3d=HankelTrans3D, **kwargs):
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
        d           :   int, 2 or 3. The number of spatial dimensions.
        hankel_trans_2d, hankel_trans_3d:
                        hankel.SymmetricFourierTransform objects, the
                        transformer. If not given, defaults to a very high 
                        precision instance (HankelTrans3D), but this should 
                        be specified if greater speed is required.
    	kwargs      :	to *func_cf*

    returns
    -------
    	1D ndarray of shape x.shape, the PDF

    """
    F = lambda j: func_cf(j, **kwargs)
    if d == 3:
        return hankel_trans_3d.transform(F, x, ret_err=False, inverse=True)
    elif d == 2:
        return hankel_trans_2d.transform(F, x, ret_err=False, inverse=True)
    else:
        raise RuntimeError("strobemodels.utils.pdf_from_cf_rad: only dimensions 2 and 3 supported")

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





