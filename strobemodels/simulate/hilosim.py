#!/usr/bin/env python
"""
hilosim.py -- simulated regular and anomalous diffusion processes in 
stroboscopic highly inclined laminated optical sheet (HiLo) geometry.

This means that a diffusion process in 3D space is only observable
(1) at regular frame intervals and (2) across a finite range
of positions in *z*. Additionally, the remaining parts of the 3D
trajectory are projected onto the 2D plane of the camera.

This module implements methods to simulate trajectories acquired in HiLo 
geometry.

"""
import os
import numpy as np 
import pandas as pd 

from strobemodels.simulate.simutils import (
    tracks_to_dataframe,
    tracks_to_dataframe_gapped
)
from strobemodels.simulate import fbm 
from strobemodels.simulate import levy
from strobemodels.utils import concat_tracks 

# Available diffusion simulators
DIFFUSION_MODELS = {
    "brownian": fbm.FractionalBrownianMotion3D,
    "fbm": fbm.FractionalBrownianMotion3D,
    "levy": levy.LevyFlight3D
}

def strobe_infinite_plane(model_obj, n_tracks, dz=0.7, loc_error=0.0, exclude_outside=True, 
    n_gaps=0, return_dataframe=True):
    """
    Simulate 3D trajectories with photoactivation in a HiLo geometry. Molecules
    are photoactivated with uniform probability in the axial column. Their displacements
    in the XY plane are observed, and the molecules are lost as soon as their z position
    lies outside of the interval (-dz/2, dz/2) during one frame interval.

    args
    ----
        model_obj       :   either a FractionalBrownianMotion3D or LevyFlight3D object,
                            the simulator
        n_tracks        :   int, the number of trajectories to simulate
        dz              :   float, thickness of the observation slice in um
        loc_error       :   float, standard deviation of normally distributed 1D
                            localization error in um
        exclude_outside :   bool, exclude localizations that lie outside of the detectable
                            z interval (-dz/2, dz/2). If False, the whole trajectories
                            are returned.
        n_gaps          :   int, the number of gap frames to tolerate in trajectories before
                            dropping them. Only applies if *exclude_outside* is True.
        return_dataframe:   bool, format results as a pandas.DataFrame rather than a 
                            numpy.ndarray

    returns
    -------
        if return_dataframe:
            pandas.DataFrame with columns ["trajectory", "frame", "z", "y", "x"]
        else:
            3D np.ndarray with shape (n_tracks, track_len, 3), the 3D positions

    """
    hz = dz / 2.0 

    # Simulate some 3D trajectories
    tracks = model_obj(n_tracks)

    # Add a random starting position in z
    tracks[:,:,0] = (tracks[:,:,0].T + np.random.uniform(-hz, hz, size=n_tracks)).T 

    # Exclude molecules outside of the slice from observation by setting their
    # values to NaN
    if exclude_outside:

        # Simulate tracking with gaps: if a particle returns to the focal volume
        # after being outside, it can be observed again and reconnected with the
        # previous part of its trajectory
        # if gaps:
        #     outside = np.abs(tracks[:,:,0]) > hz
        #     for d in range(3):
        #         tracks[:,:,d][outside] = np.nan

        if n_gaps > 0:
            gap_count = np.zeros(n_tracks, dtype=np.int64)
            dead = np.zeros(n_tracks, dtype=np.bool)

            for t in range(model_obj.track_len):
                outside = np.abs(tracks[:,t,0]) > hz
                gap_count[outside] += 1
                gap_count[~outside] = 0
                dead = np.logical_or(dead, gap_count>n_gaps)
                set_nan = np.logical_or(outside, dead)
                for d in range(3):
                    tracks[:,t,d][set_nan] = np.nan

        # Simulate tracking without gaps, so that if a molecule lies outside
        # of the focal volume at any given frame, it is lost for all future frames
        else:
            outside = np.zeros(n_tracks, dtype="bool")
            for t in range(model_obj.track_len):

                # If a molecule is observed outside the slice once, it is lost for 
                # all subsequent frame intervals
                outside = np.logical_or(outside, np.abs(tracks[:,t,0])>hz)
                for d in range(3):
                    tracks[:,t,d][outside] = np.nan 

    # Add localization error, if desired
    if loc_error != 0.0:
        tracks = tracks + np.random.normal(scale=loc_error, size=tracks.shape)

    # Format as a pandas.DataFrame, if desired
    if return_dataframe:
        return tracks_to_dataframe(tracks, kill_nan=True)
    else:
        return tracks 

def strobe_nucleus(model_obj, n_tracks, dz=0.7, loc_error=0.0, exclude_outside=True,
    n_gaps=0, nucleus_radius=5.0, bleach_prob_per_frame=0):
    """
    Simulate 3D trajectories that are photoactivated at any point in a spherical
    "nucleus" and diffuse around for some number of frames. if *return_dataframe*
    is *True*, then these trajectories are only observed when they lie within a 
    thin slice of thickness *dz* that bisects the nucleus.

    args
    ----
        model_obj       :   a FractionalBrownianMotion3D or LevyFlight3D object,
                            the simulator; or alternatively a 3D ndarray of shape
                            (n_tracks, track_len, 3), the 3D positions of each 
                            trajectory
        n_tracks        :   int, the number of trajectories to simulate
        dz              :   float, the thickness of the observation plane in um
        loc_error       :   float, standard deviation of normally distributed 1D
                            localization error in um
        exclude_outside :   bool, exclude localizations that lie outside of the detectable
                            z interval (-dz/2, dz/2). If False, the whole trajectories
                            are returned.
        n_gaps          :   int, the number of gap frames to tolerate in trajectories before
                            dropping them. Only applies if *exclude_outside* is True.
        nucleus_radius  :   float, radius of the nucleus in um. Trajectories are 
                            not allowed to cross nuclear boundaries.

    returns
    -------
        pandas.DataFrame with columns ["trajectory", "frame", "z", "y", "x"]

    """
    # Half the observation slice width
    hz = dz * 0.5

    # Nucleus diameter
    diameter = nucleus_radius * 2

    # Simulate some 3D trajectories, which start at the origin
    if isinstance(model_obj, np.ndarray):
        tracks = model_obj
    else:
        tracks = model_obj(n_tracks)


    ## STARTING POSITIONS: choose random starting positions inside a sphere of 
    # radius *nucleus_radius*

    # Angle relative to origin, sampled using the Gaussian method
    start_pos_ang = np.random.normal(size=(n_tracks, 3))
    start_pos_ang = (start_pos_ang.T / np.sqrt((start_pos_ang**2).sum(axis=1))).T 

    # Radial distance from origin
    start_pos_rad = nucleus_radius * np.cbrt(np.random.random(size=n_tracks))

    # Starting positions
    start_pos = (start_pos_ang.T * start_pos_rad).T 

    # Offset each trajectory by the starting position
    for dim in range(3):
        tracks[:,:,dim] = (tracks[:,:,dim].T + start_pos[:,dim]).T 


    ## SPECULAR REFLECTIONS: deal with trajectories that cross the nuclear boundary
    # by reflecting back into the nucleus
    for frame_idx in range(tracks.shape[1]):

        # Determine the set of points that are outside the nucleus at this time
        distance_from_origin = np.sqrt((tracks[:,frame_idx,:]**2).sum(axis=1))
        outside = distance_from_origin > nucleus_radius 

        # Get the closest point on the sphere to each of these points
        reflect_points = nucleus_radius * (tracks[outside,frame_idx,:].T / distance_from_origin[outside]).T

        # Reflect the trajectories back into the nucleus for each subsequent frame
        for g in range(frame_idx, tracks.shape[1]):
            tracks[outside, g, :] = 2 * reflect_points - tracks[outside, g, :]


    ## DEFOCALIZATION: exclude molecules outside of the slice from observation
    # by setting their values to NaN. Some of these molecules may subsequently
    # reenter and not be lost, if the number of gaps tolerated during tracking
    # is greater than 0
    if exclude_outside:
        for t in range(tracks.shape[1]):

            # Get the set of localizations that lie outside the focal plane
            # at this frame interval
            outside = np.abs(tracks[:,t,0]) > hz 

            # Set the coordinates of these localizations to NaN
            for d in range(3):
                tracks[:,t,d][outside] = np.nan 


    ## BLEACHING: stochastically and permanently bleach molecules. The bleaching
    # rate is assumed to be the same throughout the nucleus and is stationary
    # in time.
    if bleach_prob_per_frame > 0:

        bleached = np.zeros(tracks.shape[0], dtype=np.bool)

        for t in range(tracks.shape[1]):
            b = np.random.random(size=tracks.shape[0]) <= bleach_prob_per_frame
            bleached = np.logical_or(bleached, b)
            for d in range(3):
                tracks[:,t,d][bleached] = np.nan 


    ## LOCALIZATION ERROR: add normally-distributed localization error to 
    # each localization
    if loc_error != 0.0:
        tracks = tracks + np.random.normal(scale=loc_error, size=tracks.shape)


    # Format as a pandas.DataFrame
    if len(tracks) == 0:
        return pd.DataFrame([])
    else:
        return tracks_to_dataframe_gapped(tracks, n_gaps=n_gaps)

def strobe_ellipsoid(model_obj, n_tracks, dz=0.7, loc_error=0.0, exclude_outside=True,
    n_gaps=0, ellipse_dim=[10.0, 5.0, 5.0], bleach_prob_per_frame=0):
    """
    Simulate trajectories confined to stay inside of an ellipsoid.

    Unlike the simulation above, we do not use specular reflections for
    trajectories at the boundary, and instead resample jumps until one falls
    inside the bounds of the nucleus.

    args
    ----
        model_obj       :   a FractionalBrownianMotion3D or LevyFlight3D object,
                            the simulator; or alternatively a 3D ndarray of shape
                            (n_tracks, track_len, 3), the 3D positions of each 
                            trajectory
        n_tracks        :   int, the number of trajectories to simulate
        dz              :   float, the thickness of the observation plane in um
        loc_error       :   float, standard deviation of normally distributed 1D
                            localization error in um
        exclude_outside :   bool, exclude localizations that lie outside of the detectable
                            z interval (-dz/2, dz/2). If False, the whole trajectories
                            are returned.
        n_gaps          :   int, the number of gap frames to tolerate in trajectories before
                            dropping them. Only applies if *exclude_outside* is True.
        ellipse_dim     :   float, axial radii of the 3D ellipse in um. These are 
                            specified in the following order: z, y, x.
        bleach_prob_per_frame   :   float, probability to bleach on any given frame

    returns
    -------
        pandas.DataFrame with columns ["trajectory", "frame", "z", "y", "x"]

    """
    # Half the observation slice width
    hz = dz * 0.5

    # Ellipsoid squared radii
    ellipse_dim = np.asarray(ellipse_dim)
    e2 = ellipse_dim ** 2
    a = ellipse_dim[0]
    b = ellipse_dim[1]
    c = ellipse_dim[2]
    a2 = a ** 2
    b2 = b ** 2
    c2 = c ** 2

    # Simulate some 3D trajectories, which start at the origin
    tracks = model_obj(n_tracks)


    ## STARTING POSITIONS: choose random starting positions inside the
    # ellipsoid using rejection sampling

    start_pos = np.zeros((n_tracks, 3), dtype=np.float64)
    start_pos[:,0] = np.random.uniform(-a, a, size=n_tracks)
    start_pos[:,1] = np.random.uniform(-b, b, size=n_tracks)
    start_pos[:,2] = np.random.uniform(-c, c, size=n_tracks)

    distance = (start_pos**2 / e2).sum(axis=1)
    outside = distance > 1.0
    n_outside = outside.sum()
    while n_outside > 0:
        start_pos[outside,0] = np.random.uniform(-a, a, size=n_outside)
        start_pos[outside,1] = np.random.uniform(-b, b, size=n_outside)
        start_pos[outside,2] = np.random.uniform(-c, c, size=n_outside)
        distance = (start_pos**2 / e2).sum(axis=1)
        outside = distance > 1.0
        n_outside = outside.sum()

    # Offset each trajectory by the starting position
    for dim in range(3):
        tracks[:,:,dim] = (tracks[:,:,dim].T + start_pos[:,dim]).T 


    ## DEAL WITH BOUNDARY CROSSINGS
    track_len = tracks.shape[1]
    for t in range(1, tracks.shape[1]):

        # Determine which trajectories lie outside the ellipsoid
        # at this frame
        distance_from_origin = (tracks[:,t,:]**2 / e2).sum(axis=1)
        outside = distance_from_origin > 1.0
        n_outside = outside.sum()

        # Resimulate the remainder of the trajectories until none
        # of them lie outside the ellipsoid
        while n_outside > 0:
            tracks[outside, t:, :] = model_obj(n_outside)[:,1:track_len-t+1,:]
            for d in range(3):
                tracks[outside, t:, d] = (tracks[outside, t:, d].T + tracks[outside, t-1, d]).T
            distance_from_origin = (tracks[:,t,:]**2 / e2).sum(axis=1)
            outside = distance_from_origin > 1.0
            n_outside = outside.sum()
        print("Finished with %d timepoints..." % t)

    ## DEFOCALIZATION: exclude molecules outside of the slice from observation
    # by setting their values to NaN. Some of these molecules may subsequently
    # reenter and not be lost, if the number of gaps tolerated during tracking
    # is greater than 0
    if exclude_outside:
        for t in range(tracks.shape[1]):

            # Get the set of localizations that lie outside the focal plane
            # at this frame interval
            outside = np.abs(tracks[:,t,0]) > hz 

            # Set the coordinates of these localizations to NaN
            for d in range(3):
                tracks[:,t,d][outside] = np.nan 


    ## BLEACHING: stochastically and permanently bleach molecules. The bleaching
    # rate is assumed to be the same throughout the nucleus and is stationary
    # in time.
    if bleach_prob_per_frame > 0:

        bleached = np.zeros(tracks.shape[0], dtype=np.bool)

        for t in range(tracks.shape[1]):
            b = np.random.random(size=tracks.shape[0]) <= bleach_prob_per_frame
            bleached = np.logical_or(bleached, b)
            for d in range(3):
                tracks[:,t,d][bleached] = np.nan 


    ## LOCALIZATION ERROR: add normally-distributed localization error to 
    # each localization
    if loc_error != 0.0:
        tracks = tracks + np.random.normal(scale=loc_error, size=tracks.shape)


    # Format as a pandas.DataFrame
    if len(tracks) == 0:
        return pd.DataFrame([])
    else:
        return tracks_to_dataframe_gapped(tracks, n_gaps=n_gaps)

def strobe_multistate_nucleus(model, n_tracks,  model_diffusivities,
    model_occupations, track_len=10, dz=0.7, frame_interval=0.01, loc_error=0.0,
    exclude_outside=True, n_gaps=0, nucleus_radius=5.0, bleach_prob_per_frame=0,
    n_rounds=1, **model_kwargs):
    """
    Simulate multiple diffusing states inside a sphere ("nucleus"). These 
    are subject to the following:

        - specular reflections at the boundaries of the sphere

        - if *exclude_outside* is True, then only localizations than lie
          within a focal depth of *dz* at the center of the sphere are 
          observed

        - also if *exclude_outside* is True, then "defocalized" trajectories
          may reenter the focal volume on subsequent frames to contribute
          additional localizations. In those cases, the reentrant trajectories
          are counted as part of the original trajectories only if their 
          transit outside the focal frame has length equal to or less than 
          *n_gaps*. Otherwise, they are counted as separate trajectories.

        - if *bleach_prob_per_frame* is greater than 0, then trajectories
          will be bleached at a constant rate (for all diffusivities). When
          trajectories are bleached, they are permanently lost and are not
          counted for subsequent frames

    Note that trajectories may not be counted at all (because they never 
    transit through the focal volume) or may be counted multiple times (because
    they transit multiple times through the focal volume).

    args
    ----
        model               :   str, "brownian", "fbm", or "levy"
        n_tracks            :   int, the number of trajectories to simulate. 
                                Note that trajectories may not be counted at all
                                or be counted multiple times, depending on how they
                                transit through the focal volume
        model_diffusivities :   1D ndarray, the diffusivities for each state 
                                in um^2 s^-1
        model_occupations   :   1D ndarray, the fractional occupations of each state.
                                Must sum to 1.0.
        track_len           :   int, the number of frames to simulate per trajectory
                                (before defocalization/bleaching)
        dz                  :   float, focal depth in um
        frame_interval      :   float, the frame interval in seconds
        loc_error           :   float, 1D normally distributed localization error
                                in um
        exclude_outside     :   bool. If False, then no defocalization is performed
                                and trajectories may be detected at any point in 
                                the nucleus
        n_gaps              :   int, the number of gaps to tolerate before dropping
                                a trajectory
        nucleus_radius      :   float, radius of the spherical nucleus in um
        bleach_prob_per_frame:  float, the probability for a trajectory to bleach
                                on any given frame, assumed to be the same for all
                                diffusivities
        n_rounds            :   int, the number of rounds to repeat this simulation.
                                Sometimes helpful if a large number of trajectories
                                are needed. The results are concatenated as a single
                                pandas.DataFrame.
        model_kwargs        :   any additional keyword arguments to the model
                                simulator

    returns
    -------
        pandas.DataFrame with columns "trajectory", "frame", "z", "y", and "x"

    """
    # Only consider states with nonzero occupations
    model_occupations = np.asarray(model_occupations).copy()
    model_diffusivities = np.asarray(model_diffusivities).copy()
    nonzero = model_occupations > 0
    model_occupations = model_occupations[nonzero]
    model_diffusivities = model_diffusivities[nonzero]

    # Run the simulation for each of the remaining diffusivities
    results = []
    for round_idx in range(n_rounds):

        # Choose the number of trajectories in each diffusive state
        n_occ = np.random.multinomial(n_tracks, model_occupations)

        # Simulate the states
        tracks = []
        for i, D in enumerate(model_diffusivities):
            model_obj = DIFFUSION_MODELS[model](D=D, dt=frame_interval, track_len=track_len, **model_kwargs)
            tracks_state = strobe_nucleus(model_obj, n_occ[i], dz=dz, loc_error=loc_error,
                exclude_outside=exclude_outside, n_gaps=n_gaps, nucleus_radius=nucleus_radius,
                bleach_prob_per_frame=bleach_prob_per_frame)
            tracks.append(tracks_state)

        # Concatenate trajectories from both states
        if isinstance(tracks[0], pd.DataFrame):
            tracks = concat_tracks(*tracks)
        else:
            tracks = np.concatenate(tracks, axis=0)

        results.append(tracks)

    if n_rounds > 1:
        return concat_tracks(*results)
    else:
        return results[0]

def strobe_multistate_infinite_plane(model, n_tracks, model_diffusivities,
    model_occupations, track_len=10, dz=0.7, frame_interval=0.01, loc_error=0.0,
    exclude_outside=True, n_gaps=0, return_dataframe=True, **model_kwargs):
    """
    Simulate a mixture of non-interconverting diffusive states. If the 
    diffusion models have underlying special parameters (for instance, the
    stability parameter for Levy flights, or the Hurst parameter for FBMs),
    then these are assumed to be the same for each component of the mixture.
    Only the diffusivity (scale parameter for Levy flights) is assumed to be
    different.

    args
    ----
        model               :   str, either "brownian", "fbm", or "levy"
        n_tracks            :   int, the number of trajectories to simulate
        model_diffusivities :   1D ndarray, the diffusivities of each 
                                component in the mixture in um^2 s^-1
        model_occupations   :   1D ndarray, the fractional occupations of
                                each component in the mixture 
        track_len           :   int, the maximum length of trajectories to 
                                simulate
        dz                  :   float, the thickness of the focal depth in um
        frame_interval      :   float, time between frames in seconds
        loc_error           ;   float, 1D localization error in um
        exclude_outside     :   bool, drop trajectories once they leave the
                                focal volume 
        n_gaps              :   int, the number of gap frames tolerated before
                                dropping a trajectory
        model_kwargs        ;   additional keyword arguments and special parameters
                                for the underlying diffusion model

    returns
    -------
        pandas.DataFrame, the resulting trajectories

    """
    # Choose the number of trajectories in each diffusive state
    n_occ = np.random.multinomial(n_tracks, model_occupations)

    # Simulate the states
    tracks = []
    for i, D in enumerate(model_diffusivities):
        model_obj = DIFFUSION_MODELS[model](D=D, dt=frame_interval, track_len=track_len, **model_kwargs)
        tracks_state = strobe_infinite_plane(model_obj, n_occ[i], dz=dz, loc_error=loc_error,
            exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)
        tracks.append(tracks_state)

    # Concatenate trajectories from both states
    if isinstance(tracks[0], pd.DataFrame):
        tracks = concat_tracks(*tracks)
    else:
        tracks = np.concatenate(tracks, axis=0)

    return tracks 

def strobe_one_state_infinite_plane(model, n_tracks, track_len=10, dz=0.7, dt=0.01,
    loc_error=0.0, exclude_outside=True, n_gaps=0, return_dataframe=True, **model_kwargs):
    """
    Simulate a single diffusing state in a HiLo geometry.

    args
    ----
        model           :   str, one of the models in DIFFUSION_MODELS
        n_tracks        :   int, the number of trajectories to simulate
        dz              :   float, the thickness of the observation slice
        loc_error       :   float, localization error 
        exclude_outside :   bool, remove positions that fall outside the focal 
                            depth 
        n_gaps          :   int, number of gap frames to tolerate
        return_dataframe:   bool, format results as a pandas.DataFrame
        model_kwargs    :   for generating the diffusion model 

    returns
    -------
        if return_dataframe:
            pandas.DataFrame with columns ["trajectory", "frame", "z", "y", "x"]
        else:
            3D np.ndarray with shape (n_tracks, track_len, 3), the 3D positions

    """
    model_obj = DIFFUSION_MODELS[model](dt=dt, track_len=track_len, **model_kwargs)
    return strobe_infinite_plane(model_obj, n_tracks, dz=dz, loc_error=loc_error,
        exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)
    
def strobe_two_state_infinite_plane(model_0, n_tracks, model_1=None, f0=0.0, track_len=10,
    dz=0.7, dt=0.01, loc_error=0.0, exclude_outside=True, n_gaps=0, return_dataframe=True, 
    model_0_kwargs={}, model_1_kwargs={}):
    """
    Simulate two diffusing states in a HiLo geometry.

    args
    ----
        model_0         :   str, one of the models in DIFFUSION_MODELS
        model_1         :   str, one of the models in DIFFUSION_MODELS. If *None*,
                            assumed to be the same as *model_0*.
        n_tracks        :   int, the number of trajectories to simulate
        f0              :   float, the number of trajectories in the first
                            diffusing state
        track_len       :   int, the trajectory length 
        dz              :   float, thickness of the HiLo focal depth in um
        loc_error       :   float, 1D normal localization error in um
        exclude_outside :   bool, remove positions that fall outside the 
                            focal depth 
        n_gaps          :   int, the number of gap frames to tolerate during tracking
        return_dataframe:   bool, format result as a pandas.DataFrame
        model_0_kwargs  :   dict, model parameters for state 0
        model_1_kwargs  :   dict, model parameters for state 1

    returns
    -------
        if return_dataframe:
            pandas.DataFrame with columns ["trajectory", "frame", "z", "y", "x"]
        else:
            3D np.ndarray with shape (n_tracks, track_len, 3), the 3D positions

    """
    # If *model_1* is not given, default to the model for the first state
    if model_1 is None:
        model_1 = model_0 

    # Number of trajectories in each state
    N0 = np.random.binomial(n_tracks, f0)
    N1 = n_tracks - N0 

    # Generate simulators for each state
    model_obj_0 = DIFFUSION_MODELS[model_0](dt=dt, track_len=track_len, **model_0_kwargs)
    tracks_0 = strobe_infinite_plane(model_obj_0, N0, dz=dz, loc_error=loc_error,
        exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)   
    model_obj_1 = DIFFUSION_MODELS[model_1](dt=dt, track_len=track_len, **model_1_kwargs)
    tracks_1 = strobe_infinite_plane(model_obj_1, N1, dz=dz, loc_error=loc_error,
        exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)

    # Concatenate trajectories from both states
    if isinstance(tracks_0, pd.DataFrame):
        tracks = concat_tracks(tracks_0, tracks_1)
    else:
        tracks = np.concatenate([tracks_0, tracks_1], axis=0)

    return tracks 

def strobe_three_state_infinite_plane(model_0, n_tracks, model_1=None, model_2=None, f0=0.0, f1=0.0,
    track_len=10, dz=0.7, dt=0.01, loc_error=0.0, exclude_outside=True, n_gaps=False, return_dataframe=True,
    model_0_kwargs={}, model_1_kwargs={}, model_2_kwargs={}):
    """
    Simulate three diffusing states in a HiLo geometry.

    args
    ----
        model_0         :   str, one of the models in DIFFUSION_MODELS
        n_tracks        :   int, the number of trajectories to simulate
        model_1         :   str, one of the models in DIFFUSION_MODELS. If *None*,
                            default to *model_0*.
        model_2         :   str, one of the models in DIFFUSION_MODELS. If *None*,
                            default to *model_1*.
        f0              :   float, the number of trajectories in the first
                            diffusing state
        track_len       :   int, the trajectory length 
        dz              :   float, thickness of the HiLo focal depth in um
        loc_error       :   float, 1D normal localization error in um
        exclude_outside :   bool, remove positions that fall outside the 
                            focal depth 
        n_gaps          :   int, the number of gap frames to tolerate
        return_dataframe:   bool, format result as a pandas.DataFrame
        model_0_kwargs  :   dict, model parameters for state 0
        model_1_kwargs  :   dict, model parameters for state 1
        model_2_kwargs  :   dict, model parameters for state 2

    returns
    -------
        if return_dataframe:
            pandas.DataFrame with columns ["trajectory", "frame", "z", "y", "x"]
        else:
            3D np.ndarray with shape (n_tracks, track_len, 3), the 3D positions

    """
    # If not given, default to the first model
    if model_1 is None:
        model_1 = model_0 
    if model_2 is None:
        model_2 = model_1 

    # Number of trajectories in each state
    N0, N1, N2 = np.random.multinomial(n_tracks, np.array([f0, f1, 1-f0-f1]))

    # Generate simulators for each state
    model_obj_0 = DIFFUSION_MODELS[model_0](dt=dt, track_len=track_len, **model_0_kwargs)
    model_obj_1 = DIFFUSION_MODELS[model_1](dt=dt, track_len=track_len, **model_1_kwargs)
    model_obj_2 = DIFFUSION_MODELS[model_2](dt=dt, track_len=track_len, **model_2_kwargs)

    # Run the simulations
    tracks_0 = strobe_infinite_plane(model_obj_0, N0, dz=dz, loc_error=loc_error,
        exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)
    tracks_1 = strobe_infinite_plane(model_obj_1, N1, dz=dz, loc_error=loc_error,
        exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)
    tracks_2 = strobe_infinite_plane(model_obj_2, N2, dz=dz, loc_error=loc_error,
        exclude_outside=exclude_outside, n_gaps=n_gaps, return_dataframe=return_dataframe)

    # Concatenate trajectories from both states
    if isinstance(tracks_0, pd.DataFrame):
        tracks = concat_tracks(tracks_0, tracks_1, tracks_2)
    else:
        tracks = np.concatenate([tracks_0, tracks_1, tracks_2], axis=0)

    return tracks 

