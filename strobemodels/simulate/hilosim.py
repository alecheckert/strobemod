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

from strobemodels.simulate.simutils import tracks_to_dataframe
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
        model_obj = DIFFUSION_MODELS[model](dt=frame_interval, track_len=track_len, **model_kwargs)
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

