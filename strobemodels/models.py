#!/usr/bin/env python
"""
models.py -- diffusion models for the strobemodels package. This includes functions
that generate the PDF and CDF for the radial displacement distributions of different
diffusion models, such as Brownian motion, fractional Brownian motion, etc.

Most of these models are expressed as a function of *r*, the radial distance traversed
in the 2D plane of the camera, and *t*, the delay between the observation of the first
point and the second point. (*t* may take on any multiple of the frame interval.)

As a result, the CDF and PDF functions (*func*) have the signature

    pdf_or_cdf = func(rt_tuples, **kwargs)

where *rt_tuples* is a 2D ndarray of shape (n_points, 2):

    *rt_tuples[i,0]* corresponds to *r* for the i^th observation
    *rt_tuples[i,1]* corresponds to *t* for the i^th observation

This array can be generated automatically with the function 
*strobemodels.utils.generate_support*.

The output array, *pdf_or_cdf*, is a 1D ndarray of shape (n_points,) that gives the 
corresponding PDF or CDF for each observation. This functional signature is
convenient for the the scipy.optimize fitting utilities.

Another convenient form for the PDFs/CDFs is a 2D ndarray, so that 

    pdf_or_cdf[t, r]

corresponds to the PDF/CDF for the delay *t* and the jump length *r*.

"""
# Numeric
import numpy as np 

# Custom utilities
from .utils import defoc_prob_brownian

def cdf_1state_brownian(rt_tuples, D, loc_error, **kwargs):
    """
    Distribution function for the 2D radial displacements of a single-
    state Brownian motion.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the CDF
        D               :   float, diffusion coefficient in um^2 s^-1
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    return 1.0 - np.exp(-(rt_tuples[:,0]**2) / (4*(D*rt_tuples[:,1] + (loc_error**2))))

def pdf_1state_brownian(rt_tuples, D, loc_error, **kwargs):
    """
    Probability density function for the 2D radial displacements
    of a single-state Brownian motion evaluated on a finite support.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the PDF
        D               :   float, diffusion coefficient in um^2 s^-1
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    var2 = 2 * (D * rt_tuples[:,1] + (loc_error**2))
    return (rt_tuples[:,0] / var2) * np.exp(-(rt_tuples[:,0]**2) / (2*var2))

def cdf_2state_brownian_uncorr(rt_tuples, f0, D0, D1, loc_error, **kwargs):
    """
    Distribution function for the 2D radial displacements of a two-state
    Brownian motion with no state transitions.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the CDF
        f0              :   float, probability to be in state 0
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """ 
    le2 = loc_error ** 2
    r2 = rt_tuples[:,0] ** 2
    cdf_0 = 1.0 - np.exp(-r2/(4*(D0*rt_tuples[:,1] + le2)))
    cdf_1 = 1.0 - np.exp(-r2/(4*(D1*rt_tuples[:,1] + le2)))
    return f0 * cdf_0 + (1 - f0) * cdf_1 

def pdf_2state_brownian_uncorr(rt_tuples, f0, D0, D1, loc_error, **kwargs):
    """
    Probability density function for the 2D radial displacements of a two-state
    Brownian motion with no state transitions.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the PDF
        f0              :   float, probability to be in state 0
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    le2 = loc_error ** 2
    r2 = rt_tuples[:,0] ** 2
    var2_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var2_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    pdf_0 = (rt_tuples[:,0] / var2_0) * np.exp(-r2 / (2 * var2_0))
    pdf_1 = (rt_tuples[:,0] / var2_1) * np.exp(-r2 / (2 * var2_1))
    return f0 * pdf_0 + (1.0 - f0) * pdf_1 

def cdf_3state_brownian_uncorr(rt_tuples, f0, f1, D0, D1, D3, loc_error, **kwargs):
    """
    Distribution function for the 2D radial displacements of a three-state
    Brownian motion with no state transitions.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the CDF
        f0              :   float, probability to be in state 0
        f1              :   float, probability to be in state 1
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        D2              :   float, diffusion coefficient for state 2
                            in um^2 s^-1                           
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    le2 = loc_error ** 2
    r2 = rt_tuples[:,0] ** 2

    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    var_2 = 2 * (D2 * rt_tuples[:,1] + le2)

    cdf_0 = 1.0 - np.exp(-r2 / (2 * var_0))
    cdf_1 = 1.0 - np.exp(-r2 / (2 * var_1))
    cdf_2 = 1.0 - np.exp(-r2 / (2 * var_2))

    return f0*cdf_0 + f1*cdf_1 + (1-f0-f1)*cdf_2

def pdf_3state_brownian_uncorr(rt_tuples, f0, f1, D0, D1, D2, loc_error, **kwargs):
    """
    Distribution function for the 2D radial displacements of a three-state
    Brownian motion with no state transitions.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the CDF
        f0              :   float, probability to be in state 0
        f1              :   float, probability to be in state 1
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        D2              :   float, diffusion coefficient for state 2
                            in um^2 s^-1                           
        loc_error       :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    le2 = loc_error ** 2
    r2 = rt_tuples[:,0] ** 2

    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    var_2 = 2 * (D2 * rt_tuples[:,1] + le2)

    pdf_0 = (rt_tuples[:,0] / var_0) * np.exp(-r2 / (2 * var_0))
    pdf_1 = (rt_tuples[:,0] / var_1) * np.exp(-r2 / (2 * var_1))
    pdf_2 = (rt_tuples[:,0] / var_2) * np.exp(-r2 / (2 * var_2))

    return f0*pdf_0 + f1*pdf_1 * (1-f0-f1)*pdf_2

def cdf_2state_brownian_zcorr(rt_tuples, f0, D0, D1, loc_error,
    dz=0.7, frame_interval=0.01, **kwargs):
    """
    Distribution function for the 2D radial displacements of a two-state
    Brownian motion with no state transitions. In this model, state 
    occupations are corrected for defocalization of the free state. All
    of the frame intervals (unique values of rt_tuples[:,1]) must 
    be multiples of *frame_interval* for this correction.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the CDF
        f0              :   float, probability to be in state 0
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        loc_error       :   float, 1D localization error in um
        dz              :   float, the depth of the focal plane in um
        frame_interval  :   float, seconds

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    le2 = loc_error ** 2
    r2 = rt_tuples[:,0] ** 2

    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)

    # CDF for each state
    cdf_0 = 1.0 - np.exp(-r2 / (2 * var_0))
    cdf_1 = 1.0 - np.exp(-r2 / (2 * var_1))

    # Assign each observation to an integer frame
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Get the fraction of molecules in the free state 
    # remaining at each frame, after defocalization
    f_rem = defoc_prob_brownian(D1, n_frames, frame_interval, dz)

    # Use this correction to synthesize the combined CDF for both states
    result = np.empty(rt_tuples.shape[0], dtype=np.float64)
    for f in unique_frames:

        # Adjusted free fraction
        f1_adj = f_rem[f-1] * (1.0-f0)

        # Renormalize the occupations for both states
        norm_factor = f0 + f1_adj 
        f1_adj = f1_adj / norm_factor
        f0_adj = f0 / norm_factor 

        # Make the combined CDF
        in_frame = frames == f
        result[in_frame] = f0_adj * cdf_0[in_frame] + f1_adj * cdf_1[in_frame]

    return result 

def pdf_2state_brownian_zcorr(rt_tuples, f0, D0, D1, loc_error,
    dz=0.7, frame_interval=0.01, **kwargs):
    """
    Probability density function for the 2D radial displacements of a two-state
    Brownian motion with no state transitions. In this model, state 
    occupations are corrected for defocalization of the free state. All
    of the frame intervals (unique values of rt_tuples[:,1]) must 
    be multiples of *frame_interval* for this correction.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the PDF
        f0              :   float, probability to be in state 0
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        loc_error       :   float, 1D localization error in um
        dz              :   float, the depth of the focal plane in um
        frame_interval  :   float, seconds

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """   
    le2 = loc_error ** 2
    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    r2 = rt_tuples[:,0] ** 2

    # PDFs for each state
    pdf_0 = (rt_tuples[:,0]/var_0) * np.exp(-r2/(2*var_0))
    pdf_1 = (rt_tuples[:,0]/var_1) * np.exp(-r2/(2*var_1))

    # Assign each observation to an integer frame
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Given this diffusion coefficient and focal depth, get the expected
    # fraction of free molecules remaining at each time point
    f_rem = defoc_prob_brownian(D1, n_frames, frame_interval, dz)

    # Use this correction to synthesize the PDF for both states
    result = np.empty(rt_tuples.shape[0], dtype=np.float64)
    for f in unique_frames:

        # Adjusted free fraction
        f1_adj = f_rem[f-1] * (1.0-f0)

        # Renormalize the occupations for both states
        norm_factor = f0 + f1_adj 
        f1_adj = f1_adj / norm_factor
        f0_adj = f0 / norm_factor 

        # Make the combined CDF
        in_frame = frames == f
        result[in_frame] = f0_adj * pdf_0[in_frame] + f1_adj * pdf_1[in_frame]

    return result 

def cdf_3state_brownian_zcorr(rt_tuples, f0, f1, D0, D1, D2, loc_error,
    dz=0.7, frame_interval=0.01, **kwargs):
    """
    Distribution function for the 2D radial displacements of a three-state
    Brownian motion with no state transitions. In this model, state 
    occupations are corrected for defocalization of two "free" states.
    All of the frame intervals (unique values of rt_tuples[:,1]) must 
    be multiples of *frame_interval* for this correction.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the CDF
        f0              :   float, probability to be in state 0
        f1              :   float, probability to be in state 1
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        D2              :   float, diffusion coefficient for state 2
                            in um^2 s^-1               
        loc_error       :   float, 1D localization error in um
        dz              :   float, the depth of the focal plane in um
        frame_interval  :   float, seconds

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    le2 = loc_error ** 2
    r2 = rt_tuples[:,0] ** 2

    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    var_2 = 2 * (D2 * rt_tuples[:,1] + le2)

    # CDF for each state
    cdf_0 = 1.0 - np.exp(-r2 / (2 * var_0))
    cdf_1 = 1.0 - np.exp(-r2 / (2 * var_1))
    cdf_2 = 1.0 - np.exp(-r2 / (2 * var_2))

    # Assign each observation to an integer frame
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Get the fraction of molecules in each free state 
    # remaining at each frame, after defocalization
    f_rem_1 = defoc_prob_brownian(D1, n_frames, frame_interval, dz)
    f_rem_2 = defoc_prob_brownian(D2, n_frames, frame_interval, dz)

    # Use this correction to synthesize the combined CDF for both states
    result = np.empty(rt_tuples.shape[0], dtype=np.float64)
    for f in unique_frames:

        # Adjusted free fraction
        f1_adj = f_rem_1[f-1] * f1
        f2_adj = f_rem_2[f-1] * (1.0 - f0 - f1)

        # Renormalize the occupations for both states
        norm_factor = f0 + f1_adj + f2_adj 
        f2_adj = f2_adj / norm_factor 
        f1_adj = f1_adj / norm_factor
        f0_adj = f0 / norm_factor 

        # Make the combined CDF
        in_frame = frames == f
        result[in_frame] = f0_adj * cdf_0[in_frame] + \
            f1_adj * cdf_1[in_frame] + \
            f2_adj * cdf_2[in_frame]

    return result    

def pdf_3state_brownian_zcorr(rt_tuples, f0, f1, D0, D1, D2, loc_error,
    dz=0.7, frame_interval=0.01, **kwargs):
    """
    Probability density function for the 2D radial displacements of a three-state
    Brownian motion with no state transitions. In this model, state 
    occupations are corrected for defocalization of two "free" states.
    All of the frame intervals (unique values of rt_tuples[:,1]) must 
    be multiples of *frame_interval* for this correction.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the 
                            independent tuples (r, dt) at which to 
                            evaluate the PDF
        f0              :   float, probability to be in state 0
        f1              :   float, probability to be in state 1
        D0              :   float, diffusion coefficient for state 0
                            in um^2 s^-1       
        D1              :   float, diffusion coefficient for state 1
                            in um^2 s^-1
        D2              :   float, diffusion coefficient for state 2
                            in um^2 s^-1               
        loc_error       :   float, 1D localization error in um
        dz              :   float, the depth of the focal plane in um
        frame_interval  :   float, seconds

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    le2 = loc_error ** 2
    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    r2 = rt_tuples[:,0] ** 2

    # PDFs for each state
    pdf_0 = (rt_tuples[:,0]/var_0) * np.exp(-r2/(2*var_0))
    pdf_1 = (rt_tuples[:,0]/var_1) * np.exp(-r2/(2*var_1))
    pdf_2 = (rt_tuples[:,0]/var_2) * np.exp(-r2/(2*var_2))

    # Assign each observation to an integer frame
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Given this diffusion coefficient and focal depth, get the expected
    # fraction of free molecules remaining at each time point
    f_rem_1 = defoc_prob_brownian(D1, n_frames, frame_interval, dz)
    f_rem_2 = defoc_prob_brownian(D2, n_frames, frame_interval, dz)

    # Use this correction to synthesize the PDF for both states
    result = np.empty(rt_tuples.shape[0], dtype=np.float64)
    for f in unique_frames:

        # Adjusted free fraction
        f1_adj = f_rem_1[f-1] * f1
        f2_adj = f_rem_2[f-1] * (1.0 - f0 - f1)

        # Renormalize the occupations for both states
        norm_factor = f0 + f1_adj + f2_adj 
        f2_adj = f2_adj / norm_factor 
        f1_adj = f1_adj / norm_factor
        f0_adj = f0 / norm_factor 

        # Make the combined CDF
        in_frame = frames == f
        result[in_frame] = f0_adj * pdf_0[in_frame] + \
            f1_adj * pdf_1[in_frame] + \
            f2_adj * pdf_2[in_frame]

    return result 

######################
## AVAILABLE MODELS ##
######################

# Cumulative distribution functions
CDF_MODELS = {
    "one_state_brownian": cdf_1state_brownian,
    "two_state_brownian": cdf_2state_brownian_uncorr,
    "two_state_brownian_zcorr": cdf_2state_brownian_zcorr,
    "three_state_brownian": cdf_3state_brownian_uncorr,
    "three_state_brownian_zcorr": cdf_3state_brownian_zcorr,
}

# Probability density functions
PDF_MODELS = {
    "one_state_brownian": pdf_1state_brownian,
    "two_state_brownian": pdf_2state_brownian_uncorr,
    "two_state_brownian_zcorr": pdf_2state_brownian_zcorr,
    "three_state_brownian": pdf_3state_brownian_uncorr,
    "three_state_brownian_zcorr": pdf_3state_brownian_zcorr,   
}

# Identity of each fit parameter for each model 
MODEL_PARS = {
    "one_state_brownian": ["D", "loc_error"],
    "two_state_brownian": ["f0", "D0", "D1", "loc_error"],
    "two_state_brownian_zcorr": ["f0", "D0", "D1", "loc_error"],
    "three_state_brownian": ["f0", "f1", "D0", "D1", "D2", "loc_error"],
    "three_state_brownian_zcorr": ["f0", "f1", "D0", "D1", "D2", "loc_error"]
}

# Number of parameters per model
MODEL_N_PARS = {k: len(MODEL_PARS[k]) for k in MODEL_PARS.keys()}

# Naive bounds on fit parameters
MODEL_PAR_BOUNDS = {
    "one_state_brownian": (
        np.array([1.0e-8, 0.0]),
        np.array([np.inf, 0.1])
    ),
    "two_state_brownian": (
        np.array([0.0, 1.0e-8, 0.5, 0.0]),
        np.array([1.0, 0.005, np.inf, 0.1])
    ),
    "two_state_brownian_zcorr": (
        np.array([0.0, 1.0e-8, 0.5, 0.0]),
        np.array([1.0, 0.005, np.inf, 0.1])
    ),
    "three_state_brownian": (
        np.array([0.0, 0.0, 1.0e-8, 0.05, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.02, 1.0, np.inf, 0.1])
    ),
    "three_state_brownian_zcorr": (
        np.array([0.0, 0.0, 1.0e-8, 0.05, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.02, 1.0, np.inf, 0.1])
    ),
}

# Naive parameter guesses to seed fits
MODEL_GUESS = {
    "one_state_brownian": np.array([1.0, 0.035]),
    "two_state_brownian": np.array([0.3, 0.01, 1.0, 0.035]),
    "two_state_brownian_zcorr": np.array([0.3, 0.001, 1.0, 0.035]),
    "three_state_brownian": np.array([0.33, 0.33, 0.001, 0.5, 2.0, 0.035]),
    "three_state_brownian_zcorr": np.array([0.33, 0.33, 0.001, 0.5, 2.0, 0.035]),
}










