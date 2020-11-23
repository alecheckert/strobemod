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

# Fairly low-precision 3D Hankel transform, for fast model refinement
from hankel import SymmetricFourierTransform
HankelTrans3DLowPrec = SymmetricFourierTransform(ndim=3, N=500, h=0.01)
HankelTrans2DLowPrec = SymmetricFourierTransform(ndim=2, N=500, h=0.001)

# Radial projection in the HiLo geometry
from .radproj import radproj 

# Various custom utilities
from .utils import (
    defoc_prob_brownian,
    defoc_prob_fbm,
    defoc_prob_levy,
    pdf_from_cf_rad,
    get_proj_matrix,
    radnorm
)

# For profiling
global_iter_idx = 0

# Frequency domain support, for evaluation of the Levy flight density
real_support = np.arange(0.0, 100.001, 0.001)
freq_support = 2 * np.pi * np.fft.rfftfreq(real_support.shape[0], d=0.001)

def levy_flight_cf(k, alpha, scale, dt, loc_error):
    """
    Characteristic function for the radial displacements of a Levy flight
    in terms of its radial frequency coordinate *k*.

    args
    ----
        k           :   float or arraylike, the radial frequency
        alpha       :   float between 0.0 and 2.0, the stability term
        scale       :   float, the dispersion term
        dt          :   float, interval between frames
        loc_error   :   float, 1D localization error in um

    returns
    -------
        float or arraylike, the characteristic function evaluated
            at each point in *k*

    """
    return np.exp(-scale * dt * np.power(np.abs(k), alpha) - (loc_error**2) * (k**2))

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

def cdf_1state_fbm(rt_tuples, hurst, D, loc_error, frame_interval=0.01,
    D_type=4, **kwargs):
    """
    Distribution function for the 2D radial displacements of a single-
    state fractional Brownian motion.

    note
    ----
    Assumes the diffusion coefficient is "D_type 3", in the language of
    strobemodels.simulate.fbm. This means that the coefficient *D* will 
    depend on the value of *frame_interval*. While it is possible to 
    define a dispersion parameter independent of *frame_interval*, the
    advantage here is that *D* can be compared between FBMs that have
    different Hurst parameters.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the CDF
        hurst           :   float between 0.0 and 1.0, the Hurst parameter
        D               :   float, the diffusion coefficient  (um^2 s^(-2*hurst))
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, the frame interval in seconds.

    returns
    -------
        1D ndarray of shape (n_points,), the CDF 

    """
    if D_type == 3:
        D_mod = D / (hurst * np.power(frame_interval, 2 * hurst - 1))
    elif D_type == 4:
        D_mod = 2 * D / (np.power(frame_interval, 2 * hurst - 1))
    return 1.0 - np.exp(-(rt_tuples[:,0]**2) / (
        2 * D_mod * np.power(rt_tuples[:,1], 2*hurst) \
        + 4 * (loc_error**2)
    ))

def pdf_1state_fbm(rt_tuples, hurst, D, loc_error, frame_interval=0.01,
    D_type=4, **kwargs):
    """
    Probability density function for the 2D radial displacements of a single-state
    fractional Brownian motion.

    note
    ----
    Assumes the diffusion coefficient is "D_type 3", in the language of
    strobemodels.simulate.fbm. This means that the coefficient *D* will 
    depend on the value of *frame_interval*. While it is possible to 
    define a dispersion parameter independent of *frame_interval*, the
    advantage here is that *D* can be compared between FBMs that have
    different Hurst parameters.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF
        hurst           :   float between 0.0 and 1.0, the Hurst parameter
        D               :   float, the diffusion coefficient (um^2 s^(-2*hurst))
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, the frame interval in seconds.

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    if D_type == 3:
        D_mod = D / (hurst * np.power(frame_interval, 2 * hurst - 1))
    elif D_type == 4:
        D_mod = 2 * D / (np.power(frame_interval, 2 * hurst - 1))
    var2 = D_mod * np.power(rt_tuples[:,1], 2*hurst) + 2 * (loc_error**2)
    return (rt_tuples[:,0] / var2) * np.exp(-(rt_tuples[:,0]**2) / (2*var2))

def cdf_1state_levy_flight(rt_tuples, alpha, D, loc_error, dz=None, **kwargs):
    """
    Cumulative distribution function for the 2D radial displacements of a
    single-state Levy flight. 

    When *dz* is *None*, this model assumes that the focal plane is infinite
    in depth - every displacement is recorded. When dz is set to a finite number
    (say, 0.7 um), we assume that the Levy flight starts at a random position
    in the focal volume and only displacements that end inside the focal volume
    are counted for distribution of jump lengths.

    important note
    --------------
        The implementation of Levy flights in this model package requires a
        special projection that can only accommodate one binning scheme for the
        jumps: 0.0 to 20.0 um, with 0.004 um (4 nm) bins for a total of 1250
        bins per timepoint.

        *rt_tuples* must be supplied in this format.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the CDF
        alpha           :   float between 1.0 and 2.0, the stability parameter
                            for this Levy flight
        D               :   float, the dispersion parameter for the Levy flight
        loc_error       :   float, 1D localization error in um
        dz              :   float, the focal depth. If *None*, we assume that 
                            all jumps are recorded.

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    # Global real/frequency domain binning schemes
    global freq_support
    global real_support

    # Get the projection matrix for this focal depth
    proj = get_proj_matrix(dz)

    # Identify the unique frame intervals at which to evaluate the CDF 
    unique_times = np.unique(rt_tuples[:,1])

    result = np.zeros(len(unique_times) * 1250, dtype=np.float64)
    for t in unique_times:

        # The set of observations corresponding to this frame interval
        match = rt_tuples[:,1] == t

        # Evaluate the characteristic function (the next few lines are equivalent to 
        # the "radon_alt" method in the strobemodels.simulate.levy package)
        cf = 1.0j * freq_support * levy_flight_cf(freq_support, alpha, D, t, loc_error)

        # Transform to real space, and take all bins up to 20 um
        pdf = -real_support[:20001] * np.fft.irfft(cf, n=real_support.shape[0])[:20001]

        # Linearly interpolate the center of each bin
        pdf = 0.5 * (pdf[1:] + pdf[:-1])

        # Combine 1 nm bins into 4 nm bins
        pdf = pdf[0::4] + pdf[1::4] + pdf[2::4] + pdf[3::4]

        # Normalize on the full 0-20 um interval
        pdf /= pdf.sum()

        # Project into 2D
        pdf = (proj @ pdf)[:1250]

        # Accumulate to get the CDF
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        result[match] = cdf 

    return result 

def pdf_1state_levy_flight_alt(rt_tuples, alpha, D, loc_error, **kwargs):
    le2 = loc_error ** 2
    unique_times = np.unique(rt_tuples[:,1])
    result = np.zeros(rt_tuples.shape[0])
    for t in unique_times:
        match = rt_tuples[:,1] == t
        func = lambda k_: np.exp(-D * t * np.power(np.abs(k_), alpha) \
            - le2 * (k_ ** 2))
        result[match] = np.abs(HankelTrans2DLowPrec.transform(func,
            rt_tuples[match,0], inverse=True, ret_err=False))
        result[match] = result[match] * rt_tuples[match, 0]
        result[match] = result[match] / result[match].sum()
    return result 

def cdf_1state_levy_flight_alt(rt_tuples, alpha, D, loc_error, **kwargs):
    le2 = loc_error ** 2
    unique_times = np.unique(rt_tuples[:,1])
    result = np.zeros(rt_tuples.shape[0])
    for t in unique_times:
        match = rt_tuples[:,1] == t
        func = lambda k_: np.exp(-D * t * np.power(np.abs(k_), alpha) \
            - le2 * (k_ ** 2))
        result[match] = np.abs(HankelTrans2DLowPrec.transform(func,
            rt_tuples[match,0], inverse=True, ret_err=False))
        result[match] = result[match] * rt_tuples[match, 0]
        result[match] = np.cumsum(result[match] / result[match].sum())
    return result 


def pdf_1state_levy_flight(rt_tuples, alpha, D, loc_error, dz=None, **kwargs):
    """
    Probability density function for the 2D radial displacements of a
    single-state Levy flight. 

    When *dz* is *None*, this model assumes that the focal plane is infinite
    in depth - every displacement is recorded. When dz is set to a finite float
    (say, 0.7 um), we assume that the Levy flight starts at a random position
    in the focal volume and only displacements that end inside the focal volume
    are counted for distribution of jump lengths.
    
    important note
    --------------
        The implementation of Levy flights in this model package requires a
        special projection that can only accommodate one binning scheme for the
        jumps: 0.0 to 20.0 um, with 0.004 um (4 nm) bins for a total of 1250
        bins per timepoint.

        *rt_tuples* must be supplied in this format.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF
        alpha           :   float between 1.0 and 2.0, the stability parameter
                            for this Levy flight
        D               :   float, the dispersion parameter for the Levy flight
        loc_error       :   float, 1D localization error in um
        dz              :   float, the focal depth. If *None*, we assume that 
                            all jumps are recorded.

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    # Global real/frequency domain binning schemes
    global freq_support
    global real_support

    # Get the projection matrix for this focal depth
    proj = get_proj_matrix(dz)

    # Identify the unique frame intervals at which to evaluate the CDF 
    unique_times = np.unique(rt_tuples[:,1])

    result = np.zeros(len(unique_times) * 1250, dtype=np.float64)
    for t in unique_times:

        # The set of observations corresponding to this frame interval
        match = rt_tuples[:,1] == t

        # Evaluate the characteristic function (the next few lines are equivalent to 
        # the "radon_alt" method in the strobemodels.simulate.levy package)
        cf = 1.0j * freq_support * levy_flight_cf(freq_support, alpha, D, t, loc_error)

        # Transform to real space, and take all bins up to 20 um
        pdf = -real_support[:20001] * np.fft.irfft(cf, n=real_support.shape[0])[:20001]

        # Linearly interpolate the center of each bin
        pdf = 0.5 * (pdf[1:] + pdf[:-1])

        # Combine 1 nm bins into 4 nm bins
        pdf = pdf[0::4] + pdf[1::4] + pdf[2::4] + pdf[3::4]

        # Normalize on the full 0-20 um interval
        pdf /= pdf.sum()

        # Project into 2D
        pdf = (proj @ pdf)[:1250]

        # Normalize
        result[match] = pdf / pdf.sum()

    return result 

def cdf_2state_levy_flight(rt_tuples, alpha, f0, D0, D1, loc_error, dz=None,
    frame_interval=0.01, **kwargs):
    """
    Cumulative distribution function for the 2D radial displacements of 
    a two-state Levy flight in the XY plane of a microscope.

    This model assumes that the rate of state transitions is negligible
    and that the stability parameter *alpha* is the same for both states.

    When *dz* is *None*, this model assumes that the focal plane is infinite
    in depth - every displacement is recorded. When dz is set to a finite float
    (say, 0.7 um), we assume that the Levy flight starts at a random position
    in the focal volume and only displacements that end inside the focal volume
    are counted for distribution of jump lengths.

    important note
    --------------
        The implementation of Levy flights in this model package requires a
        special projection that can only accommodate one binning scheme for the
        jumps: 0.0 to 20.0 um, with 0.004 um (4 nm) bins for a total of 1250
        bins per timepoint.

        *rt_tuples* must be supplied in this format.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the CDF
        alpha           :   float between 1.0 and 2.0, the stability parameter
                            for this Levy flight
        f0              :   float, the fraction of molecules in the slower
                            diffusing state
        D0              :   float, dispersion for the slower diffusing state
        D1              :   float, dispersion for the faster diffusing state 
        loc_error       :   float, 1D localization error in um
        dz              :   float, the focal depth. If *None*, we assume that 
                            all jumps are recorded.

    returns
    -------
        1D ndarray of shape (n_points,), the CDF 

    """
    # Global real/frequency domain binning schemes
    global freq_support
    global real_support

    # Get the projection matrix for this focal depth
    proj = get_proj_matrix(dz)

    # Assign each observation to a frame interval
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Evaluate the defocalization function for each diffusing state
    if dz is None:
        f_rem_0 = np.ones(len(unique_frames))
        f_rem_1 = np.ones(len(unique_frames))
    else:
        f_rem_0 = defoc_prob_levy(D0, alpha, n_frames, frame_interval, dz)
        f_rem_1 = defoc_prob_levy(D1, alpha, n_frames, frame_interval, dz)

    # State occupations adjusted for defocalization
    f_adj_0 = f0 * f_rem_0 
    f_adj_1 = (1-f0) * f_rem_1 
    norm = f_adj_0 + f_adj_1 
    f_adj_0 = f_adj_0 / norm 

    # Synthesize the mixed state CDF for each frame interval
    result = np.zeros(rt_tuples.shape[0], dtype=np.float64)
    for f in unique_frames:

        # The set of observations corresponding to this frame interval
        match = frames == f

        # Evaluate the characteristic functions for each state
        cf0 = 1.0j * freq_support * levy_flight_cf(freq_support, alpha, D0, f*frame_interval, loc_error)
        cf1 = 1.0j * freq_support * levy_flight_cf(freq_support, alpha, D1, f*frame_interval, loc_error)

        # Transform to real space (taking bins up to 20.0 um)
        pdf0 = -real_support[:20001] * np.fft.irfft(cf0, n=real_support.shape[0])[:20001]
        pdf1 = -real_support[:20001] * np.fft.irfft(cf1, n=real_support.shape[0])[:20001]

        # Linearly interpolate the center of each bin
        pdf0 = 0.5 * (pdf0[1:] + pdf0[:-1])
        pdf1 = 0.5 * (pdf1[1:] + pdf1[:-1])

        # Aggregate the 1 nm bins into 4 nm bins
        pdf0 = pdf0[0::4] + pdf0[1::4] + pdf0[2::4] + pdf0[3::4]
        pdf1 = pdf1[0::4] + pdf1[1::4] + pdf1[2::4] + pdf1[3::4]

        # Normalize
        pdf0 /= pdf0.sum()
        pdf1 /= pdf1.sum()       

        # Project into 2D
        pdf0 = (proj @ pdf0)[:1250]
        pdf1 = (proj @ pdf1)[:1250]

        # Combine the CDFs and renormalize
        cdf_comb = f_adj_0[f-1] * np.cumsum((pdf0/pdf0.sum())) + (1 - f_adj_0[f-1]) * np.cumsum((pdf1/pdf1.sum()))
        result[match] = cdf_comb / cdf_comb[-1]

    return result

def pdf_2state_levy_flight(rt_tuples, alpha, f0, D0, D1, loc_error, dz=None,
    frame_interval=0.01, **kwargs):
    """
    Probability density function for the 2D radial displacements of 
    a two-state Levy flight in the XY plane of a microscope.

    This model assumes that the rate of state transitions is negligible
    and that the stability parameter *alpha* is the same for both states.

    When *dz* is *None*, this model assumes that the focal plane is infinite
    in depth - every displacement is recorded. When dz is set to a finite float
    (say, 0.7 um), we assume that the Levy flight starts at a random position
    in the focal volume and only displacements that end inside the focal volume
    are counted for distribution of jump lengths.

    important note
    --------------
        The implementation of Levy flights in this model package requires a
        special projection that can only accommodate one binning scheme for the
        jumps: 0.0 to 20.0 um, with 0.004 um (4 nm) bins for a total of 1250
        bins per timepoint.

        *rt_tuples* must be supplied in this format.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF
        alpha           :   float between 1.0 and 2.0, the stability parameter
                            for this Levy flight
        f0              :   float, the fraction of molecules in the slower
                            diffusing state
        D0              :   float, dispersion for the slower diffusing state
        D1              :   float, dispersion for the faster diffusing state 
        loc_error       :   float, 1D localization error in um
        dz              :   float, the focal depth. If *None*, we assume that 
                            all jumps are recorded.

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    # Global real/frequency domain binning schemes
    global freq_support
    global real_support

    # Get the projection matrix for this focal depth
    proj = get_proj_matrix(dz)

    # Assign each observation to a frame interval
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Evaluate the defocalization function for each diffusing state
    if dz is None:
        f_rem_0 = np.ones(len(unique_frames))
        f_rem_1 = np.ones(len(unique_frames))
    else:
        f_rem_0 = defoc_prob_levy(D0, alpha, n_frames, frame_interval, dz)
        f_rem_1 = defoc_prob_levy(D1, alpha, n_frames, frame_interval, dz)

    # State occupations adjusted for defocalization
    f_adj_0 = f0 * f_rem_0 
    f_adj_1 = (1-f0) * f_rem_1 
    norm = f_adj_0 + f_adj_1 
    f_adj_0 = f_adj_0 / norm 

    # Synthesize the mixed state PDF for each frame interval
    result = np.zeros(rt_tuples.shape[0], dtype=np.float64)
    for f in unique_frames:

        # The set of observations corresponding to this frame interval
        match = frames == f

        # Evaluate the characteristic functions for each state
        cf0 = 1.0j * freq_support * levy_flight_cf(freq_support, alpha, D0, f*frame_interval, loc_error)
        cf1 = 1.0j * freq_support * levy_flight_cf(freq_support, alpha, D1, f*frame_interval, loc_error)

        # Transform to real space (taking bins up to 20.0 um)
        pdf0 = -real_support[:20001] * np.fft.irfft(cf0, n=real_support.shape[0])[:20001]
        pdf1 = -real_support[:20001] * np.fft.irfft(cf1, n=real_support.shape[0])[:20001]

        # Linearly interpolate the center of each bin
        pdf0 = 0.5 * (pdf0[1:] + pdf0[:-1])
        pdf1 = 0.5 * (pdf1[1:] + pdf1[:-1])

        # Aggregate the 1 nm bins into 4 nm bins
        pdf0 = pdf0[0::4] + pdf0[1::4] + pdf0[2::4] + pdf0[3::4]
        pdf1 = pdf1[0::4] + pdf1[1::4] + pdf1[2::4] + pdf1[3::4]

        # Normalize
        pdf0 /= pdf0.sum()
        pdf1 /= pdf1.sum()       

        # Project into 2D
        pdf0 = (proj @ pdf0)[:1250]
        pdf1 = (proj @ pdf1)[:1250]

        # Combine the PDFs and renormalize
        pdf_comb = f_adj_0[f-1] * (pdf0/pdf0.sum()) + (1 - f_adj_0[f-1]) * (pdf1/pdf1.sum())
        result[match] = pdf_comb / pdf_comb.sum()

    return result

def cdf_1state_levy_flight_hankel(rt_tuples, alpha, D, loc_error, **kwargs):
    """
    Distribution function for the 2D radial displacements of a single-state
    Levy flight. This model assumes that the focal depth is infinite - that
    we can observe the Levy flight regardless of the z-position.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF 
        alpha           :   float between 1.0 and 2.0, the stability parameter
                            for this Levy flight
        D               :   float, dispersion parameter
        loc_error       :   float, localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    # Identify the unique frame intervals at which to evaluate the CDF
    unique_times = np.unique(rt_tuples[:,1])

    # Shift all of the bins shown by half a bin size. We'll approximate
    # the PMF for that bin as the PDF in the center of the bin, then 
    # accumulate them to get the CDF.
    bin_size = rt_tuples[1,0] - rt_tuples[0,0]
    r_centers = rt_tuples[:,0] - bin_size * 0.5

    # Evaluate the PDF for each timepoint
    result = np.zeros(rt_tuples.shape[0], dtype=np.float64)
    for t in unique_times:

        # The set of data points that match this timepoint
        match = rt_tuples[:, 1] == t
        r = r_centers[match]

        # Get the 2D radial density
        cf_func = lambda k: levy_flight_cf(k, alpha, D, t, loc_error)
        pdf = pdf_from_cf_rad(cf_func, r, d=2)

        # Marginalize on the angular component (2D)
        pdf = radnorm(r, pdf, d=2)

        # Accumulate to get the CDF
        result[match] = np.cumsum(pdf)

    global global_iter_idx 
    global_iter_idx += 1
    print("CDF: iter_idx %d" % global_iter_idx)

    return result 

def pdf_1state_levy_flight_hankel(rt_tuples, alpha, D, loc_error, **kwargs):
    """
    Probability density function for the 2D radial displacements of a single-state
    Levy flight. This model assumes that the focal depth is infinite - that we can 
    observe the Levy flight regardless of z-position.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF 
        alpha           :   float between 1.0 and 2.0, the stability parameter
                            for this Levy flight
        D               :   float, dispersion parameter
        loc_error       :   float, localization error in um

    returns
    -------
        1D ndarray of shape (n_points,), the PDF 

    """
    # Identify the unique frame intervals at which to evaluate the PDF
    unique_times = np.unique(rt_tuples[:,1])

    # Evaluate the PDF for each timepoint
    result = np.zeros(rt_tuples.shape[0], dtype=np.float64)
    for t in unique_times:

        # The set of data points that match this timepoint
        match = rt_tuples[:, 1] == t
        r = rt_tuples[match, 0]

        # Get the 2D radial density
        cf_func = lambda k: levy_flight_cf(k, alpha, D, t, loc_error)
        pdf = pdf_from_cf_rad(cf_func, r, d=2)

        # Marginalize on the angular component (2D)
        pdf = radnorm(r, pdf, d=2)

        # Save this part of the curve
        result[match] = pdf 

    # Divide the result by the bin size, to approximate the PDF 
    # rather than the PMF 
    bin_size = rt_tuples[1,0] - rt_tuples[0,0]
    result /= bin_size 

    return result 

def cdf_2state_fbm_uncorr(rt_tuples, hurst, f0, D0, D1, loc_error, frame_interval=0.01,
    **kwargs):
    """
    Distribution function for the 2D radial displacements of a two-state
    fractional Brownian motion. Both states are assumed to have the same 
    Hurst parameter.

    See the Note in *cdf_1state_fbm* and *pdf_1state_fbm*, which also 
    applies here.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the CDF
        hurst           :   float between 0.0 and 1.0, the Hurst parameter
        f0              :   float, fraction of molecules in state 0
        D0              :   float, diffusion coefficient for state 0 (um^2 s^(-2*hurst))
        D1              :   float, diffusion coefficient for state 1 (um^2 s^(-2*hurst))
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, frame interval in seconds

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    D0_mod = D0 / (hurst * np.power(frame_interval, 2 * hurst - 1))
    D1_mod = D1 / (hurst * np.power(frame_interval, 2 * hurst - 1))

    r2 = rt_tuples[:,0]**2
    t_h2 = np.power(rt_tuples[:,1], 2*hurst)

    var2_0 = D0_mod * t_h2 + 2 * (loc_error**2)
    var2_1 = D1_mod * t_h2 + 2 * (loc_error**2)

    cdf_0 = 1.0 - np.exp(-r2 / (2 * var2_0))
    cdf_1 = 1.0 - np.exp(-r2 / (2 * var2_1))

    return f0 * cdf_0 + (1 - f0) * cdf_1

def pdf_2state_fbm_uncorr(rt_tuples, hurst, f0, D0, D1, loc_error, frame_interval=0.01,
    **kwargs):
    """
    Probability density function for the 2D radial displacements of a two-state
    fractional Brownian motion. Both states are assumed to have the same 
    Hurst parameter.

    See the Note in *cdf_1state_fbm* and *pdf_1state_fbm*, which also 
    applies here.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF 
        hurst           :   float between 0.0 and 1.0, the Hurst parameter
        f0              :   float, fraction of molecules in state 0
        D0              :   float, diffusion coefficient for state 0 (um^2 s^(-2*hurst))
        D1              :   float, diffusion coefficient for state 1 (um^2 s^(-2*hurst))
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, frame interval in seconds

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    D0_mod = D0 / (hurst * np.power(frame_interval, 2 * hurst - 1))
    D1_mod = D1 / (hurst * np.power(frame_interval, 2 * hurst - 1))

    r2 = rt_tuples[:,0]**2
    t_h2 = np.power(rt_tuples[:,1], 2*hurst)

    var2_0 = D0_mod * t_h2 + 2 * (loc_error**2)
    var2_1 = D1_mod * t_h2 + 2 * (loc_error**2)

    pdf_0 = (rt_tuples[:,0] / var2_0) * np.exp(-r2 / (2 * var2_0))
    pdf_1 = (rt_tuples[:,0] / var2_1) * np.exp(-r2 / (2 * var2_1))

    return f0 * pdf_0 + (1 - f0) * pdf_1

def cdf_2state_fbm_zcorr(rt_tuples, hurst, f0, D0, D1, loc_error, frame_interval=0.01,
    dz=0.7, **kwargs):
    """
    Distribution function for the 2D radial displacements of a two-state
    fractional Brownian motion. Both states are assumed to have the same 
    Hurst parameter. The molecules are assumed to start with uniform probability
    in a focal volume of depth *dz*, and are lost at the first frame interval
    that they lie outside of the focal volume.

    See the Note in *cdf_1state_fbm* and *pdf_1state_fbm*, which also 
    applies here.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the CDF
        hurst           :   float between 0.0 and 1.0, the Hurst parameter
        f0              :   float, fraction of molecules in state 0
        D0              :   float, diffusion coefficient for state 0 (um^2 s^(-2*hurst))
        D1              :   float, diffusion coefficient for state 1 (um^2 s^(-2*hurst))
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, frame interval in seconds
        dz              :   float, depth of the focal volume in um

    returns
    -------
        1D ndarray of shape (n_points,), the CDF

    """
    D0_mod = D0 / (hurst * np.power(frame_interval, 2 * hurst - 1))
    D1_mod = D1 / (hurst * np.power(frame_interval, 2 * hurst - 1))

    r2 = rt_tuples[:,0]**2
    t_h2 = np.power(rt_tuples[:,1], 2*hurst)

    var2_0 = D0_mod * t_h2 + 2 * (loc_error**2)
    var2_1 = D1_mod * t_h2 + 2 * (loc_error**2)

    cdf_0 = 1.0 - np.exp(-r2 / (2 * var2_0))
    cdf_1 = 1.0 - np.exp(-r2 / (2 * var2_1))
    result = np.zeros(cdf_0.shape, dtype=np.float64)

    # Get the total number of frames in the movie
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Get the fraction of defocalized particles for each state
    frac_remain_state_0 = defoc_prob_fbm(D0, hurst, n_frames, frame_interval, dz)
    frac_remain_state_1 = defoc_prob_fbm(D1, hurst, n_frames, frame_interval, dz)

    # For each frame, adjust the state occupations to account for defocalization
    for frame in unique_frames:

        # The set of observations corresponding to this frame interval
        in_frame = frames == frame 

        # The adjusted state fractions
        f0_adj = f0 * frac_remain_state_0[frame-1]
        f1_adj = (1-f0) * frac_remain_state_1[frame-1]
        norm = f0_adj + f1_adj 
        f0_adj = f0_adj / norm 
        f1_adj = f1_adj / norm 

        # Make the mixed CDF
        result[in_frame] = f0_adj * cdf_0[in_frame] + f1_adj * cdf_1[in_frame]

    return result

def pdf_2state_fbm_zcorr(rt_tuples, hurst, f0, D0, D1, loc_error, frame_interval=0.01,
    dz=0.7, **kwargs):
    """
    Distribution function for the 2D radial displacements of a two-state
    fractional Brownian motion. Both states are assumed to have the same 
    Hurst parameter. The molecules are assumed to start with uniform probability
    in a focal volume of depth *dz*, and are "lost" at the first frame interval
    that they lie outside of the focal volume.

    See the Note in *cdf_1state_fbm* and *pdf_1state_fbm*, which also 
    applies here.

    args
    ----
        rt_tuples       :   2D ndarray, shape (n_points, 2), the independent
                            tuples (r, dt) at which to evaluate the PDF
        hurst           :   float between 0.0 and 1.0, the Hurst parameter
        f0              :   float, fraction of molecules in state 0
        D0              :   float, diffusion coefficient for state 0 (um^2 s^(-2*hurst))
        D1              :   float, diffusion coefficient for state 1 (um^2 s^(-2*hurst))
        loc_error       :   float, 1D localization error in um
        frame_interval  :   float, frame interval in seconds
        dz              :   float, depth of the focal volume in um

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """   
    D0_mod = D0 / (hurst * np.power(frame_interval, 2 * hurst - 1))
    D1_mod = D1 / (hurst * np.power(frame_interval, 2 * hurst - 1))

    r2 = rt_tuples[:,0]**2
    t_h2 = np.power(rt_tuples[:,1], 2*hurst)

    var2_0 = D0_mod * t_h2 + 2 * (loc_error**2)
    var2_1 = D1_mod * t_h2 + 2 * (loc_error**2)
    
    pdf_0 = (rt_tuples[:,0] / var2_0) * np.exp(-r2 / (2 * var2_0))
    pdf_1 = (rt_tuples[:,0] / var2_1) * np.exp(-r2 / (2 * var2_1))

    result = np.zeros(pdf_0.shape, dtype=np.float64)

    # Get the total number of frames in the movie
    frames = (rt_tuples[:,1] / frame_interval).round(0).astype(np.int64)
    n_frames = frames.max()
    unique_frames = np.arange(1, n_frames+1)

    # Get the fraction of defocalized particles for each state
    frac_remain_state_0 = defoc_prob_fbm(D0, hurst, n_frames, frame_interval, dz)
    frac_remain_state_1 = defoc_prob_fbm(D1, hurst, n_frames, frame_interval, dz)

    # For each frame, adjust the state occupations to account for defocalization
    for frame in unique_frames:

        # The set of observations corresponding to this frame interval
        in_frame = frames == frame 

        # The adjusted state fractions
        f0_adj = f0 * frac_remain_state_0[frame-1]
        f1_adj = (1-f0) * frac_remain_state_1[frame-1]
        norm = f0_adj + f1_adj 
        f0_adj = f0_adj / norm 
        f1_adj = f1_adj / norm 

        # Make the mixed CDF
        result[in_frame] = f0_adj * pdf_0[in_frame] + f1_adj * pdf_1[in_frame]

    return result   

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

def cdf_3state_brownian_uncorr(rt_tuples, f0, f1, D0, D1, D2, loc_error, **kwargs):
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

    return f0*pdf_0 + f1*pdf_1 + (1-f0-f1)*pdf_2

def cdf_2state_brownian_zcorr(rt_tuples, f0, D0, D1, loc_error,
    dz=0.7, frame_interval=0.01, n_gaps=0, **kwargs):
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
        n_gaps          :   int, the number of gaps allowed during tracking

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
    f_rem = defoc_prob_brownian(D1, n_frames, frame_interval, dz, n_gaps=n_gaps)

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
    dz=0.7, frame_interval=0.01, n_gaps=0, **kwargs):
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
        n_gaps          :   int, the number of gaps allowed during tracking

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
    f_rem = defoc_prob_brownian(D1, n_frames, frame_interval, dz, n_gaps=n_gaps)

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
    dz=0.7, frame_interval=0.01, n_gaps=0, **kwargs):
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
        n_gaps          :   int, the number of gap frames allowed during
                            tracking

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
    f_rem_1 = defoc_prob_brownian(D1, n_frames, frame_interval, dz, n_gaps=n_gaps)
    f_rem_2 = defoc_prob_brownian(D2, n_frames, frame_interval, dz, n_gaps=n_gaps)

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
    dz=0.7, frame_interval=0.01, n_gaps=0, **kwargs):
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
        n_gaps          :   int, the number of gap frames allowed during
                            tracking

    returns
    -------
        1D ndarray of shape (n_points,), the PDF

    """
    le2 = loc_error ** 2
    var_0 = 2 * (D0 * rt_tuples[:,1] + le2)
    var_1 = 2 * (D1 * rt_tuples[:,1] + le2)
    var_2 = 2 * (D2 * rt_tuples[:,1] + le2)
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
    f_rem_1 = defoc_prob_brownian(D1, n_frames, frame_interval, dz, n_gaps=n_gaps)
    f_rem_2 = defoc_prob_brownian(D2, n_frames, frame_interval, dz, n_gaps=n_gaps)

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

def pdf_specdiffuse_model(rt_tuples, diffusivities, D_occs, frame_interval=0.01, dz=0.7,
    loc_error=0.0):
    """
    Evaluate the probability density function for a mixed Brownian model with
    any number of non-interconverting states, accounting for defocalization.

    args
    ----
        rt_tuples           :   2D ndarray of shape (N, 2), the set of (radial displacement
                                in um, delay in seconds) tuples at which to evaluate the PDF
        diffusivities       :   1D ndarray of shape (M,), the set of diffusivities
        D_occs              :   1D ndarray of shape (M,), the set of state occupations
        frame_interval      :   float, time between frames in seconds
        dz                  :   float, the thickness of the observation slice in um
        loc_error           :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (N,), the PDF evaluated at each point of the support

    """
    N = rt_tuples.shape[0]
    M = len(diffuvisities)

    # Get the unique time intervals present in this set of data
    unique_dt = np.unique(rt_tuples[:,1])
    n_frames = len(unique_dt)

    # Evaluate the defocalization probabilities for each diffusive state
    F_remain = np.zeros((M, n_frames), dtype=np.float64)
    for i, D in enumerate(diffusivities):
        F_remain[i,:] = defoc_prob_brownian(D, n_frames, frame_interval,
            dz, n_gaps=0)

    # Multiply by the state occupation estimates
    F_remain = (F_remain.T * D_occs).T 

    # Normalize on each frame interval
    F_remain = F_remain / F_remain.sum(axis=0)

    # Evaluate the PDFs for each state
    pdfs = np.zeros(N, dtype=np.float64)
    r2 = rt_tuples[:,0] ** 2
    for i, D in enumerate(diffusivities):

        pdf_D = np.zeros(N, dtype=np.float64)

        for j, dt in enumerate(unique_dt):

            # Evaluate the naive PDF for this diffusivity
            match = rt_tuples[:,1] == dt 
            sig2 = 2 * (D * dt + loc_error**2)
            pdf_D[match] = rt_tuples[match, 0] * np.exp(-r2[match] / (2 * sig2)) / sig2 
            pdf_D[match] = pdf_D[match] * F_remain[i,j]

        pdfs += pdf_D 

    return pdfs 

def cdf_specdiffuse_model(rt_tuples, diffusivities, D_occs, frame_interval=0.01, dz=0.7,
    loc_error=0.0):
    """
    Evaluate the cumulative distribution function for a mixed Brownian model
    with any number of non-interconverting states, accounting for defocalization.

    args
    ----
        rt_tuples           :   2D ndarray of shape (N, 2), the set of (radial displacement
                                in um, delay in seconds) tuples at which to evaluate the PDF
        diffusivities       :   1D ndarray of shape (M,), the set of diffusivities
        D_occs              :   1D ndarray of shape (M,), the set of state occupations
        frame_interval      :   float, time between frames in seconds
        dz                  :   float, the thickness of the observation slice in um
        loc_error           :   float, 1D localization error in um

    returns
    -------
        1D ndarray of shape (N,), the PDF evaluated at each point of the support

    """
    N = rt_tuples.shape[0]
    M = len(diffusivities)

    # Get the unique time intervals present in this set of data
    unique_dt = np.unique(rt_tuples[:,1])
    n_frames = len(unique_dt)

    # Evaluate the defocalization probabilities for each diffusive state
    F_remain = np.zeros((M, n_frames), dtype=np.float64)
    for i, D in enumerate(diffusivities):
        F_remain[i,:] = defoc_prob_brownian(D, n_frames, frame_interval,
            dz, n_gaps=0)

    # Multiply by the state occupation estimates
    F_remain = (F_remain.T * D_occs).T 

    # Normalize on each frame interval
    F_remain = F_remain / F_remain.sum(axis=0)

    # Evaluate the CDFs for each state
    cdfs = np.zeros(N, dtype=np.float64)
    r2 = rt_tuples[:,0] ** 2
    for i, D in enumerate(diffusivities):

        cdf_D = np.zeros(N, dtype=np.float64)

        for j, dt in enumerate(unique_dt):
            sig2 = 2 * (D * dt + loc_error**2)
            match = rt_tuples[:,1] == dt 
            cdf_D[match] = 1.0 - np.exp(-r2[match] / (2 * sig2))
            cdf_D[match] = cdf_D[match] * F_remain[i,j]

        cdfs += cdf_D 

    return cdfs 

######################
## AVAILABLE MODELS ##
######################

# Cumulative distribution functions
CDF_MODELS = {
    "one_state_brownian": cdf_1state_brownian,
    "one_state_fbm": cdf_1state_fbm,
    "one_state_levy_flight": cdf_1state_levy_flight,
    "one_state_levy_flight_alt": cdf_1state_levy_flight_alt,
    "one_state_levy_hankel": cdf_1state_levy_flight_hankel,
    "two_state_brownian": cdf_2state_brownian_uncorr,
    "two_state_brownian_zcorr": cdf_2state_brownian_zcorr,
    "two_state_fbm": cdf_2state_fbm_uncorr,
    "two_state_fbm_zcorr": cdf_2state_fbm_zcorr,
    "two_state_levy_flight": cdf_2state_levy_flight,
    "three_state_brownian": cdf_3state_brownian_uncorr,
    "three_state_brownian_zcorr": cdf_3state_brownian_zcorr,
}

# Probability density functions
PDF_MODELS = {
    "one_state_brownian": pdf_1state_brownian,
    "one_state_fbm": pdf_1state_fbm,
    "one_state_levy_flight": pdf_1state_levy_flight,
    "one_state_levy_flight_alt": pdf_1state_levy_flight_alt,
    "one_state_levy_hankel": pdf_1state_levy_flight_hankel,
    "two_state_brownian": pdf_2state_brownian_uncorr,
    "two_state_brownian_zcorr": pdf_2state_brownian_zcorr,
    "two_state_fbm": pdf_2state_fbm_uncorr,
    "two_state_fbm_zcorr": pdf_2state_fbm_zcorr,
    "two_state_levy_flight": pdf_2state_levy_flight,
    "three_state_brownian": pdf_3state_brownian_uncorr,
    "three_state_brownian_zcorr": pdf_3state_brownian_zcorr,   
}

# Identity of each fit parameter for each model 
MODEL_PARS = {
    "one_state_brownian": ["D", "loc_error"],
    "one_state_fbm": ["hurst", "D", "loc_error"],
    "one_state_levy_flight": ["alpha", "D", "loc_error"],
    "one_state_levy_flight_alt": ["alpha", "D", "loc_error"],
    "one_state_levy_hankel": ["alpha", "scale", "loc_error"],
    "two_state_brownian": ["f0", "D0", "D1", "loc_error"],
    "two_state_brownian_zcorr": ["f0", "D0", "D1", "loc_error"],
    "two_state_fbm": ["hurst", "f0", "D0", "D1", "loc_error"],
    "two_state_fbm_zcorr": ["hurst", "f0", "D0", "D1", "loc_error"],
    "two_state_levy_flight": ["alpha", "f0", "D0", "D1", "loc_error"],
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
    "one_state_fbm": (
        np.array([1.0e-8, 1.0e-8, 0.0]),
        np.array([1.0, np.inf, 0.1])
    ),
    "one_state_levy_flight": (
        np.array([1.0, 1.0e-8, 0.0]),
        np.array([2.0, np.inf, 0.1])
    ),
    "one_state_levy_flight_alt": (
        np.array([1.0, 1.0e-8, 0.0]),
        np.array([2.0, np.inf, 0.1])
    ),
    "one_state_levy_hankel": (
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, np.inf, 0.1])
    ),
    "two_state_brownian": (
        np.array([0.0, 1.0e-8, 0.5, 0.0]),
        np.array([1.0, 0.005, np.inf, 0.1])
    ),
    "two_state_brownian_zcorr": (
        np.array([0.0, 1.0e-8, 0.5, 0.0]),
        np.array([1.0, 0.005, np.inf, 0.1])
    ),
    "two_state_fbm": (
        np.array([0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([1.0, 1.0, 0.05, np.inf, 0.1])
    ),
    "two_state_fbm_zcorr": (
        np.array([0.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([1.0, 1.0, 0.05, np.inf, 0.1])
    ),
    "two_state_levy_flight": (
        np.array([1.0, 0.0, 0.0, 0.1, 0.0]),
        np.array([2.0, 1.0, 0.1, np.inf, 0.1])
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
    "one_state_fbm": np.array([0.5, 1.0, 0.035]),
    "one_state_levy_flight": np.array([2.0, 1.0, 0.035]),
    "one_state_levy_flight_alt": np.array([2.0, 1.0, 0.035]),
    "one_state_levy_hankel": np.array([2.0, 1.0, 0.035]),
    "two_state_brownian": np.array([0.3, 0.01, 1.0, 0.035]),
    "two_state_brownian_zcorr": np.array([0.3, 0.001, 1.0, 0.035]),
    "two_state_fbm": np.array([0.5, 0.3, 0.001, 1.0, 0.035]),
    "two_state_fbm_zcorr": np.array([0.5, 0.3, 0.001, 1.0, 0.035]),
    "two_state_levy_flight": np.array([2.0, 0.3, 0.01, 1.0, 0.035]),
    "three_state_brownian": np.array([0.33, 0.33, 0.001, 0.5, 2.0, 0.035]),
    "three_state_brownian_zcorr": np.array([0.33, 0.33, 0.001, 0.5, 2.0, 0.035]),
}

