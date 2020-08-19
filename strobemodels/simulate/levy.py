#!/usr/bin/env python
"""
levy.py -- simulate 3D Levy flights

"""
import os
import numpy as np 
from scipy.stats import levy_stable 
from strobemodels.simulate.simutils import (
    sample_sphere
)
from strobemodels.utils import (
    HankelTrans3D,
    spline,
    pdf_from_cf_rad,
    radnorm 
)

class LevyFlight3D(object):
    """
    Simulate a 3D Levy flight using an approximate inverse CDF approach.

    description
    -----------
        LevyFlight3D simulates a Markovian diffusion process whose 3D
        radial displacements are distributed according to a Levy stable
        random variable, and whose angular displacements are isotropically
        distributed on the surface of the unit sphere.

    init
    ----
        alpha       :   float between 0.0 and 2.0, the stability parameter
                        for the Levy flight
        scale       :   float, the dispersion parameter. For alpha = 2.0,
                        this is equivalent to the square root of the 
                        diffusion coefficient.
        dt          :   float, the interval between frames in seconds
        interpolation_range :   (float, float, float), arguments to 
                        np.linspace generating the support to use by default
                        for CDF sampling. The default range (0.0, 10.0, 10001)
                        works well up to a diffusion coefficient of about 80.0
                        (scale = sqrt(80)) for Brownian motion.

    example simulation
    ------------------

        # Generate the simulator object (here, Brownian motion)
        L = LevyFlight3D(2.0, 1.0)

        # Generate 100000 trajectories of length 10
        tracks = L(100000, 10)

    methods
    -------
        cf          :   evaluate the characteristic function
        cdf_rad     :   evaluate the CDF for the 3D radial displacements
        pdf_rad     :   evaluate the PDF for the 3D radial displacements
        cdf_rad_dev :   evaluate the first derivative of the CDF for the 
                        3D radial displacements
        inverse_cdf :   evaluate the inverse CDF function for arguments
                        between 0.0 and 1.0

    """
    def __init__(self, alpha, scale, dt=0.01, track_len=10, interpolation_range=(0.0, 10.0, 10001)):
        self.alpha = alpha 
        self.scale = scale
        self.dt = dt 
        self.track_len = track_len 
        self.interpolation_range = interpolation_range 

    def __call__(self, N, track_len=None, n_iter=10):
        """
        Simulate *N* instances of this Levy flight.

        args
        ----
            N           :   int, the number of Levy flights to simulate
            track_len   :   int, the length of each trajectory to 
                            simulate
            n_iter      :   int, the number of iterations of Newton's
                            method to use for inverse CDF sampling

        returns
        -------
            3D ndarray, shape (N, traj_len, 3), the ZYX positions
                of each Levy flight at each timepoint

        """
        if track_len is None:
            track_len = self.track_len 

        # Generate the radial displacements
        p = np.random.random(size=(N * (track_len-1)))
        r = self.inverse_cdf(p, n_iter=n_iter).reshape((N, track_len-1))

        # Generate the angular displacements
        v = sample_sphere((N, track_len), d=3)

        # The first position for each trajectory is always zero
        v[:,0,:] = 0

        # Multiply radial and angular displacements
        for t in range(1, track_len):
            v[:,t,:] = (v[:,t,:].T * r[:,t-1]).T 

        # Accumulate the displacements to generate trajectories
        v = np.cumsum(v, axis=1)
        return v 

    def cf(self, k):
        """
        Evaluate the characteristic function for the radial displacements
        of the Levy flight.

        args
        ----
            k       :   real-valued 1D ndarray, evaluation points

        returns
        -------
            1D ndarray, dtype float64, the characteristic function

        """
        return np.exp(-self.scale * self.dt * np.power(np.abs(k), self.alpha))

    def cdf_rad(self, r):
        """
        Approximate the CDF for the 3D radial displacements.

        args
        ----
            r       :   1D ndarray, points at which to evaluate CDF

        returns
        -------
            1D ndarray, the CDF 

        """
        if not hasattr(self, "_cdf_rad"):
            self.gen_cdf_rad()
        return self._cdf_rad(r)

    def pdf_rad(self, r):
        """
        Approximate the PDF for the 3D radial displacements.

        args
        ----
            r       :   1D ndarray, points at which to evaluate PDF

        returns
        -------
            1D ndarray, the PDF 

        """       
        if not hasattr(self, "_pdf_rad"):
            self.gen_pdf_rad()
        return self._pdf_rad(r)

    def cdf_rad_dev(self, r):
        """
        Approximate the first derivative for the CDF for the 3D radial
        displacements.

        args
        ----
            r       :   1D ndarray, evaluation points

        returns
        -------
            1D ndarray, the first derivative of the CDF

        """
        if not hasattr(self, "_cdf_rad_dev"):
            self.gen_cdf_rad_dev()
        return self._cdf_rad_dev(r)

    def gen_pdf_rad(self, r=None):
        """
        Generate a spline approximation to the PDF of the radial displacements
        for the Levy flight.

        args
        ----
            r       :   1D ndarray, real space support

        returns
        -------
            scipy.interpolate.InterpolatedUnivariateSpline, the spline
                approximation to the PDF 

        """
        if r is None:
            r = np.linspace(*self.interpolation_range)
        pdf = pdf_from_cf_rad(self.cf, r)
        pdf = radnorm(r, pdf, d=3)
        self._pdf_rad = spline(r, pdf/pdf.sum())
        return self._pdf_rad 

    def gen_cdf_rad(self, r=None):
        """
        Generate a spline approximation to the CDF of the radial displacements
        for the Levy flight.

        args
        ----
            r       :   1D ndarray, real space support

        returns
        -------
            scipy.interpolate.InterpolatedUnivariateSpline, the spline
                approximation to the CDF 

        """
        if r is None:
            r = np.linspace(*self.interpolation_range)
        pdf = pdf_from_cf_rad(self.cf, r)
        pdf = radnorm(r, pdf, d=3)
        cdf = np.cumsum(pdf)
        self._cdf_rad = spline(r, cdf/cdf[-1])
        return self._cdf_rad 

    def gen_cdf_rad_dev(self, r=None):
        """
        Generate a spline approximation to the first derivative of the CDF for
        the radial displacements of this Levy flight.

        args
        ----
            r       :   1D ndarray, real space support

        returns
        -------
            scipy.interpolate.InterpolatedUnivariateSpline

        """
        if not hasattr(self, "_cdf_rad"):
            self.gen_cdf_rad(r=r)
        self._cdf_rad_dev = self._cdf_rad.derivative(1)
        return self._cdf_rad_dev 

    def inverse_cdf(self, p, n_iter=20):
        """
        Return the inverse CDF of the 3D radial displacements of the Levy
        flight.

        args
        ----
            p       :   1D ndarray, floats between 0.0 and 1.0
            n_iter  :   int, the number of iterations of Newton's
                        algorithm to run

        returns
        -------
            float, the inverse CDF at the point

        """
        # Starting point - closest to the sampled value
        r = np.linspace(*self.interpolation_range)

        # Guarantee monotonicity, necessary for numpy.digitize
        pdf = self.pdf_rad(r)
        pdf[pdf<0] = 0
        cdf = np.cumsum(pdf)

        # Take the closest bin to each point as the initial guess
        r0 = r[np.digitize(p, cdf)]

        # Do a few rounds of Newton to refine the guesses
        iter_idx = 0
        for i in range(n_iter):
            r0 += -0.5 * (self.cdf_rad(r0) - p) / self.cdf_rad_dev(r0)

        return r0 



