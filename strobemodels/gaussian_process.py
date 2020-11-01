#!/usr/bin/env python
"""
gaussian_process.py -- Gaussian process representations of 
regular and fractional Brownian motion for inference

"""
import numpy as np 
from scipy.special import ndtri 

class GaussianProcess(object):
    """
    A Gaussian process. 

    init
    ----
        cov_func        :   function that returns the 
                            covariance of single time points
                            or vectors of time points
        mean_func       :   function that returns the mean 
                            value of single time points or 
                            vectors of time points. 

    note
    ----
        cov_func should be defined so that, if passed 1.0 and 2.0,
            it returns a float corresponding to the covariance 
            evaluated at 1.0 and 2.0

            If passed np.array([1, 2]) and np.array([3, 4]), it 
            should instead return the covariance matrix

                [[Cov(1, 3), Cov(1, 4)],
                 [Cov(2, 3), Cov(2, 4)]]

    methods
    -------


    """
    def __init__(self, cov_func, mean_func=None):
        if mean_func is None:
            mean_func = lambda x: 0 if not isinstance(x, np.ndarray) \
                else np.zeros(x.shape)
        self.cov_func = cov_func 
        self.mean_func = mean_func 
        self.is_conditioned = False 

    def construct_covariance(self, times):
        """
        Given a set of timepoints, construct the covariance 
        matrix corresponding to these times. 

        """
        return self.cov_func(times, times)

    def condition(self, times, values, errors=None):
        """
        args
        ----
            times       :   1D ndarray of shape (n,) 
            values      :   1D ndarray of shape (n,)
            errors      :   1D ndarray of shape (n,) or 
                            float, if the same for all 
                            observations. Specified as 
                            Gaussian standard deviations.

        returns
        -------

        """
        # Sanitize 
        assert times.shape == values.shape
        if not errors is None:
            if isinstance(errors, np.ndarray):
                assert errors.shape == times.shape
            else:
                errors = errors * np.ones(times.shape)
        else:
            errors = np.zeros(times.shape)

        # If this Gaussian process has already been conditioned
        # on some points, add the new points
        if self.is_conditioned:

            # Replace overlapping points
            overlap = np.array([True for t in self.cond_times if t in times])
            if overlap.sum() > 0:
                self.cond_times = self.cond_times[~overlap]
                self.cond_values = self.cond_values[~overlap]
                self.cond_errors = self.cond_errors[~overlap]

            # Concatenate old observations with new
            self.cond_times = np.concatenate((self.cond_times, times))
            self.cond_values = np.concatenate((self.cond_values, values))
            self.cond_errors = np.concatenate((self.cond_errors, errors**2))

        else:
            self.cond_times = times 
            self.cond_values = values 
            self.cond_errors = errors**2
        
        self.E11 = self.cov_func(self.cond_times, self.cond_times) + np.diag(self.cond_errors)
        self.E11_inv = np.linalg.inv(self.E11)
        self.mean_1 = self.mean_func(self.cond_times)
        self.cond_values_shift = self.cond_values - self.mean_1 
        self.is_conditioned = True 

    def pdf(self, times, values):
        """
        Evaluate the joint probability density function of 
        the process at some set of points.

        args
        ----
            times       :   1D ndarray of shape (m,)
            values      :   1D ndarray of shape (m,)

        returns
        -------
            1D ndarray, dtype float64, shape (m,)

        """
        # Covariance matrix
        E = self.cov_func(times, times)
        E_inv = np.linalg.inv(E)

        # Mean vector
        mean = self.mean_func(times)

        # Evaluate the multivariate normal PDF 
        values_shift = values - mean 
        return np.exp(-0.5 * (values_shift * (E_inv @ values_shift)).sum()) / \
            (np.power(2*np.pi, 0.5) * np.power(np.linalg.det(E), 0.5))

    def conditional_mean(self, times):
        """
        Return mean value of the conditioned Gaussian process at 
        a new set of time points.

        args
        ----
            times       :   1D ndarray of shape (m,)

        returns
        -------
            1D ndarray, dtype float64, shape (m,)

        """
        if not self.is_conditioned:
            return self.mean_func(times)
        else:
            E21 = self.cov_func(times, self.cond_times)
            return self.mean_func(times) + E21 @ \
                self.E11_inv.dot(self.cond_values_shift)

    def conditional_inverse_cdf(self, times, p):
        """
        args
        ----
            times       :   1D ndarray of shape (m,)
            p           :   float between 0.0 and 1.0

        returns
        -------
            1D ndarray of shape (m,), the inverse CDF
                for the conditional density of each point

        """
        n = len(times)
        obs_means = np.zeros(n, dtype=np.float64)
        obs_vars = np.zeros(n, dtype=np.float64)       

        # For times that coincide with the conditioning set
        coincide = np.isin(times, self.cond_times)
        print("Number of coinciding time points: %d" % coincide.sum())
        for i in coincide.nonzero()[0]:
            t = times[i] 
            j = (self.cond_times == t).nonzero()[0][0]
            obs_means[i] = self.cond_values[j]
            if self.cond_errors[j] > 0:
                obs_vars[i] = self.cond_errors[j]
            else:
                pass  # no error associated with this measurement

        # For times that do not coincide with the conditioning set
        for i, t in enumerate(times):
            if not coincide[i]:
                time = np.array([t])
                E12 = self.cov_func(self.cond_times, time)
                E21 = E12.T 
                E22 = self.cov_func(time, time)
                mean_2 = self.mean_func(time)
                obs_means[i] = mean_2 + E21 @ (self.E11_inv @ self.cond_values_shift)
                obs_vars[i] = E22 - E21 @ (self.E11_inv @ E12)

        return obs_means + np.sqrt(obs_vars) * ndtri(p)

    def conditional_inverse_cdf_v2(self, times, p):
        """
        Potentially more efficient? Actually not, when *times*
        has cardinality greater than 1000, and it's more numerically
        unstable than *conditional_inverse_cdf* too. 

        """
        E12 = self.cov_func(self.cond_times, times)
        E21 = E12.T 
        E22 = self.cov_func(times, times)
        mean = self.mean_func(times) + E21 @ (self.E11_inv @ self.cond_values_shift)
        E = E22 - E21 @ (self.E11_inv @ E12)
        return mean + np.sqrt(np.diagonal(E)) * ndtri(p)

    def conditional_pdf(self, times, values):
        """
        Evaluate the joint conditional probability density function of the 
        conditioned Gaussian process at a new set of values. 

        args
        ----
            times       :   1D ndarray of shape (m,)
            values      :   1D ndarray of shape (m,)

        returns
        -------
            1D ndarray, dtype float64, shape (m,)

        """
        if not self.is_conditioned:
            return self.pdf(times, values)
        else:
            E21 = self.cov_func(times, self.cond_times)
            E12 = E21.T 
            E = self.cov_func(times, times) - E21 @ (self.E11_inv @ E12)
            mean = self.mean_func(times) + E21 @ (self.E11_inv @ self.cond_values_shift)
            return np.exp(-0.5 * (mean * (np.linalg.inv(E) @ mean)).sum()) / \
                (np.power(2 * np.pi * np.linalg.det(E), 0.5))

    def disjoint_conditional_pdfs(self, time, values):
        """
        Evaluate the conditional PDF of this Gaussian process on 
        a particular time point, against one or more values.

        args
        ----
            time        :   float, seconds
            values      :   1D ndarray, shape (m,)

        returns
        -------
            1D ndarray, shape (m,)

        """
        time = np.array([time])
        if not self.is_conditioned:
            mean = self.mean_func(time)[0]
            sig2 = self.cov_func(time, time)[0,0]
        elif time in self.cond_times:
            i = (self.cond_times == time).nonzero()[0][0]
            sig2 = self.cond_errors[i]
            mean = self.cond_values[i]
        else:
            E21 = self.cov_func(time, self.cond_times)
            E12 = E21.T 
            E = self.cov_func(time, time) - E21 @ (self.E11_inv @ E12)
            mean = (self.mean_func(time) + E21 @ (self.E11_inv @ self.cond_values_shift))[0]
            sig2 = E[0,0]

        return np.exp(-(values-mean)**2 / (2 * sig2)) / np.sqrt(2 * np.pi * sig2)

    def decondition(self):
        """
        Delete the current conditioning data.

        """
        del self.E11
        del self.E11_inv 
        del self.mean_1 
        del self.cond_values_shift 
        self.is_conditioned = False 

class FractionalBrownianMotion(GaussianProcess):
    """
    A fractional Brownian motion, constructed as a Gaussian process.

    """
    def __init__(self, D=1.0, hurst=0.5, dt=0.01, D_type=1):

        self.D = D 
        self.hurst = hurst 
        self.dt = dt 
        self.D_type = D_type
        self.D_mod = D / (2 * hurst * np.power(dt, 2 * hurst - 1))

        # Define the covariance function
        def cov_func(t0, t1):
            S, T = np.meshgrid(t1, t0)
            if self.D_type == 1:
                return self.D * (
                    np.power(T, 2 * hurst)
                    + np.power(S, 2 * hurst)
                    - np.power(np.abs(T-S), 2*hurst)
                )
            elif self.D_type == 3:
                return self.D_mod * (
                    np.power(T, 2 * hurst)
                    + np.power(S, 2 * hurst)
                    - np.power(np.abs(T-S), 2*hurst)
                )

        super(FractionalBrownianMotion, self).__init__(cov_func)


