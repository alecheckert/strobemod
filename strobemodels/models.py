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

# Dataframes
import pandas as pd 

def 



