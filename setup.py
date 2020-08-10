#!/usr/bin/env python
"""
setup.py

"""
import setuptools

setuptools.setup(
    name="strobemod",
    version="1.0",
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="Diffusion modeling for stroboscopic particle tracking",
    packages=setuptools.find_packages(),
)
