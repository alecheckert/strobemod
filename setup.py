#!/usr/bin/env python
"""
setup.py -- install the strobemodels package in the current
environment and also unzip some of the larger data files

"""
import os
import setuptools
from zipfile import ZipFile

# User's home directory
HOME_DIR = os.path.expanduser("~")

# Directory for local user binaries
LOCAL_DIR = os.path.join(HOME_DIR, ".local")
BIN_DIR = os.path.join(LOCAL_DIR, "bin")

# Source code for Dirichlet process Gibbs samplers
SRC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src")

# Target executables for same
EXECS = ["gs_dp_diff", "gs_dp_diff_defoc"]

# Potential user .bashrc files
BASHRCS = [os.path.join(HOME_DIR, f) for f in [".bash_profile", ".bashrc"]]
PATHLINE = '\nexport PATH=$PATH:"{}"'.format(BIN_DIR)

# Build the gs_dp_diff program for evaluation of 
# a Dirichlet process mixture model
GS_DP_DIFF_INSTALLED = False
try:
    # Make directory for binaries if it doesn't already exist
    for d in [LOCAL_DIR, BIN_DIR]:
        if not os.path.isdir(d):
            os.mkdir(d)

    # Compile the executables
    os.system("make -f {}".format(os.path.join(SRC_DIR, "makefile")))
    for _exec in EXECS:
        os.rename(
            os.path.join(SRC_DIR, _exec),
            os.path.join(BIN_DIR, _exec)
        )

    # Add the binary directory to PATH
    for bashrc in BASHRCS:
        if os.path.isfile(bashrc):
            with open(bashrc, "r") as f:
                flines = f.read()
            if not PATHLINE.replace("\n", "") in flines:
                with open(bashrc, "a") as f:
                    f.write(PATHLINE)
            break
    GS_DP_DIFF_INSTALLED = True
except:
    print("WARNING: gs_dp_diff not installed")

# Unzip large data files
REPO_DIR = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(REPO_DIR, "strobemodels", "data")
targets = [
    os.path.join(DATA_DIR, "fbm_defoc_splines.zip"),
    os.path.join(DATA_DIR, "abel_transform_dz-0.5um.zip"),
    os.path.join(DATA_DIR, "abel_transform_dz-0.6um.zip"),
    os.path.join(DATA_DIR, "abel_transform_dz-0.7um.zip"),
    os.path.join(DATA_DIR, "free_abel_transform.zip"),
    os.path.join(DATA_DIR, "abel_transforms_range-20um.zip")
]
for t in targets:
    print("unzipping {}...".format(t))
    out_path = os.path.split(t)[0]
    with ZipFile(t) as f:
        f.extractall(out_path)

# Install
setuptools.setup(
    name="strobemodels",
    version="1.0",
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="Diffusion modeling for stroboscopic particle tracking",
    packages=setuptools.find_packages(),
)

# User message
if GS_DP_DIFF_INSTALLED:
    print("\nIMPORTANT:\n" \
        "Successfully installed gs_dp_diff and gs_dp_diff_defoc.\n" \
        "If you want to run Dirichlet process mixture models, check\n" \
        "that these executabes exist by doing the following:\n" \
        "\n\t1. Open a new terminal.\n" \
        "\t2. Enter 'gs_dp_diff' or 'gs_dp_diff_defoc'. If configured\n" \
        "\tcorrectly, a docstring should print to the terminal.\n" \
        "\nIf you get an 'executable not found', add the executables\n" \
        "at ~/.local/bin to your $PATH.\n")
else:
    print("\n\nWARNING:\nNecessary binaries gs_dp_diff and gs_pd_diff_defoc\n" \
        "NOT installed. Both source code files at strobemodels/src will\n" \
        "need to be compiled (with -std>=c++11) and placed into PATH\n" \
        "before running any Dirichlet processes.\n")
