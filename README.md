# strobemodels
Diffusion models for stroboscopic particle tracking

### Install

`strobemodels` requires the following Python packages, which (apart from `hankel`) are standard stuff:
- `numpy`
- `pandas`
- `scipy`
- `dask`
- `tqdm`
- `matplotlib`
- `seaborn`
- `hankel`
- `nose2` (or `pytest`, or another test runner)

If you need a place to start and you're using `conda`, a possible `conda` environment is in the `strobe_env.yml` file. To build this environment, do
```
  conda env create -f strobe_env.yml
```

Switch to the environment:
```
  conda activate strobe_env
```

Then install `strobemodels` with
```
  python setup.py develop
```

`strobemodels` is in active development. The `develop` option will track changes in the source files as new versions are pulled.

### Running the testing suite

From the `strobemodels` repo, run `nose2`. (Or `pytest`, or another rest runner if you prefer.)

`strobemodels` must be installed in the current `conda` environment to run tests.

### Getting started

(coming soon)


