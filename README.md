# strobemodels
Diffusion models for stroboscopic particle tracking

### Install

`strobemodels` requires the following Python packages:
- `numpy`
- `pandas`
- `scipy`
- `dask`
- `tqdm`
- `scikit-image`
- `matplotlib`
- `seaborn`
- `hankel`
- `nose2` (or `pytest`, or another test runner)

If you're using `conda`, a possible `conda` environment is in the `strobe_env.yml` file. To build this environment, do
```
  conda env create -f strobe_env.yml
```

Switch to the environment:
```
  conda activate strobe_env
```

Install `strobemodels` with
```
  python setup.py develop
```

`strobemodels` is in active development. The `develop` option will track changes in the source files as new versions are pulled.

### Running the testing suite

From the `strobemodels` repo, run `nose2`. (Or `pytest`, or another rest runner if you prefer.)

### Getting started

(coming soon)


