# Developing

To contribute developing `Optima`, we have a structured developing mode in order to
have almost the same environment for everyone. Below we describe the workflow to develop
`Optima`. All the instructions below should be executed inside the Project root directory.

## Guidelines for developing mode installation

### 1. `conda` environment

We use [conda](https://docs.conda.io/en/latest/) to generate and manage environments.
Please follow the instructions therein to install it locally. We recommend you to install
`miniconda` configured for Python 3 (`miniconda3`).

### 2. Extending `conda` with `conda-devenv`

We use [conda-devenv](https://github.com/ESSS/conda-devenv) to extend `conda` capabilities
and define environments variables in developing mode. With `miniconda3` properly installed and running,
just execute in your console:

```console
$ conda install -c conda-forge conda-devenv
``` 

The above command will install `conda-devenv` in your `miniconda3` base environment.

### 3. Creating an environment for `Optima` development

Now, we can create an environment to develop `Optima`, with all dependencies installed in this environment.
From your `miniconda3` base environment with `conda-devenv` installed, run the following in your console:

```console
$ conda devenv
```

After that, you will have a `conda` environment called `optima` properly configured. A successful installation
should display in your console the message:

```console
#
# To activate this environment, use
#
#     $ conda activate optima
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### 4. Building `Optima`

Activate `optima` conda environment with:

```console
$ conda activate optima
```

Then run in your console:

```console
$ cmake -P install
```

A successful installation will build `Optima` and install its Python bindings. If everything is fine, you are able
to execute the following in your console without errors:

```console
$ python -c 'import optima'
```

### 5. Testing

At last, but importantly, you have to run the tests. Every contribution to the code should be tested. The `master`
branch should be always passing all the tests. With `Optima` and its Python bindings properly installed inside `optima`
conda environment, run in your console (in the project root directory):

```console
pytest . -n auto
```

## Questions? Problems?

Please feel free to contact us or open an issue. Thanks in advance!