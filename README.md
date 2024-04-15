Implements Levin's method for products of spherical and cylindrical Bessel functions.

## Installation
For starters you first clone the directory via:
```shell
git clone git@github.com:rreischke/levin_bessel.git
```

Then navigate to the cloned directory
```shell
cd levin_bessel
conda env create -f conda_env.yaml
conda activate levinpower_env
pip install .
```
On some Linux servers you will have to install ``gxx_linux-64`` by hand and the installation will not work. This usually shows the following error message in the terminal:
``
gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory
``
If this is the case just install it by typing
```shell
 conda install -c conda-forge gxx_linux-64
```
and redo the ``pip`` installation.

If you do not want to use the conda environment make sure that you have ``boost`` and ``gsl`` installed.
You can install both via ``conda``:
```shell
conda install -c conda-forge gsl
conda install -c conda-forge gxx_linux-64
conda install conda-forge::boost
git clone git@github.com:rreischke/OneCovariance.git
cd levin_bessel    
pip install .
```

## Examples
There are two notebooks in the ``test`` directory for the case of a single Bessel function and products of two or three Bessel functions.

