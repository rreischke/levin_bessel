Installation
============

With pip
------------
The library is available via PyPi installation, so you can just install it using 

.. code-block:: bash

    pip install pylevin

which installs the levin package and you are ready to go. Note that you need certain packages which you have to install via ``conda-forge``, in particular you will need the GSL, BOOST and GXX (on some linux servers) to be installed. You can grab them all via

.. code-block:: bash

    conda install -c conda-forge gsl
    conda install -c conda-forge gxx_linux-64
    conda install conda-forge::boost

if they are not installed already.


From source
------------

Alternatively you first clone the directory via:

.. code-block:: bash
    
    git clone git@github.com:rreischke/levin_bessel.git

then navigate to the cloned directory

.. code-block:: bash

    cd levin_bessel
    conda env create -f conda_env.yaml
    conda activate levin_env
    pip install .

On some Linux servers you will have to install ``gxx_linux-64`` by hand and the installation will not work. This usually shows the following error message in the terminal:
``gcc: fatal error: cannot execute 'cc1plus': execvp: No such file or directory``.
If this is the case just install it by typing

.. code-block:: bash

     conda install -c conda-forge gxx_linux-64

and redo the ``pip`` installation.

