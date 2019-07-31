Installation
============

This note briefly describes how to install and use *neuro_shooting*. We recommend installing *neuro_shooting* using `conda <http://docs.conda.io>`__. We also recommend using a conda virtual environment to keep the installation clean.

neuro_shooting requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*neuro_shooting* is based on the following:

  - python >= 3.6
  - pytorch >= 1.0

It runs on Linux and on OSX. Windows installs might be possible, but have not been tested (hence will likely not work out of the box) and are not officially supported.
    
anaconda installation
^^^^^^^^^^^^^^^^^^^^^

If you have not done so yet, install anaconda. Simply follow the installation instructions `here <https://www.anaconda.com/download>`__. This will provide you with the conda package manager which will be used to create a virtual environment (if desired, we highly recommend it) and to install all python packages that *neuro_shooting* depends on.

Creating a conda virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is best to install everything into a conda virtual environment. This can be done as follows.

.. code::

   conda create --name neuro_shooting python=3.7 pip
   conda activate neuro_shooting

If you later want to remove this environment again, this can be done by executing

.. code::

   conda remove --name neuro_shooting --all
   
   
Neuro_shooting installation via conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. todo::
   Add the actual conda channel here, once we have it

*neuro_shooting* can conveniently be installed via *conda*. Once you have activated your desired conda environment (for example, the conda virtual environment created above) *neuro_shooting* can be installed by executing

.. code::
   
   conda install -vv -k -c http://anonymous_link.com/download/local_conda_channel -c pytorch -c conda-forge -c anaconda neuro_shooting=0.1.0

Once installed *neuro_shooting* simply needs to be imported in your python program via

.. code::
   
   import neuro_shooting
   

Neuro_shooting development installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is desirable to install *neuro_shooting* for development purposes. To do this, first download the git repository

.. code::

   git clone https://github.com/ANONYMOUS_GITHUB_ACCOUNT/neuro_shooting.git

The repository's main folder contains a setup.py file (see `python setup file <https://github.com/kennethreitz/setup.py>`_) for the setup file *mermaid's* is based on and a general explanation of its use. For development purposes then simply execute

.. code::

   cd neuro_shooting
   python setup.py develop

This will install all library links and all missing packages and will allow mermaid imports with the exception of the `torchdiffeq` package which can be installed via

.. code::

   pip install git+https://github.com/rtqichen/torchdiffeq
   


Creating the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is created via `sphinx <http://www.sphinx-doc.org/>`__. To build it first install graphviz (on OSX: `brew install graphviz` or via conda, see below). If you installed via the developer option (via `setup.py`) you will also need to install *pandoc* (should be auto installed via conda). This can be done by following the instructions `here <https://pypi.org/project/pypandoc/>`__ (pypandoc will be auto installed via `setup.py`) or by installing it manually via conda:

.. code::

   conda install -c conda-forge pandoc

Graphviz can also be installed via conda if desired:

.. code::

   conda install -c anaconda graphviz

Then execute the following to make the documentation

.. code::

   cd neuro_shooting
   cd docs
   make html


This will create the docs in `build/html`.

Running the code
^^^^^^^^^^^^^^^^

The simplest way to start is to look example scripts in the `demos/` directory or
to run the examples from the `jupyter` directory which contains various example jupyter notebooks.
You can run the jupyter notebooks as follows (should be intalled if you installed
via conda or `python setup.py develop` as described above):

.. code::

   cd neuro_shooting
   cd jupyter
   jupyter notebook



   
