# neuro_shooting
Shooting approaches for deep neural networks

# Neuro_shooting development installation

Sometimes it is desirable to install *neuro_shooting* for development purposes. To do this, first download the git repository

   git clone https://github.com/ANONYMOUS_GITHUB_ACCOUNT/neuro_shooting.git

The repository's main folder contains a setup.py file (see `python setup file https://github.com/kennethreitz/setup.py). 
For development purposes then simply execute

   cd neuro_shooting
   python setup.py develop

This will install all library links and all missing packages and will allow mermaid imports with the exception of the `torchdiffeq` package which can be installed via

   pip install git+https://github.com/rtqichen/torchdiffeq
   
Once done you simply import the library as

import neuro_shooting


