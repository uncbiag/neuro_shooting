# neuro_shooting
Shooting approaches for deep neural networks.

# Neuro_shooting development installation

Sometimes it is desirable to install *neuro_shooting* for development purposes. To do this, first download the git repository

```
   git clone https://github.com/ANONYMOUS_GITHUB_ACCOUNT/neuro_shooting.git
```

The repository's main folder contains a setup.py file (see [python setup file](https://github.com/kennethreitz/setup.py "python setup file")). For development purposes then simply execute

```
   cd neuro_shooting
   python setup.py develop
```

This will install all library links and all missing packages and will allow neuro_shooting imports with the exception of the `torchdiffeq` package which can be installed via

```
   pip install git+https://github.com/rtqichen/torchdiffeq
```

Once done you simply import the library as

```
import neuro_shooting
```

There is also some rudimentary documentation available. This documentation can currently be compiled via the following

```
cd neuro_shooting
cd docs
make html
```

Then simply open index.html which will be in the build/html directory (of the docs subdirectory).

It is now also possible to add hooks to shooting blocks. Most easily results are displayed via tensorboard. Once tensorboard output exists you can simply start a tensorboard server by typing

```
tensorboard --logdir=runs
```

(assuming the tensorboard output is in the directory `runs`).
You can look at the results by opening

```
http://localhost:6006
```

in your webbrowser.
