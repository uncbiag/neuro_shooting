# neuro_shooting

This is the official repository for   

F.X. Vialard and R. Kwitt and S. Wei and M. Niethammer         
**A Shooting Formulation of Deep Learning**    
*NeurIPS 2020*    
[Paper](https://proceedings.neurips.cc//paper/2020/file/89562dccfeb1d0394b9ae7e09544dc70-Paper.pdf)

In case you use the code for your research, please use the following BibTeX entry:

```
@inproceedings{Vialard20a,
   author    = {F.X.~Vialard and R.~Kwitt and S.~Wei and M.~Niethammer},
   title     = {A Shooting Formulation of Deep Learning},
   booktitle = {NeurIPS},
   year      = {2020}}
```

# Development installation

The (PyTorch-based) code has, so far, been tested on a system running Ubuntu Linux 18.04, with four NVIDIA GeForce RTX 2080 Ti cards using PyTorch 1.4 and CUDA 10.1. We do recommend a setup using [Anaconda Python](https://www.anaconda.com/products/individual). 

To install `neuro_shooting` for development purposes, first clone the git repository via

```
git clone https://github.com/uncbiag/neuro_shooting.git
```

The repository's main folder contains a `setup.py` file (see [python setup file](https://github.com/kennethreitz/setup.py "python setup file")). For development purposes then simply execute

```bash
cd neuro_shooting
python setup.py develop
```

This will install all library links and all missing packages and will allow `neuro_shooting` imports with the exception of the `torchdiffeq` package which can be installed via

```bash
pip install git+https://github.com/rtqichen/torchdiffeq
```

Once done, you simply import the library as

```python
import neuro_shooting
```

In case you use Anaconda Python you can also install all required packages (see `requirements.txt` and `torchdiffeq`) by hand, clone the repository and add the base folder via

```bash
conda develop neuro_shooting
```

# Experiments/Demos

The experiments from the NeurIPS 2020 paper can be found in the `demos/neurips20` folder, see [here](demos/neurips20).

# Documentation

There is also some rudimentary documentation available. This documentation can currently be compiled via the following

```bash
cd neuro_shooting
cd docs
make html
```

Then simply open `index.html` which will be in the `build/html` directory (of the `docs` subdirectory).

# Tensorboard support

It is now also possible to add hooks to shooting blocks. Most easily results are displayed via tensorboard. Once tensorboard output exists you can simply start a tensorboard server by typing

```bash
tensorboard --logdir=runs
```

(assuming the tensorboard output is in the directory `runs`).
You can look at the results by opening

```
http://localhost:6006
```

in your webbrowser.
