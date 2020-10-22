# neuro_shooting

Shooting approaches for deep neural networks. In case you use the code for your research, please use the following
BibTeX entry.

```
@inproceedings{Vialard20a,
   author    = {F.X.~Vialard and R.~Kwitt and S.~Wei and M.~Niethammer},
   title     = {A Shooting Formulation of Deep Learning},
   booktitle = {NeurIPS},
   year      = {2020}}
```

# neuro_shooting development installation

Sometimes it is desirable to install *neuro_shooting* for development purposes. To do this, first download the git repository

```
   git clone https://github.com/uncbiag/neuro_shooting.git
```

The repository's main folder contains a setup.py file (see [python setup file](https://github.com/kennethreitz/setup.py "python setup file")). For development purposes then simply execute

```
   cd neuro_shooting
   python setup.py develop
```

This will install all library links and all missing packages and will allow `neuro_shooting` imports with the exception of the `torchdiffeq` package which can be installed via

```
   pip install git+https://github.com/rtqichen/torchdiffeq
```

Once done, you simply import the library as

```
import neuro_shooting
```

# Running demos and recreating some of the results in the paper

The examples can be run based on code in the *demos* directory

### Cubic and quadratic-like function fitting

The main python scripts to run these two examples is *simple_functional_mapping_example.py*. It comes with various command 
line configuration options, but the base configuration should already be reasonable. Here are examples how to run the scripts:

```
    python simple_functional_mapping_example.py --fcn cubic --custom_parameter_freezing 
        --unfreeze_parameters_at_iter 50 --optimize_over_data_initial_conditions --niter 500
    python simple_functional_mapping_example.py --fcn quadratic --custom_parameter_freezing 
        --unfreeze_parameters_at_iter 50 --optimize_over_data_initial_conditions --niter 500
``` 

If you are interested in how the results for the paper where obtained (that are the basis for the boxplots for example) have a look
at
- run_long_iteration_experiments_simple_functional_mapping.py
- run_long_iteration_experiments_simple_functional_mapping_rnn.py

These two python scripts specify the entire experimental setting for the cubic and the quadratic-like functions for 
all the four models in the paper.

### Spiral example

The main python script to run the spiral example is *sprial.py*. As for the cubic and the quadratic-like
examples the default parameters should already be pretty good. It can be run as follow:
```
    python spiral.py --seed 0 --shooting_model updown_universal --niters 2000 
        --optional_weight 10 --save_figures --viz_freq 50 --validate_with_long_range 
        --optimize_over_data_initial_conditions --inflation_factor 32 --viz 
        --nr_of_particles 25 --custom_parameter_freezing --unfreeze_parameters_at_iter 50
```

If you are interested in how the results for the paper where obtained (that are the basis for the boxplots for example) have a look
at

- run_experiments_spiral.py
- run_experiments_spiral_rnn.py
 
These two python scripts specify the entire experimental setting for the spiral for all the four models in the paper.

### Rotating MNIST and bouncing balls

We are in the process of cleaning up our Jupyter notebooks for these examples and will make them available shortly.

# Documentation

There is also some rudimentary documentation available. This documentation can currently be compiled via the following

```
cd neuro_shooting
cd docs
make html
```

Then simply open index.html which will be in the build/html directory (of the docs subdirectory).

# Tensorboard support

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
