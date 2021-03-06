# NeurIPS 2020 Experiments

The following sections describe how to run (and reproduce) the experimental results from 

F.X. Vialard and R. Kwitt and S. Wei and M. Niethammer         
**A Shooting Formulation of Deep Learning**    
*NeurIPS 2020*    
[Paper](https://proceedings.neurips.cc//paper/2020/file/89562dccfeb1d0394b9ae7e09544dc70-Paper.pdf)


**Note**: We assume, for demonstration, that we are working with the repository checked out into `/scratch/neuro_shooting`. Obviously, you can clone the repo to any place you like, but you have to adjust the path to your setting.

# Overview

* [Cubic and quadratic like function fitting experiment](#cubic-and-quadratic-like-function-fitting-experiment)
* [Spiral experiment](#spiral-experiment)
* [Concentric circles experiment](#concentric-circles-experiment)
* [Rotated MNIST experiment](#rotated-mnist-experiment)
* [Bouncing balls experiment](#bouncing-balls-experiment)

# Cubic and quadratic like function fitting experiment

The main python script to run these two examples is `simple_functional_mapping_example.py`. It comes with various command-line 
configuration options, but the base configuration should already be reasonable. Here are examples how to run the scripts:

```bash
cd /scratch/neuro_shooting/demos/neurips20
python simple_functional_mapping_example.py \
   --fcn cubic \
   --custom_parameter_freezing \
   --unfreeze_parameters_at_iter 50 \ 
   --optimize_over_data_initial_conditions \
   --niter 500

python simple_functional_mapping_example.py \
   --fcn quadratic \
   --custom_parameter_freezing 
   --unfreeze_parameters_at_iter 50 \
   --optimize_over_data_initial_conditions \
   --niter 500
``` 

If you are interested in how the results for the paper where obtained (i.e., the basis for the boxplots for example) have a look
at

- `run_long_iteration_experiments_simple_functional_mapping.py` and
- `run_long_iteration_experiments_simple_functional_mapping_rnn.py`

These two python scripts specify the entire experimental setting for the cubic and the quadratic-like functions for 
all the four models in the paper.

# Spiral experiment

The main python script to run the spiral example is `sprial.py`. As for the cubic and the quadratic-like
examples, the default parameters should already be pretty good. It can be run as follows:

```bash
cd /scratch/neuro_shooting/demos/neurips20
python spiral.py \
   --seed 0 \
   --shooting_model updown_universal \
   --niters 2000 \
   --optional_weight 10 \
   --save_figures \
   --viz_freq 50 \
   --validate_with_long_range \
   --optimize_over_data_initial_conditions \
   --inflation_factor 32 \
   --viz \
   --nr_of_particles 25 \
   --custom_parameter_freezing \
   --unfreeze_parameters_at_iter 50
```

If you are interested in how the results for the paper where obtained (i.e., the basis for the boxplots for example) have a look
at

- `run_experiments_spiral.py` and
- `run_experiments_spiral_rnn.py`
 
These two python scripts specify the entire experimental setting for the spiral for all the four models in the paper.


# Concentric circles experiment

In the concenctric circles experiment, we use synthetic data (generated from two concentric circles; one circle = one class) to train a simple binary classifier that uses a  **dyn. w particles UpDown** model (see paper) and a simple affine layer.

### Training

To run the concentric circles experiment, see `cc.py`. For example, to train a classifer based on the **dyn. w. particles UpDown** model of the paper using an inflation factor ($\alpha$) of 20, run:

```bash
cd /scratch/neuro_shooting/demos/neurips20
python cc.py \
   --shooting_dim 2 \
   --method rk4 \
   --stepsize 0.1 \
   --pw 1.5 \
   --shooting_model updown_universal \
   --nr_of_particles 20 \
   --gpu 0 \
   --batch_size 128 \
   --inflation_factor 20 \
   --sim_weight 1.0 \
   --lr 5e-3 \
   --niters 20 \
   --save_model model_20
```

For comparison, you can then, e.g., also experiment with an inflation factor of $\alpha=5$. The Juyter notebook [`Analysis-ConcentricCircles.ipynb`](Analysis-ConcentricCircles.ipynb) contains the analyses performed in the paper.

### Pre-trained models

You can also download 10 pre-trained models (five models for $\alpha=5$ and five models for $\alpha=20$) using

```bash
cd /scratch/neuro_shooting/demos/neurips20
python download_concentric_circles_runs.py
```

This creates a subfolder `concentric_circles_runs` which contains the models. The [`Analysis-ConcentricCircles.ipynb`](Analysis-ConcentricCircles.ipynb) Jupyter notebook can then be directly executed. It contains an analysis of how often each model could perfectly fit the training data.

# Rotated MNIST experiment

To experiment with the *Rotated MNIST* data, check out the `rot.py` file.

### Data

To get the rotated MNIST data (and also used in [Yildiz et al., NeurIPS 19](https://papers.nips.cc/paper/2019/hash/99a401435dcb65c4008d3ad22c8cdad0-Abstract.html)), go to the `data` folder and, from there, execute 

```bash
cd /scratch/neuro_shooting/data
python download_rotated_mnist.py
```

This will download and unzip the data. **Note** that automatic downloading requires two packages, `gdown` and `tarfile` which can easily be installed via

```bash
pip install gdown
pip install tarfile
```

### Training

To run experiments with the particle shooting approach of the paper (using the *UpDown* model) you can use:

```bash
cd /scratch/neuro_shooting/demos/neurips20
python rot.py \
    --gpu 0 \
    --verbose \
    --lr 5e-3 \
    --method dopri5 \
    --batch_size 25 \
    --norm_weight 0.05 \
    --sim_weight 1.0 \
    --inflation_factor 10 \
    --pw 1.0 \
    --nr_of_particles 100 \
    --optimize_over_data_initial_conditions \
    --shooting_dim 20 \
    --niters 500 \
    --n_skip 4 \
    --i_eval 3 \
    --save \
    --shooting_model updown_universal
```

In case you want to experiment with the *static direct* version (i.e., not using any particlces), add the `--use_particle_free_rnn_mode` command line parameter.

*This will create a subfolder (within the current directory) called `runs`. Once model training has finished, you can then run the [`RotatedMNIST_Analysis.ipynb`](RotatedMNIST_Analysis.ipynb) notebook that allows to load all tracked stats and plot some visualization.*

### Pre-trained models

You can also download some pre-trained models via 

```bash
cd /scratch/neuro_shooting/demos/neurips20
python download_rotated_mnist_runs.py
```

which will create a directory `rotated_mnist_runs`. Again, the [`RotatedMNIST_Analysis.ipynb`](RotatedMNIST_Analysis.ipynb) can then be used. In particular, we include 5 trained models for the **dyn. w. particles UpDown model** and the **static direct UpDown model**. 

# Bouncing balls experiment

To experiment with the *Bouncing Balls* data, check out the `balls.py` file.

### Data

To get the rotated MNIST data (and also used in [Yildiz et al., NeurIPS 19](https://papers.nips.cc/paper/2019/hash/99a401435dcb65c4008d3ad22c8cdad0-Abstract.html)), go to the `data` folder and, from there, execute 

```bash
cd /scratch/neuro_shooting/data
python download_bouncing_balls.py
```

### Training

To run experiments with the particle shooting approach of the paper (using the *UpDown* model) you can use:

```bash
cd /scratch/neuro_shooting/demos/neurips20
python balls.py \
   --gpu 0 \
   --verbose \
   --lr 1e-3 \
   --method dopri5 \
   --shooting_model updown_universal \
   --batch_size 25 \
   --sim_weight 1.0 \
   --norm_weight 0.0005 \
   --inflation_factor 20 \
   --pw 1.0 \
   --nr_of_particles 100 \
   --optimize_over_data_initial_conditions \
   --shooting_dim 40 \
   --niters 100 \
   --save_to balls_experiment
  ```
  
All results will be saved to the `balls_experiment` subfolder. In the Jupyter notebook [`Analysis-BouncingBalls.ipynb`](Analysis-BouncingBalls.ipynb), you can then use this path to get result statistics and show some visualizations (as in the paper).

### Pre-trained models

You can also download some pre-trained models via 

```bash
cd /scratch/neuro_shooting/data
python download_bouncing_balls_runs.py
```

which will create a directory `bouncing_balls_runs`. Again, the [`Analysis-BouncingBalls.ipynb`](Analysis-BouncingBalls.ipynb) can then be used to get statistics and visualizations. In particular, we include one trained **dyn. w. particles UpDown model** (trained with settings above).





