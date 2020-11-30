The following sections describe how to run (and reproduce) the results from the paper.

* [Cubic and quadratic like function fitting](#cubic-and-quadratic-like-function-fitting)

## Cubic and quadratic like function fitting

The main python script to run these two examples is `simple_functional_mapping_example.py`. It comes with various command-line 
configuration options, but the base configuration should already be reasonable. Here are examples how to run the scripts:

```bash
python simple_functional_mapping_example.py \
   --fcn cubic --custom_parameter_freezing 
   --unfreeze_parameters_at_iter 50 \ 
   --optimize_over_data_initial_conditions \
   --niter 500
      
python simple_functional_mapping_example.py \
   --fcn quadratic --custom_parameter_freezing 
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

## Spiral example

The main python script to run the spiral example is `sprial.p`. As for the cubic and the quadratic-like
examples, the default parameters should already be pretty good. It can be run as follows:

```bash
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


## Concentric circles

```bash
for run in `seq 10`; do
  for inf in ${INFLATION[*]}; do
    CMD="$PYTHON cc.py \
      --shooting_dim 2 \
      --method rk4 \
      --stepsize 0.1 \
      --pw 1.5 \
      --shooting_model updown_universal \
      --nr_of_particles 20 \
      --gpu 1 \
      --batch_size 128 \
      --inflation_factor ${inf}\
      --sim_weight 1.0 \
      --lr 5e-3 \
      --niters 20 \
      --save_model model_${inf}_run_${run}"
    echo $CMD
    $CMD
  done
done
```

## Rotated MNIST 

To experiment with the *Rotated MNIST* data, check out the `rot.py` file.

### Data

To get the rotated MNIST data (and also used in [Yildiz et al., NeurIPS 19](https://papers.nips.cc/paper/2019/hash/99a401435dcb65c4008d3ad22c8cdad0-Abstract.html)), go to the `data` folder and, from there, execute 

```bash
python download_rotated_mnist.py
```

This will download and unzip the data. **Note** that automatic downloading requires two packages, `gdown` and `tarfile` which can easily be installed via

```bash
pip install gdown
pip install tarfile
```

### Running experiments

To run experiments with the particle shooting approach of the paper (using the *UpDown* model) you can use:

```bash
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

*This will create a subfolder (within the current directory) called `runs`. Once model training has finished, you can then run the `RotatedMNIST_Analysis.ipynb` notebook that allows to load all tracked stats and plot some visualization.*

### Experimenting with pre-trained models

You can also download some pre-trained models via 

```bash
python download_rotated_mnist_runs.py
```

which will create a directory `rotated_mnist_runs`. Again, the ``RotatedMNIST_Analysis.ipynb` can then be used. In particular, we include 5 trained models for the **dyn. w. particles UpDown model** and the **static direct UpDown model**. 
