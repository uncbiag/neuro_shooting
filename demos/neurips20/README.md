# Rotated MNIST 

To experiment with the *Rotated MNIST* data, check out the `rot.py` file.

## Data

To get the rotated MNIST data used in the paper (and also used in [Yildiz et al., NeurIPS 19](https://papers.nips.cc/paper/2019/hash/99a401435dcb65c4008d3ad22c8cdad0-Abstract.html)), go to the `data` folder and, from there, execute 

```bash
python download_rotated_mnist.py
```

This will download and unzip the data. **Note** that automatic downloading requires two packages, `gdown` and `tarfile` which can easily be installed via

```bash
pip install gdown
pip install tarfile
```

## Running experiments

To run experiments with the particle shooting approach of the paper (using the *UpDown* model), take a look at `rot.py`. In particular, you can run, e.g.,

```bash
python rot.py \
    --gpu 1 \
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

*This will create a subfolder (within the `rotated_mnist` directory) called `runs`. Once model training has finished, you can then run the `RotatedMNIST_Analysis.ipynb` notebook that allows to load all tracked stats and plot some visualization.*

## Experimenting with pre-trained models

You can also download some pre-trained models via 

```bash
python download_rotated_mnist_runs.py
```

which will create a directory `rotated_mnist_runs`. Again, the ``RotatedMNIST_Analysis.ipynb` can then be used. In particular, we include 5 trained models for the **dyn. w. particles UpDown model** and the **static direct UpDown model**. 

### Concentric circles

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

