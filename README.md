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

The (PyTorch-based) code has, so far, been tested on a system running Ubuntu Linux 18.04, with four NVIDIA GeForce RTX 2080 Ti cards using PyTorch 1.7.0 and CUDA 10.1.

We do recommend a setup using [Anaconda Python](https://www.anaconda.com/products/individual). The following describes the full setup. We assume Anaconda Python will be installed in `/scratch/anaconda` and `neuro_shooting` will be reside in `/scratch/neuro_shooting`.

```
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
<FOLLOW INSTRUCTIONS TO INSTALL in /scratch/anaconda>
source /scratch/anaconda/bin/activate
```

Next, clone the `neuro_shooting` git repository via

```
cd /scratch/
git clone https://github.com/uncbiag/neuro_shooting.git
```

Install PyTorch (here 1.7.0 and CUDA 10.1) and the `torchdiffeq` package via:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install git+https://github.com/rtqichen/torchdiffeq
```

Install the remaining dependencies and install `neuro_shooting` in development mode via

```bash
cd /scratch/neuro_shooting
pip install -r requirements.txt
conda develop /scratch/neuro_shooting 
```

Test your installation via

```python
import neuro_shooting
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
