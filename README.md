Symplectic Learning for Hamiltonian Neural Networks
========
Copyright © 2021 SpaceAble. This work is published under the terms of the
[Creative Commons BY-NC-SA 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
See also the file ```LICENSE```.

This repository hosts the open source code for the article "Symplectic Learning for Hamiltonian Neural Networks",
authored by Marco David and Florian Méhats, available [on the arXiv](https://arxiv.org/abs/2106.11753). An address for possible
correspondence related to this work can be found in the article PDF file. 

#### Dependencies & Installation

This project was written in Python 3.9. Its dependencies are specified in the `environment.yml` file
contained in this repository, which can be used to automatically create an anaconda virtual environment.

To run the code locally, first clone the repository using ```git clone https://github.com/SpaceAbleOrg/symplectic-hnn.git```,
then ```cd symplectic-hnn```.
Create the anaconda virtual environment using ```conda env create -f environment.yml``` and activate it as
```conda activate main3.9```.

To run the training code, run for example ```python train.py pendulum --loss_type euler-symp```.
To run the evaluation code, launch one of the Jupyter Notebooks in the ```plotting``` folder. Please consult the below
explanation of the repository structure and notably the file ```model/args.py``` for a detailed description of the
possible commands and arguments.


#### Pre-trained models and datasets

The [release v1.0](https://github.com/SpaceAbleOrg/symplectic-hnn/releases/tag/v1.0) contains the synthetic datasets as well as the trained models
that we used for the article. The datasets are pickled Python dictionaries (`*.shnndata`), which should be unpickled
manually. Contrarily, the trained models' `state_dict` was saved together with the used arguments via `torch.save` in a
`*.tar` file. This file can simply be loaded by ```HNN.load``` through specification of the correct arguments (i.e. 
name, the loss type, the value of h and, if applicable, the noise strength) that
were initially used to train the model.


#### Repository Structure

The main source code for our models (an adaption of [the first implementation of HNNs](https://github.com/greydanus/hamiltonian-nn)
by Greydanus et al.)
is contained in the directory `model`. Other than the neural networks built with PyTorch, it contains the
various symplectic and non-symplectic loss functions, an abstract base class to generate data sets (including
example subclasses, notably those used in our work) and an argument parser. All this code is polymorphic and
can be easily interfaced or extended.

Further, the root directory contains several modules or scripts useful for training and evaluation:
- The file `train.py` is the central training script and can be run using any of the arguments listed in `args.py`
(also see below for an example).
- The file `parallelize.py` builds on the training script, allowing to launch multiple threads training different models
in parallel. It also provides a simple text prompt for the main arguments `name`, `loss_type` and `h`.
- The file `metrics.py` provides a collection of functions that evaluate a given trained model (either by calculating
the error of the learned Hamiltonian or by rolling out specific predicted trajectories).
- The file `visualize.py` contains some useful functions and global constants for producing the plots included
in the submitted article.

Finally, the directory `plotting` contains three Jupyter Notebooks that were used to generate our plots based on our
trained models. Each notebook builds on the `visualize.py` module.
