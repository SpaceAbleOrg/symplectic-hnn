# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import os
import sys
import torch
import numpy as np

from nn_models import MLP
from hnn import HNN
from experiment.data import get_dataset
from utils import choose_loss
from args import get_args


def train(args):
    # SETUP NETWORK AND OPTIMIZER
    # Create the standard MLP with args.dim inputs and one single scalar output, the Hamiltonian
    nn_model = MLP(args.dim, args.hidden_dim, output_dim=1, nonlinearity=args.nonlinearity)
    # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
    model = HNN(nn_model)
    # Create a standard optimizer
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    # Load the symplectic (or not) loss function
    loss = choose_loss(args.loss_type)(args)

    # LOAD DATASET AND PREPARE TENSOR OBJECTS
    # TODO choose dynamically which dataset to load, to not have to duplicate this field only ever changing one line
    data = get_dataset(seed=args.seed, noise_std=args.noise, delta_t=args.delta_t)
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, 2) ???
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, 2) ???
    t = data['t']
    test_t = data['test_t']

    # DO VANILLA TRAINING LOOP
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        # train step, find loss and optimize
        loss = loss(model, x, t)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        test_loss = loss(model, test_x, test_t)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dist = loss(model, x, t, return_dist=True)
    test_dist = loss(model, test_x, test_t, return_dist=True)
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return model, stats


# This function is generic, but needs to run in this "main" file to setup the path variables
# and define the save_directory properly.
def setup_args():
    # Setup directory of this file as working (save) directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    sys.path.append(parent_dir)

    # Load arguments
    args = get_args(dimension, name, this_dir)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


if __name__ == "__main__":
    # DEFINE PROBLEM SPECIFIC CONSTANTS
    # -- These are defaults, they can be overwritten using the argument parsers
    dimension = 2  # number of (generalized) coordinates of the Hamiltonian
    name = "spring"  # name of the problem or dataset

    # SETUP AND LOAD ARGUMENTS
    args = setup_args()

    # RUN THE MAIN FUNCTION
    model, _ = train(args)

    # SAVE - TODO
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    if args.symplectic_euler:
        label += '-syeu'
    elif args.symplectic_midpoint:
        label += '-symp'
    else:
        label += '-fweu'
    label += '-n' + str(args.noise)
    label += '-t' + str(args.delta_t)
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)
