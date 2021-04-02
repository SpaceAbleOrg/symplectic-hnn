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
from utils import choose_loss, choose_data
from args import get_args


def train(args):
    # Load the data set loader/generator
    data_loader = choose_data(args.name)(args.h, args.noise)

    # SETUP NETWORK AND OPTIMIZER
    # Create the standard MLP with data_loader.dimension() inputs and one single scalar output, the Hamiltonian
    nn_model = MLP(data_loader.dimension(), args.dim_hidden_layer, output_dim=1, nonlinearity=args.nonlinearity)
    # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
    model = HNN(nn_model)
    # Create a standard optimizer
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    # Load the symplectic (or not) loss function
    loss_fct = choose_loss(args.loss_type)(args)

    # LOAD DATASET AND PREPARE TENSOR OBJECTS
    data = data_loader.get_dataset(seed=args.seed)
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, 2) ???
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, 2) ???
    t = data['t']
    test_t = data['test_t']

    # DO VANILLA TRAINING LOOP
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        # train step, find loss and optimize
        loss = loss_fct(model, x, t)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        test_loss = loss_fct(model, test_x, test_t)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dist = loss_fct(model, x, t, return_dist=True)
    test_dist = loss_fct(model, test_x, test_t, return_dist=True)
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
    args = get_args()

    # Set the save directory if nothing is given
    if not args.save_dir:
        args.save_dir = this_dir + '/experiment-' + args.name

    # Store dimension directly in args, for future convenience of loss functions
    data_class = choose_data(args.name)
    args.dim = data_class.dimension()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


if __name__ == "__main__":
    # SETUP AND LOAD ARGUMENTS
    args = setup_args()

    # RUN THE MAIN FUNCTION
    model, _ = train(args)

    # SAVE
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = args.name + '-' + args.loss_type + '-h' + str(args.h)
    if args.noise > 0:
        label += '-n' + str(args.noise)
    path = '{}/{}.tar'.format(args.save_dir, label)
    torch.save(model.state_dict(), path)
