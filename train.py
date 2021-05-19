# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import os
import sys
import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model.loss import OneStepLoss
from model.hnn import HNN
from model.args import get_args
from model.data import choose_data
from utils import save_path, to_pickle, from_pickle


# This function is generic, but needs to run in a top-level file to setup the path variables
# and define the save_directory properly.
def setup(args, save_dir_prefix='/experiment-'):
    # Setup directory of this file as working (save) directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    sys.path.append(parent_dir)

    # Set the save directory if nothing is given
    if not args.save_dir:
        args.save_dir = this_dir + save_dir_prefix + args.name

    # Store data_class directly in args for future access, and dimension for future convenience (eg of loss functions)
    args.data_class = choose_data(args.name)
    args.dim = args.data_class.dimension()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def load_data(args):
    data_path = save_path(args, ext='shnndata', incl_loss=False)
    if args.new_data or not os.path.exists(data_path):
        if args.verbose:
            print("Generating a new data set...")

        data_loader = args.data_class(args.h, args.noise)
        data = data_loader.get_dataset(seed=args.seed, samples=args.data_samples,
                                       test_split=args.test_split, print_args=args)
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        to_pickle(data, data_path)
    else:
        if args.verbose:
            print("Loading the existing data set...")
        data = from_pickle(data_path)

    return data


def train(model, data, args):
    # Create a standard optimizer
    # optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    optim = torch.optim.AdamW(model.parameters(), args.learn_rate, betas=(0.9, 0.999), eps=1e-08,
                              weight_decay=0.01, amsgrad=True)

    # Load the symplectic (or not) loss function
    loss_fct = OneStepLoss(args)  # Choosing the actual loss_type is hidden in this constructor

    # Prepare objects from dataset dictionary
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)  # shape (set_size, dim)
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)  # shape (set_size, dim)
    t = data['t']
    test_t = data['test_t']

    # DO VANILLA TRAINING LOOP
    if args.verbose:
        print("Begin training loop...")

    best_model, best_test_loss = model, np.infty
    stats = {'train_loss': [], 'test_loss': []}
    writer = SummaryWriter()
    for step in range(args.epochs + 1):

        # Use stochastic gradient descent (SGD) with args.batch_size
        # – – TODO indexing into tensors is slow and not supported (?) on GPU
        # for ixs in torch.split(torch.arange(x.shape[0]), args.batch_size):
        #     ...
        #     loss = loss_fct(model, x[ixs], t[ixs])
        #     ...

        # train step, find loss and optimize
        model.train()
        loss = loss_fct(model, x, t)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        model.eval()
        train_loss_val = loss_fct(model, x, t)
        test_loss_val = loss_fct(model, test_x, test_t)

        if step > args.epochs/2 and test_loss_val < best_test_loss:
            best_model = copy.deepcopy(model)
            best_test_loss = test_loss_val

        # logging with tensorboard
        #writer.add_scalar("Loss/Train", train_loss_val, step)
        #writer.add_scalar("Loss/Test", test_loss_val, step)

        # logging manually
        stats['train_loss'].append(train_loss_val)
        stats['test_loss'].append(test_loss_val)

        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss_val, test_loss_val))

    # Final evaluation using the best_model
    train_dist = loss_fct(best_model, x, t, return_dist=True)
    test_dist = loss_fct(best_model, test_x, test_t, return_dist=True)
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return best_model, stats


def train_main(args):
    # SETUP ENV AND ARGUMENTS
    args = setup(args)

    # CREATE THE EMPTY MODEL
    model = HNN.create(args)

    # LOAD DATA SET
    data = load_data(args)

    # RUN THE MAIN FUNCTION TO TRAIN THE MODEL
    model, loss_log = train(model, data, args)

    # SAVE
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    torch.save({'args': args, 'model': model.state_dict(), 'stats': loss_log}, save_path(args))


def train_if_missing(args, save_dir_prefix='/experiment-'):
    args = setup(args, save_dir_prefix=save_dir_prefix)
    if not os.path.exists(save_path(args)):
        train_main(args)


if __name__ == "__main__":
    """ This file can be run with one well-defined set of arguments. To run for several configurations at once,
        please resort to the parallelize.py file. """
    train_main(get_args())
