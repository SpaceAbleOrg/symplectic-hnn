# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import os
import copy
import torch
import numpy as np

from model.loss import OneStepLoss
from model.hnn import get_hnn
from utils import setup_args, save_path, to_pickle, from_pickle


def train(model, data, args):
    # Create a standard optimizer
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)
    # Load the symplectic (or not) loss function
    loss_fct = OneStepLoss(args)  # Choosing the actual loss_type is hidden in this constructor

    # Prepare objects from dataset dictionary
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)  # shape (set_size, dim)
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)  # shape (set_size, dim)
    t = data['t']
    test_t = data['test_t']

    if args.verbose:
        print("Begin training loop...")

    best_model, best_test_loss = model, np.infty

    # DO VANILLA TRAINING LOOP
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.epochs + 1):

        # Use stochastic gradient descent (TODO (??) to verify if that's reall SGD) with args.batch_size
        for ixs in torch.split(torch.arange(x.shape[0]), args.batch_size):
            # train step, find loss and optimize
            model.train()
            loss = loss_fct(model, x[ixs], t[ixs])
            loss.backward()
            optim.step()
            optim.zero_grad()

        # run test data
        model.eval()
        test_loss = loss_fct(model, test_x, test_t)
        test_loss_val = test_loss.item()

        if test_loss_val < best_test_loss:
            best_model = copy.deepcopy(model)
            best_test_loss = test_loss_val

        # logging
        #stats['train_loss'].append(loss.item())
        #stats['test_loss'].append(test_loss_val)
        if args.verbose and step % args.print_every == 0:
            train_loss_val = loss_fct(model, x, t)
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, train_loss_val, test_loss_val))

    # Final evaluation using the best_model
    train_dist = loss_fct(best_model, x, t, return_dist=True)
    test_dist = loss_fct(best_model, test_x, test_t, return_dist=True)
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
          .format(train_dist.mean().item(), train_dist.std().item() / np.sqrt(train_dist.shape[0]),
                  test_dist.mean().item(), test_dist.std().item() / np.sqrt(test_dist.shape[0])))

    return best_model, stats


if __name__ == "__main__":
    # SETUP AND LOAD ARGUMENTS, CREATE EMPTY MODEL
    args = setup_args()
    model = get_hnn(args)

    # LOAD DATA SET
    data_path = save_path(args, ext='shnndata')
    if args.new_data or not os.path.exists(data_path):
        print("Generating a new data set...")
        data_loader = args.data_class(args.h, args.noise)
        data = data_loader.get_dataset(seed=args.seed, samples=args.data_samples,
                                       test_split=args.test_split, print_args=args)
        print()
        to_pickle(data, data_path)
    else:
        print("Loading the existing data set...")
        data = from_pickle(data_path)

    # RUN THE MAIN FUNCTION TO TRAIN THE MODEL
    if not args.new_data:
        model, _ = train(model, data, args)

        # SAVE
        os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
        torch.save({'args': args, 'model': model.state_dict()}, save_path(args))
