# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import os
import torch
import numpy as np

from model.standard_nn import MLP
from model.hnn import HNN
from utils import setup_args, choose_loss, choose_data, save_path


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

    if args.verbose:
        print("Generating data set...")

    # LOAD DATASET AND PREPARE TENSOR OBJECTS
    print_every = args.print_every if args.verbose else None
    data = data_loader.get_dataset(seed=args.seed, samples=args.data_samples,
                                   test_split=args.test_split, print_every=print_every)
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, dim)
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, dim)
    t = data['t']
    test_t = data['test_t']

    if args.verbose:
        print("Begin training loop...")

    # DO VANILLA TRAINING LOOP
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):

        # TODO split training data into smaller batches for faster training
        #       open question: do smaller batches also yield better training (or do the gradients just add up, for
        #       the same overall effect during the gradient descent)

        # train step, find loss and optimize
        model.train()
        loss = loss_fct(model, x, t)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        model.eval()
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


if __name__ == "__main__":
    # SETUP AND LOAD ARGUMENTS
    args = setup_args()

    # RUN THE MAIN FUNCTION
    model, _ = train(args)

    # SAVE
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    torch.save(model.state_dict(), save_path(args))
