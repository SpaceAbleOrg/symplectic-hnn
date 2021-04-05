# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(description=None)

    # MAIN ARGUMENT (non-optional)
    parser.add_argument('name', type=str, help='choose the system and data set')

    # DEFAULT ML ARGUMENTS (all optional)
    #parser.add_argument('--dim', default=dimension, type=int, help='dimensionality of input tensor')
    # The dimension (of the input tensor) is now determined by the chosen system/data set via the 'name' arg
    parser.add_argument('--dim_hidden_layer', default=200, type=int, help='dimension of each hidden layer of the MLP')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')

    # SCIENTIFIC ARGUMENTS (all optional)
    parser.add_argument('--loss_type', default='midpoint', type=str,
                        help='choose the symplectic integration method used in training')
    parser.add_argument('--noise', default=0.0, type=float, help='how much noise to include in the training data')
    parser.add_argument('--h', default=0.1, type=float, help='time step between data points in the training data')

    # OTHER ARGUMENTS (all optional)
    # Use time in seconds as random seed by default
    parser.add_argument('--seed', default=int(time.time()), type=int, help='random seed')
    parser.add_argument('--save_dir', default=None, type=str, help='where to save the trained model')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')

    return parser.parse_args()
