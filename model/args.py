# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

import time
import itertools

from argparse import ArgumentParser, Namespace


class UpdatableNamespace(Namespace):
    @staticmethod
    def get(namespace):
        return UpdatableNamespace(**namespace.__dict__)

    def __or__(self, other):
        if isinstance(other, Namespace):
            return UpdatableNamespace(**(self.__dict__ | other.__dict__))
        elif isinstance(other, dict):
            return UpdatableNamespace(**(self.__dict__ | other))
        else:
            raise TypeError(f"unsupported operand type(s) for |: '{type(self)}' and '{type(other)}'")


def get_args():
    parser = ArgumentParser(description=None)

    # MAIN ARGUMENT (non-optional)
    parser.add_argument('name', type=str, help='choose the system and data set')

    # DEFAULT ML ARGUMENTS (all optional)
    #parser.add_argument('--dim', default=dimension, type=int, help='dimensionality of input tensor')
    # The dimension (of the input tensor) is now determined by the chosen system/data set via the 'name' arg
    parser.add_argument('--hidden_layers', default=1, type=int, help='number of hidden layers of the MLP')
    parser.add_argument('--hidden_dim', default=200, type=int, help='dimension of each hidden layer of the MLP')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--batch_size', default=64, type=int, help='number of data points used together in one batch')

    # SCIENTIFIC ARGUMENTS (all optional)
    parser.add_argument('--loss_type', default='midpoint', type=str,
                        help='choose the symplectic integration method used in training')
    parser.add_argument('--noise', default=0.0, type=float, help='how much noise to include in the training data')
    parser.add_argument('--h', default=0.1, type=float, help='time step between data points in the training data')
    parser.add_argument('--new_data', dest='new_data', action='store_true',
                        help='do not use stored dataset, re-create a full dataset from scratch instead')
    parser.add_argument('--data_samples', default=3000, type=int, help='number of samples in datasets')
    parser.add_argument('--test_split', default=0.2, type=float, help='percentage size of testing/validation dataset')

    # OTHER ARGUMENTS (all optional)
    # Use time in seconds as random seed by default
    parser.add_argument('--seed', default=int(time.time()), type=int, help='random seed')
    parser.add_argument('--save_dir', default=None, type=str, help='where to save the trained model')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')

    args = parser.parse_args()  # is a Namespace object
    return UpdatableNamespace.get(args)


def process_list(input_string):
    return map(str.strip, input_string.split(','))


def custom_product(name_list=(None,), loss_type_list=(None,), h_list=(None,), noise_list=(None,)):
    for name, loss_type, h, noise in itertools.product(name_list, loss_type_list, h_list, noise_list):
        args = {}
        if name:
            args['name'] = name
        if loss_type:
            args['loss_type'] = loss_type
        if h:
            args['h'] = float(h)
        if noise:
            args['noise'] = float(noise)
        yield args


def prompt():
    name = input("Which model (data set name) do you want to use ?")

    loss_type_list = process_list(input("Which numerical method for training (default midpoint) ?"))
    h_list = process_list(input("Which step size h (default 0.1) ?"))
    # noise = process_list(input("Which level of noise (default none) ?"))

    yield from custom_product(name_list=(name,), loss_type_list=loss_type_list, h_list=h_list)


def load_args(custom_prod=None):
    """ Loads all possible combinations of arguments provided by the user. Returns a generator object. """
    # Load arguments
    args = get_args()

    # Allow for prompt
    if args.name == "prompt":
        for prompt_res in prompt():
            yield args | prompt_res
            # read the dict union as: args, updated and overwritten with the keys/values from prompt_res
    elif custom_prod is not None:
        for custom_args in custom_prod:
            yield args | custom_args
            # read the dict union as: args, updated and overwritten with the keys/values from custom_args
    else:
        yield args
