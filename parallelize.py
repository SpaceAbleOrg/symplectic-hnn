# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

from joblib import Parallel, delayed

from model.args import get_args, custom_product

from utils import process_list
from train import train_main


def prompt():
    name_list = process_list(input("Which model (data set name) do you want to use ?"))
    loss_type_list = process_list(input("Which numerical method for training (default midpoint) ?"))
    h_list = process_list(input("Which step size h (default 0.1) ?"))
    # noise = process_list(input("Which level of noise (default none) ?"))

    yield from custom_product(name_list=name_list, loss_type_list=loss_type_list, h_list=h_list)


def load_args(custom_prod=None, base_args=None):
    """ Loads all possible combinations of arguments provided by the user. Returns a generator object.

        f no custom argument combinations are given, the generator will only yield one object containing
         the arguments passed via the command line. """
    if not base_args:
        # Load arguments
        base_args = get_args()

    if not custom_prod:
        # If none, just yield one time, the value 'yield args | {}' (i.e. args updated by nothing)
        # In this case, load_args() returns a generator that contains one element, which is just get_args()
        custom_prod = [{}]

    for custom_args in custom_prod:
        yield base_args | custom_args
        # read the dict union '|' as: args, updated and overwritten with the keys/values from custom_args


def train_parallel(arg_iterable, joblib_verbose=False):
    # n_jobs = 6 since the VPS has six vCPUs
    Parallel(n_jobs=6, verbose=joblib_verbose)(delayed(train_main)(args) for args in arg_iterable)


if __name__ == "__main__":
    train_parallel(load_args(custom_prod=prompt()))
