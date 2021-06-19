from torch import cuda
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

        If no custom argument combinations are given, the generator will only yield one object containing
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


def train_parallel(arg_iterable, n_jobs=4, joblib_verbose=False):
    if n_jobs > 4:
        print("WARNING: ")
    Parallel(n_jobs=n_jobs, verbose=joblib_verbose)(delayed(train_main)(args) for args in arg_iterable)


if __name__ == "__main__":
    # On the Virtual Private Server used to train these models, there is only 1 GPU, so parallelizing has no advantage.
    # Thus, only parallelize when CUDA is not available. A single run (model + dataset) easily uses 2-3 GB of memory,
    # so keep n_jobs <= 4 in case of a CPU and parallelization.
    if cuda.is_available():
        print("Training on GPU. Assuming only 1 kernel is available. Hence, will train all models sequentially.")
        for args in load_args(custom_prod=prompt()):
            print("=====================================")
            print(f"Next model: Now training {args.name}, h={args.h} with {args.loss_type}...")
            train_main(args)
    else:
        print("Training on CPU. Will parallelize the training of all possible parameter combinations.")
        train_parallel(load_args(custom_prod=prompt()))
