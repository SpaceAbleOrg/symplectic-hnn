# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

# NOTE: This file needs to remain in the top-level project directory!

import os
import sys
import pickle
import itertools
from argparse import Namespace
import numpy as np
import torch
from torch.nn import functional
# import zipfile
# import imageio
# import shutil
# from PIL import Image

from model.args import get_args
from model.data import HarmonicOscillator, NonlinearPendulum, FermiPastaUlam


def process_list(input_string):
    return map(str.strip, input_string.split(','))


def prompt():
    name = input("Which model (data set name) do you want to use ?")

    loss_type_list = process_list(input("Which numerical method for training (default midpoint) ?"))
    h_list = process_list(input("Which step size h (default 0.1) ?"))
    # noise = process_list(input("Which level of noise (default none) ?"))

    args_list = []
    for loss_type, h in itertools.product(loss_type_list, h_list):
        args = {'name': name}
        if loss_type:
            args['loss_type'] = loss_type
        if h:
            args['h'] = float(h)
        # if noise:
        #    args['noise'] = float(noise)

        args_list.append(args)

    return args_list


def load_args():
    """ Loads all possible combinations of arguments provided by the user. Returns a generator object. """
    # Load arguments
    args = get_args()

    if isinstance(args, Namespace):
        args = args.__dict__

    # Allow for prompt
    if args['name'] == "prompt":
        for prompt_res in prompt():
            yield Namespace(**(args | prompt_res))
            # read the dict union as: args, updated and overwritten with the keys/values from prompt_res
    else:
        yield Namespace(**args)


# This function is generic, but needs to run in a top-level file to setup the path variables
# and define the save_directory properly.
def setup(args):
    # Setup directory of this file as working (save) directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)
    sys.path.append(parent_dir)

    # Set the save directory if nothing is given
    if not args.save_dir:
        args.save_dir = this_dir + '/experiment-' + args.name

    # Store data_class directly in args for future access, and dimension for future convenience (eg of loss functions)
    args.data_class = choose_data(args.name)
    args.dim = args.data_class.dimension()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


# def rk4(fun, y0, t, dt, *args, **kwargs):
#     dt2 = dt / 2.0
#     k1 = fun(y0, t, *args, **kwargs)
#     k2 = fun(y0 + dt2 * k1, t + dt2, *args, **kwargs)
#     k3 = fun(y0 + dt2 * k2, t + dt2, *args, **kwargs)
#     k4 = fun(y0 + dt * k3, t + dt, *args, **kwargs)
#     dy = dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
#     return dy
#
#
# def read_lipson(experiment_name, save_dir):
#     desired_file = experiment_name + ".txt"
#     with zipfile.ZipFile('{}/invar_datasets.zip'.format(save_dir)) as z:
#         for filename in z.namelist():
#             if desired_file == filename and not os.path.isdir(filename):
#                 with z.open(filename) as f:
#                     data = f.read()
#     return str(data)
#
#
# def str2array(string):
#     lines = string.split('\\n')
#     names = lines[0].strip("b'% \\r").split(' ')
#     dnames = ['d' + n for n in names]
#     names = ['trial', 't'] + names + dnames
#     data = [[float(s) for s in l.strip("' \\r,").split()] for l in lines[1:-1]]
#
#     return np.asarray(data), names


def to_pickle(thing, path):  # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing


def choose_helper(dict, name, choose_what="Input"):
    if name in dict.keys():
        return dict[name]
    else:
        raise ValueError(f"{choose_what} not recognized: {name}. Possibilities are: " + ", ".join(dict.keys()))


def choose_nonlinearity(name):
    nonlinearities = {'tanh': torch.tanh,
                      'relu': torch.relu,
                      'sigmoid': torch.sigmoid,
                      'softplus': torch.nn.functional.softplus,
                      'selu': torch.nn.functional.selu,
                      'elu:': torch.nn.functional.elu,
                      'swish': lambda x: x * torch.sigmoid(x)
                      }

    return choose_helper(nonlinearities, name, choose_what="Nonlinearity")


# This function cannot be moved inside data.py because that would cause a circular import!
def choose_data(name):
    datasets = {'spring': HarmonicOscillator,
                'pendulum': NonlinearPendulum,
                'fpu': FermiPastaUlam  # FPU = Fermi-Pasta-Ulam (see GNI book)
                }

    return choose_helper(datasets, name, choose_what="Data set name")


def save_path(args, pltname='', ext='tar', incl_h=True, incl_loss=True):
    label = args.name
    if incl_h:
        label += '-h' + str(args.h)
    if incl_loss:
        label += '-' + args.loss_type
    if pltname:
        label = pltname + '-' + label
    if args.noise > 0:
        label += '-n' + str(args.noise)
    return '{}/{}.{}'.format(args.save_dir, label, ext)


# def make_gif(frames, save_dir, name='pendulum', duration=1e-1, pixels=None, divider=0):
#     '''Given a three dimensional array [frames, height, width], make
#     a gif and save it.'''
#     temp_dir = './_temp'
#     os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
#     for i in range(len(frames)):
#         im = (frames[i].clip(-.5, .5) + .5) * 255
#         im[divider, :] = 0
#         im[divider + 1, :] = 255
#         if pixels is not None:
#             im = Image.fromarray(im).resize(pixels)  # TODO Test – Line edited because deprecated functions were removed
#         imageio.imwrite(temp_dir + '/f_{:04d}.png'.format(i), im)  # TODO Test – Line also edited, see above
#
#     images = []
#     for file_name in sorted(os.listdir(temp_dir)):
#         if file_name.endswith('.png'):
#             file_path = os.path.join(temp_dir, file_name)
#             images.append(imageio.imread(file_path))
#     save_path = '{}/{}.gif'.format(save_dir, name)
#     png_save_path = '{}.png'.format(save_path)
#     imageio.mimsave(save_path, images, duration=duration)
#     os.rename(save_path, png_save_path)
#
#     shutil.rmtree(temp_dir)  # remove all the images
#     return png_save_path
