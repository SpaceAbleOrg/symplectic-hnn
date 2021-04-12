# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

# NOTE: This file needs to remain in the top-level project directory!

import os
import sys
import pickle
import zipfile
import imageio
import shutil
from PIL import Image

from model.args import get_args
from model.loss import *
from model.data import *


# This function is generic, but needs to run in a top-level file to setup the path variables
# and define the save_directory properly.
def setup_args():
    # Load arguments
    args = get_args()

    # Allow for prompt
    if args.name == "prompt":
        args.name = input("Which model (data set name) do you want to use ?")
        loss_type = input("Which numerical method for training (default midpoint) ?")
        if loss_type:
            args.loss_type = loss_type
        h = input("Which step size h (default 0.1) ?")
        if h:
            args.h = float(h)
        noise = input("Which level of noise (default none) ?")
        if noise:
            args.noise = float(noise)

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


# TODO Replace all these giant cases by dictionaries and simply index them!
def choose_nonlinearity(name):
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


# TODO Replace all these giant cases by dictionaries and simply index them!
#       Maybe even register the string-format name automatically from within the respective classes...
def choose_loss(name):
    if name == 'euler-symp':
        loss = EulerSympLoss
    elif name == 'midpoint':
        loss = MidpointLoss
    else:
        raise ValueError("loss function not recognized")
    return loss


# TODO Replace all these giant cases by dictionaries and simply index them!
#       Maybe even register the string-format name automatically from within the respective classes...
def choose_data(name):
    if name == 'spring':
        data_loader = HarmonicOscillator
    elif name == 'pendulum':
        data_loader = NonlinearPendulum
    elif name == 'fpu':  # FPU = Fermi-Pasta-Ulam (see GNI book)
        data_loader = FermiPastaUlam
    else:
        raise ValueError("data set not recognized")
    return data_loader


def save_path(args, ext='tar'):
    label = args.name + '-' + args.loss_type + '-h' + str(args.h)
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
