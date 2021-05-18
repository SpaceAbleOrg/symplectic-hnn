# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski


import pickle
import torch
from torch.nn import functional

# import zipfile
# import imageio
# import shutil
# from PIL import Image


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


def process_list(input_string):
    return map(str.strip, input_string.split(','))


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
