import pickle
import torch
from torch.nn import functional


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
