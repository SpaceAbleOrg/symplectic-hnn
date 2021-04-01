# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data import get_dataset
from utils import L2_loss, rk4

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=200, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=2000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='spring', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--noise', default=0.1, type=float, help='how much noise to include in the training data')
    parser.add_argument('--delta_t', default=0.1, type=float, help='timestep between data points used to train')
    parser.add_argument('--symplectic_euler', dest='symplectic_euler', action='store_true', help='choose training method')
    parser.add_argument('--symplectic_midpoint', dest='symplectic_midpoint', action='store_true', help='choose training method')

    parser.set_defaults(feature=True)
    return parser.parse_args()


def shift_data(tens):
    # asserts dimension [nb of samples, nb of data points, 2] for now
    new_tens = torch.stack((tens[:, :-1, 0], tens[:, 1:, 1]), dim=-1)
    return new_tens  # returns one line less: shape [nb of samples, nb of data points - 1, 2]


def calc_single_loss(x, t, model, args, return_dist=False):
    h = torch.tensor(np.diff(t, axis=1)).float()  # t[:, 1:] - t[:, :-1]  # shape (batch=25, t-evals)

    # Calculate finite differences as approximation of the derivatives
    # -> The division of two tensors with a different number of axes requires the extra dimension to be the first one,
    # -> not the last like one would intuitively expect. Hence we need this bulky permutation business.
    diff = (x[:, 1:] - x[:, :-1]).permute(2, 0, 1)
    dxdt_hat = (diff / h).permute(1, 2, 0)

    # Calculate prediction for derivatives of neural network
    if args.symplectic_euler:
        x = shift_data(x)
    elif args.symplectic_midpoint:
        x = (x[:, 1:] + x[:, :-1]) / 2

    x_flat = x.flatten(0, 1)

    dxdt_flat = model.rk4_time_derivative(x_flat) if args.use_rk4 else model.time_derivative(x_flat)
    dxdt = dxdt_flat.view(x.shape)

    if (not args.symplectic_euler) and (not args.symplectic_midpoint):
        dxdt = dxdt[:, 1:, :]  # cut off one element from the list, due to finite differences used.
        # If args.symplectic is true, this was already done during shift_data above.

    loss = L2_loss(dxdt, dxdt_hat, mean=not return_dist)

    return loss


def train(args):
  # set random seed
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # init model and optimizer
  if args.verbose:
    print("Training baseline model:" if args.baseline else "Training HNN model:")

  output_dim = args.input_dim if args.baseline else 2
  nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
  model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=args.baseline)
  optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

  # arrange data
  noise_std = args.noise
  data = get_dataset(seed=args.seed, noise_std=noise_std, delta_t=args.delta_t)

  x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, 2) ???
  test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)  # shape (batch_size, 2) ???

  t = data['t']
  test_t = data['test_t']

  # vanilla train loop
  stats = {'train_loss': [], 'test_loss': []}
  for step in range(args.total_steps+1):

    # train step, find loss and optimize
    loss = calc_single_loss(x, t, model, args)
    loss.backward() ; optim.step() ; optim.zero_grad()

    # run test data
    test_loss = calc_single_loss(test_x, test_t, model, args)

    # logging
    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    if args.verbose and step % args.print_every == 0:
      print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

  train_dist = calc_single_loss(x, t, model, args, return_dist=True)
  test_dist = calc_single_loss(test_x, test_t, model, args, return_dist=True)
  print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
    .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
            test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

  return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    if args.symplectic_euler:
        label += '-syeu'
    elif args.symplectic_midpoint:
        label += '-symp'
    else:
        label += '-fweu'
    label += '-n' + str(args.noise)
    label += '-t' + str(args.delta_t)
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)