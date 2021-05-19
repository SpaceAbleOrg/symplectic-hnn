# Symplectic Hamiltonian Neural Networks | 2021
# Florian Méhats and Marco David

import os
import torch
from joblib import Parallel, delayed

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from utils import save_path
from model.args import custom_product
from model.loss import choose_scheme
from model.hnn import HNN, CorrectedHNN

from train import setup, train_if_missing


def hamiltonian_error_grid(model, data_loader):
    dim = data_loader.dimension()
    n = dim//2

    cmin, cmax = data_loader.plot_boundaries()
    p, q = np.linspace(cmin, cmax), np.linspace(cmin, cmax)
    P, Q = np.meshgrid(*([p]*n), *([q]*n))

    y_flat = np.stack((P, Q), axis=-1).reshape(-1, 2)  # reshape (50, 50, 2) to (2500, 2)
    H_flat = model.forward(torch.tensor(y_flat, dtype=torch.float32, requires_grad=True)).detach().cpu().numpy()
    H = H_flat.reshape(*P.shape)

    # Calculate the correct Hamiltonian on the grid
    H_grid = np.array([data_loader.bundled_hamiltonian(y) for y in y_flat]).reshape(*P.shape)

    # Calculate the global constant up to which the model predicts H
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)
    H0_pred = model.forward(zero_tensor).detach().cpu().numpy()

    # Return the meshgrid space and the Hamiltonian error
    # VERSION 1
    return P, Q, H - H0_pred - H_grid

    # VERSION 2
    #diff = H - H_grid
    #return P, Q, diff - diff.mean()


def hamiltonian_error_sampled(model, data_loader, N=2000):
    # Create an array y0 of N samples in the shape (N, dim) where dim is the dim of the specific problem, i.e.
    dim = data_loader.dimension()
    y0_list = [data_loader.random_initial_value() for _ in range(N)]
    y0 = torch.tensor(np.array(y0_list), dtype=torch.float32, requires_grad=True)

    # Also create a zero tensor of shape (1, dim)
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)

    H_pred = model.forward(y0).detach().cpu().numpy()
    H0 = model.forward(zero_tensor).detach().cpu().numpy()
    H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in y0_list])

    # VERSION 1
    return H_pred - H0 - H_exact

    # VERSION 2
    #diff = H_pred - H_exact
    #return diff - diff.mean()
