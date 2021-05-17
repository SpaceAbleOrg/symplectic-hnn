# Symplectic Hamiltonian Neural Networks | 2021
# Florian Méhats and Marco David

import os
import torch
from joblib import Parallel, delayed

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from utils import setup, save_path
from model.args import load_args, custom_product
from model.loss import choose_scheme
from model.hnn import HNN, CorrectedHNN
from train import train_main


def hamiltonian_error_grid(model, data_loader):
    cmin, cmax = data_loader.plot_boundaries()
    p, q = np.linspace(cmin, cmax), np.linspace(cmin, cmax)
    P, Q = np.meshgrid(p, q)

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


def hamiltonian_error_random(model, data_loader, N=2000):
    # Create an array of 1000 samples in the shape (1000, dim) where dim is the dim of the specific problem, i.e.
    dim = data_loader.dimension()

    y0_list = [data_loader.random_initial_value() for _ in range(N)]
    y0 = torch.tensor(np.array(y0_list), dtype=torch.float32, requires_grad=True)
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)

    H_pred = model.forward(y0).detach().cpu().numpy()
    H0 = model.forward(zero_tensor).detach().cpu().numpy()
    H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in y0_list])

    # VERSION 1
    return H_pred - H0 - H_exact

    # VERSION 2
    #diff = H_pred - H_exact
    #return diff - diff.mean()


def train_missing_model(args):
    args = setup(args)
    this_args = args | {'new_data': False}
    if not os.path.exists(save_path(this_args)):
        train_main(this_args)


# THIS FILE CANNOT BE RUN WITH 'prompt' AS ITS NAME.
if __name__ == "__main__":
    hs = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]
    methods = ['euler-forw', 'euler-symp', 'midpoint']
    errors_sampled, errors_corrected = [], []
    # TODO Average the errors on the mesh grid and compare to the random sample

    # TRAIN MISSING MODELS
    args_list = list(load_args(custom_prod=custom_product(h_list=hs, loss_type_list=methods)))
    _ = Parallel(n_jobs=-1, verbose=True)(delayed(train_missing_model)(args) for args in args_list)

    # Set up the subplots
    N = len(args_list) + 1
    fig = plt.figure(figsize=(6*N, 6), facecolor='white', dpi=300)
    ax = [fig.add_subplot(1, N, i+1, projection='3d') for i in range(N-1)]  # kwarg useful sometimes: aspect='equal'
    axN = fig.add_subplot(1, N, N, frameon=True)

    # Calculate the actual Hamiltonian errors
    for i, args in enumerate(args_list):
        args = setup(args)

        # Loads the model automatically, and rewrites all other args,
        # i.e. all except name, loss_type, h, noise, to match this model
        model, args = HNN.load(args)
        scheme = choose_scheme(args.loss_type)(args)
        corrected_model = CorrectedHNN.get(model, scheme, args.h)
        data_loader = args.data_class(args.h, args.noise)
        dim = data_loader.dimension()

        # TO BE CHANGED BY THE USER:
        use_model = model  #corrected_model

        # ----- MAP THE HAMILTONIAN ERROR ON A MESHGRID -----
        P, Q, H_err = hamiltonian_error_grid(use_model, data_loader)

        CS = ax[i].plot_surface(P, Q, H_err, cmap=cm.coolwarm)
        ax[i].set_title(f"Hamiltonian Error for $h={args.h}$ in phase space")
        ax[i].set_zlim(-1, 1)
        #ax[i].clabel(CS, inline=True, fontsize=10)

        # ----- SAMPLE HAMILTONIAN ERROR FROM RANDOM INITIAL VALUES -----
        H_err = hamiltonian_error_random(model, data_loader)
        H_err_corr = hamiltonian_error_random(corrected_model, data_loader)

        mean, std = np.abs(H_err).mean(), np.abs(H_err).std()
        errors_sampled.append((mean, std))
        print(f"h: {args.h}, Error on the Hamiltonian: {mean:.3f} ± {std:.3f}")

        mean_corr, std_corr = np.abs(H_err_corr).mean(), np.abs(H_err_corr).std()
        errors_corrected.append((mean_corr, std_corr))
        print(f"h: {args.h}, Error on the corrected Hamiltonian: {mean_corr:.3f} ± {std_corr:.3f}")

    # Error Plot for all h's
    err = np.array(errors_sampled)
    errc = np.array(errors_corrected)
    axN.loglog(hs, hs, '--', color='lightgrey', label=r'$\varepsilon = h$')  # y = px, straight line with slope of order p=1
    axN.loglog(hs, [h**2 for h in hs], '--', color='lightgrey', label=r'$\varepsilon = h^2$')  # same for p=2
    axN.errorbar(hs, err[:, 0], yerr=err[:, 1], fmt='o-', label=r'$\varepsilon_H$')
    axN.errorbar(hs, errc[:, 0], yerr=errc[:, 1], fmt='o-', label=r'$\varepsilon_{\tilde H}$')

    axN.set_xlabel("Discretization Step $h$", fontsize=14)
    axN.set_ylabel(r"Error $\varepsilon$", rotation=0, fontsize=14)
    axN.set_title("Average Error of Hamiltonian vs. Time Step $h$ \n (Averaged over the relevant phase space $\Omega$)",
                  pad=10)
    axN.legend()

    fig.tight_layout()
    plt.savefig(save_path(args_list[0], pltname='herr', ext='pdf', incl_h=False))
    plt.show()
