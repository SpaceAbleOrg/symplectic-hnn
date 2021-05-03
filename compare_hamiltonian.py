# Symplectic Hamiltonian Neural Networks | 2021
# Florian Méhats and Marco David

import os
import torch
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt

from utils import setup, load_args, save_path
from model.loss import choose_scheme
from model.hnn import HNN, CorrectedHNN
from train import train_main


def hamiltonian_error_grid(model, data_loader):
    cmin, cmax = data_loader.plot_boundaries()
    p, q = np.linspace(cmin, cmax), np.linspace(cmin, cmax)
    P, Q = np.meshgrid(p, q)

    y_flat = np.stack((P, Q), axis=-1).reshape(-1, 2)  # reshape (50, 50, 2) to (2500, 2)
    H_flat = model.forward(torch.tensor(y_flat, dtype=torch.float32, requires_grad=True)).detach().numpy()
    H = H_flat.reshape(*P.shape)

    # Calculate the correct Hamiltonian on the grid
    H_grid = np.array([data_loader.bundled_hamiltonian(y) for y in y_flat]).reshape(*P.shape)

    # Calculate the global constant up to which the model predicts H
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)
    H0_pred = model.forward(zero_tensor).data.numpy()

    # Return the meshgrid space and the Hamiltonian error
    return P, Q, H - H0_pred - H_grid


def hamiltonian_error_random(model, data_loader, N=2000):
    # Create an array of 1000 samples in the shape (1000, dim) where dim is the dim of the specific problem, i.e.
    dim = data_loader.dimension()

    y0_list = [data_loader.random_initial_value() for j in range(N)]
    y0 = torch.tensor(np.array(y0_list), dtype=torch.float32, requires_grad=True)
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)

    H_pred = model.forward(y0).detach().numpy()
    H0 = model.forward(zero_tensor).detach().numpy()
    H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in y0_list])

    return H_pred - H0 - H_exact


# THIS FILE CANNOT BE RUN WITH 'prompt' AS ITS NAME.
if __name__ == "__main__":
    hs = [0.8, 0.4, 0.2, 0.1, 0.05]
    errors_sampled = []
    # TODO Average the errors on the mesh grid and compare to the random sample

    # TODO Rewrite using generator power like in load_args... Incorporate the below parallelization code in one logic
    args = setup(next(load_args()))

    # Train the models in parallel (!) if they do not exist
    def f(h):
        args.h = h
        return not os.path.exists(save_path(args))

    def g(h):
        args.h = h
        args.new_data = False
        args.verbose = False
        train_main(args)

    h_missing = list(filter(f, hs))
    print(h_missing)
    results = Parallel(n_jobs=-1, verbose=True)(delayed(g)(h) for h in h_missing)

    # Set up the subplots
    N = len(hs)+1
    fig = plt.figure(figsize=(6*N, 6), facecolor='white', dpi=300)
    ax = [fig.add_subplot(1, N, i+1, frameon=True) for i in range(N)]  # kwarg useful sometimes: aspect='equal'

    # Calculate the actual Hamiltonian errors
    for i, h in enumerate(hs):
        args.h = h

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

        CS = ax[i].contour(P, Q, H_err)
        ax[i].set_title(f"Hamiltonian Error for $h={h}$ in phase space")
        ax[i].clabel(CS, inline=True, fontsize=10)

        # ----- SAMPLE HAMILTONIAN ERROR FROM RANDOM INITIAL VALUES -----
        H_err = hamiltonian_error_random(use_model, data_loader)

        mean, std = np.abs(H_err).mean(), np.abs(H_err).std()
        errors_sampled.append((mean, std))
        print(f"h: {h}, Error on the Hamiltonian: {mean:.3f} ± {std:.3f}")

    # Error Plot for all h's
    err = np.array(errors_sampled)
    ax[N-1].loglog(hs, [h**2 for h in hs], 'r-')  # y = px, straight line with slope of order p
    ax[N-1].errorbar(hs, err[:, 0], yerr=err[:, 1], fmt='X-')

    ax[N-1].set_xlabel("$h$", fontsize=14)
    ax[N-1].set_ylabel(r"$\varepsilon_H$", rotation=0, fontsize=14)
    ax[N-1].set_title("Average Error of Hamiltonian vs. Time Step $h$ \n (Averaged over the relevant phase space $\Omega$)",
              pad=10)

    fig.tight_layout()
    plt.savefig(save_path(args, pltname='herr', ext='pdf', incl_h=False))
    plt.show()
