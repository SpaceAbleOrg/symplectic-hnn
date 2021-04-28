# Symplectic Hamiltonian Neural Networks | 2021
# Florian Méhats and Marco David

import os
import torch
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt

from utils import setup_args, save_path
from model.hnn import load_model
from train import main


if __name__ == "__main__":
    hs = [0.8, 0.4, 0.2, 0.1, 0.05]
    errors_sampled = np.zeros(len(hs))
    errors_grid = np.zeros(len(hs))

    args = setup_args()

    # Train the models in parallel (!) if they do not exist
    def f(h):
        args.h = h
        return not os.path.exists(save_path(args))

    def g(h):
        args.h = h
        args.new_data = False
        args.verbose = False
        main(args)

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
        model, args = load_model(args)
        data_loader = args.data_class(args.h, args.noise)

        # ----- MAP THE HAMILTONIAN ERROR ON A MESHGRID -----
        cmin, cmax = data_loader.plot_boundaries()
        p, q = np.linspace(cmin, cmax), np.linspace(cmin, cmax)
        P, Q = np.meshgrid(p, q)
        y_flat = np.stack((P, Q), axis=-1).reshape(-1, 2)  # reshape (50, 50, 2) to (2500, 2)
        H_flat = model.forward(torch.tensor(y_flat, dtype=torch.float32)).detach().numpy()
        H = H_flat.reshape(*P.shape)

        # Calculate the correct Hamiltonian on the grid AND the global constant up to which the model predicts H
        H_grid = np.array([data_loader.bundled_hamiltonian(y) for y in y_flat]).reshape(*P.shape)
        H0 = model.forward(torch.tensor(np.zeros(data_loader.dimension()), dtype=torch.float32)).data.numpy()

        H_err = H - H0 - H_grid
        CS = ax[i].contour(P, Q, H_err)
        ax[i].set_title(f"Hamiltonian Error for $h={h}$ in phase space")
        ax[i].clabel(CS, inline=True, fontsize=10)

        # ----- SAMPLE HAMILTONIAN ERROR FROM RANDOM INITIAL VALUES -----
        # Create an array of 1000 samples in the shape (1000, dim) where dim is the dim of the specific problem,
        # accessible e.g. via data_loader.dimension().
        y0_list = [data_loader.random_initial_value() for j in range(1000)]
        y0 = np.array(y0_list)
        H_pred = model.forward(torch.tensor(y0, dtype=torch.float32)).data.numpy()
        H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in y0_list])

        # TODO What does this next line do; how does the calculation differ ?
        # errors[hi] = np.mean(np.abs(tabH - np.mean(tabH) - (tabHy0-np.mean(tabHy0))))

        error_list = H_pred - H0 - H_exact
        errors_sampled[i] = np.mean(np.abs(error_list))
        print(f"h: {h}, Error on the Hamiltonian: {errors_sampled[i]}")

    # Error Plot for all h's
    ax[N-1].loglog(hs, errors_sampled, 'X-')
    ax[N-1].loglog(hs, hs, 'r-')  # y = x, straight line

    ax[N-1].set_xlabel("$h$", fontsize=14)
    ax[N-1].set_ylabel(r"$\varepsilon_H$", rotation=0, fontsize=14)
    ax[N-1].set_title("Average Error of Hamiltonian vs. Time Step $h$ \n (Averaged over the relevant phase space $\Omega$)",
              pad=10)

    fig.tight_layout()
    plt.savefig(save_path(args, pltname='herr', ext='pdf', incl_h=False))
    plt.show()
