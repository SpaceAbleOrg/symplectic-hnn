# Symplectic Hamiltonian Neural Networks | 2021
# Florian Méhats and Marco David

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import setup_args, save_path, load_model


if __name__ == "__main__":
    hs = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    errors = np.zeros(len(hs))

    args = setup_args()

    for i, h in enumerate(hs):
        args.h = h

        # Loads the model automatically, and rewrites all other args,
        # i.e. all except name, loss_type, h, noise, to match this model
        model, args = load_model(args)

        data_loader = args.data_class(args.h, args.noise)

        N = 1000
        tab, tabH, tabH0, tabHy0 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        for j in range(N):
            y0 = data_loader.random_initial_value()

            tabH[j] = model.forward(torch.tensor(y0, dtype=torch.float32)).data.numpy()
            tabH0[j] = model.forward(torch.tensor(0 * y0, dtype=torch.float32)).data.numpy()
            tabHy0[j] = data_loader.bundled_hamiltonian(y0)

        # errors[hi] = np.mean(np.abs(tabH - np.mean(tabH) - (tabHy0-np.mean(tabHy0))))
        errors[i] = np.mean(np.abs(tabH - tabH0 - tabHy0))
        print(f"h: {h}, Error on the Hamiltonian: {errors[i]}")

    plt.loglog(hs, errors, 'X-')
    plt.loglog(hs, hs, 'r-')  # y = x, straight line

    plt.xlabel("$h$", fontsize=14)
    plt.ylabel(r"$\varepsilon_H$", rotation=0, fontsize=14)
    plt.title("Average Error of Hamiltonian vs. Time Step $h$ \n (Averaged over the relevant phase space $\Omega$)",
              pad=10)

    plt.savefig(save_path(args, pltname='herr', ext='pdf'))
    plt.show()
