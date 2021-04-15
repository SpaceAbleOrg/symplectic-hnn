# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import setup_args, save_path
from model.hnn import get_hnn


if __name__ == "__main__":
    tabh = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    taberr = np.zeros(len(tabh))

    for hi in range(len(tabh)):
        args = setup_args()
        args.h = tabh[hi]

        if tabh[hi] in [0.2, 0.1]:
            args.hidden_dim = 400

        model = get_hnn(args)
        model.load_state_dict(torch.load(save_path(args)))

        data_loader = args.data_class(args.h, args.noise)

        Ntest = 1000
        tab, tabH, tabH0, tabHy0 = np.zeros(Ntest), np.zeros(Ntest), np.zeros(Ntest), np.zeros(Ntest)
        for i in range(Ntest):
            y0 = data_loader.random_initial_value()
            # y0 = 2 * np.random.rand(2) - 1
            # y0 = y0 / np.linalg.norm(y0) * (0.1 + np.random.rand())
            # theta = 2 * np.pi * (np.random.rand() - 1 / 2)
            # p = 2 * np.random.rand() - 1.
            # y0 = np.array([p, theta])

            tabH[i] = model.forward(torch.tensor(y0, dtype=torch.float32)).data.numpy()
            tabH0[i] = model.forward(torch.tensor(0 * y0, dtype=torch.float32)).data.numpy()
            tabHy0[i] = data_loader.bundled_hamiltonian(y0)

        # taberr[hi]=np.mean(np.abs(tabH-np.mean(tabH)-(tabHy0-np.mean(tabHy0))))
        taberr[hi] = np.mean(np.abs(tabH - tabH0 - tabHy0))
        print('h: ', args.h, ', Error on the Hamiltonian: ', taberr[hi])

    plt.loglog(tabh, taberr, 'X-')
    plt.loglog(tabh, tabh, 'r-')
    plt.show()
