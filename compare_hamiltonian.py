# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils import setup_args, save_path
from model.standard_nn import MLP
from model.hnn import HNN
from model.data import get_t_eval


def load_model(args):
    # Create the standard MLP with args.dim inputs and one single scalar output, the Hamiltonian
    nn_model = MLP(args.dim, args.dim_hidden_layer, output_dim=1, nonlinearity=args.nonlinearity)
    # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
    model = HNN(nn_model)
    # Load saved state using the standard save_path
    model.load_state_dict(torch.load(save_path(args)))

    return model


if __name__ == "__main__":
    tabh=[0.8,0.4,0.2,0.1,0.05,0.025]
    taberr=np.zeros(len(tabh))
    for hi in range(len(tabh)):
        args = setup_args()
        args.h=tabh[hi]
        if args.h==0.025:
            args.dim_hidden_layer=300
        model = load_model(args)
        data_loader = args.data_class(args.h, args.noise)

        Ntest=1000
        tab=np.zeros(Ntest)
        tabH=np.zeros(Ntest)
        tabH0 = np.zeros(Ntest)
        tabHy0=np.zeros(Ntest)
        for i in range(Ntest):
            #y0 = 2 * np.random.rand(2) - 1
            #y0 = y0 / np.linalg.norm(y0) * (0.1 + np.random.rand())
            theta = 2 * np.pi * (np.random.rand() - 1 / 2)
            p = 2 * np.random.rand() - 1.
            y0=np.array([p, theta])

            tabH[i] = model.forward(torch.tensor(y0, dtype=torch.float32)).data.numpy()
            tabH0[i] = model.forward(torch.tensor(0 * y0, dtype=torch.float32)).data.numpy()
            tabHy0[i] = data_loader.bundled_hamiltonian(y0)
        #taberr[hi]=np.mean(np.abs(tabH-np.mean(tabH)-(tabHy0-np.mean(tabHy0))))
        taberr[hi] = np.mean(np.abs(tabH - tabH0 - tabHy0))
        print('h: ',args.h,', Error on the Hamiltonian: ',taberr[hi])
    plt.loglog(tabh,taberr,'X-')
    plt.loglog(tabh,tabh,'r-')
    plt.show()
