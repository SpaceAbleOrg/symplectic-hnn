# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

from sklearn.model_selection import train_test_split

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    H = p**2 + q**2 # spring hamiltonian (linear oscillator)
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(t_span=[0,3], radius=None, y0=None, noise_std=0.1, delta_t=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])/delta_t))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2-1
    if radius is None:
        radius = np.random.rand()*0.9 + 0.1 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius, allows for y0 to just give a direction (i.e. point on unit circle)

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]

    # EDIT: Numerical "Derivatives" i.e. finite differences
    # Using symmetric difference quotients here, to prevent bias either way
    # This also allows to equally delete the first _and_ last data point from q and p

    # NOT NECESSARY ANYMORE ?
    # TODO ERROR: The Symplectic training of a geometric HNN requires first-differences to actually be
    # TODO          equivalent to symplectic integrations (although I don't know which one, right or left first differences ?)
    #dqdt = (q[2:] - q[:-2]) / (t_eval[2:] - t_eval[:-2])
    #dpdt = (p[2:] - p[:-2]) / (t_eval[2:] - t_eval[:-2])

    # Original Method: Analytic Derivative using dynamics_fn
    #dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    #dydt = np.stack(dydt).T
    #dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    #return q[1:-1], p[1:-1], dqdt, dpdt, t_eval[1:-1]
    return q, p, t_eval

#def get_dataset(seed=0, samples=50, test_split=0.5, geometric_shift=False, noise_std=0.1, **kwargs):
def get_dataset(seed=0, samples=50, test_split=0.2, noise_std=0.1, delta_t=0.1, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs, ts = [], [], []
    for s in range(samples):
        q, p, t = get_trajectory(noise_std=noise_std, delta_t=delta_t, **kwargs)
        #print(x.shape, y.shape, dx.shape, dy.shape)
        #if geometric_shift:
        #    xs.append(np.stack([x[1:], y[:-1]]).T)
        #    dxs.append(np.stack([dx[1:], dy[:-1]]).T)
        #else:
        xs.append(np.stack([q, p]).T)
        #dxs.append(np.stack([dx, dy]).T)
        ts.append(t)
        
    data['x'] = np.array(xs)
    #data['dx'] = np.array(dxs)#.squeeze()

    # Also add t to the data (likely all rows will be the same, though)
    data['t'] = np.array(ts)

    #print(data['x'].shape, data['t'].shape)  # debug

    # make a train/test split
    t_train, t_test, x_train, x_test = train_test_split(data['t'], data['x'], test_size=test_split)
    split_data = {'t': t_train, 'test_t': t_test, 'x': x_train, 'test_x': x_test}
    return split_data

def get_analytic_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field