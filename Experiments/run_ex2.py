import jax
from jax import random, grad, jit, vmap
from jax.config import config
from jax.lib import xla_bridge
import jax.numpy as np
# import neural_tangents as nt
# from neural_tangents import stax
from jax.example_libraries import stax
from jax.example_libraries import optimizers
import os

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import pandas as pd

import time

import numpy as onp


# %%
# Utils

fplot = lambda x : np.fft.fftshift(np.log10(np.abs(np.fft.fft(x))))

# Signal makers

def sample_random_signal(key, decay_vec):
  N = decay_vec.shape[0]
  raw = random.normal(key, [N, 2]) @ np.array([1, 1j])
  signal_f = raw * decay_vec
  signal = np.real(np.fft.ifft(signal_f))
  return signal


# Network \
def progression(m, max = 50):
    omega = np.arange(m) / (m - 1)
    return max ** omega

progression(3, 15/2.46)

def make_network(num_layers, num_channels, ntk_params=True, num_outputs=1):
  layers = []
  for i in range(num_layers-1):
    if ntk_params:
        layers.append(stax.Dense(num_channels))
    else:
        layers.append(stax.Dense(num_channels))
    layers.append(stax.Relu)
  layers.append(stax.Dense(num_outputs))
  return stax.serial(*layers)

# Encoding 

def compute_ntk(x, avals, bvals, kernel_fn):
    x1_enc = input_encoder(x, avals, bvals)
    x2_enc = input_encoder(np.array([0.], dtype=np.float32), avals, bvals)
    out = np.squeeze(kernel_fn(x1_enc, x2_enc, 'ntk'))
    return out


input_encoder = lambda x, a, b: np.concatenate([a * np.sin((2.*np.pi*x[...,None]) * b), 
                                                a * np.cos((2.*np.pi*x[...,None]) * b)], axis=-1) / np.linalg.norm(a)
# input_encoder_2 = lambda x, a, b: 

def predict_psnr_basic(kernel_fn, train_fx, test_fx, train_x, train_y, test_x, test_y, t_final, eta=None):  
  g_dd = kernel_fn(train_x, train_x, 'ntk')
  g_td = kernel_fn(test_x, train_x, 'ntk')
  train_predict_fn = nt.predict.gradient_descent_mse(g_dd, train_y[...,None], g_td)
  train_theory_y, test_theory_y = train_predict_fn(t_final, train_fx[...,None], test_fx[...,None])

  calc_psnr = lambda f, g: -10. * np.log10(np.mean((f-g)**2))
  return calc_psnr(test_y, test_theory_y[:,0]), calc_psnr(train_y, train_theory_y[:,0])

predict_psnr_basic = jit(predict_psnr_basic, static_argnums=(0,))


def train_model(rand_key, network_size, lr, iters, 
                train_input, test_input, test_mask, optimizer, ab, name=''):
    if ab is None:
        ntk_params = False
    else:
        ntk_params = True
    init_fn, apply_fn = make_network(*network_size, ntk_params=ntk_params)

    if ab is None:
        run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, x[...,None] - .5)))
    else:
        run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, input_encoder(x, *ab))))
    model_loss = jit(lambda params, ab, x, y: .5 * np.sum((run_model(params, ab, x) - y) ** 2))
    model_psnr = jit(lambda params, ab, x, y: -10 * np.log10(np.mean((run_model(params, ab, x) - y) ** 2)))
    model_grad_loss = jit(lambda params, ab, x, y: jax.grad(model_loss)(params, ab, x, y))

    opt_init, opt_update, get_params = optimizer(lr)
    opt_update = jit(opt_update)

    if ab is None:
        _, params = init_fn(rand_key, (-1, 1))
    else:
        _, params = init_fn(rand_key, (-1, input_encoder(train_input[0], *ab).shape[-1]))
    opt_state = opt_init(params)

    pred0 = run_model(get_params(opt_state), ab, test_input[0])
    pred0_f = np.fft.fft(pred0)

    train_psnrs = []
    test_psnrs = []
    theories = []
    xs = []
    errs = []
    for i in tqdm(range(iters), desc=name):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), ab, *train_input), opt_state)

        if i % 20 == 0:
            train_psnr = model_psnr(get_params(opt_state), ab, *train_input)
            test_psnr = model_psnr(get_params(opt_state), ab, test_input[0][test_mask], test_input[1][test_mask])

            if np.isnan(train_psnr) or np.isnan(test_psnr):
                break
            # if ab is None:
            #     train_fx = run_model(get_params(opt_state), ab, train_input[0])
            #     test_fx = run_model(get_params(opt_state), ab, test_input[0][test_mask])
            #     theory = predict_psnr_basic(kernel_fn, train_fx, test_fx, train_input[0][...,None]-.5, train_input[1], test_input[0][test_mask][...,None], test_input[1][test_mask], i*lr)
            # else:
            #     test_x = input_encoder(test_input[0][test_mask], *ab)
            #     train_x = input_encoder(train_input[0], *ab)

            #     train_fx = run_model(get_params(opt_state), ab, train_input[0])
            #     test_fx = run_model(get_params(opt_state), ab, test_input[0][test_mask])
            #     theory = predict_psnr_basic(kernel_fn, train_fx, test_fx, train_x, train_input[1], test_x, test_input[1][test_mask], i*lr)


            train_psnrs.append(train_psnr)
            test_psnrs.append(test_psnr)
            # print(train_psnr, test_psnr)
            # theories.append(theory)
            pred = run_model(get_params(opt_state), ab, train_input[0])
            errs.append(pred - train_input[1])
            xs.append(i)
    return get_params(opt_state), apply_fn, train_psnrs, test_psnrs, errs, np.array(theories), xs

# %% [markdown]
# # Make fig 2

# %%
N_train = 500
data_power = 1

network_size = (4, 1024)

learning_rate = 0.0005
sgd_iters = 24000



# %%
def sample_random_powerlaw(key, N, power):
  coords = np.float32(np.fft.ifftshift(1 + N//2 - np.abs(np.fft.fftshift(np.arange(N)) - N//2)))
  decay_vec = coords ** -power
  decay_vec = onp.array(decay_vec)
  decay_vec[N//4:] = 0
  return sample_random_signal(key, decay_vec)

def sinusoid(x, freq):
  return np.sin(2 * np.pi * freq * (x + 0.21))

rand_key = random.PRNGKey(42)
subkey, subkey2 = random.split(rand_key)




M = 2
N = N_train

s = sinusoid(np.linspace(0,1.,N*M,endpoint=False), 21)
# s2 = sinusoid(np.linspace(0,1.,N*M//2,endpoint=False), 5)
#merge s and s2
# s = np.concatenate([s2, s2])

# s = s.reshape(N,M)
# s.shape
# plt.plot(s)
x_test = np.float32(np.linspace(0,1.,N*M,endpoint=False))
# x_train = x_test[::M]
# s = (s-s.min()) / (s.max()-s.min()) - .5
# plt.plot(x_test, s)

# %%
class RandomSignal:
  """ Generate a signal with a domain N given number of regions with random values.
      For each region, the signal value is a random number between -1 and 1."""
  def __init__(self, N, regions) -> None:
      self.N = N
      self.regions = regions
      self.region_size = N // regions
      self.region_values = random.uniform(subkey, (regions,), minval=-1, maxval=1)

  def get_value(self, x):
    x = np.int32(((x/self.N) * self.regions) % self.regions)
    return self.region_values[x]

signal = RandomSignal(1.0, 10)
s = signal.get_value(np.linspace(0,1.,N*M,endpoint=False))
# plt.plot(x_test, s)


# %%
rand_key = random.PRNGKey(0)

config.update('jax_disable_jit', False)

# Signal
M = 5
N = N_train
x_test = np.float32(np.linspace(0,1.,N*M,endpoint=False))
# get N_train random points from x_test
# sample 90% from the second half of the signal
index_first_half = random.choice(subkey, np.arange(0,N*M//2), shape=(int(50),), replace=True)
index_second_half = random.choice(subkey2, np.arange(N*M//2,N*M), shape=(int(450),), replace=True)
index = np.concatenate([index_first_half, index_second_half])
# get only 1/M test points of the first half and all of the second half
# first = np.arange(0,N*M//2,M)
# index = np.concatenate([first, np.arange(N*M//2,N*M)])
x_train = x_test[index]

test_mask = onp.ones(len(x_test), bool)
test_mask[index] = 0
# test_mask[np.arange(0,x_test.shape[0],M)] = 0

# s = sample_random_powerlaw(rand_key, N*M, data_power) 
# s = (s-s.min()) / (s.max()-s.min()) - .5
s = sinusoid(np.linspace(0,1.,N*M//2,endpoint=False), 0.3)
s2 = sinusoid(np.linspace(0,1.,N*M//2,endpoint=False), 15)
# s = sinusoid(np.linspace(0,1.,N*M,endpoint=False), 5)

#merge s and s2


# signal = RandomSignal(1.0, 250)
# s2 = signal.get_value(np.linspace(0,1.,N*M//2,endpoint=False))
s = np.concatenate([s, s2])
# Kernels
# bvals = np.float32(np.arange(1, N//2+1))
# g = 10
# bvals = 1/((1/(50))*g**np.arange(0, 3))
# bvals = progression(6, 500/2.818203040529665)
# bvals=np.array([0.5, 2])
bvals = progression(3, 15/2.464039000347777)
# bvals = 1/((1/(12))*g**np.arange(0, 9))
# bvals = np.abs(random.normal(rand_key, [128]) * 10)
# print(bvals)

ab_dict = {}
# ab_dict = {r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0, 1]}
p = 1
ab_dict = {r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [p]}
# ab_dict[r'$p = \infty$'] = (np.eye(bvals.shape[0])[0], bvals)
# ab_dict['No mapping'] = None


# Train the networks
# rand_key, *ensemble_key = random.split(rand_key, len(ab_dict))

outputs = {k : train_model(rand_key, network_size, learning_rate, sgd_iters, 
                           (x_train, s[index]), (x_test, s), test_mask,
                           optimizer=optimizers.sgd, ab=ab_dict[k], name=k) for k in ab_dict}
# save outputs to csv
import numpy as np2
for k in outputs:
  np2.savetxt(f'outputs/{k}.csv', outputs[k][2], delimiter=',')
  np2.savetxt(f'outputs/{k}_test.csv', outputs[k][3], delimiter=',')

# outputs['$p = 0$'][2][-1], outputs['$p = 0$'][3][-1]
# outputs['No mapping'][2][-1], outputs['No mapping'][3][-1]


# %%
# run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, input_encoder(x, *ab))))
print(outputs[f'$p = {p}$'][2][-1], outputs[f'$p = {p}$'][3][-1])
# res = outputs['No mapping'][1](outputs['No mapping'][0], x_test[...,None] - .5)

res2 = outputs[f'$p = {p}$'][1](outputs[f'$p = {p}$'][0], input_encoder(x_test, *ab_dict[f'$p = {p}$']))
# len(x_test)
plt.plot(x_train, s[index], 'o')
# s[::M]
plt.plot(x_test, res2)
# plt.show()
# plt.plot(x_test, res)
plt.plot(x_test, s)
plt.show()