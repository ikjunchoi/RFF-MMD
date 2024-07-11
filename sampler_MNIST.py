from utils import download_mnist, load_mnist_7x7, load_mnist_28x28
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from functools import partial

def sampler_mnist(
    m,
    n,
    key,
    mix_rate = 0.1,
    downsample = True
):
    if downsample == True:
        with open('mnist_dataset/mnist_7x7_P.data', 'rb') as handle:
            P = pickle.load(handle)
        with open('mnist_dataset/mnist_7x7_Q.data', 'rb') as handle:
            Q = pickle.load(handle)
    else:
        with open('mnist_dataset/mnist_28x28_P.data', 'rb') as handle:
            P = pickle.load(handle)
        with open('mnist_dataset/mnist_28x28_Q.data', 'rb') as handle:
            Q = pickle.load(handle)
    d= P.shape[1]
    # X
    key, subkey = random.split(key)
    X = jax.random.choice(subkey, P, shape=(m,), replace=True, axis=0)

    #Y
    subkeys = random.split(key, num=4)
    n_Q = random.binomial(subkeys[0], n, p=mix_rate).astype(int)
    n_P = n - n_Q

    if n_Q == 0:
        Y = jax.random.choice(subkeys[1], P, shape=(n_P,), replace=True, axis=0)
    elif n_Q == n:
        Y = jax.random.choice(subkeys[1], Q, shape=(n_Q,), replace=True, axis=0)
    else:
        from_P = jax.random.choice(subkeys[1], P, shape=(n_P,), replace=True, axis=0)
        from_Q = jax.random.choice(subkeys[2], Q, shape=(n_Q,), replace=True, axis=0)
        Y = jnp.vstack((from_P,from_Q))
        Y = random.permutation(subkeys[3],Y)

    return X,Y

    
    

