from functools import partial
import jax
import jax.numpy as jnp
from jax import vmap, random, jit
import os, sys

import pickle
from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt


@partial(jit, static_argnums=(2))
def median_heuristic(X, Y, l):
    Z = jnp.concatenate((X, Y))
    distances = jax_distances(Z, Z, l, matrix=False)
    median = jnp.median(distances)
    return median



def jax_distances(X, Y, l, max_samples=None, matrix=False):
    """
    This function is extracted from the following
    https://github.com/antoninschrab/mmdagg-paper
    which is under the MIT License.
    """
    if l == "l1":

        def dist(x, y):
            z = x - y
            return jnp.sum(jnp.abs(z))

    elif l == "l2":

        def dist(x, y):
            z = x - y
            return jnp.sqrt(jnp.sum(jnp.square(z)))

    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(Y[:max_samples], X[:max_samples])
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]

        
@partial(jit, static_argnums=(1,2,3,4))
def gaussian_data_generator(key=random.PRNGKey(42),
    size=40, 
    d=10, 
    mu=0, 
    Sigma=1
):
    mean = mu * jnp.ones((d,))
    cov = Sigma * jnp.eye(d)
    return random.multivariate_normal(key, mean, cov, shape=(size,))


@partial(jit, static_argnums=(1,2,3,4,5))
def gaussian_data_generator2(key=random.PRNGKey(42),
    size=40,  
    Sigma=1,
    d=10,
    epsilon=0.1,
    j=5
):
    mean = jnp.zeros((d,))
    mean=mean.at[:j].set(epsilon * jnp.ones((j,)))
    cov = Sigma * jnp.eye(d)
    return random.multivariate_normal(key, mean, cov, shape=(size,))



def result_viewer(experiment_name,RFFMMD12=True):
    scale = 0.8
    f, axs = plt.subplots(sharey=True,figsize=(8,6))
    f.tight_layout()
    f.subplots_adjust(wspace=0.1, hspace=0.45)
    
    fs = 18

    markersize = 3.5
    
    if RFFMMD12 == True:
        tests_names = ["MMD",  # MMD
                       "RFFMMD(R=10)",  # RFFMMD(R=10)
                       "RFFMMD(R=200)",  # RFFMMD(R=200)
                       "incMMD(R'=100)",  # incMMD(R'=100)
                       "incMMD(R'=200)",  # incMMD(R'=200)
                       "lMMD",  # lMMD
                       r"bMMD(b=${n}^{1/2}_1$)"  # bMMD
                      ]
    else:
        tests_names = ["MMD",  # MMD
                       "RFFMMD(R=200)",  # RFFMMD(R=200)
                       "RFFMMD(R=1000)",  # RFFMMD(R=1000)
                       "incMMD(R'=100)",  # incMMD(R'=100)
                       "incMMD(R'=200)",  # incMMD(R'=200)
                       "lMMD",  # lMMD
                       r"bMMD(b=${n}^{1/2}_1$)"  # bMMD
                      ]
    
    styles = [
        'solid',  # MMD
        'dashed', # RFFMMD(R=10)
        'dashed', # RFFMMD(R=200)
        'dotted', # incMMD(R'=100)
        'dotted', # incMMD(R'=200)
        'dotted', # lMMD
        'dotted'  # bMMD
    ]
    
    markers = np.array(["o", # MMD
                        "d", # RFFMMD(R=10)
                        "d", # RFFMMD(R=200)
                        "^", # incMMD(R'=100)
                        "^", # incMMD(R'=200)
                        "^", # lMMD
                        "^"  # bMMD
                        ])
    
    colors = np.array(["C0", # MMD
                       "C8", # RFFMMD(R=10)
                       "C1", # RFFMMD(R=200)
                       "C5", # incMMD(R'=100)
                       "C7", # incMMD(R'=200)
                       "C6", # lMMD
                       "C2"  # bMMD
                      ])
    linewidths=np.array([3,  # MMD
                         2,  # except MMD
                         2,
                         2,
                         2,
                         2,
                         2
                        ])
    
    power = np.load(f"results/{experiment_name}_power.npy") 
    varying = np.load(f"results/{experiment_name}_varying.npy") 
    for j in range(len(tests_names)):
        axs.plot(varying, power[j], color=colors[j], marker=markers[j], linestyle=styles[j], label=tests_names[j], markersize=markersize)
    axs.set_ylabel("Power", labelpad=10, fontsize=fs+2)
    axs.set_ylim(-0.05, 1.05)
    axs.set_yticks([0, 0.5, 1])
    axs.set_title(experiment_name, fontsize=fs, pad=10)
    axs.tick_params(axis='x', which='major', labelsize=fs - 1)
    
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    order_legend = range(len(tests_names))
    axs.legend(
        [handles[index] for index in order_legend],
        [labels[index] for index in order_legend],
        fontsize=fs+1,
        ncol=3,
        handleheight=0.5,
        labelspacing=0.4,
        columnspacing=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.5),
    )



    
    """
    *The below codes are taken from https://github.com/antoninschrab/mmdagg-paper, 
    which was extracted from https://github.com/MPI-IS/tests-wo-splitting under The MIT License.
    
    *These are slight modifications of them tailored to our settings.

    
    download_mnist : Download MNIST dataset and downsample it to 7x7 images,
    save the downsampled dataset as mnist_7x7.data in the
    mnist_dataset directory.

    load_mnist_7x7 : Returns P and Q_list where P consists of downsampled images of even digits, 
    and Q consists images of odd digits.

    load_mnist_28x28 : Returns P and Q_list where P consists of images of even digits, 
    and Q consists images of odd digits.
    
    These functions (load_mnist_7x7, load_mnist_28x28) should only be run after download_mnist().
    
    """

def download_mnist():

    X, y = fetch_openml("mnist_784", return_X_y=True)
    X = jnp.array(X)
    X = X / 255
    digits = {}
    for i in range(10):
        digits[str(i)] = []
    for i in range(len(y)):
        digits[y[i]].append(X[i])
    digits_7x7 = {}
    digits_28x28 = {}
    for i in range(10):
        current = jnp.array(digits[str(i)])
        n = len(current)
        # make the dataset 2D again
        current = jnp.reshape(current, (n, 28, 28))
        digits_28x28[str(i)] = jnp.reshape(current, (n, 28*28))
        current = jnp.reshape(current, (n, 7, 4, 7, 4))
        current = current.mean(axis=(2, 4))
        digits_7x7[str(i)] = jnp.reshape(current, (n, 49))

    X = digits_7x7
    P = jnp.vstack((X['0'], X['2'], X['4'], X['6'], X['8']))
    Q = jnp.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))

    path = "mnist_dataset/mnist_7x7_P.data"
    f = open(path, 'wb')
    pickle.dump(P, f)
    f.close()

    path = "mnist_dataset/mnist_7x7_Q.data"
    f = open(path, 'wb')
    pickle.dump(Q, f)
    f.close()

    X = digits_28x28
    P = jnp.vstack((X['0'], X['2'], X['4'], X['6'], X['8']))
    Q = jnp.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))

    path = "mnist_dataset/mnist_28x28_P.data"
    f = open(path, 'wb')
    pickle.dump(P, f)
    f.close()

    path = "mnist_dataset/mnist_28x28_Q.data"
    f = open(path, 'wb')
    pickle.dump(Q, f)
    f.close()

def load_mnist_7x7():

    with open('mnist_dataset/mnist_7x7.data', 'rb') as handle:
        X = pickle.load(handle)
    P  = np.vstack(
        (X['0'], X['2'], X['4'], X['6'], X['8'])
    )
    Q = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))
    return P, Q

def load_mnist_28x28():
    with open('mnist_dataset/mnist_28x28.data', 'rb') as handle:
        X = pickle.load(handle)
    P  = np.vstack(
        (X['0'], X['2'], X['4'], X['6'], X['8'])
    )
    Q_raw = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))
    return P, Q_raw






