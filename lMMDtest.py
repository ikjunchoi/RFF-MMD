import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, random, jit
from functools import partial

"""
This code corresponds to the method presented in Gretton et al. (2012).
"""

@partial (jit, static_argnums=(5,6,7))
def lMMDtest(
    X,
    Y,
    key,
    seed,
    bandwidth = 1,
    alpha = 0.05,
    kernel ='gaussian',
    B = 1999
):

    m, d = X.shape
    n = Y.shape[0]
    
    Z_0 = jnp.concatenate((X,Y), axis = 0) #(m+n) x d

    Z_0_mul= jnp.tile(Z_0,(B,1,1)) # B x (m+n) x d

    subkeys = random.split(key, num=B)
    Z_perm = vmap(random.permutation)(subkeys,Z_0_mul)

    Z_total = jnp.zeros((B+1,m+n,d)) # (B+1) x (m+n) x d
    
    Z_total = Z_total.at[0].set(Z_0) # should include the non-permuted case Z=(X,Y)
    Z_total = Z_total.at[1:].set(Z_perm)

    X_total = Z_total[:,0:m,:]
    Y_total = Z_total[:,m:,:]

    stat_total = vmap(lambda x,y: lMMDstat(x, y, bandwidth, kernel))(X_total,Y_total) # (B+1)

    test_stat = stat_total[0]
    q_index = jnp.array(jnp.ceil((B+1)*(1-alpha)), int)-1
    q = jnp.sort(stat_total)[q_index]
    Delta = (test_stat>q).astype(int)
    
    return Delta

@partial (jit, static_argnums=(3))
def lMMDstat(X, Y, bandwidth=1, kernel='gaussian'):
    m, d = X.shape
    n = Y.shape[0]
    n = min(n, m)
    b = 2
    r = n%b 
    n = int(n-r) # drop the n%b terms 
    X, Y = X[:n], Y[:n] 
    
    # obtain the gram matrix 

    # compute the block mmd statistic
    num_blocks = n//b 
    
    X_reshape=jnp.reshape(X,(num_blocks ,b ,d))
    Y_reshape=jnp.reshape(Y,(num_blocks ,b ,d))
    stat = (vmap(lambda x,y: MMD2unbiased(x, y, kernel=kernel, bandwidth=bandwidth))(X_reshape,Y_reshape)).mean()
    
    return stat

@partial (jit, static_argnums=(3))
def MMD2unbiased(X, Y, bandwidth=1, kernel='gaussian'):
    if kernel == 'gaussian':
        l = 'l2'
    Kxx = kernel_func(X, X, l, kernel, bandwidth) 
    Kyy = kernel_func(Y, Y, l, kernel, bandwidth)
    Kxy = kernel_func(X, Y, l, kernel, bandwidth) 

    n, m = len(X), len(Y)

    term1 = Kxx.sum()
    term2 = Kyy.sum()
    term3 = 2*Kxy.mean()

    term1 -= jnp.trace(Kxx)
    term2 -= jnp.trace(Kyy)
    MMD2 = (term1/(n*(n-1)) + term2/(m*(m-1)) - term3)
    
    return MMD2 
    
def kernel_func(X, Y, l, kernel_type, bandwidth):
    pairwise_matrix = jax_distances(X, Y, l, matrix=True)
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2":
        return  jnp.exp(-d ** 2)

def jax_distances(X, Y, l, max_samples=None, matrix=False):
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