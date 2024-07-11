import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from functools import partial


"""
This code corresponds to the method presented in Schrab et al. (2023), and is taken from https://github.com/antoninschrab/mmdagg.

"""

@partial(jit, static_argnums=(5,6,7,8,9))
def MMDtest(
    X,
    Y,
    key,
    seed,
    bandwidth = 1,
    alpha = 0.05,
    kernel ='gaussian',
    stat_type = 'U',
    B = 1999,
    value=False
):
    
    m, d = X.shape
    n = Y.shape[0]

    Z = jnp.concatenate((X,Y),axis = 0) # (m+n) x d

    # feature mapping
    if kernel == 'gaussian':
        l = 'l2'

    pairwise_matrix = jax_distances(Z, Z, l, matrix=True)

    #distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
    #bandwidth = jnp.median(distances)

    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
    K = K.at[jnp.diag_indices(K.shape[0])].set(0)
    
    # permutation test
    I_1 = jnp.concatenate((jnp.ones(m),jnp.zeros(n)))
    I = jnp.tile(I_1,(B+1,1)) # (B+1) x (m+n)
    
    key, subkey = random.split(key)
    I_X = random.permutation(subkey, I, axis = 1, independent=True)
    I_X=I_X.at[0].set(I_1) # should include the non-permuted case Z=(X,Y)
    I_Y = 1-I_X
    
    V11 = I_X-I_Y
    V11 = V11.transpose()
    V10 = I_X
    V10 = V10.transpose()
    V01 = -I_Y
    V01 = V01.transpose()

    
    if stat_type == 'U': 
        test_stats = (
            jnp.sum(V10 * (K @ V10), 0) * ((n - m + 1) / (m * n)) / (m - 1)
            + jnp.sum(V01 * (K @ V01), 0) * ((m - n + 1) / (m * n)) / (n - 1)
            + jnp.sum(V11 * (K @ V11), 0) / (m * n)
            )  
#        test_stats = (
#                    jnp.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
#                    + jnp.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
#                    + jnp.sum(V11 * (K @ V11), 0) / (m * n)
#                ) 

    MMD2 = test_stats[0]
    q_index = jnp.array(jnp.ceil((B+1)*(1-alpha)), int)-1
    q = jnp.sort(test_stats)[q_index]
    
    Delta = (MMD2>q).astype(int) # 1 implies reject the null, 0 implies accept the null
    if value==False:
        return Delta
    elif value==True:
        return Delta,MMD2

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


def kernel_matrix(pairwise_matrix, l, kernel_type, bandwidth):
    """
    Compute kernel matrix for a given kernel_type and bandwidth. 

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel_type: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel_type must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2":
        return  jnp.exp(-d ** 2)
    elif kernel_type == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif (kernel_type == "matern_0.5_l1" and l == "l1") or (kernel_type == "matern_0.5_l2" and l == "l2") or (kernel_type == "laplace" and l == "l1"):
        return  jnp.exp(-d)
    elif (kernel_type == "matern_1.5_l1" and l == "l1") or (kernel_type == "matern_1.5_l2" and l == "l2"):
        return (1 + jnp.sqrt(3) * d) * jnp.exp(- jnp.sqrt(3) * d)
    elif (kernel_type == "matern_2.5_l1" and l == "l1") or (kernel_type == "matern_2.5_l2" and l == "l2"):
        return (1 + jnp.sqrt(5) * d + 5 / 3 * d ** 2) * jnp.exp(- jnp.sqrt(5) * d)
    elif (kernel_type == "matern_3.5_l1" and l == "l1") or (kernel_type == "matern_3.5_l2" and l == "l2"):
        return (1 + jnp.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * jnp.sqrt(7) / 3 / 5 * d ** 3) * jnp.exp(- jnp.sqrt(7) * d)
    elif (kernel_type == "matern_4.5_l1" and l == "l1") or (kernel_type == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * jnp.exp(- 3 * d)
    else:
        raise ValueError(
            'The values of l and kernel_type are not valid.'
        )