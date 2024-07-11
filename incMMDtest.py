import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from jax.flatten_util import ravel_pytree
from functools import partial
import itertools
import psutil
import GPUtil as gputil
import warnings


"""
This code corresponds to the method presented in Schrab et al. (2022), and is taken from https://github.com/antoninschrab/agginc.

"""

@partial(jit, static_argnums=(4, 5, 6, 7, 8, 9))
def incMMDtest(
    X, 
    Y,
    key,
    seed=42,
    R=200,
    alpha=0.05,
    batch_size="auto",
    memory_percentage=0.4,
    B=500, 
    return_dictionary=False,
    bandwidth=None,
):
    """
    Efficient test for two-sample (MMD), independence (HSIC) and goodness-of-fit 
    (KSD) testing, using median bandwidth (no aggregation).
    
    Given the appropriate data for the type of testing, 
    return 0 if the test fails to reject the null (i.e. same distribution, independent, fits the data), 
    or return 1 if the test rejects the null (i.e. different distribution, dependent, does not fit the data).
    
    Parameters
    ----------
    agginc: str
        "mmd" or "hsic" or "ksd"
    X : array_like
        The shape of X must be of the form (N_X, d_X) where N_X is the number
        of samples and d_X is the dimension.
    Y : array_like
        The shape of Y must be of the form (N_Y, d_Y) 
        where N_Y is the number of samples and d_Y is the dimension.
        Case agginc = "mmd": Y is the second sample, we must have d_X = d_Y.
        Case agginc = "hsic": Y is the paired sample, we must have N_X = N_Y.
        Case agginc = "ksd": Y is the score of X, we must have N_Y = N_X and d_Y = d_X.
    R : int
        Number of superdiagonals to consider. 
        If R >= min(N_X, N_Y) - 1 then the complete U-statistic is computed in quadratic time.
    alpha: float
        The level alpha must be between 0 and 1.
    batch_size : int or None or str
        The memory cost consists in storing an array of shape (batch_size, R * N - R * (R - 1) / 2)
        where batch_size is between 1 and B.
        Using batch_size = "auto", calculates automatically the batch_size which uses 80% of the memory.
        For faster runtimes but using more memory, use batch_size = None (equivalent to batch_size = B)
        By decreasing batch_size from B, the memory cost is reduced but the runtimes increase.
    memory_percentage: float
        The value of memory_percentage must be between 0 and 1.
        It is used when batch_size = "auto", the batch_size is calculated automatically 
        to use memory_percentage of the memory.
    B: int
        B is the number of wild bootstrap samples to approximate the quantiles.
    seed: int 
        Random seed used for the randomness of the Rademacher variables.
    return_dictionary: bool
        If true, a dictionary is also returned containing the test out, the kernel, the bandwidth, 
        the statistic, the statistic quantile, the p-value and the p-value threshold value (level).
   bandwidth: float or list or None
        If bandwidths is None, the bandwidth used is the median heuristic.
        Otherwise, the bandwidth provided is used instead.
        If agg_type is "mmd" or "ksd", then bandwidth needs to be a float.
        If agg_type is "hsic", then bandwidths should be a list 
        containing 2 floats (bandwidths for X and Y).

        
    Returns
    -------
    output : int
        0 if the Inc test fails to reject the null (i.e. same distribution, independent, fits the data), 
        1 if the Inc test rejects the null (i.e. different distribution, dependent, does not fit the data).
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the output of the Inc test, the kernel, the bandwidth, 
        the statistic, the quantile, the p-value and the p-value threshold (level).
    
    
    Examples
    --------
    
    >>> # MMDInc
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1
    >>> output = inc("mmd", X, Y)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = inc("mmd", X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'Bandwidth': 3.391918659210205,
     'Kernel Gaussian': True,
     'MMD': 0.9845684170722961,
     'MMD quantile': 0.007270246744155884,
     'MMDInc test reject': True,
     'p-value': 0.0019960079807788134,
     'p-value threshold': 0.05000000074505806}
    
    >>> # HSICInc
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = 0.5 * X + random.uniform(subkeys[1], shape=(500, 10))
    >>> output = inc("hsic", X, Y)
    >>> output
    Array(0, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = inc("hsic", X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'Bandwidth X': 1.2791297435760498,
     'Bandwidth Y': 1.4075509309768677,
     'HSIC': 0.00903838686645031,
     'HSIC quantile': 0.0005502101266756654,
     'HSICInc test reject': True,
     'Kernel Gaussian': True,
     'p-value': 0.0019960079807788134,
     'p-value threshold': 0.05000000074505806}
    
    >>> # KSDInc
    >>> perturbation = 0.5
    >>> rs = np.random.RandomState(0)
    >>> X = rs.gamma(5 + perturbation, 5, (500, 1))
    >>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
    >>> score_X = score_gamma(X, 5, 5)
    >>> X = jnp.array(X)
    >>> score_X = jnp.array(score_X)
    >>> output = inc("ksd", X, score_X)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = inc("ksd", X, score_X, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> from agginc.jax import human_readable_dict
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'Bandwidth': 10.13830852508545,
     'KSD': 2.4731751182116568e-05,
     'KSD quantile': 5.930277438892517e-06,
     'KSDInc test reject': True,
     'Kernel IMQ': True,
     'p-value': 0.0019960079807788134,
     'p-value threshold': 0.05000000074505806}
    """
        
    # bandwidth: use provided one or compute median heuristic
    if bandwidth is not None:
        bandwidths = jnp.array(bandwidth).reshape(1)
    else:
        max_samples=500
        distances = jax_distances(X, Y, max_samples)
        median_bandwidth = jnp.median(distances)
        bandwidths = jnp.array([median_bandwidth])

        
    # compute all h-values
    h_values, index_i, index_j, N = compute_h_MMD_values(
        X, Y, R, bandwidths, return_indices_N=True
    )
    
    # compute bootstrap and original statistics
    bootstrap_values, original_value = compute_bootstrap_values(
        h_values, index_i, index_j, N, B, seed, batch_size, return_original=True, memory_percentage=memory_percentage
    )
    original_value = original_value[0]
    
    # compute quantile
    assert bootstrap_values.shape[0] == 1
    bootstrap_1 = jnp.column_stack([bootstrap_values, original_value])
    bootstrap_1_sorted = jnp.sort(bootstrap_1)
    quantile = bootstrap_1_sorted[0, (jnp.ceil((B + 1) * (1 - alpha))).astype(int) - 1]
    
    # reject if original_value > quantile
    reject_stat_val = original_value > quantile

    return (reject_stat_val).astype(int)


def create_indices(N, R):
    """
    Return lists of indices of R superdiagonals of N x N matrix
    
    This function can be modified to compute any type of incomplete U-statistic.
    """
    index_X = list(
        itertools.chain(*[[i for i in range(N - r)] for r in range(1, R + 1)])
    )
    index_Y = list(
        itertools.chain(*[[i + r for i in range(N - r)] for r in range(1, R + 1)])
    )
    return index_X, index_Y


def compute_h_MMD_values(X, Y, R, bandwidths, return_indices_N=False):
    """
    Compute h_MMD values.

    inputs:
        X (m,d)
        Y (n,d)
        R int
        bandwidths (#bandwidths,)

    output (#bandwidths, R * N - R * (R - 1) / 2)
    """
    N = min(X.shape[0], Y.shape[0])
    assert X.shape[1] == Y.shape[1]

    index_i, index_j = create_indices(N, R)
    
    norm_Xi_Xj = jnp.linalg.norm(X[jnp.array(index_i)] - X[jnp.array(index_j)], axis=1) ** 2
    norm_Xi_Yj = jnp.linalg.norm(X[jnp.array(index_i)] - Y[jnp.array(index_j)], axis=1) ** 2
    norm_Yi_Xj = jnp.linalg.norm(Y[jnp.array(index_i)] - X[jnp.array(index_j)], axis=1) ** 2
    norm_Yi_Yj = jnp.linalg.norm(Y[jnp.array(index_i)] - Y[jnp.array(index_j)], axis=1) ** 2

    h_values = jnp.zeros((bandwidths.shape[0], norm_Xi_Xj.shape[0]))
    for r in range(bandwidths.shape[0]):
        K_Xi_Xj_b = jnp.exp(-norm_Xi_Xj / bandwidths[r] ** 2)
        K_Xi_Yj_b = jnp.exp(-norm_Xi_Yj / bandwidths[r] ** 2)
        K_Yi_Xj_b = jnp.exp(-norm_Yi_Xj / bandwidths[r] ** 2)
        K_Yi_Yj_b = jnp.exp(-norm_Yi_Yj / bandwidths[r] ** 2)
        h_values = h_values.at[r].set(K_Xi_Xj_b - K_Xi_Yj_b - K_Yi_Xj_b + K_Yi_Yj_b)

    if return_indices_N:
        return h_values, index_i, index_j, N
    else:
        return h_values

def compute_bootstrap_values(
    h_values, index_i, index_j, N, B, seed, batch_size="auto", return_original=False, memory_percentage=0.8
):
    """
    Compute B bootstrap values.

    inputs:
        h_values, index_i, index_j = compute_h_XXX_values(...)
        h_values (#bandwidths, R * N - R * (R - 1) / 2)
        N int
        B int
        seed int


    output (#bandwidths, B)
    """
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    epsilon = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(N, B))
    
    # Bootstrap values can be computed as follows
    # with memory cost of storing an array (e_values) 
    # of size (R * N - R * (R - 1) / 2, B)
    
    # e_values = epsilon[index_i] * epsilon[index_j]
    # bootstrap_values = h_values @ e_values
    
    # Instead we use batches to store only arrays
    # of size (R * N - R * (R - 1) / 2, batch_size)
    # where batch_size is automatically chosen to use 80% of the memory
    # In the experiments of the paper, batch_size = None (i.e. batch_size = B) has been used
    # Larger batch_size increases the memory cost and decreases computational time
    
    if batch_size == None:
        batch_size = B
    elif batch_size == "auto":
        # Automatically compute the batch size depending on cpu/gpu memory 
        if "gpu" in str(jax.devices()[0]).lower() and len(gputil.getGPUs()) > 0:
            memory = gputil.getGPUs()[0].memoryTotal * 1048576 # bytes
        else:
            memory = psutil.virtual_memory().total # bytes
        memory_single_array = jnp.zeros(h_values.shape[0]).nbytes
        batch_size = int(memory * memory_percentage / memory_single_array)
    bootstrap_values = jnp.zeros((h_values.shape[0], epsilon.shape[1]))
    i = 0
    index = 0
    while index + batch_size < B or i == 0:
        index = i * batch_size
        epsilon_b = epsilon[:, index : index + batch_size]
        e_values_b = epsilon_b[jnp.array(index_i)] * epsilon_b[jnp.array(index_j)]
        bootstrap_values = bootstrap_values.at[:, index : index + batch_size].set(h_values @ e_values_b)
        i += 1
    bootstrap_values = bootstrap_values / len(index_i)

    if return_original:
        original_value = h_values @ jnp.ones(h_values.shape[1]) / len(index_i)
        return bootstrap_values, original_value
    else:
        return bootstrap_values

@partial(jit, static_argnums=(2,))
def jax_distances(X, Y, max_samples):
    def dist(x, y):
        z = x - y
        return jnp.sqrt(jnp.sum(jnp.square(z)))
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    output = output[jnp.triu_indices(output.shape[0])]
    return output