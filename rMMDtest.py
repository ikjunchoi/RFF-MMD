import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, random, jit
from functools import partial

@partial(jit, static_argnums=(5,6,7,8,9))
def rMMDtest(
    X,
    Y,
    key,
    seed,
    bandwidth = 1,
    alpha = 0.05,
    kernel ='gaussian',
    R = 20,
    stat_type = 'U',
    B = 1999
):
    '''
    [Input and output]
        (Input)
            X: array_like
                The shape of X must be of the form (m, d) where m is the number
                of samples and d is the dimension.
            Y: array_like
                The shape of Y must be of the form (n, d) where n is the number
                of samples and d is the dimension.

        (Output)
            Delta: int
                The value of delta must be 0 or 1;
                    return 0 if the test ACCEPTS the null
                 or return 1 if the test REJECTS the null 


    [Parameters]
        key:
            Random key for the randomness of permutations and Fourier features.
        seed:
            Dummy parameter (to match the format of other tests).
        alpha: scalar
            The value of alpha must be between 0 and 1.
        bandwidth: scalar:
            The value of bandwidth must be between 0 and 1.
        kernel: str
            The value of kernel must be, "gaussian", ...
        R: int
            The number of random Fourier features.
        stat_type: str
            The value must be 'U' or 'V'; U-statistic or V-statistic.
        B: int
            The number of simulated test statistics to approximate the quantiles.
    '''
    
    m, d = X.shape
    n = Y.shape[0]

    Z = jnp.concatenate((X,Y),axis = 0) # (m+n) x d

    # feature mapping
    key, subkey_feature, subkey_permutation = random.split(key, num=3)
    
    if kernel == 'gaussian':
        omegas = jnp.sqrt(2) / bandwidth  * random.normal(subkey_feature, (R, d))

    omegas_Z = jnp.dot(Z, omegas.T) # (m+n) x R
    cos_feature = (1/jnp.sqrt(R)) * jnp.cos(omegas_Z) # (m+n) x R
    sin_feature = (1/jnp.sqrt(R)) * jnp.sin(omegas_Z) # (m+n) x R
    psi_Z = jnp.concatenate((cos_feature, sin_feature), axis=1) # (m+n) x 2R

    
    # permutation test
    I_1 = jnp.concatenate((jnp.ones(m),jnp.zeros(n)))
    I = jnp.tile(I_1,(B+1,1)) # (B+1) x (m+n)
    I_X = random.permutation(subkey_permutation,I,axis = 1, independent=True)
    I_X=I_X.at[0].set(I_1) # should include the non-permuted case Z=(X,Y)
    I_Y = 1-I_X
    
    bar_Z_B_piX = (1/m) * I_X @ psi_Z # (B+1) x 2R;
    bar_Z_B_piY = (1/n) * I_Y @ psi_Z # (B+1) x 2R;
    T = bar_Z_B_piX-bar_Z_B_piY # (B+1) x 2R
    V = jnp.sum(T ** 2, axis=1) # (B+1, )
    if stat_type == 'V':
        test_stats = V
    
    elif stat_type == 'U':
        W = 1/(m-1) - (1/(m-1)) * jnp.sum(bar_Z_B_piX ** 2,axis=1) + 1/(n-1) - (1/(n-1)) * jnp.sum(bar_Z_B_piY ** 2,axis=1)
        U = V - W
        test_stats = U

    rMMD2 = test_stats[0]
    q_index = jnp.array(jnp.ceil((B+1)*(1-alpha)), int)-1
    q = jnp.sort(test_stats)[q_index]

    Delta = (rMMD2>q).astype(int) # 1 implies reject the null, 0 implies accept the null

    return Delta