from sklearn.gaussian_process.kernels import RBF, PairwiseKernel, DotProduct, Kernel
from sklearn.datasets import fetch_mldata
from sklearn.utils.random import sample_without_replacement
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from pathlib import Path
import pickle
from random import SystemRandom
import time
from dppy.finite_dpps import FiniteDPP

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f s' % self.secs)


class FastMixinKernel(Kernel):
    def __init__(self, gp_kernel, pairwise_kernel):
        self.gp_kernel = gp_kernel
        self.pairwise_kernel = pairwise_kernel

    def __call__(self, X, Y=None, **kwargs):
        return self.pairwise_kernel(X, Y, **kwargs)

    def diag(self, X):
        return self.gp_kernel.diag(X)

    def is_stationary(self):
        return self.gp_kernel.is_stationary()

mnist8m = np.load('/dev/shm/mnist8m.npz')
X_all = mnist8m['X']
X_all /= np.linalg.norm(X_all, axis=1).max()

n_list = np.geomspace(100000, 999999, 10)

urandom_seed = SystemRandom().randrange(99999)
r = np.random.RandomState(urandom_seed)

sigma = np.sqrt(3*X_all.shape[1])
dot_func = FastMixinKernel(
    DotProduct(sigma_0=0),
    PairwiseKernel(gamma=1/np.square(sigma), metric='linear', pairwise_kernels_kwargs={'n_jobs':1})
)

res = np.zeros((len(n_list), 4))

desired_k = 20


for (i_cand, n_cand) in enumerate(n_list):
    result = []
    print(n_cand)
    I_train = sample_without_replacement(n_population=X_all.shape[0],
                                         n_samples=int(n_cand),
                                         random_state=r)

    X_train = X_all[I_train, :]
    n = X_train.shape[0]

    with Timer(verbose=False) as t_vfx:
        vfx_dpp_sampler = FiniteDPP(kernel_type='likelihood', L_eval_X_data=(dot_func, X_train))
        S_vfx = vfx_dpp_sampler.sample_exact('vfx',
                                             rls_oversample_bless=2.5,
                                             rls_oversample_dppvfx=2,
                                             random_state=r,
                                             desired_expected_size=desired_k,
                                             verbose=False)
    result.append({'n': n, 'alg': 'vfx', 'time': t_vfx.secs, 'k': len(S_vfx)})

    with Timer(verbose=False) as t_vfx_resample:
        S_vfx_resample = vfx_dpp_sampler.sample_exact('vfx',
                                                      rls_oversample_bless=2.5,
                                                      rls_oversample_dppvfx=2,
                                                      random_state=r,
                                                      desired_expected_size=desired_k,
                                                      verbose=False)
    result.append({'n': n, 'alg': 'vfx_resample', 'time': t_vfx_resample.secs,'k': len(S_vfx_resample)})

    if X_train.shape[0] <= 200000000:
        with Timer(verbose=False) as t_exact:
            L_gram_factor = X_train.T * np.sqrt(vfx_dpp_sampler.intermediate_sample_info.alpha_star)
            exact_dpp_sample = FiniteDPP(kernel_type='likelihood', L_gram_factor=L_gram_factor)
            S_exact = exact_dpp_sample.sample_exact()
            assert len(S_exact)
        result.append({'n': n, 'alg': 'exact', 'time': t_exact.secs, 'k': len(S_exact)})

        with Timer(verbose=False) as t_exact_resample:
            S_exact_resample = exact_dpp_sample.sample_exact()
        result.append({'n': n, 'alg': 'exact_resample', 'time': t_exact_resample.secs, 'k': len(S_exact_resample)})

    with tempfile.NamedTemporaryFile(prefix=f'succ_{urandom_seed:05d}_', suffix='.pickle', dir=Path.home() / 'data/result', delete=False) as file:
        pickle.dump(result, file)

pass
