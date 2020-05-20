from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.datasets import fetch_openml
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

mnist = fetch_openml('mnist_784', version=1, cache=True, data_home='~/python/datasets/')

n_list = np.linspace(1000, 70000, 15)

urandom_seed = SystemRandom().randrange(99999)
r = np.random.RandomState(urandom_seed)

sigma = np.sqrt(3*mnist.data.shape[1])
dot_func = FastMixinKernel(
    RBF(sigma),
    PairwiseKernel(gamma=1/np.square(sigma), metric='rbf', pairwise_kernels_kwargs={'n_jobs':-2})
)

res = np.zeros((len(n_list), 4))

desired_k = 10


for (i_cand, n_cand) in enumerate(n_list):
    result = []
    print(n_cand)
    I_train = sample_without_replacement(n_population=mnist.data.shape[0],
                                         n_samples=int(n_cand),
                                         random_state=r)

    X_train = mnist.data[I_train, :]/255.
    n = X_train.shape[0]

    with Timer(verbose=False) as t_vfx:
        vfx_dpp_sampler = FiniteDPP(kernel_type='likelihood', L_eval_X_data=(dot_func, X_train))
        S_vfx = vfx_dpp_sampler.sample_exact('vfx',
                                             rls_oversample_bless=5,
                                             rls_oversample_dppvfx=10,
                                             random_state=r,
                                             desired_expected_size=desired_k,
                                             verbose=False)
    result.append({'n': n, 'alg': 'vfx', 'time': t_vfx.secs, 'k': len(S_vfx)})

    with Timer(verbose=False) as t_vfx_resample:
        S_vfx_resample = vfx_dpp_sampler.sample_exact('vfx',
                                                      rls_oversample_bless=5,
                                                      rls_oversample_dppvfx=10,
                                                      random_state=r,
                                                      desired_expected_size=desired_k,
                                                      verbose=False)
    result.append({'n': n, 'alg': 'vfx_resample', 'time': t_vfx_resample.secs,'k': len(S_vfx_resample)})

    with Timer(verbose=False) as t_mc:
        exact_mcmc_sample = FiniteDPP(kernel_type='likelihood', L=dot_func(X_train) * vfx_dpp_sampler.intermediate_sample_info.alpha_star)
        S_mcmc = exact_mcmc_sample.sample_mcmc(mode='AED',
                                               s_init=r.permutation(I_train.shape[0])[:len(S_vfx_resample)].tolist(),
                                               nb_iter=I_train.shape[0] * len(S_vfx_resample))
    result.append({'n': n, 'alg': 'mcmc', 'time': t_mc.secs, 'k': len(S_mcmc)})

    if X_train.shape[0] <= 16000:
        with Timer(verbose=False) as t_exact:
            L = dot_func(X_train) * vfx_dpp_sampler.intermediate_sample_info.alpha_star
            E,U = np.linalg.eigh(L)
            exact_dpp_sample = FiniteDPP(kernel_type='likelihood', L_eig_dec=(E, U))
            S_exact = exact_dpp_sample.sample_exact()
        result.append({'n': n, 'alg': 'exact', 'time': t_exact.secs, 'k': len(S_exact)})

        with Timer(verbose=False) as t_exact_resample:
            exact_dpp_sample = FiniteDPP(kernel_type='likelihood', L_eig_dec=(E, U))
            S_exact_resample = exact_dpp_sample.sample_exact()
        result.append({'n': n, 'alg': 'exact_resample', 'time': t_exact_resample.secs, 'k': len(S_exact_resample)})

    with tempfile.NamedTemporaryFile(prefix=f'run_final_{urandom_seed:05d}_', suffix='.pickle', dir=Path.home() / 'data/result', delete=False) as file:
        pickle.dump(result, file)

pass
