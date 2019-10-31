# DPP-VFX: Very Fast and eXact DPP sampler
Material related to the NeurIPS 2019 paper
["Exact sampling of determinantal point processes with sublinear time preprocessing"](https://arxiv.org/abs/1905.13476)
by [Michał Dereziński](https://users.soe.ucsc.edu/~mderezin/), [Calandriello Daniele](https://scholar.google.com/citations?user=R7c1UMMAAAAJ), and [Michal Valko](http://researchers.lille.inria.fr/~valko/hp/).

## Resources

|Link | Resource|
|---|---|
| [ArXiv](https://arxiv.org/abs/1905.13476) | Paper
| [Poster](https://github.com/LCSL/dpp-vfx/blob/master/poster.pdf) | Poster
| [Algorithm code](https://github.com/guilgautier/DPPy) | Implemented in `DPPy` ver. 0.3 or later
| Experiment code | This repository

## Algorithm code
`dpp-vfx` is the first sub-linear time exact DPP sampler. It is implmented
as part of the [`DPPy`](https://github.com/guilgautier/DPPy) library, which
can be installed using the command

```
pip install dppy
```

An example usage on some gaussian data with linear kernel is

```python
from dppy.finite_dpps import FiniteDPP
from dppy.utils import example_eval_L_linear
from numpy.random import randn

N = 1000
r = 100

DPP = FiniteDPP('likelihood',
                **{'L_eval_X_data': (example_eval_L_linear, randn(N, r))})

for _ in range(10):
    sample = DPP.sample_exact('vfx')
    print(sample)

for _ in range(10):
    sample = DPP.sample_exact_k_dpp(size=k, mode='vfx')
    print(sample)
```

## Experiments code
The experiments reported in the paper can be reproduced using the files `exp_dppy_mnist_first_sample.py`
and `exp_dppy_mnist_successive_sample.py`, with the file `postprocess_dppy_experiments.py`
containing the `matplotlib` code necessary to generate the plots.

The experiments require:
* Access to a scientific python stack, including `numpy`, `scipy`, and `sklearn`
* `DPPy` version 0.3 or higher, available on [github](https://github.com/guilgautier/DPPy) and [PyPi](https://pypi.org/project/dppy/)

The experiments also require access to the [MNIST8M ](https://leon.bottou.org/projects/infimnist) (now called infinite MNIST) dataset.
To reproduce the experiments, download the dataset's [source file](https://leon.bottou.org/_media/projects/infimnist.tar.gz),
and convert it from the MNIST or `svmlight` format to a numpy `.npz` first.

## Citation
If you use `dpp-vfx` or the related experiments code please cite:
```
@incollection{neurips2019dppvfx,
title = {Exact sampling of determinantal point processes with sublinear time preprocessing,
author = {Micha{\l} Derezi\'{n}ski and Daniele Calandriello and Michal Valko},
booktitle = {Advances in Neural Information Processing Systems 32},
year = {2019},
}
```

## Contact
For any question, you can contact daniele.calandriello@iit.it or mderezin@berkeley.edu
