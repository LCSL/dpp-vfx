import numpy as np
import pickle
import glob
import sys
import matplotlib.pyplot as plt
import os
import tempfile
np.seterr(all='raise')

def bar_plot(arr, title, fig, mk='o', mkevery=1, color=None):
    x_new = arr[:,0].squeeze()
    y_new = arr[:,1].squeeze()
    ax = fig.axes
    if color is None:
        color = ax.plot(x_new, y_new, label=title, marker=mk, markevery=mkevery)[-1].get_color()
    else:
        ax.plot(x_new, y_new, label=title, marker=mk, markevery=mkevery, color=color)

    y_new_err = arr[:,2].squeeze()
    ax.fill_between(x_new, y_new - y_new_err, y_new + y_new_err, facecolor=color, alpha=0.25)

    return fig

alg_list = {"vfx_resample": '#1f77b4',
            "vfx": '#ff7f0e',
            "mcmc": '#2ca02c',
            "exact_resample": '#d62728',
            "exact": '#9467bd'}

cmap=plt.get_cmap("tab10")

exp_prefixes = ["first", "succ"]

for exp_prefix in exp_prefixes:
    f_list = glob.glob('result/' + exp_prefix + '_*')

    fig = plt.figure().add_subplot(111)

    for alg in alg_list.keys():
        if exp_prefix == "succ" and not('resample' in alg):
            continue

        n_list = []
        for f in f_list:
            with open(f, "rb") as fin:
                point = pickle.load(fin)
                for p in point:
                    if p['alg'] == alg:
                        n_list.append(p['n'])

        n_list = sorted(list(set(n_list)))
        res_arr = np.zeros((len(n_list), 3))
        res_arr[:, 0] = np.array(n_list)
        for (i,n) in enumerate(n_list):
            time_list = []
            k_list = []
            for f in f_list:
                with open(f, "rb") as fin:
                    point = pickle.load(fin)
                    for p in point:
                        if p['alg'] == alg and p['n'] == n:
                            time_list.append(p['time'])
                            k_list.append(p['k'])

            point_arr = np.array(time_list)
            res_arr[i, 1:] = np.array([point_arr.mean(), 1.96*point_arr.std()/np.sqrt(len(point_arr)) if len(point_arr) > 1 else 100])
            print(f'alg {alg} n {n} k {np.mean(k_list)}')

        bar_plot(res_arr, alg, fig, color=alg_list[alg])

    ax = fig.axes
    ax.set_xlabel('$n$', fontsize=17, fontname='Times New Roman')
    ax.set_ylabel('sec', fontsize=17, fontname='Times New Roman')

    if exp_prefix == "first":
        plt.ylim([-1, 40])
    else:
        plt.ylim([-.2, 4])
    plt.legend()
    plt.draw()
    fig.legend(loc='best', fontsize=14)
    fig.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig('result/fig_runtime_repro_' + exp_prefix + '.pdf')

