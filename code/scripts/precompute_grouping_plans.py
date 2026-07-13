#!/usr/bin/env python3
"""
Phase 1: Precompute all grouping plans and save to .npz
Runs in a separate process to avoid BLAS contamination.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import time
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.custom_grouping import (
    spearman_correlation_matrix, mi_proxy_matrix,
    ordered_grouping, random_grouping, mst_cut_clustering,
    hem_coarsening, anti_clustering, maximin_dispersion,
    score_stratified, rebalance_groups,
)
from data_provider.data_factory import data_dict
import math

METHODS = [
    'ordered', 'random', 'finch_like', 'coarsening',
    'mi_based', 'anti_clustering', 'score_stratified', 'maximin_dispersion'
]

DATASETS = {
    'traffic': dict(
        root_path='./dataset/traffic/', data_path='traffic.csv',
        enc_in=862, e_layers=4, data='custom', num_groups=32,
    ),
    'electricity': dict(
        root_path='./dataset/electricity/', data_path='electricity.csv',
        enc_in=321, e_layers=3, data='custom', num_groups=16,
    ),
    'solar_AL': dict(
        root_path='./dataset/Solar/', data_path='solar_AL.txt',
        enc_in=137, e_layers=2, data='Solar', num_groups=16,
    ),
}

SEEDS = list(range(2021, 2051))


def load_train_data(dataset_name):
    ds = DATASETS[dataset_name]
    Data = data_dict[ds['data']]
    dataset = Data(
        root_path=ds['root_path'], data_path=ds['data_path'],
        flag='train', size=[96, 48, 96], features='M', target='OT',
        timeenc=1, freq='h',
    )
    return dataset.data_x  # (T, N)


def compute_one(args_tuple):
    """Worker: compute one (method, seed) grouping plan."""
    method, seed, N, G, S_spearman, S_mi, X_train = args_tuple
    alpha = 1.5

    if method == 'ordered':
        result = ordered_grouping(N, G)
    elif method == 'random':
        result = random_grouping(N, G, seed=seed)
    elif method == 'score_stratified':
        result = score_stratified(X_train, G, pred_len=96, seed=seed)
    elif method == 'finch_like':
        D = 1 - S_spearman
        result = mst_cut_clustering(D, G)
    elif method == 'coarsening':
        result = hem_coarsening(S_spearman, G, seed=seed)
    elif method == 'mi_based':
        result = hem_coarsening(S_mi, G, seed=seed)
    elif method == 'anti_clustering':
        result = anti_clustering(S_spearman, G, seed=seed)
    elif method == 'maximin_dispersion':
        result = maximin_dispersion(S_spearman, G, seed=seed)
    else:
        raise ValueError(method)

    # Rebalancing for similarity-based methods
    if method in ('finch_like', 'coarsening', 'anti_clustering', 'maximin_dispersion'):
        S = S_spearman
    elif method == 'mi_based':
        S = S_mi
    else:
        S = None

    if S is not None:
        avg = N / G
        M_max = math.ceil(avg * (1 + math.sqrt((alpha - 1) * (G - 1))))
        max_size = max(len(g) for g in result["groups"])
        if max_size > M_max:
            result = rebalance_groups(
                result["groups"], S, target_G=G,
                alpha=alpha, min_size=1, verbose=False
            )

    return (method, seed, np.array(result['perm'], dtype=np.int64))


def main():
    output_path = './test_results/probing_classifier/grouping_plans.npz'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_workers = min(mp.cpu_count(), 32)
    print(f"Workers: {n_workers}", flush=True)

    all_plans = {}
    total_start = time.time()

    for dataset_name in DATASETS:
        ds = DATASETS[dataset_name]
        N = ds['enc_in']
        G = ds['num_groups']

        print(f"\n{'='*50}", flush=True)
        print(f"Dataset: {dataset_name} (N={N}, G={G})", flush=True)

        X_train = load_train_data(dataset_name)
        print(f"  Train shape: {X_train.shape}", flush=True)

        # Precompute similarity matrices
        t0 = time.time()
        S_spearman = spearman_correlation_matrix(X_train, absolute=True)
        print(f"  Spearman: {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        S_mi = mi_proxy_matrix(X_train)
        print(f"  MI proxy: {time.time()-t0:.1f}s", flush=True)

        # Fast methods (sequential)
        fast = {'ordered', 'random', 'score_stratified', 'finch_like',
                'coarsening', 'mi_based'}
        for method in METHODS:
            if method not in fast:
                continue
            t0 = time.time()
            for seed in SEEDS:
                _, _, perm = compute_one(
                    (method, seed, N, G, S_spearman, S_mi, X_train))
                key = f"{dataset_name}__{method}__{seed}"
                all_plans[key] = perm
            print(f"  {method}: {len(SEEDS)} in {time.time()-t0:.1f}s",
                  flush=True)

        # Slow methods (parallel)
        slow_tasks = []
        for method in METHODS:
            if method not in ('anti_clustering', 'maximin_dispersion'):
                continue
            for seed in SEEDS:
                slow_tasks.append(
                    (method, seed, N, G, S_spearman, S_mi, X_train))

        if slow_tasks:
            print(f"  Parallelizing {len(slow_tasks)} slow plans "
                  f"({n_workers} workers)...", flush=True)
            t0 = time.time()
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(compute_one, slow_tasks)
            for method, seed, perm in results:
                key = f"{dataset_name}__{method}__{seed}"
                all_plans[key] = perm
            print(f"  anti_clustering + maximin_dispersion: "
                  f"{time.time()-t0:.1f}s (parallel)", flush=True)

    # Save
    np.savez_compressed(output_path, **all_plans)
    elapsed = time.time() - total_start
    print(f"\nSaved {len(all_plans)} plans to {output_path}", flush=True)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)", flush=True)


if __name__ == '__main__':
    main()
