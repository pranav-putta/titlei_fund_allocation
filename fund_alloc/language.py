import math
import multiprocessing
import os
import time
from dataclasses import field
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from mltoolkit import argclass, parse_args
import codecs

from tqdm import tqdm

from fund_alloc.util_fns import dgaussian_fn


@argclass
class Arguments:
    num_trials: int = field(default=1000)
    global_rho: float = field(default=0.1)
    weights: List[float] = field(default_factory=lambda: ([0.085, 0.274, 0.02].copy()))
    language: str = field(default="Hispanic")

    use_constraints: bool = field(default=True)
    num_workers: int = field(default=4)


def compute_epsilons_per_level(rho, weights):
    rhos = rho * np.array(weights) * 1 / 1426
    epsilons = rhos + 2 * np.sqrt(-rhos * np.log(10 ** -10))
    return epsilons


def compute_alloc(df, key_vaclep='VACLEP', key_illit='ILLIT', key_geo_pop='GEO_POP'):
    leppct = (100 * (df[key_vaclep] / df[key_geo_pop])).round(decimals=1)
    flag5 = df['pred_flag5'] = leppct >= 5
    flag10 = df['pred_flag10'] = df[key_vaclep] >= 10000
    flagedu = df['pred_flagedu'] = (df[key_illit] / df[key_vaclep]).round(decimals=4) >= 0.0131

    coverage = (flag5 | flag10) & flagedu
    return leppct, coverage


def compute_noisy_allocs(df, epsilons, _args):
    S = math.sqrt(3)
    # no. of voting age citizens with limited english proficiency AND didn't graduate 5th grade
    df['noisy_illit'] = df['ILLIT'].apply(lambda x: max(int(dgaussian_fn(S / epsilons[0])(x)), 1))
    # no. of voting age citizens
    df['noisy_geo_pop'] = df['GEO_POP'].apply(lambda x: max(int(dgaussian_fn(S / epsilons[0])(x)), 1))
    # no. of voting age citizens with limited english proficiency
    df['noisy_vaclep'] = df['VACLEP'].apply(lambda x: max(int(dgaussian_fn(S / epsilons[0])(x)), 1))
    # set up noisy allocator function
    leppct, coverage = compute_alloc(df, key_vaclep='noisy_vaclep', key_illit='noisy_illit',
                                     key_geo_pop='noisy_geo_pop')
    return leppct, coverage, df['noisy_illit'], df['noisy_geo_pop'], df['noisy_vaclep']


def main():
    # configuration
    args: Arguments = parse_args(Arguments)

    # load in language data
    with codecs.open('datasets/sect203_Determined_Areas_Only.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
        df = pd.read_csv(fdata)
        df = df[df['LEVEL'] == 'County']
        df['GEO_POP'] = df.groupby('S203_GEOID')['VACIT'].transform(lambda x: x.iloc[0])
        df = df.dropna(subset=['VACIT', 'VACLEP'])

    df['pred_leppct'], df['pred_cov'] = compute_alloc(df)
    df = df[df['LANGUAGE'] == args.language]
    # compute dp versions
    epsilons = compute_epsilons_per_level(args.global_rho, args.weights)
    print(f'epsilons (county, state, national): {epsilons}')

    # compute for N trials
    start = time.time()
    avgs = None

    f = partial(compute_noisy_allocs, df, epsilons)
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for results in tqdm(pool.imap(f, range(args.num_trials)), total=args.num_trials):
            if avgs is None:
                avgs = [pd.Series(np.zeros(len(df)), index=results[i].index) for i in range(len(results))]
            avgs = [avgs[i] + (results[i] / args.num_trials) for i in range(len(results))]

    end = time.time()
    print(end - start, 's')
    # take the average of the noisy allocations and update the dataframe
    leppct, coverage, noisy_illit, noisy_geopop, noisy_vaclep = avgs
    df['noisy_leppct'] = leppct
    df['noisy_illit'] = noisy_illit
    df['noisy_geopop'] = noisy_geopop
    df['noisy_vaclep'] = noisy_vaclep
    df['noisy_coverage'] = coverage > 0.5

    if not os.path.exists('out/'):
        os.makedirs('out/', exist_ok=True)

    file_name = f'out/dataL1K/df_noisy_out_language_dp_rho={args.global_rho}.csv'
    print(f'saving to {file_name}')
    df.to_csv(file_name)


if __name__ == '__main__':
    main()
