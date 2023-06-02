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
    # TODO; finish this
    rhos = rho * np.array(weights) * 1 / 4726
    epsilons = rhos + 2 * np.sqrt(-rhos * np.log(10 ** -10))
    return epsilons


def compute_non_noisy_alloc(df):
    # total_voting_age_citizens_by_county = df.groupby('S203_GEOID')['VACIT'].sum()
    # total_limit_english_no_5th_grade_by_county = df.groupby('S203_GEOID')['ILLIT'].sum()
    # total_limit_english = df.groupby('S203_GEOID')['VACLEP'].sum()
    total_voting_age_citizens_by_county = df['VACIT']
    total_limit_english_no_5th_grade_by_county = df['ILLIT']
    total_limit_english = df['VACLEP']
    flag5 = (total_limit_english / total_voting_age_citizens_by_county) > 0.05
    flag10 = total_limit_english > 10000
    flag_national = (total_limit_english_no_5th_grade_by_county / total_voting_age_citizens_by_county) > 0.0131

    coverage = (flag5 | flag10) & flag_national
    return coverage


def compute_allocs(df):
    total_voting_age_citizens_by_county = df['noisy_vacit']
    total_limit_english_no_5th_grade_by_county = df['noisy_illit']
    total_limit_english = df['noisy_vaclep']
    leppct = total_limit_english / total_voting_age_citizens_by_county
    flag5 = leppct > 0.05
    flag10 = total_limit_english > 10000
    flag_national = (total_limit_english_no_5th_grade_by_county / total_voting_age_citizens_by_county) > 0.0131

    coverage = (flag5 | flag10) & flag_national
    return leppct, flag5, flag10, flag_national, coverage


def compute_noisy_allocs(df, epsilons, _args):
    S = math.sqrt(3)
    # no. of voting age citizens with limited english proficiency AND didn't graduate 5th grade
    df['noisy_illit'] = df['gt_illit'].apply(lambda x: max(int(dgaussian_fn(S / epsilons[0])(x)), 1))

    # no. of voting age citizens
    df['noisy_vacit'] = df['gt_vacit'].apply(lambda x: max(int(dgaussian_fn(S / epsilons[0])(x)), 1))

    # no. of voting age citizens with limited english proficiency
    df['noisy_vaclep'] = df['gt_vaclep'].apply(lambda x: max(int(dgaussian_fn(S / epsilons[0])(x)), 1))

    # set up noisy allocator function
    return compute_allocs(df)


def main():
    # configuration
    args: Arguments = parse_args(Arguments)

    # load in language data
    with codecs.open('datasets/sect203_Determined_Areas_Only.csv', 'r', encoding='utf-8', errors='ignore') as fdata:
        df = pd.read_csv(fdata)
        df = df[df['LANGUAGE'] == args.language]
        df = df.dropna(subset=['VACIT', 'VACLEP', 'ILLIT'])
        df = df[df['LEVEL'] == 'County']

    df['pred_illrat'] = df['ILLIT'] / df['VACLEP']
    df['pred_leppct'] = df['VACLEP'] / df['S203_GEOID'].map(df.groupby('S203_GEOID')['VACIT'].sum())

    df['gt_vaclep'] = df['VACLEP']
    df['gt_vacit'] = df['VACIT']
    df['gt_illit'] = df['ILLIT']

    df['gt_coverage'] = compute_non_noisy_alloc(df)

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
    leppct, flag5, flag10, flag_national, coverage = avgs
    df['noisy_leppct'] = leppct
    df['noisy_flag5'] = flag5 > 0.5
    df['noisy_flag10'] = flag10 > 0.5
    df['noisy_flag_national'] = flag_national > 0.5
    df['noisy_coverage'] = coverage > 0.5

    if not os.path.exists('out/'):
        os.makedirs('out/', exist_ok=True)

    file_name = f'out/df_noisy_out_language_dp_rho={args.global_rho}.csv'
    print(f'saving to {file_name}')
    df.to_csv(file_name)
    print('errors: ', np.sum(df['noisy_coverage'] != df['gt_coverage']))
    print('regular errors: ', np.sum(df['gt_coverage'] != df['FLAG_COV']))
    print('flag5 errors: ', np.sum(df['noisy_flag5'] != df['FLAG5']))


if __name__ == '__main__':
    main()
