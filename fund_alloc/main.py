import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from constraint_opt import solve_hierarchical_counts
from data_util import get_inputs, get_sppe, compute_adj_sppe
from util_fns import dgaussian_fn


def compute_basic_alloc(formula_pop, total_pop, adj_sppe_, total_funds):
    eligible_mask = (formula_pop > 10) & ((formula_pop / total_pop) > 0.02)
    basic_auth_amt = eligible_mask * formula_pop * adj_sppe_
    basic_alloc_amt = total_funds * (basic_auth_amt / basic_auth_amt.sum())
    return basic_alloc_amt


def compute_concentrated_alloc(formula_pop, total_pop, adj_sppe_, total_funds):
    eligible_mask = (formula_pop > 6500) | ((formula_pop / total_pop) > 0.15)
    concentrated_auth_amt = eligible_mask * formula_pop * adj_sppe_
    concentrated_alloc_amt = total_funds * (concentrated_auth_amt / concentrated_auth_amt.sum())
    return concentrated_alloc_amt


def targeted_grant_weight(eligible, total):
    # calculate weighted count based on counts
    wec_counts = np.zeros(len(eligible))
    for r, w in {
        (1, 691): 1.0,
        (692, 2262): 1.5,
        (2263, 7851): 2.0,
        (7852, 35514): 2.5,
        (35514, float('inf')): 3.0
    }.items():
        wec_counts += (eligible >= r[0]) * (np.minimum(r[1], eligible) - r[0] + 1) * w

    # calculate weighted count based on proportions
    wec_props = np.zeros(len(eligible))
    prop = eligible / total
    prop[np.isnan(prop)] = 0
    for r, w in {
        (0, 0.1558): 1.0,
        (0.1558, 0.2211): 1.75,
        (0.2211, 0.3016): 2.5,
        (0.3016, 0.3824): 3.25,
        (0.3824, float('inf')): 4.0
    }.items():
        wec_props += (prop >= r[0]) * (np.minimum(r[1], prop) - r[0]) * w * total

    # take the higher weighted eligibility count
    return np.maximum(wec_counts, wec_props)


def compute_targeted_alloc(formula_pop, total_pop, adj_sppe_, total_funds):
    weighted_formula_pop = targeted_grant_weight(formula_pop, total_pop)
    eligible_mask = (formula_pop > 10) & ((formula_pop / total_pop) > 0.05)

    targeted_authorization_amount = eligible_mask * weighted_formula_pop * adj_sppe_
    targeted_allocation = total_funds * (targeted_authorization_amount / targeted_authorization_amount.sum())

    return targeted_allocation


def compute_alloc(df, formula_col, total_col, adj_sppe_, total_funds):
    eligible_mask = (df[formula_col] > 10) & (
            df[formula_col] / df[total_col] > 0.02)
    official_authorization_amount = eligible_mask * df[formula_col] * adj_sppe_
    official_allocation = total_funds * (official_authorization_amount / official_authorization_amount.sum())
    return official_allocation


def compute_noisy_alloc(df, epsilons, use_constraints, adj_sppe_, total_available_funds,
                        official_state_formula_population, official_national_population, _args):
    basic_funds, concentration_funds, target_funds = total_available_funds
    df['noisy_children_formula_count'] = df['official_children_formula_count'].apply(
        lambda x: max(int(dgaussian_fn(x, 2.0 / epsilons[0])), 1))
    df['noisy_children_count'] = df['official_children_count'].apply(
        lambda x: max(int(dgaussian_fn(x, 2.0 / epsilons[0])), 1))
    noisy_state_formula_count = official_state_formula_population.apply(
        lambda x: max(int(dgaussian_fn(x, 1.0 / epsilons[1])), 1))
    noisy_national_formula_count = max(int(dgaussian_fn(official_national_population, 1.0 / epsilons[2])), 1)
    if use_constraints:
        national, state, df['noisy_children_formula_count'] = solve_hierarchical_counts(
            pd.Series([noisy_national_formula_count]),
            noisy_state_formula_count,
            df['noisy_children_formula_count'])

    noisy_children_formula_count = df['noisy_children_formula_count']
    noisy_children_count = df['noisy_children_count']

    # compute allocation
    basic_alloc = compute_basic_alloc(noisy_children_formula_count, noisy_children_count, adj_sppe_, basic_funds)
    concentration_alloc = compute_concentrated_alloc(noisy_children_formula_count, noisy_children_count, adj_sppe_,
                                                     concentration_funds)
    target_alloc = compute_targeted_alloc(noisy_children_formula_count, noisy_children_count, adj_sppe_,
                                          target_funds)

    return basic_alloc, concentration_alloc, target_alloc


def main():
    # configuration
    num_trials = 1000
    global_rho = 0.1
    weights = np.array([0.085, 0.274, 0.02])
    use_constraints = True

    # load in school district data and state expenditure data
    df = get_inputs(2021, use_official_children=True)
    sppe = get_sppe('datasets/sppe18.xlsx')

    df = df.join(sppe['ppe'].rename('sppe'), how='left')
    df = df.dropna(subset=['sppe'])
    df.sppe = df.sppe.astype(float)

    adj_sppe_, _ = compute_adj_sppe(df.sppe)

    # compute total funds actually available ; this is public information
    basic_total_available_funds = df['basic_alloc'].sum()
    concentration_total_available_funds = df['concentration_alloc'].sum()
    targeted_total_available_funds = df['targeted_alloc'].sum()

    # real counts
    official_county_formula_population = df['official_children_formula_count']
    official_state_formula_population = official_county_formula_population.groupby('State FIPS Code').sum()
    official_national_population = official_state_formula_population.sum()
    total_pop_count = df['official_children_count']

    # calculate with our formula the allocation amounts for each grant
    basic_official_alloc = compute_basic_alloc(official_county_formula_population, total_pop_count, adj_sppe_,
                                               basic_total_available_funds)
    concentration_official_alloc = compute_concentrated_alloc(official_county_formula_population, total_pop_count,
                                                              adj_sppe_,
                                                              concentration_total_available_funds)
    targeted_official_alloc = compute_targeted_alloc(official_county_formula_population, total_pop_count, adj_sppe_,
                                                     targeted_total_available_funds)

    # compute dp versions
    rhos = global_rho * weights
    epsilons = rhos + 2 * np.sqrt(-rhos * np.log(10 ** -10))

    print(f'epsilons (county, state, national): {epsilons}')

    # set up noisy allocator function
    total_available_funds = (
        basic_total_available_funds,
        concentration_total_available_funds,
        targeted_total_available_funds
    )
    f = partial(compute_noisy_alloc, df, epsilons, use_constraints, adj_sppe_, total_available_funds,
                official_state_formula_population, official_national_population)

    # compute for N trials
    with multiprocessing.Pool(processes=4) as pool:
        noisy_allocs = list(tqdm(pool.imap(f, range(num_trials)), total=num_trials))

    # take the average of the noisy allocations and update the dataframe
    noisy_basic_alloc, noisy_concentration_alloc, noisy_target_alloc = list(zip(*noisy_allocs))
    avg_noisy_basic_alloc = pd.Series.sum(pd.concat(noisy_basic_alloc, axis=1), axis=1) / num_trials
    avg_noisy_concentration_alloc = pd.Series.sum(pd.concat(noisy_concentration_alloc, axis=1), axis=1) / num_trials
    avg_noisy_target_alloc = pd.Series.sum(pd.concat(noisy_target_alloc, axis=1), axis=1) / num_trials

    df['avg_noisy_basic_alloc'] = avg_noisy_basic_alloc
    df['avg_noisy_concentration_alloc'] = avg_noisy_concentration_alloc
    df['avg_noisy_target_alloc'] = avg_noisy_target_alloc

    df['calculated_basic_alloc'] = basic_official_alloc
    df['calculated_concentrated_alloc'] = concentration_official_alloc
    df['calculated_targeted_alloc'] = targeted_official_alloc

    df.to_csv(f'out/df_noisy_out_census_dp_rho={global_rho}.csv')


if __name__ == '__main__':
    main()
