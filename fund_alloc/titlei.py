from collections import defaultdict

from titlei_util import *
from mltoolkit import parse_args
from gpu_dp import gpu_dgaussian_samples
from constraint_opt import solve_hierarchical_counts, FormulaCounts
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_epsilons_per_level(rho, weights):
    rhos = rho * np.array(weights)
    epsilons = rhos + 2 * np.sqrt(-rhos * np.log(10 ** -10))
    return epsilons


def fast_compute_basic_alloc(formula_pop, total_pop, adj_sppe_, total_funds):
    eligible_mask = (formula_pop > 10) & ((formula_pop / total_pop) > 0.02)
    basic_auth_amt = eligible_mask * formula_pop * torch.tensor(adj_sppe_[:, None], device=device)
    basic_alloc_amt = total_funds * (basic_auth_amt / basic_auth_amt.sum(dim=0))
    return basic_alloc_amt


def fast_compute_concentrated_alloc(formula_pop, total_pop, adj_sppe_, total_funds):
    eligible_mask = (formula_pop > 6500) | ((formula_pop / total_pop) > 0.15)
    concentrated_auth_amt = eligible_mask * formula_pop * torch.tensor(adj_sppe_[:, None], device=device)
    concentrated_alloc_amt = total_funds * (concentrated_auth_amt / concentrated_auth_amt.sum(dim=0))
    return concentrated_alloc_amt


def fast_targeted_grant_weight(eligible, total):
    # calculate weighted count based on counts
    wec_counts = torch.zeros(eligible.shape, device=device)
    for r, w in {
        (1, 691): 1.0,
        (692, 2262): 1.5,
        (2263, 7851): 2.0,
        (7852, 35514): 2.5,
        (35514, float('inf')): 3.0
    }.items():
        wec_counts += (eligible >= r[0]) * (torch.minimum(torch.tensor(r[1]), eligible) - r[0] + 1) * w

    # calculate weighted count based on proportions
    wec_props = torch.zeros(eligible.shape, device=device)
    prop = eligible / total
    prop[torch.isnan(prop)] = 0
    for r, w in {
        (0, 0.1558): 1.0,
        (0.1558, 0.2211): 1.75,
        (0.2211, 0.3016): 2.5,
        (0.3016, 0.3824): 3.25,
        (0.3824, float('inf')): 4.0
    }.items():
        wec_props += (prop >= r[0]) * (torch.minimum(torch.tensor(r[1]), prop) - r[0]) * w * total

    # take the higher weighted eligibility count
    return torch.maximum(wec_counts, wec_props)


def fast_compute_targeted_alloc(formula_pop, total_pop, adj_sppe_, total_funds):
    weighted_formula_pop = fast_targeted_grant_weight(formula_pop, total_pop)
    eligible_mask = (formula_pop > 10) & ((formula_pop / total_pop) > 0.05)

    targeted_authorization_amount = eligible_mask * weighted_formula_pop * torch.tensor(adj_sppe_[:, None],
                                                                                        device=device)
    targeted_authorization_amount[torch.isnan(targeted_authorization_amount)] = 0
    targeted_allocation = total_funds * (targeted_authorization_amount / targeted_authorization_amount.sum(dim=0))

    return targeted_allocation


def fast_compute_noisy_counts(real_child_formula, real_child, real_state_formula, real_national_formula, epsilons,
                              num_samples, adj_sppe, real_basic, real_target, real_concentrated,
                              max_gpu_size=int(250e6), hierarchal=False):
    # number of rows in the dataframe
    length = real_child.shape[0]
    max_gpu_samples = int(max_gpu_size / length)
    total_samples = num_samples

    noisy_keys = ['noisy_children_formula_count', 'noisy_children_count', 'noisy_state_formula_count',
                  'noisy_national_formula_count']
    cols = [real_child_formula, real_child, real_state_formula, real_national_formula]
    cols_tensors = [torch.tensor(col.values, dtype=torch.float32, device=device)[:, None] for col in cols]
    scales = [2.0 / epsilons[0], 2.0 / epsilons[0], 1.0 / epsilons[1], 1.0 / epsilons[2]]
    data = list(zip(noisy_keys, cols_tensors, scales))

    # compute the 0 indexed state mask
    sm = torch.tensor(real_child_formula.index.get_level_values('State FIPS Code').tolist()).to(torch.int)
    state_codes = torch.tensor(real_state_formula.index.get_level_values('State FIPS Code').values).to(torch.int)
    for i, code in enumerate(state_codes):
        sm[sm == code] = i

    results = defaultdict(lambda: [].copy())
    avgs = defaultdict(lambda: [].copy())

    avg_noisy_child_formula = torch.zeros(length, device=device)
    avg_noisy_child = torch.zeros(length, device=device)
    avg_noisy_state_formula = torch.zeros(len(real_state_formula))
    avg_noisy_national_formula = torch.zeros(len(real_national_formula))

    for i in tqdm(range(0, total_samples, max_gpu_samples)):
        # compute the noisy samples on GPU
        count = min(max_gpu_samples, total_samples - i)
        for key, col, scale in data:
            noise = gpu_dgaussian_samples(scale, 'cuda', (len(col), count))
            results[key] = (col + noise)

        # compute hierarchical solutions
        # ncf = noisy child formula, ncc = noisy child count, nsf = noisy state formula, nnf = noisy national formula
        ncf, ncc, nsf, nnf = [results[key] for key in noisy_keys]

        avg_noisy_child += ncc.sum(dim=1)
        # sm = state mask, give each state
        if hierarchal:
            with multiprocessing.Pool(processes=8) as pool:
                f = solve_hierarchical_counts
                inputs = [(nnf[:, i], nsf[:, i], sm, ncf[:, i]) for i in range(ncf.shape[1])]
                for results in tqdm(pool.imap(f, inputs), total=ncf.shape[1]):
                    results: FormulaCounts
                    # compute basic, concentration, and target allocations
                    avg_noisy_child_formula += results.county

        basic_alloc = fast_compute_basic_alloc(ncf, ncc, adj_sppe, real_basic)
        concentration_alloc = fast_compute_concentrated_alloc(ncf, ncc, adj_sppe, real_concentrated)
        target_alloc = fast_compute_targeted_alloc(ncf, ncc, adj_sppe, real_target)

        avgs['noisy_basic_alloc'].append(basic_alloc.mean(dim=1)[:, None])
        avgs['noisy_concentration_alloc'].append(concentration_alloc.mean(dim=1)[:, None])
        avgs['noisy_target_alloc'].append(target_alloc.mean(dim=1)[:, None])

        # average it all out
        for key in results.keys():
            avgs[key].append(results[key].mean(dim=1)[:, None])

        num_samples -= count

    # concat the results
    for key in avgs.keys():
        avgs[key] = torch.cat(avgs[key], dim=1).mean(dim=1).cpu()

    return avgs


def main():
    # configuration
    args: TitleIArguments = parse_args(TitleIArguments)
    weights = [0.085, 0.274, 0.02]
    weights = [w * 1 / 1426 for w in weights]
    stuff = [(2.56, weights), (1.0, weights), (0.1, weights)]
    hierarchical = False
    # compute epsilon versions
    epsilons = compute_epsilons_per_level(args.global_rho, weights)

    for rho, weights in stuff:
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
        official_national_formula_population = pd.Series([official_state_formula_population.sum()])
        official_county_population = df['official_children_count']

        # calculate with our formula the allocation amounts for each grant
        basic_official_alloc = compute_basic_alloc(official_county_formula_population, official_county_population,
                                                   adj_sppe_,
                                                   basic_total_available_funds)
        concentration_official_alloc = compute_concentrated_alloc(official_county_formula_population,
                                                                  official_county_population,
                                                                  adj_sppe_,
                                                                  concentration_total_available_funds)
        targeted_official_alloc = compute_targeted_alloc(official_county_formula_population, official_county_population,
                                                         adj_sppe_,
                                                         targeted_total_available_funds)

        print(f'epsilons (county, state, national): {epsilons}')

        results = fast_compute_noisy_counts(official_county_formula_population, official_county_population,
                                            official_state_formula_population, official_national_formula_population,
                                            epsilons,
                                            args.num_trials,
                                            adj_sppe_,
                                            basic_total_available_funds,
                                            concentration_total_available_funds,
                                            targeted_total_available_funds, hierarchal=False)

        df['noisy_children_formula_count'] = pd.Series(results['noisy_children_formula_count'].numpy().flatten(),
                                                       index=df.index)
        df['noisy_children_count'] = pd.Series(results['noisy_children_count'].numpy().flatten(),
                                               index=df.index)
        df['noisy_basic_alloc'] = pd.Series(np.array(results['noisy_basic_alloc']).flatten(), index=df.index)
        df['noisy_concentration_alloc'] = pd.Series(np.array(results['noisy_basic_alloc']).flatten(),
                                                    index=df.index)
        df['noisy_target_alloc'] = pd.Series(np.array(results['noisy_basic_alloc']).flatten(), index=df.index)
        df['calculated_basic_alloc'] = pd.Series(basic_official_alloc, index=df.index)
        df['calculated_concentration_alloc'] = pd.Series(concentration_official_alloc, index=df.index)
        df['calculated_target_alloc'] = pd.Series(targeted_official_alloc, index=df.index)

        # save the dataframe to a file
        filename = f'out/data1M/df_noisy_out_census_dp_rho={rho}_hierarchical={hierarchical}.csv'
        with open(filename, 'w+') as f:
            # write the metadata as comments
            f.write(f'# epsilon={epsilons}, rho={args.global_rho}\n')

            # save the dataframe to the file without headers
            df.to_csv(f)


if __name__ == '__main__':
    main()
