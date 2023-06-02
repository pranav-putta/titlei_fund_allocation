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


def fast_compute_noisy_counts(real_child_formula, real_child, real_state_formula, real_national_formula, epsilons,
                              num_samples, adj_sppe, real_basic, real_target, real_concentrated,
                              max_gpu_size=int(750e6), hierarchal=False):
    # use gpu to compute noisy counts

    # number of rows in the dataframe
    length = real_child.shape[0]
    max_gpu_samples = int(max_gpu_size / length)

    noisy_keys = ['noisy_children_formula_count', 'noisy_children_count', 'noisy_state_formula_count',
                  'noisy_national_formula_count']
    cols = [real_child_formula, real_child, real_state_formula, real_national_formula]
    cols_tensors = [torch.tensor(col.values, dtype=torch.float32, device=device)[:, None] for col in cols]
    scales = [2.0 / epsilons[0], 2.0 / epsilons[0], 1.0 / epsilons[1], 1.0 / epsilons[2]]
    data = zip(noisy_keys, cols_tensors, scales)

    # compute the 0 indexed state mask
    sm = torch.tensor(real_child_formula.index.get_level_values('State FIPS Code').tolist()).to(torch.int)
    state_codes = torch.tensor(real_state_formula.index.get_level_values('State FIPS Code').values).to(torch.int)
    for i, code in enumerate(state_codes):
        sm[sm == code] = i

    results = {'noisy_children_formula_count': [],
               'noisy_children_count': [],
               'noisy_state_formula_count': [],
               'noisy_national_formula_count': []}

    avg_noisy_child_formula = torch.zeros(length)
    avg_noisy_child = torch.zeros(length)
    avg_noisy_state_formula = torch.zeros(len(real_state_formula))
    avg_noisy_national_formula = torch.zeros(len(real_national_formula))

    while num_samples > 0:
        # compute the noisy samples on GPU
        count = min(num_samples, max_gpu_samples)
        for key, col, scale in data:
            noise = gpu_dgaussian_samples(scale, 'cuda', (len(col), count))
            results[key].append((col + noise).cpu())

        # compute hierarchical solutions
        # ncf = noisy child formula, ncc = noisy child count, nsf = noisy state formula, nnf = noisy national formula
        ncf, ncc, nsf, nnf = [results[key][-1] for key in noisy_keys]
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

        num_samples -= max_gpu_samples

    # average the results
    avg_noisy_state_formula /= num_samples
    avg_noisy_national_formula /= num_samples
    avg_noisy_child_formula /= num_samples
    avg_noisy_child /= num_samples

    basic_alloc = compute_basic_alloc(avg_noisy_child_formula, avg_noisy_child, adj_sppe, real_basic)
    concentration_alloc = compute_concentrated_alloc(avg_noisy_child_formula, avg_noisy_child, adj_sppe,
                                                     real_concentrated)
    target_alloc = compute_targeted_alloc(avg_noisy_child_formula, avg_noisy_child, adj_sppe,
                                          real_target)

    return {'noisy_children_formula_count': avg_noisy_child_formula,
            'noisy_children_count': avg_noisy_child,
            'noisy_state_formula_count': avg_noisy_state_formula,
            'noisy_national_formula_count': avg_noisy_national_formula,
            'noisy_basic_alloc': basic_alloc,
            'noisy_concentration_alloc': concentration_alloc,
            'noisy_target_alloc': target_alloc}


def main():
    # configuration
    args: TitleIArguments = parse_args(TitleIArguments)
    weights = [0.085, 0.274, 0.02]
    weights = [w * 1 / 4726 for w in weights]
    stuff = [(2.56, weights), (1.0, weights), (0.1, weights)]
    hierarchical = False

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

        # compute dp versions
        epsilons = compute_epsilons_per_level(args.global_rho, args.weights)

        print(f'epsilons (county, state, national): {epsilons}')

        # set up noisy allocator function
        total_available_funds = (
            basic_total_available_funds,
            concentration_total_available_funds,
            targeted_total_available_funds
        )

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
        # save the dataframe to a file
        filename = f'out/data/df_noisy_out_census_dp_rho={rho}_hierarchical={hierarchical}.csv'
        with open(filename, 'w+') as f:
            # write the metadata as comments
            f.write(f'# epsilon={epsilons}, rho={args.global_rho}\n')

            # save the dataframe to the file without headers
            df.to_csv(f, header=None, index=None)


if __name__ == '__main__':
    main()
