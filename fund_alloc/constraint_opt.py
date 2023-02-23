import numpy as np
import pandas as pd
from gurobipy import *


def greedy_fix_one_off(agg, counts):
    diff = agg - sum(counts)
    for i in range(abs(int(diff))):
        counts.iloc[i] += diff / abs(diff)
    return counts


def verify_sums(national, state, county):
    state_groups = state.reset_index().groupby("State FIPS Code").groups
    county_groups = county.reset_index().groupby('State FIPS Code').groups

    # check sum(state) == national
    if sum(national) != sum(state):
        return False

    for state_idx, indices in county_groups.items():
        if sum(county.iloc[indices]) != sum(state.iloc[state_groups[state_idx]]):
            return False
    return True


def solve_hierarchical_counts(national_counts: pd.Series, state_counts: pd.Series, county_counts: pd.Series):
    opt_type = GRB.CONTINUOUS
    n1, n2, n3 = len(national_counts), len(state_counts), len(county_counts)
    model = Model()

    x1 = model.addVars(n1, lb=[0] * n1, vtype=opt_type)
    x2 = model.addVars(n2, lb=[0] * n2, vtype=opt_type)
    x3 = model.addVars(n3, lb=[0] * n3, vtype=opt_type)

    model.addConstr(quicksum(x3) == x1[0], name='county->national')
    model.addConstr(quicksum(x2) == x1[0], name='state->national')

    state_groups = state_counts.reset_index().groupby("State FIPS Code").groups
    county_groups = county_counts.reset_index().groupby('State FIPS Code').groups.items()

    for state_idx, indices in county_groups:
        model.addConstr(quicksum([x3[i] for i in indices]) == x2[state_groups[state_idx][0]],
                        name=f'county->state:{state_idx}')

    obj = 0
    _list_national, _list_state, _list_county = list(national_counts), list(state_counts), list(county_counts)
    for count, var in [(_list_national, x1), (_list_state, x2), (_list_county, x3)]:
        for i in range(len(count)):
            obj += (var[i] - count[i]) ** 2

    model.setObjective(obj)
    model.setParam('OutputFlag', False)
    model.setParam('OptimalityTol', 1e-6)
    model.optimize()

    national_counts.update(pd.Series([int(np.round(x1[i].x)) for i in range(n1)], index=national_counts.index))
    state_counts.update(pd.Series([int(np.round(x2[i].x)) for i in range(n2)], index=state_counts.index))
    county_counts.update(pd.Series([int(np.round(x3[i].x)) for i in range(n3)], index=county_counts.index))

    # greedily fix 1-off errors
    state_counts = (greedy_fix_one_off(national_counts[0], state_counts))
    assert sum(state_counts) == sum(national_counts)
    for state_idx, indices in county_groups:
        county_counts.update(pd.Series(greedy_fix_one_off(
            state_counts.iloc[state_groups[state_idx][0]],
            county_counts.iloc[indices]), index=county_counts.index[indices]))

    assert verify_sums(national_counts, state_counts, county_counts)

    return national_counts, state_counts, county_counts
