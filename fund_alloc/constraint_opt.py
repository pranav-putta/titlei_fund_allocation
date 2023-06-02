from dataclasses import dataclass

import numpy as np
import pandas as pd
import gurobipy as gp
import torch


# from pyscipopt import Model, quicksum
# import pyomo.environ as pyo

@dataclass
class FormulaCounts:
    county: torch.Tensor
    state: torch.Tensor
    national: torch.Tensor


def greedy_fix_one_off(agg, counts, indices):
    diff = agg - sum(counts[indices])
    for i in range(abs(int(diff))):
        counts[indices[i % len(indices)]] += diff / abs(diff)
    return counts


def verify_sums(national, state, state_masks, county):
    # check sum(state) == national
    if sum(national) != sum(state):
        return False

    # check sum(county) == state
    for i in range(state_masks.min(), state_masks.max() + 1):
        if sum(county[state_masks == i]) != state[i]:
            return False
    return True


def solve_hierarchical_counts(args):
    national_counts, state_counts, state_mask, county_counts = args
    # convert tensors into float types
    national_counts = national_counts.float()
    state_counts = state_counts.float()
    county_counts = county_counts.float()

    with gp.Env(empty=True) as env:
        env.start()

        model = gp.Model(env=env)
        opt_type = gp.GRB.CONTINUOUS
        n1, n2, n3 = len(national_counts), len(state_counts), len(county_counts)

        x1 = model.addVars(n1, lb=[0] * n1, vtype=opt_type)
        x2 = model.addVars(n2, lb=[0] * n2, vtype=opt_type)
        x3 = model.addVars(n3, lb=[0] * n3, vtype=opt_type)

        model.addConstr(gp.quicksum(x3) == x1[0], name='county->national')
        model.addConstr(gp.quicksum(x2) == x1[0], name='state->national')

        for i in range(state_mask.min(), state_mask.max() + 1):
            model.addConstr(gp.quicksum(x3[i] for i in torch.argwhere(state_mask == i)[:, 0].tolist()) == x2[i],
                            name=f'county->state:{i}')

        obj = 0
        for count, var in [(national_counts, x1), (state_counts, x2), (county_counts, x3)]:
            for i in range(len(count)):
                obj += (var[i] - count[i]) ** 2

        model.setObjective(obj)
        model.setParam('OutputFlag', False)
        model.setParam('OptimalityTol', 1e-6)
        model.optimize()

        # greedily fix 1-off errors
        # import pdb
        # pdb.set_trace()
        # round everything to nearest integer
        national_counts = torch.round(torch.tensor([x1[i].x for i in range(n1)]))
        state_counts = torch.round(torch.tensor([x2[i].x for i in range(n2)]))
        county_counts = torch.round(torch.tensor([x3[i].x for i in range(n3)]))

        state_counts = (greedy_fix_one_off(national_counts[0], state_counts, list(range(len(state_counts)))))
        assert sum(state_counts) == sum(national_counts)
        for i in range(state_mask.min(), state_mask.max() + 1):
            indices = torch.argwhere(state_mask == i)[:, 0].tolist()
            county_counts = greedy_fix_one_off(state_counts[i], county_counts, indices)

        assert verify_sums(national_counts, state_counts, state_mask, county_counts)

    return FormulaCounts(county=county_counts, state=state_counts, national=national_counts)
