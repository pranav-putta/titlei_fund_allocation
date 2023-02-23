import os
from typing import Tuple

import numpy as np
import pandas as pd


def get_official(path, sheet, header, columns):
    allocs = pd.read_excel(path, sheet_name=sheet, header=header)
    allocs = allocs.iloc[1:, :len(columns)]
    allocs.columns = columns
    return allocs


def median_cv(total_pop: float) -> float:
    """
    Based on the table given here:
    https://www.census.gov/programs-surveys/saipe/guidance/district-estimates.html
    Args:
        total_pop (float): Total population of district.
    Returns:
        float: Coefficient of variation.
    """
    if total_pop <= 2500:
        return 0.67
    elif total_pop <= 5000:
        return 0.42
    elif total_pop <= 10000:
        return 0.35
    elif total_pop <= 20000:
        return 0.28
    elif total_pop <= 65000:
        return 0.23
    return 0.15


def split_leaids(leaids: pd.Series):
    # get the last seven digits of the ID
    leaids = leaids.astype(str).str[-7:]
    return leaids.str[-5:].astype(int), leaids.str[:-5].astype(int)


def get_official_combined(path: str) -> pd.DataFrame:
    """Load official Dept Ed data.
    Args:
        path (str): Path of file.
    Returns:
        pd.DataFrame: Return cleaned dataframe.
    """
    allocs = get_official(
        path, "Allocations", 10, [
            "Sort C",
            "State FIPS Code",
            "State",
            "LEAID",
            "Name",
            "Basic Hold Harmless",
            "Concentration Hold Harmless",
            "Targeted Hold Harmless",
            "EFIG Hold Harmless",
            "Total Hold Harmless",
            "Basic Alloc",
            "Concentration Alloc",
            "Targeted Alloc",
            "EFIG Alloc",
            "Total Alloc",
            "Hold Harmless Percentage",
            "Resident Pop."
        ]
    )
    counts = get_official(
        path, "Populations", 7, [
            "Sort C",
            "State FIPS Code",
            "State",
            "LEAID",
            "Name",
            "Total Formula Count",
            "5-17 Pop.",
            "Percent Formula",
            "Basic Eligibles",
            "Concentration Eligibles",
            "Targeted Eligibles",
            "Weighted Counts Targeted",
            "Weighted Counts EFIG"
        ]
    )
    combined = allocs.set_index("LEAID").join(
        counts.drop(columns=[
            "Sort C",
            "State FIPS Code",
            "State",
            "Name"
        ]).set_index("LEAID"),
        how="inner"
    ).reset_index()
    combined.loc[:, "State FIPS Code"] = \
        combined["State FIPS Code"].astype(int)
    combined["District ID"], _ = split_leaids(combined["LEAID"].astype(int))

    combined = combined.rename(columns={'Basic Alloc': 'basic_alloc', 'Targeted Alloc': 'targeted_alloc',
                                        'Concentration Alloc': 'concentration_alloc',
                                        'Resident Pop.': 'official_population_count',
                                        '5-17 Pop.': 'official_children_count',
                                        'Total Formula Count': 'official_children_formula_count'})
    combined = combined[
        ['State FIPS Code', 'District ID', 'State', 'Name', 'basic_alloc',
         'targeted_alloc',
         'concentration_alloc',
         'official_population_count',
         'official_children_count',
         'official_children_formula_count']]
    return combined.set_index(["State FIPS Code", "District ID"])


def get_saipe(path: str) -> pd.DataFrame:
    """Get district-level SAIPE data.
    Args:
        path (str): Path to file.
    Returns:
        pd.DataFrame: Cleaned district-level data.
    """
    saipe = pd.read_excel(path, header=2) \
        .set_index(["State FIPS Code", "District ID"])
    saipe["cv"] = saipe.apply(
        lambda x: median_cv(x["Estimated Total Population"]),
        axis=1
    )
    # ground truth - assume SAIPE 2019 is ground truth
    saipe = saipe.rename(columns={
        "Estimated Total Population": "saipe_population_count",
        "Estimated Population 5-17": "saipe_children_count",
        "Estimated number of relevant children 5 to 17 years old in poverty"
        " who are related to the householder": "saipe_children_formula_count"
    })
    return saipe


def get_county_saipe(path: str):
    """Get county-level SAIPE data.
    Args:
        path (str): Path to file.
    Returns:
        pd.DataFrame: Cleaned county-level data.
    """
    saipe = pd.read_excel(path, header=3, usecols=[
        "State FIPS Code",
        "County FIPS Code",
        "Name",
        "Poverty Estimate, All Ages",
        "Poverty Percent, All Ages",
        "Poverty Estimate, Age 5-17 in Families",
        "Poverty Percent, Age 5-17 in Families"
    ]).replace('.', np.NaN).fillna(0.0, )

    # convert county FIPS codes to district ids
    saipe["District ID"] = saipe["County FIPS Code"]
    # convert to saipe district column names
    saipe["Estimated Total Population"] = \
        saipe["Poverty Estimate, All Ages"].astype(float) \
        / (saipe["Poverty Percent, All Ages"].astype(float) / 100)
    saipe["Estimated Population 5-17"] = \
        saipe["Poverty Estimate, Age 5-17 in Families"].astype(float) \
        / (saipe["Poverty Percent, Age 5-17 in Families"].astype(float) / 100)
    saipe[
        'Estimated number of relevant children 5 to 17 years old '
        'in poverty who are related to the householder'
    ] = saipe["Poverty Estimate, Age 5-17 in Families"]

    saipe["cv"] = saipe.apply(
        lambda x: median_cv(x["Estimated Total Population"]),
        axis=1
    )

    return saipe.set_index(["State FIPS Code", "District ID"]).drop(columns=[
        "Poverty Estimate, All Ages",
        "Poverty Percent, All Ages",
        "Poverty Estimate, Age 5-17 in Families",
        "Poverty Percent, Age 5-17 in Families"
    ])


def compute_adj_sppe(sppe, congress_cap=0.4, adj_sppe_bounds=None, adj_sppe_bounds_efig=None):
    """Calculate adjusted SPPE using Sonnenberg, 2016 pg. 18 algorithm.
    """
    # Get baseline average across all 50 states and territories
    if adj_sppe_bounds_efig is None:
        adj_sppe_bounds_efig = [0.34, 0.46]
    if adj_sppe_bounds is None:
        adj_sppe_bounds = [0.32, 0.48]
    average = np.round(
        sppe.groupby("State FIPS Code").first().mean(),
        decimals=2
    )
    # Each state’s and each territory’s SPPE is multiplied by the
    # congressional cap and rounded to the second decimal place
    # (for dollars and cents).
    scaled = np.round(sppe * congress_cap, decimals=2)
    # No state recieves above/below the bounds set by law
    adj_sppe_trunc = scaled.clip(
        # bound by some % of the average, given in the law - round to cents
        *np.round(np.array(adj_sppe_bounds) * average, decimals=2)
    )
    adj_sppe_efig = scaled.clip(
        # bound %s are different for EFIG
        *np.round(np.array(adj_sppe_bounds_efig) * average, decimals=2)
    )
    return adj_sppe_trunc, adj_sppe_efig


def get_sppe(path):
    fips_codes = pd.read_csv(f"datasets/fips_codes.csv").rename(
        columns={
            'FIPS': 'State FIPS Code'
        }
    )
    # quirk of original data file - need to change DC's name for join
    fips_codes.loc[fips_codes["Name"] == "District of Columbia", "Name"] = \
        "District Of Columbia Public Schools"
    sppe = pd.read_excel(path, header=2, engine='openpyxl') \
        .rename(columns={"Unnamed: 0": "Name"})[["Name", "ppe"]]
    return sppe.merge(fips_codes, on="Name", how="right") \
        .set_index("State FIPS Code")


def district_id_from_name(df, name, state=None):
    if state:
        df = df.loc[state, :]
    ind = df[df["Name"] == name].index.get_level_values("District ID")
    if len(ind) == 0:
        raise Exception("No districts with the name", name)
    if len(ind) > 1:
        raise Exception("Multiple district IDs with the name", name)
    return ind[0]


def average_saipe(
        year: int, year_lag: int, verbose=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get SAIPE averaged over `year_lag` years.

    Args:
        year (int): Most recent year.
        year_lag (int): Years to average over.
        verbose (bool, optional): Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Averaged SAIPE - district- and
            county-level.
    """
    if verbose:
        print(
            "Averaging SAIPEs:",
            [
                f"saipe{str(year)[2:]}"
                for year in range(year - year_lag, year + 1)
            ]
        )
    combined = _average_saipe([
        get_saipe(f"../datasets/saipe{str(year)[2:]}.xls")
        for year in range(year - year_lag, year + 1)
    ])
    combined_county = _average_saipe([
        get_county_saipe(f"../../dataset/county_saipe{str(year)[2:]}.xls")
        for year in range(year - year_lag, year + 1)
    ])
    return combined, combined_county


def district_id_from_name(df, name, state=None):
    if state:
        df = df.loc[state, :]
    ind = df[df["Name"] == name].index.get_level_values("District ID")
    if len(ind) == 0:
        raise Exception("No districts with the name", name)
    if len(ind) > 1:
        raise Exception("Multiple district IDs with the name", name)
    return ind[0]


def impute_missing(original, update, verbose=True):
    """Impute data for missing LEAs from another file.
    """
    for c in [c for c in original.columns if c not in update.columns]:
        update.loc[:, c] = np.nan
    update_reduced = update.loc[
        update.index.difference(original.index),
        original.columns
    ]
    imputed = pd.concat([
        original,
        update_reduced
    ])
    if verbose:
        print(
            "[INFO] Successfully imputed",
            len(update.index.difference(original.index)),
            "new indices"
        )
    return imputed


def get_inputs(
        year,
        baseline="prelim",
        avg_lag=0,
        verbose=True,
        use_official_children=False
):
    """Load the inputs for calculating title i allocations

    Args:
        year (int): _description_
        baseline (str, optional): what official dep ed file to use as a
            baseline. Options are "prelim", "final", and "revfinal." Defaults
            to "prelim".
        avg_lag (int, optional): Whether to average, and by how many years.
            Defaults to 0.
        verbose (bool, optional): Defaults to True.
        use_official_children (bool, optional): Whether to use the official
            # total children from Dep Ed, instead of the SAIPE # of children.
            # of children in poverty will always be from SAIPE. Designed for
            validation purposes. Defaults to False.

    Returns:
        pd.DataFrame: combined dataframe of inputs for use in an allocator.
    """
    # official ESEA data
    if year < 2020:
        print("[WARN] Using official data for 2020 instead.")
        official_year = 2020
    else:
        official_year = year

    official = get_official_combined(os.path.join(
        f"../datasets/{baseline}_{str(official_year)[2:]}.xls"
    ))

    # join with Census SAIPE
    if avg_lag > 0:
        saipe, county_saipe = average_saipe(year - 2, avg_lag, verbose=verbose)
    else:
        saipe = get_saipe(f"../datasets/saipe{str(year - 2)[2:]}.xls")
        county_saipe = get_county_saipe(
            f"../datasets/county_saipe{str(year - 2)[2:]}.xls"
        )
    # for some reason, NY naming convention different...
    # fixing that here
    county_saipe.rename(index={
        district_id_from_name(county_saipe, c, 36):
            district_id_from_name(official, c, 36)
        for c in [
            "Bronx County",
            "Kings County",
            "New York County",
            "Queens County",
            "Richmond County"
        ]
    }, level='District ID', inplace=True)
    saipe_stacked = impute_missing(saipe, county_saipe, verbose=verbose)

    # calculate coefficient of variation
    inputs = official.join(saipe_stacked.drop(columns="Name"), how="inner") \
        .astype({'Name': 'string'})

    return inputs


def _average_saipe(saipes):
    combined = pd.concat(saipes)
    # convert cv to variance
    combined["stderr"] = \
        (combined[
             'Estimated number of relevant children 5 to 17 years old '
             'in poverty who are related to the householder'
         ] * combined["cv"]).pow(2)
    agg = combined \
        .groupby(["State FIPS Code", "District ID"]) \
        .agg({
        'Name': 'first',
        'Estimated Total Population': 'mean',
        'Estimated Population 5-17': 'mean',
        'Estimated number of relevant children 5 to 17 years old '
        'in poverty who are related to the householder': 'mean',
        # variance of average of iid Gaussian is sum of variance over n^2
        'stderr': lambda stderr: np.sum(stderr) / (len(saipes) ** 2)
    })
    # convert variance back to cv
    agg['cv'] = np.sqrt(agg['stderr']) / agg["Estimated Total Population"]
    return agg.drop(columns='stderr')



