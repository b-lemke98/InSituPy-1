from numbers import Number
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from insitupy.plotting.expression_along_axis import _bin_data, _select_data
from insitupy.utils.utils import convert_to_list


# Functions
def total_variation(values):
    return np.sum(np.abs(np.diff(values)))

def random_permutation_tv(expr):
    random_order = np.random.permutation(np.arange(len(expr)))
    expr_random = expr[random_order]

    return total_variation(expr_random)

def filter_outliers(data, threshold=1.5):
    """
    Remove values that lie significantly outside the IQR.

    Args:
        data (numpy.ndarray): The input array.
        threshold (float, optional): The multiplier for the IQR to define outliers. Default is 1.5.

    Returns:
        numpy.ndarray: The filtered array with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Calculate the lower and upper bounds
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Filter the array to remove outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    return filtered_data

def calculate_total_variation_pval(
    adata,
    obs_val,
    genes,
    cell_type_column,
    cell_type,
    xlim,
    parallel,
    bin_data: bool = False,
    resolution: Number = 5,
    n_sim: int = 10000,
    min_expression: Optional[Number] = None,
    n_jobs: int = 8
    ):

    genes = convert_to_list(genes)

    # get data
    data = _select_data(
        adata=adata,
        obs_val=obs_val,
        genes=genes, cell_type_column=cell_type_column, cell_type=cell_type, xlim=xlim,
        sort=True, verbose=False
        )

    if bin_data:
        binned_data = _bin_data(data=data, resolution=resolution, plot=False)

    pvals = []
    for gene in tqdm(genes):
        print(gene)
    #for gene in genes:
        # data = _select_data(adata=adata,
        #             obs_val=obs_val,
        #             genes=gene, cell_type_column=cell_type_column, cell_type=cell_type, xlim=xlim,
        #             sort=True, verbose=False
        #             )



        # filter for minimum expression if threshold given
        if min_expression is not None:
            data = data[data[gene] >= min_expression].copy()

        # extract values necessary for further analysis
        expr_sorted = data[gene].values
        axis = data["axis"].values

        # if bin_data:
        #     binned_data = _bin_data(
        #         expr=data[gene],
        #         axis=data["axis"],
        #         resolution=resolution,
        #         plot=False
        #         )

        #     # extract the min-max scaled data for calculating the total variation
        #     scaled_expr = binned_data["minmax"]
        #     #tv_gene = binned_total_variation(expr=expr_sorted, axis=axis, resolution=resolution)
        # else:
        #     # scale values to 0-1
        #     scaler = MinMaxScaler()
        #     scaled_expr = scaler.fit_transform(expr_sorted)

        scaled_expr = scaled_expr.values

        tv_gene = total_variation(values=scaled_expr)
        # simulation of random total variations
        # speed up computation with joblib
        if parallel:
            #random_tvs = np.array(Parallel(n_jobs=8)(delayed(random_permutation_tv)(expr_sorted) for _ in range(n_sim)))
            random_tvs = np.array(Parallel(n_jobs=n_jobs)(delayed(random_permutation_tv)(scaled_expr)
                                                    for _ in range(n_sim)))
        else:
            random_tvs = np.array([
                random_permutation_tv(
                    scaled_expr
                    )
                for _ in range(n_sim)
                ])
        random_tvs_filtered = filter_outliers(random_tvs)
        n = len(random_tvs_filtered)
        pvals.append(np.sum(random_tvs_filtered <= tv_gene) / n)
    return pd.Series(pvals, index=genes)