import warnings
from typing import List, Literal, Union

import mellon
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
from tqdm import tqdm


def _calc_kernel_density(
    data: Union[np.ndarray, List],
    mode: Literal["gauss", "mellon"] = "gauss",
    verbose: bool = False
    ):
    """
    Calculate the kernel density estimation for the given data.

    Args:
        data (Union[np.ndarray, List]): Input data for density estimation.
        mode (Literal["gauss", "mellon"], optional): The mode of density estimation.
            "gauss" for Gaussian KDE using scipy, "mellon" for Mellon density estimator.
            Defaults to "gauss".
        verbose (bool, optional): If True, print statements will be used to indicate the mode.
            Defaults to False.

    Returns:
        np.ndarray: The estimated density values.

    Raises:
        UserWarning: If an invalid mode is provided.
    """
    # Make sure the data is a numpy array
    data = np.array(data)

    if mode == "mellon":
        if verbose:
            print("Using Mellon density estimator.")
        # Fit and predict log density
        model = mellon.DensityEstimator()
        density = model.fit_predict(data)
    elif mode == "gauss":
        if verbose:
            print("Using Gaussian KDE.")
        try:
            kde = gaussian_kde(data.T, bw_method="scott")
            density = kde(data.T)
        except LinAlgError:
            # return only NaN values - this happens if the data is not big enough
            density = np.empty(len(data))
            density[:] = np.nan

    else:
        warnings.warn(f"Invalid mode '{mode}' provided. Please use 'gauss' or 'mellon'.")
        return None

    return density

def calc_grouped_log_density(
    adata,
    groupby: str,
    log_density_key: str = "density",
    clipped_log_density_key: str = "density_clipped",
    mode: Literal["gauss", "mellon"] = "gauss",
    inplace: bool = False
):
    """
    Calculate the log density and clipped log density for groups in the AnnData object.

    Args:
        adata (AnnData): The annotated data matrix.
        groupby (str): The column in `adata.obs` to group by.
        log_density_key (str, optional): The key under which to store the log density in `adata.obsm`.
            Defaults to "log_density".
        clipped_log_density_key (str, optional): The key under which to store the clipped log density in `adata.obsm`.
            Defaults to "log_density_clipped".
        mode (Literal["gauss", "mellon"], optional): The mode of density estimation.
            "gauss" for Gaussian KDE using scipy, "mellon" for Mellon density estimator.
            Defaults to "gauss".
        inplace (bool, optional): If True, modify `adata` in place. If False, return a copy of `adata` with the modifications.
            Defaults to False.

    Returns:
        AnnData: The modified AnnData object with added log density and clipped log density.
    """
    if inplace:
        _adata = adata
    else:
        _adata = adata.copy()

    # Initialize lists to store results
    log_density_df = pd.DataFrame(index=_adata.obs_names)

    # Iterate over unique values in the groupby column
    for group in tqdm(_adata.obs[groupby].unique()):
        # Select the respective values in adata.obsm["spatial"]
        group_mask = _adata.obs[groupby] == group
        spatial_data = _adata.obsm["spatial"][group_mask]

        # Fit and predict density
        density = _calc_kernel_density(spatial_data, mode=mode)

        # create pandas series from results
        density_series = pd.Series(
            data=density,
            index=_adata.obs_names[group_mask],
            name=group
            )



        # Store results in dataframes
        log_density_df[group] = density_series

    # clip the data
    quantiles_df = log_density_df.quantile([0.05, 1])
    log_density_df_clipped = log_density_df.clip(
        lower=quantiles_df.iloc[0],
        upper=quantiles_df.iloc[1],
        axis=1
        )

    # Sort the dataframes and add them to adata
    _adata.obsm[f"{log_density_key}-{mode}"] = log_density_df
    _adata.obsm[f"{clipped_log_density_key}-{mode}"] = log_density_df_clipped

    if not inplace:
        return _adata


# def calc_grouped_log_density(
#     adata,
#     groupby: str,
#     log_density_key: str = "log_density",
#     clipped_log_density_key: str = "log_density_clipped",
#     mode: Literal["gauss", "mellon"] = "gauss",
#     inplace: bool = False
# ):
#     """
#     Calculate the log density and clipped log density for groups in the AnnData object.

#     Args:
#         adata (AnnData): The annotated data matrix.
#         groupby (str): The column in `adata.obs` to group by.
#         log_density_key (str, optional): The key under which to store the log density in `adata.obsm`.
#             Defaults to "log_density".
#         clipped_log_density_key (str, optional): The key under which to store the clipped log density in `adata.obsm`.
#             Defaults to "log_density_clipped".
#         mode (Literal["gauss", "mellon"], optional): The mode of density estimation.
#             "gauss" for Gaussian KDE using scipy, "mellon" for Mellon density estimator.
#             Defaults to "gauss".
#         inplace (bool, optional): If True, modify `adata` in place. If False, return a copy of `adata` with the modifications.
#             Defaults to False.

#     Returns:
#         AnnData: The modified AnnData object with added log density and clipped log density.
#     """
#     if inplace:
#         _adata = adata
#     else:
#         _adata = adata.copy()

#     # Initialize lists to store results
#     log_density_list = []
#     clipped_log_density_list = []

#     # Iterate over unique values in the groupby column
#     for group in tqdm(_adata.obs[groupby].unique()):
#         # Select the respective values in adata.obsm["spatial"]
#         group_mask = _adata.obs[groupby] == group
#         spatial_data = _adata.obsm["spatial"][group_mask]

#         # Fit and predict density
#         density = _calc_kernel_density(spatial_data, mode=mode)

#         # Store results in lists
#         log_density_list.append(
#             pd.DataFrame(
#                 {groupby: density},
#                 index=adata.obs_names[group_mask]
#             ))
#         clipped_log_density_list.append(
#             pd.DataFrame(
#                 {groupby: np.clip(density, *np.quantile(density, [0.05, 1]))},
#                 index=adata.obs_names[group_mask]
#             ))

#     log_density_df = pd.concat(log_density_list)
#     clipped_log_density_df = pd.concat(clipped_log_density_list)

#     # Sort the dataframes and add them to adata
#     _adata.obsm[f"{log_density_key}-{mode}"] = log_density_df.loc[_adata.obs_names]
#     _adata.obsm[f"{clipped_log_density_key}-{mode}"] = clipped_log_density_df.loc[_adata.obs_names]

#     if not inplace:
#         return _adata
