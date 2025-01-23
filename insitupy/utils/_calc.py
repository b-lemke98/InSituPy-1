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

def calc_density(
    adata,
    groupby: str,
    mode: Literal["gauss", "mellon"] = "gauss",
    clip: bool = True,
    inplace: bool = False
):
    """
    Calculate the spatial density for groups in the AnnData object. Groups could be e.g. cell types in the sample.
    Spatial coordinates are expected to be saved in `adata.obsm["spatial"]`.

    Args:
        adata (AnnData): The annotated data matrix.
        groupby (str): The column in `adata.obs` to group by.
        mode (Literal["gauss", "mellon"], optional): The mode of density estimation.
            "gauss" for Gaussian KDE using scipy, "mellon" for Mellon density estimator.
            Defaults to "gauss".
        clip (bool, optional): If True, clip the density values to the 5th and 95th percentile.
        inplace (bool, optional): If True, modify `adata` in place. If False, return a copy of `adata` with the modifications.
            Defaults to False.

    Returns:
        AnnData: The modified AnnData object with added density values.
    """
    if inplace:
        _adata = adata
    else:
        _adata = adata.copy()

    # Initialize lists to store results
    density_df = pd.DataFrame(index=_adata.obs_names)

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
        density_df[group] = density_series

    if clip:
        # clip the data
        quantiles_df = density_df.quantile([0.05, 1])
        density_df_clipped = density_df.clip(
            lower=quantiles_df.iloc[0],
            upper=quantiles_df.iloc[1],
            axis=1
            )

        _adata.obsm[f"density-{mode}"] = density_df_clipped

    else:
        _adata.obsm[f"density-{mode}"] = density_df

    if not inplace:
        return _adata
