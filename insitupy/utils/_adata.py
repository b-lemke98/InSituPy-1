import anndata
import numpy as np


def _extract_groups(adata, groupby, groups=None, extract_uns=False, uns_key='spatial', uns_exclusion_pattern=None,
    return_mask=False, strip=False):

    '''
    Function to extract a group from a dataframe or anndata object.
    If `anndata_object` is True it expects the dataframe in `adata.obs`
    '''

    # convert groups into list
    if groups is not None:
        groups = [groups] if isinstance(groups, str) else list(groups)

    if type(adata) == anndata.AnnData:
        anndata_object = True
    elif type(adata) == pd.DataFrame:
        anndata_object = False
        extract_uns = False
        strip = False
    else:
        raise ValueError("Unknown type of input object.")

    # select dataframe on which to filter on
    if anndata_object:
        obs = adata.obs
    else:
        obs = adata

    # check if filtering is wanted
    filtering = True
    if groupby is None:
        filtering = False

    # # check if we want to filter `.uns`
    # if uns_exclusion_pattern is not None:
    #     extract_uns = True

    if groupby in obs.columns or not filtering:

        # create filtering mask for groups
        if groupby is None:
            mask = np.full(len(adata), True) # no filtering
        else:
            mask = obs[groupby].isin(groups).values

        # filter dataframe or anndata object
        if anndata_object:
            adata = adata[mask, :].copy()
        else:
            adata = adata.loc[mask, :].copy()

        if len(adata) == 0:
            print("Subset variables '{}' not in groupby '{}'. Object not returned.".format(groups, groupby))
            return
        elif filtering:
            # check if all groups are in groupby category
            groups_found = [group for group in groups if group in obs[groupby].unique()]
            groups_notfound = [group for group in groups if group not in groups_found]

            if len(groups_found) != len(groups):
                print("Following groups were not found in column {}: {}".format(groupby, groups_notfound))

            if extract_uns or uns_exclusion_pattern is not None:
                new_uns = {key:value for (key,value) in adata.uns[uns_key].items() if np.any([group in key for group in groups])}

                if uns_exclusion_pattern is not None:
                    new_uns = {key:value for (key,value) in new_uns.items() if uns_exclusion_pattern not in key}
                adata.uns[uns_key] = new_uns

        if strip:
            # remove annotations in .uns and obsp
            stores = [adata.uns, adata.obsp]
            for s in stores:
                keys = list(s.keys())
                for k in keys:
                    del s[k]

        if return_mask:
            return adata, mask
        else:
            return adata

    else:
        print("Subset category '{}' not found".format(groupby))
        return