import pandas as pd
from scipy.stats import wasserstein_distance
import numpy as np

def get_perfect_mesh_similarity(case_ids: pd.DataFrame):

    mesh_similarity_score = pd.DataFrame()

    for focus_case_name in case_ids.unique():
        mesh_similarity_score.loc[
            len(mesh_similarity_score), "mesh_id_1"
        ] = focus_case_name

        mesh_similarity_score.loc[
            len(mesh_similarity_score) - 1, "mesh_similarity_score"
        ] = 1

    return mesh_similarity_score

def calculate_one_vs_one_similarity(
    cell_set: pd.DataFrame,
    cluster_prediction_name: str
):
    """
    This similaritys score is used in the design similarity. TBH I don't know
    if it's different but it seems faster.
    """
    cell_set_counts = cell_set.groupby(by=['case_id', cluster_prediction_name]).size().reset_index().rename(columns={0 : "cells_in_cluster"})
    cell_set_case_id_cell_counts = cell_set.groupby(by=['case_id']).size().reset_index().rename(columns={0 : "total_cell_count"})
    cell_set_counts = pd.merge(cell_set_counts, cell_set_case_id_cell_counts, on='case_id')
    cell_set_counts['fraction_of_total_case_cells_in_cluster'] = \
        (
            cell_set_counts["cells_in_cluster"]
            / cell_set_counts["total_cell_count"]
        )

    similarity_score = cell_set_counts.pivot(
        columns=cluster_prediction_name,
        index='case_id',
        values="fraction_of_total_case_cells_in_cluster"
    ).fillna(0.0).min(axis=0).sum()

    return similarity_score

def calculate_emd(data1, data2, var: list[str] = None):
  
    if var:
        return [wasserstein_distance(data1[var_name], data2[var_name]) for var_name in var]
    else:
        return wasserstein_distance(data1, data2)


def _calc_mp_emd(
        cell_set: tuple[ int, pd.DataFrame],
        feature_names: list[str],
        focus_case_id: str,
        ) -> tuple[int, float]:

    data = cell_set[1]

    data = data[['case_id']+feature_names].copy()

    focus_data = data[data['case_id']==focus_case_id]
    compare_data = data[data['case_id']!=focus_case_id]

    if compare_data.empty:
        return (cell_set[0], np.nan)

    emd = np.sum(calculate_emd(focus_data, compare_data, var=feature_names))

    return (cell_set[0], 1 - emd)


def _calc_mp_similarity(
        cell_set: tuple[ int, pd.DataFrame],
        cluster_prediction_name: str,
        ) -> tuple[int, float]:
    """for calculating the similiarity score in parallel

    Args:
        cell_set (tuple[ int, pd.DataFrame]): data to do calculation. 
            must have cluster labels
        cluster_prediction_name (str): _description_

    Returns:
        tuple[int, float]: the cell set id and the similarity score

    Example:
        with multiprocessing.pool.Pool(12) as pool: # uses all your processors
            one_case_results = pool.starmap(
                _calc_mp_similarity, 
                zip(grouped_data, repeat("cluster_predictions"))
                )
    """
    data = cell_set[1]

    data = data[['case_id', cluster_prediction_name]].copy()

    similarity = calculate_one_vs_one_similarity(data, cluster_prediction_name=cluster_prediction_name)

    return (cell_set[0], similarity)

def calculate_offset_distance(
        data_to_cluster: pd.DataFrame,
        feature_names: list[str],
        focus_case_name: str,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """calculates the offset distance between two different cases. The
    Manhattan distance is used.

    Args:
        data_to_cluster (pd.DataFrame): the data to cluster containing the
            features_names, and case_id
        feature_names (list[str]): the names of the features for clustering
        focus_case_name str: the name of the focus case

    Returns:
        pd.DataFrame: the offset distances and the offset features
    """
    #make a copy so we don't modified the actual dataframe
    data_to_cluster = data_to_cluster.copy()

    # first just find the median for the focus dataset
    focus_data_only = data_to_cluster[data_to_cluster['case_id']==focus_case_name].copy()
    focus_data_median = focus_data_only.groupby('coarse_mesh_parent_cell_id')[feature_names].median()
    focus_data_median = focus_data_median.add_prefix("focus_median_")

    focus_column_names = list(focus_data_median.columns.values)
    focus_data_median = focus_data_median.reset_index() 

    data_to_cluster = pd.merge(
        data_to_cluster.reset_index(drop=False),
        focus_data_median,
        on='coarse_mesh_parent_cell_id',
    ).set_index('index')

    # now find the median for the rest of the data
    all_feature_median_values = data_to_cluster.groupby(by=['case_id','coarse_mesh_parent_cell_id'])[feature_names + focus_column_names].median()

    all_feature_median_values = all_feature_median_values[feature_names] - all_feature_median_values[focus_column_names].values

    offset_distances = all_feature_median_values.rename(
        columns={current:f"offset_distance_{current}" for current in all_feature_median_values.columns}
    ).reset_index()

    offset_distance_names = [col for col in offset_distances.columns if "offset_distance_" in col]
    offset_feature_names = [f"{name}_offset" for name in feature_names]

    data_to_cluster = pd.merge(
        data_to_cluster[['case_id', 'cell_id', 'coarse_mesh_parent_cell_id',] + feature_names].reset_index(drop=False),
        offset_distances,
        on=['case_id', 'coarse_mesh_parent_cell_id',]
    ).set_index("index")

    data_to_cluster[offset_feature_names] = data_to_cluster[feature_names] - data_to_cluster[offset_distance_names].values
    data_to_cluster['offset_distance_adjustment'] = data_to_cluster[offset_distance_names].abs().sum(axis=1)

    return data_to_cluster[offset_feature_names + offset_distance_names + ['offset_distance_adjustment']]
