"""This script should be run to create the final similarity score for cases.
It must be run after creating the similarity scores using the script: src\ground_effect_design_study\process_ground_effect_w_sim_scores.py
"""

import multiprocessing
import multiprocessing.pool  # add the submodule here!
import os
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from design_similarity.similarity_scores import _calc_mp_emd, _calc_mp_similarity

# format is MM_DD_YYYY (I had saved my recent data with its processing date. Adapt as needed.)
DATE_TO_IMPORT = "01_30_2026"

base_dir = os.path.join(
    "/PATH_TO_PROCESSED_ML_CLUSTER_PREDICTIONS",
    f"wig_predictions_{DATE_TO_IMPORT}",
)
assert os.path.exists(base_dir), "PATH DOES NOT EXIST. CHECK DATE."

clustering_data = pd.read_parquet(
    os.path.join(base_dir, "data_to_cluster_w_feature_distances.parquet")
)
# normal_clustering_results = pd.read_parquet(os.path.join(base_dir, "normal_all_clusters.parquet"))
offset_clustering_results = pd.read_parquet(
    os.path.join(base_dir, "offset_all_clusters.parquet")
)

# clustering_data['normal_cluster_labels'] = normal_clustering_results
clustering_data["offset_cluster_labels"] = offset_clustering_results


# Loading the YAML file
with open(os.path.join(base_dir, "reference_case.yaml"), "r") as file:
    config = yaml.safe_load(file)

FOCUS_CASE_ID = config["FOCUS_CASE"]
print(FOCUS_CASE_ID)

print("making regular scores")
for case_count, compare_case in enumerate(clustering_data["case_id"].unique()):

    print(compare_case)
    compare_and_focus = clustering_data[
        clustering_data["case_id"].isin([FOCUS_CASE_ID, compare_case])
    ].copy()
    compare_and_focus = compare_and_focus.drop(
        columns="similarity_score", errors="ignore"
    )

    grouped_data = list(compare_and_focus.groupby("coarse_mesh_parent_cell_id"))

    with multiprocessing.pool.Pool(12) as pool:  # uses all your processors
        one_case_results = pool.starmap(
            _calc_mp_similarity, zip(grouped_data, repeat("normal_cluster_labels"))
        )

    one_case_results = pd.DataFrame(
        np.array(one_case_results),
        columns=["coarse_mesh_parent_cell_id", "similarity_score"],
    )

    compare_and_focus = pd.merge(
        compare_and_focus.reset_index(drop=False),
        one_case_results,
        on="coarse_mesh_parent_cell_id",
    ).set_index("index")

    clustering_data.loc[compare_and_focus.index, "similarity_score"] = (
        compare_and_focus["similarity_score"].values
    )

print("Making offset cluster scores")
for case_count, compare_case in enumerate(clustering_data["case_id"].unique()):

    print(compare_case)
    compare_and_focus = clustering_data[
        clustering_data["case_id"].isin([FOCUS_CASE_ID, compare_case])
    ].copy()
    compare_and_focus = compare_and_focus.drop(
        columns="offset_similarity_score", errors="ignore"
    )

    grouped_data = list(compare_and_focus.groupby("coarse_mesh_parent_cell_id"))

    with multiprocessing.pool.Pool(12) as pool:  # uses all your processors
        one_case_results = pool.starmap(
            _calc_mp_similarity, zip(grouped_data, repeat("offset_cluster_labels"))
        )

    one_case_results = pd.DataFrame(
        np.array(one_case_results),
        columns=["coarse_mesh_parent_cell_id", "offset_similarity_score"],
    )

    compare_and_focus = pd.merge(
        compare_and_focus.reset_index(drop=False),
        one_case_results,
        on="coarse_mesh_parent_cell_id",
    ).set_index("index")

    clustering_data.loc[compare_and_focus.index, "offset_similarity_score"] = (
        compare_and_focus["offset_similarity_score"].values
    )

clustering_data["adjusted_offset_similarity_score"] = (
    clustering_data["offset_similarity_score"]
    - clustering_data["offset_distance_adjustment"]
)


print("Starting the thresholding")
focus_data = clustering_data[clustering_data["case_id"] == FOCUS_CASE_ID].copy()

offset_distance_column_names = [
    col for col in clustering_data if "offset_distance" in col
][:-1]

feature_threshold = 0.05  # 5% of the magnitude

for compare_case_id in clustering_data["case_id"].unique():
    print(compare_case_id)
    if compare_case_id == FOCUS_CASE_ID:
        continue

    compare_data = clustering_data[clustering_data["case_id"] == compare_case_id]

    both_features_below_threshold = (
        compare_data[offset_distance_column_names].abs() < feature_threshold
    ).all(axis=1)

    clustering_data.loc[compare_data.index, "threshold_similarity_score"] = np.where(
        both_features_below_threshold & (compare_data["similarity_score"] < 0.8),
        1,
        compare_data["similarity_score"],
    )
    clustering_data.loc[
        compare_data.index, "threshold_adjusted_offset_similarity_score"
    ] = np.where(
        both_features_below_threshold
        & (compare_data["adjusted_offset_similarity_score"] < 0.8),
        1,
        compare_data["adjusted_offset_similarity_score"],
    )

# This was used to calculate the Earth's movers distance for the research paper.
print("Calculating EMD")

feature_names = ["U_0_norm", "U_1_norm"]

for case_count, compare_case in enumerate(clustering_data["case_id"].unique()):

    print(compare_case)
    compare_and_focus = clustering_data[
        clustering_data["case_id"].isin([FOCUS_CASE_ID, compare_case])
    ].copy()
    compare_and_focus = compare_and_focus.drop(
        columns="similarity_emd", errors="ignore"
    )
    compare_and_focus = compare_and_focus[
        ["case_id", "cell_id", "coarse_mesh_parent_cell_id"] + feature_names
    ]

    grouped_data = list(compare_and_focus.groupby("coarse_mesh_parent_cell_id"))

    with multiprocessing.pool.Pool(12) as pool:  # uses all your processors
        one_case_results = pool.starmap(
            _calc_mp_emd,
            zip(grouped_data, repeat(feature_names), repeat(FOCUS_CASE_ID)),
        )

    one_case_results = pd.DataFrame(
        np.array(one_case_results),
        columns=["coarse_mesh_parent_cell_id", "similarity_emd"],
    )

    compare_and_focus = pd.merge(
        compare_and_focus.reset_index(drop=False),
        one_case_results,
        on="coarse_mesh_parent_cell_id",
    ).set_index("index")

    clustering_data.loc[compare_and_focus.index, "similarity_emd"] = compare_and_focus[
        "similarity_emd"
    ].values

print("Saving the data")
# export the similarity scores


columns_to_save = [
    "case_id",
    "cell_id",
    "coarse_mesh_parent_cell_id",
    # 'similarity_score',
    # 'offset_similarity_score',
    # 'adjusted_offset_similarity_score',
    # 'threshold_similarity_score',
    "threshold_adjusted_offset_similarity_score",
    "similarity_emd",
]
clustering_data[columns_to_save].to_parquet(
    os.path.join(base_dir, "similarity_scores.parquet")
)
