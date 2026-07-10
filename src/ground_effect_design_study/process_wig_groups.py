
import gc
import multiprocessing
import os
from copy import deepcopy
from datetime import date
from itertools import product, repeat

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from projects.paper3.design_similarity.cluster_model_training import \
    train_models
from projects.paper3.design_similarity.localization import \
    cluster_all_meshes_radially
from projects.paper3.design_similarity.similarity_scores import \
    calculate_offset_distance


def setup_case_variables():
    base_dir = "/mnt/d/openfoam/airfoils_2d/ground_effect_study_6e6"

    chunk_dir = "/mnt/d/ground_effect_processed_data_python/parquet_chunks"

    return base_dir, chunk_dir

def import_raw_data():
    cfd_data_raw = pd.read_parquet(os.path.join("/mnt/d/ground_effect_processed_data_python", "ground_effect_study_raw_data_gold_11092025.parquet"))

    return cfd_data_raw

def make_cell_sets(
    focus_case: str,
    cfd_data_raw: pd.DataFrame,
    selected_cases: list[str] = None,
    ):
    n_radial_clusters = 131 
    n_sub_clusters = 50  
    random_state = 2  

    regionalized_data = cfd_data_raw
    if selected_cases is not None:
        selected_cases.append(focus_case)
        regionalized_data = regionalized_data[regionalized_data['case_id'].isin(selected_cases)]

    cluster_all_meshes_radially(
        focus_case,
        regionalized_data,
        n_radial_clusters=n_radial_clusters,
        n_sub_clusters=n_sub_clusters,
        random_state=random_state,
    )

    regionalized_data['coarse_mesh_parent_cell_id'] = regionalized_data['coarse_mesh_parent_cell_id'].astype(np.int64)

    return regionalized_data

# def normalize_targets(regionalized_data: pd.DataFrame):
#     features = ['U_0', "U_1"]

#     data_to_cluster = regionalized_data.copy()
#     data_to_cluster[FEATURES_TRANS] = StandardScaler().fit_transform(data_to_cluster[features])

#     return data_to_cluster

def median_normalize_velocity(
    data_to_cluster: pd.DataFrame,
    feature_names: list[str],
    ):
    data_to_cluster = data_to_cluster.copy()
    focus_data_only = data_to_cluster[data_to_cluster['case_id']==FOCUS_CASE].copy()

    focus_data_only['u_feature_mag'] = np.sqrt(np.sum(focus_data_only[feature_names]**2, axis=1))

    focus_data_median = focus_data_only.groupby("coarse_mesh_parent_cell_id")['u_feature_mag'].median()
    focus_data_median = focus_data_median.reset_index()

    data_to_cluster = pd.merge(
        data_to_cluster.reset_index(drop=False),
        focus_data_median,
        on='coarse_mesh_parent_cell_id',
    ).set_index('index')
  
    return data_to_cluster[feature_names]/np.tile(data_to_cluster['u_feature_mag'].values.reshape(-1,1), (1,2))

def _predict_one_cluster(cell_set):
        pred_data = cell_set[1][FEATURES_TRANS]

        model = deepcopy(trained_models[np.where(trained_models[:,0]==cell_set[0])[0][0], 1])

        if model is None:
            return pd.DataFrame(np.ones(pred_data.shape[0])*np.nan, index=pred_data.index)
        
        return pd.DataFrame(model.predict(pred_data), index=pred_data.index)

def _predict_one_offset_cluster(
    cell_set: tuple[int, pd.DataFrame],
    feature_names: list[str],
    trained_models: list, 
    ):
        feature_names_offset = [f"{name}_offset" for name in feature_names]
        # the model expects the names to be the same as when it was trained.
        feature_name_map = {new_name : old_name for new_name, old_name in zip(feature_names_offset, feature_names)}
        pred_data = cell_set[1][feature_names_offset].rename(columns=feature_name_map)

        model = deepcopy(trained_models[np.where(trained_models[:,0]==cell_set[0])[0][0], 1])

        if model is None:
            return pd.DataFrame(np.ones(pred_data.shape[0])*np.nan, index=pred_data.index)
        
        return pd.DataFrame(model.predict(pred_data), index=pred_data.index)

def make_offset_cluster_predictions(
    data_to_cluster: pd.DataFrame,
    feature_names: list[str],
    trained_models: np.ndarray,
):

    print('Grouping Data Before Predicting')
    grouped_data = list(data_to_cluster.groupby("coarse_mesh_parent_cell_id"))
    print("Data Grouped")

    print("Starting Predictions")
    with multiprocessing.pool.Pool(12) as pool: 
        all_case_clusters_separate = pool.starmap(_predict_one_offset_cluster, zip(grouped_data, repeat(feature_names), repeat(trained_models)))

    return all_case_clusters_separate

def make_cluster_predictions(
    data_to_cluster: pd.DataFrame,

):
    print('Grouping Data Before Predicting')
    grouped_data = list(data_to_cluster.groupby("coarse_mesh_parent_cell_id"))
    print("Data Grouped")

    print("Starting Predictions")
    with multiprocessing.pool.Pool(12) as pool: 
        all_case_clusters_separate = pool.map(_predict_one_cluster, grouped_data)

    return all_case_clusters_separate

def make_cluster_predictions_sp(
    data_to_cluster: pd.DataFrame,

):
    print('Grouping Data Before Predicting')
    grouped_data = list(data_to_cluster.groupby("coarse_mesh_parent_cell_id"))
    print("Data Grouped")

    print("Starting Predictions")
    all_case_clusters_separate = []
    for group in grouped_data:
        all_case_clusters_separate.append(_predict_one_cluster(group))

    return all_case_clusters_separate

def save_cluster_prediction_data(
    all_case_clusters_separate: np.ndarray[pd.DataFrame],
):
    # for ii, cluster_chunk in enumerate(all_case_clusters_separate):
    #     cluster_chunk.to_parquet(
    #         os.path.join("/mnt/d/ground_effect_processed_data_python/clustered_chunks",
    #                     f"cluster_chunk_{ii}_11_10_2025.parquet"))
        
    all_cluster_predictions_joined = pd.DataFrame()

    for ii, cluster_chunk in enumerate(all_case_clusters_separate):
        all_cluster_predictions_joined = pd.concat([all_cluster_predictions_joined, cluster_chunk], axis=0)
    
    return all_cluster_predictions_joined

def make_output_folder(base_dir: str, base_name: str):
    """
    Creates a folder. If the name exists, appends an incrementing number
    (e.g., folder, folder_1, folder_2).
    """
    TODAYS_DATE = date.today()
    TODAYS_DATE = TODAYS_DATE.strftime("%m_%d_%Y")
    
    folder_name = os.path.join(base_dir,f"{base_name}_{TODAYS_DATE}")
    counter = 1

    # Loop until we find a name that doesn't exist
    while os.path.exists(folder_name):
        folder_name = os.path.join(base_dir, f"{base_name}_{TODAYS_DATE}_{counter}")
        counter += 1

    # Create the directory
    try:
        os.makedirs(folder_name)
        print(f"Successfully created folder: '{folder_name}'")
    except OSError as e:
        print(f"Error creating folder: {e}")

    return folder_name

import time

if __name__ == "__main__":
    CASE_IDS_TO_DROP = [f"{ii}0{bb}" for ii, bb in product([2,4,6,8], [10,20,30,40])]
    CASE_IDS_TO_DROP.extend([f"0{ii}{bb}" for ii, bb in product([1,2,3,4,6,8], [10,20,30,40])])

    OUTPUT_DIRECTORY = make_output_folder("/mnt/d/ground_effect_processed_data_python/processed_predictions", "wig_predictions")

    # FEATURES_TRANS = ['U_0_trans', "U_1_trans"]
    FEATURES_TRANS = ['U_0_norm', "U_1_norm"]

    # SELECTED_CASES = [name for name in cfd_data_raw['case_id'].unique() if "8440" in name]
    SELECTED_CASES = [
    "naca_0010_ground_effect_04",
    "naca_0010_ground_effect_02",
    "naca_0010_ground_effect_015",
    "naca_0010_ground_effect_01",
    "naca_0010_ground_effect_0075",
    "naca_0010_ground_effect_005",
    "naca_0010_ground_effect_0025",
    "naca_0010_ground_effect_1",

    "naca_0020_ground_effect_04",
    "naca_0020_ground_effect_02",
    "naca_0020_ground_effect_015",
    "naca_0020_ground_effect_01",
    "naca_0020_ground_effect_0075",
    "naca_0020_ground_effect_005",
    "naca_0020_ground_effect_0025",

    "naca_4410_ground_effect_1",
    "naca_4410_ground_effect_04",
    "naca_4410_ground_effect_02",
    "naca_4410_ground_effect_015",
    "naca_4410_ground_effect_01",
    "naca_4410_ground_effect_0075",
    "naca_4410_ground_effect_005",
    "naca_4410_ground_effect_0025",

    "naca_8440_ground_effect_1",
    "naca_8440_ground_effect_04",
    "naca_8440_ground_effect_02",
    "naca_8440_ground_effect_015",
    "naca_8440_ground_effect_01",
    "naca_8440_ground_effect_0075",
    "naca_8440_ground_effect_005",
    "naca_8440_ground_effect_0025",
    ]
    SELECTED_CASES = None
    FOCUS_CASE_FOR_REGIONALIZATION = 'naca_0010_ground_effect_1'
    FOCUS_CASE = "0010"

    design_data = {
        "FOCUS_CASE": FOCUS_CASE,
        "FOCUS_CASE_FOR_REGIONALIZATION": FOCUS_CASE_FOR_REGIONALIZATION,
    }

    file_path = os.path.join(OUTPUT_DIRECTORY,"reference_case.yaml")

    with open(file_path, "w") as file:
        yaml.dump(design_data, file, default_flow_style=False, sort_keys=False)

    print(f"Successfully exported design study to {file_path}")

    base_dir, chunk_dir = setup_case_variables()

    cfd_data_raw = import_raw_data()

    print("Mapping case_id to naca")
    case_names = cfd_data_raw['case_id'].unique()
    case_id_to_naca = {case_id : naca for case_id, naca in zip(case_names, pd.Series(case_names).str.split("_").str[1].tolist())}
    cfd_data_raw['naca'] = cfd_data_raw['case_id'].map(case_id_to_naca)
    cfd_data_raw = cfd_data_raw[~cfd_data_raw['naca'].isin(CASE_IDS_TO_DROP)]

    print("Data Imported")
    
    start_time = time.time()
    print(f"Start time {start_time}")

    regionalized_data = make_cell_sets(
        FOCUS_CASE_FOR_REGIONALIZATION,
        cfd_data_raw,
        selected_cases=SELECTED_CASES,
        )

    del cfd_data_raw
    gc.collect()
    
    print("Shifting to group mode")
    print(regionalized_data[regionalized_data['case_id']=='naca_0010_ground_effect_1'].shape)
    # this lines turns all cases into naca groups
    # regionalized_data['case_id'] = regionalized_data['case_id'].str.split("_").str[1]
    #this line turns only the focus case into a naca group
    regionalized_data.loc[regionalized_data['naca']==FOCUS_CASE, 'case_id'] = regionalized_data[regionalized_data['naca']==FOCUS_CASE]['naca']
    print(regionalized_data[regionalized_data['case_id']==FOCUS_CASE].shape)
    
    print("Cell Sets Created")

    data_to_cluster = regionalized_data[['case_id', 'cell_id', 'coarse_mesh_parent_cell_id', 'U_0', 'U_1', 'C_0', 'C_1', 'y_wall']].copy()
    del regionalized_data
    gc.collect()
    
    feature_names = ['U_0_norm', 'U_1_norm']
    data_to_cluster[feature_names] = median_normalize_velocity(
        data_to_cluster,  
        feature_names=['U_0', 'U_1'])

    print("Data Normalized")
    offset_features = calculate_offset_distance(
        data_to_cluster[['case_id', 'coarse_mesh_parent_cell_id', 'cell_id', 'U_0_norm', 'U_1_norm', 'U_0', 'U_1', 'C_0', 'C_1', 'y_wall']],
        feature_names=feature_names,
        focus_case_name=FOCUS_CASE,
    )
    offset_feature_names = offset_features.columns.values[:len(feature_names)].tolist()
    data_to_cluster[offset_features.columns] = offset_features

    print("Saving offset_distances")
    
    export_feature_names = ['case_id', 'cell_id', 'coarse_mesh_parent_cell_id', 'C_0', 'C_1', 'y_wall'] \
        + ['U_0_norm', 'U_1_norm'] \
        + offset_features.columns.values.tolist()

    del offset_features
    gc.collect()

    data_to_cluster[export_feature_names].to_parquet(
            os.path.join(OUTPUT_DIRECTORY,f"data_to_cluster_w_feature_distances.parquet")
                         )
    # data_to_cluster = pd.read_parquet(
    #         os.path.join(OUTPUT_DIRECTORY,f"data_to_cluster_w_feature_distances.parquet")
    #                      )
    # # offset_feature_names = offset_features.columns.values[:len(feature_names)].tolist()
    # offset_feature_names = ['U_0_norm_offset', 'U_1_norm_offset']

    print("Starting model training")

    trained_models = train_models(
        data_to_cluster,
        FOCUS_CASE,
        features_trans=FEATURES_TRANS,
    )

    print("Models Trained")
    # print("STARTING REGULAR PREDICTIONS")
    # all_case_clusters_separate = make_cluster_predictions(
    #     data_to_cluster[['coarse_mesh_parent_cell_id', 'case_id', 'cell_id'] + FEATURES_TRANS],
    # )

    # print("REGULAR Data Clustered. Now Saving")

    # all_cluster_predictions_joined = save_cluster_prediction_data(all_case_clusters_separate)

    # all_cluster_predictions_joined.to_parquet(os.path.join(OUTPUT_DIRECTORY,f"normal_all_clusters.parquet"))
    
    print("STARTING OFFSET")
    all_case_clusters_separate = make_offset_cluster_predictions(
        data_to_cluster[['coarse_mesh_parent_cell_id', 'case_id', 'cell_id'] + offset_feature_names],
        feature_names=FEATURES_TRANS,
        trained_models=trained_models,
    )

    print("OFFSET Data Clustered. Now Saving")

    all_cluster_predictions_joined = save_cluster_prediction_data(all_case_clusters_separate)

    all_cluster_predictions_joined.to_parquet(os.path.join(OUTPUT_DIRECTORY, f"offset_all_clusters.parquet"))
