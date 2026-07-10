import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import RadiusNeighborsClassifier
from tqdm import tqdm

DISTS = np.array([0.05, 0.15, 0.20, 0.3 ])

def select_best_mod_agglo_model(X: pd.DataFrame):
    """selects the best AgglomerativeClustering
    and trains a model in the modified prediction style

    Args:
        X (X: pd.DataFrame): _description_

    Returns:
        sklearn model: the trained mod agglo model
    """
    best_score = []
    for distance_threshold in DISTS:
        agglo_model = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=distance_threshold)

        try:
            y_pred = agglo_model.fit_predict(X)
            sil_score = silhouette_score(X, y_pred)
            best_score.append(round(sil_score, 3))
        except:
            best_score.append(0.5)
            continue

    agglo_model = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=DISTS[np.argmax(best_score)])

    agglo_model.fit(X)

    agglo_prediction_model = RadiusNeighborsClassifier(
        radius=0.1,
        weights='uniform',
        outlier_label=-1,
        n_jobs=-1
        )

    agglo_prediction_model.fit(X, agglo_model.labels_)

    return agglo_prediction_model, best_score


def train_models(
    data_to_cluster: pd.DataFrame,
    focus_case: str,
    features_trans: list,
):
    """
    it's faster to train the models all at once and then call them as needed.
    """

    focus_data = data_to_cluster[data_to_cluster['case_id']==focus_case].copy()

    grouped_data_list = list(focus_data.groupby("coarse_mesh_parent_cell_id"))

    def process_cell_set(cell_set):
        cell_set_data = cell_set[1]

        subset_focus = cell_set_data[features_trans].copy()
        
        if (subset_focus.shape[0] <= 3):
            return (cell_set[0], None)
    
        mod_agglo_model, _ = select_best_mod_agglo_model(subset_focus[features_trans])
        
        return (cell_set[0], mod_agglo_model)

    trained_models = []
    for group in tqdm(grouped_data_list):
        trained_models.append(process_cell_set(group))

    trained_models = np.array(trained_models)

    return trained_models 