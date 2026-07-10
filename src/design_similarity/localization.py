import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def geometric_progression_clustering(
    data,
    y_start,
    r: float = 1.1,
    y_training_end: float = 999_999,
    training_max_layer: float = 999_999,
):

    cluster_labels = np.ones_like(data) * np.nan

    layer_height = y_start
    total_thickness = 0
    for layer_number in range(0, 10000):
        print(layer_height, total_thickness)

        # when predicting on new data don't make radial layers past
        # the training layers. everything past that training radial distance
        # is just the max layer number of the training
        if layer_height >= y_training_end:
            print("capping layer height at ", y_training_end, " ", training_max_layer)
            layer_number = training_max_layer

        cluster_labels = np.where(
            (data >= total_thickness) & (data <= total_thickness + layer_height),
            layer_number,
            cluster_labels,
        )

        total_thickness += layer_height

        layer_height = layer_height * r

        if (data.values <= total_thickness).sum() >= data.shape[0]:
            break
    # return the labels and last layer height
    return cluster_labels, total_thickness


def cluster_all_meshes_radially(
    parent_mesh_id: str,
    all_cfd_data: pd.DataFrame,
    n_sub_clusters: int = 20,
    first_layer_height: int = 0.005,
    n_radial_clusters: int = 151,
    random_state: int = 0,
):
    """this clusters the mesh radially then inside of each
    radial cluster then clusters them by x,y. It basically makes
    rings and then chops up the rings into pieces. good for one
    solid body mesh segmentation.

    Haven't test for if you had complex geometries like a body
    and a wing or a two element wing.

    Args:
        parent_mesh_id (str): give the id of the base design that you want
        to use to then cluster the rest. Kmean is fit to this design and
        .predict is used on all the others.
        all_cfd_data (pd.DataFrame): the full CFD for all the results
        n_sub_clusters (int, optional): the number of clusters that each
        radial cluster should be segmented into. Defaults to 20.
        n_radial_clusters (int, optional): the number of rings to make.
             Defaults to 151.
        random_state (int, optional): you know what this means. Defaults to 0.
    """

    # radial_cluster_model = make_pipeline(StandardScaler(), KMeans(n_radial_clusters, random_state=random_state))
    sub_cluster_model = make_pipeline(
        StandardScaler(), KMeans(n_sub_clusters, random_state=random_state + 42)
    )

    radial_features = ["y_wall"]
    sub_features = ["C_0", "C_1"]
    sub_features = [
        "pol_theta",
        "pol_radius",
    ]

    theta, rad = cart2pol(
        all_cfd_data["C_0"],
        all_cfd_data["C_1"],
    )

    all_cfd_data["pol_theta"] = theta
    all_cfd_data["pol_radius"] = rad

    training_data = all_cfd_data[all_cfd_data["case_id"] == parent_mesh_id].copy()

    expansion_ratio = 1.1

    training_data["radial_labels"], total_thickness = geometric_progression_clustering(
        training_data[radial_features], y_start=first_layer_height, r=expansion_ratio
    )
    max_training_label_number = np.max(training_data["radial_labels"])

    radial_clusters = geometric_progression_clustering(
        all_cfd_data[radial_features],
        y_start=first_layer_height,
        r=expansion_ratio,
        y_training_end=total_thickness,
        training_max_layer=max_training_label_number,
    )[0]

    all_cfd_data["coarse_mesh_parent_cell_id"] = np.nan
    all_cfd_data["radial_labels"] = radial_clusters

    for label_index, label in tqdm(enumerate(np.unique(radial_clusters))):
        sub_training_data = training_data[training_data["radial_labels"] == label]

        if sub_training_data.empty:
            continue

        sub_cluster_model.fit(sub_training_data[sub_features])

        sub_clusters = sub_cluster_model.predict(
            all_cfd_data[all_cfd_data["radial_labels"] == label][sub_features]
        )
        sub_clusters = (sub_clusters + 1) + label_index * n_sub_clusters

        all_cfd_data.loc[
            all_cfd_data["radial_labels"] == label, "coarse_mesh_parent_cell_id"
        ] = sub_clusters
