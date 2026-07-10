# Aerodynamic Design Similarity for Ground Effect (WIG) Airfoils

This repository contains a Python-based data processing and machine learning pipeline designed to evaluate the aerodynamic design similarity of Computational Fluid Dynamics (CFD) meshes. Specifically, this codebase is tailored for studying airfoils in ground effect (Wing-in-Ground, or WIG), identifying localized fluid behavior similarities, and calculating objective similarity scores using clustering algorithms and the Earth Mover's Distance (EMD).

## 🚀 Key Features

* **Custom Mesh Localization:** Segments aerodynamic meshes into distinct regions using geometric progression (mimicking CFD boundary layer inflation) and polar K-Means clustering.
* **Hybrid ML Clustering:** Automatically optimizes and trains clustering models using a combination of `AgglomerativeClustering` and `RadiusNeighborsClassifier` to accurately predict local flow regimes.
* **Feature Normalization & Offset:** Normalizes fluid velocity features locally against a reference "focus case" and computes spatial/feature offset distances.
* **Similarity Scoring:** Quantifies design similarities between airfoils by computing cluster overlap fractions and Earth Mover's Distance (Wasserstein distance) in parallel.

## 📁 Repository Structure

### Core Pipeline Scripts
These scripts are the main entry points for running the data processing and similarity evaluation.

* **`process_ground_effect_case.py`**: The primary pipeline script. It imports raw Parquet CFD data, generates localized cell sets, normalizes features, trains the clustering models, and runs predictions (both standard and offset) across multiple processors.
* **`process_wig_groups.py`**: A variation of the main pipeline script designed to process data in groups (e.g., mapping cases directly to their base NACA profiles, such as `0010` or `4410`) to perform generalized group evaluations.
* **`process_ground_effect_w_sim_scores.py`**: The final step in the pipeline. It takes the clustered predictions from the previous scripts and computes the actual similarity metrics (fraction of matching cells and EMD) between the focus case and comparison cases. Includes logic for a 5% offset tolerance threshold.

### Modules (`design_similarity/`)
These modules contain the underlying mathematical and machine learning logic utilized by the pipeline.

* **`localization.py`**: Handles the physical segmentation of the CFD mesh. Contains `geometric_progression_clustering` to build outward radial bands from the airfoil wall and circumferential sub-clustering.
* **`cluster_model_training.py`**: Contains the logic to perform a grid search on distance thresholds for `AgglomerativeClustering` using silhouette scores, and wraps it in a `RadiusNeighborsClassifier` to allow for `.predict()` functionality on new, unseen data chunks.
* **`similarity_scores.py`**: Contains helper functions for data normalization (median scaling), offset distance calculations (Manhattan distance of features), and the multiprocessing wrappers for evaluating similarity and EMD (`scipy.stats.wasserstein_distance`).

## 🛠️ Installation & Dependencies

To run this pipeline, you will need a Python environment with the following packages installed:

```bash
pip install numpy pandas scikit-learn scipy pyyaml matplotlib seaborn tqdm pyarrow fastparquet