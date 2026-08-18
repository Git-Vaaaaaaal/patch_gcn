"""
Converts per-slide CSV embeddings to PyTorch Geometric graph files (.pt).

Expected CSV format (one row per patch):
    x, y, feat_0, feat_1, ..., feat_N
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENCODER_CFG = {
    "prism":   dict(in_shape=2560, tiles_subdir="features_virchow",   slide_subdir="slide_features_prism",  slide_csv="prism_encoder.csv"),
    "titan":   dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_titan",  slide_csv="titan_encoder.csv"),
    "feather": dict(in_shape=768,  tiles_subdir="features_conch_v15", slide_subdir="slide_features_feather", slide_csv="feather_encoder.csv"),
    "gpfm": dict(in_shape=1024,  tiles_subdir="features_gpfm", slide_subdir="", slide_csv=""),
    "musk": dict(in_shape=1024,  tiles_subdir="features_musk", slide_subdir="", slide_csv=""),
    "openmidnight": dict(in_shape=1536,  tiles_subdir="features_openmidnight", slide_subdir="", slide_csv=""),
    "hibou_l": dict(in_shape=1024,  tiles_subdir="features_hibou_l", slide_subdir="", slide_csv=""),
    "virchow2": dict(in_shape=2560,  tiles_subdir="features_virchow2", slide_subdir="", slide_csv=""),
}

def build_graph(csv_path: str, k: int) -> Data:
    df = pd.read_csv(csv_path)

    coord_cols = ['x', 'y']
    feat_cols = [c for c in df.columns if c not in coord_cols]

    coords = df[coord_cols].values.astype(np.float32)      # [N, 2]
    features = df[feat_cols].values.astype(np.float32)     # [N, D]

    # Spatial KNN graph (euclidean distance on patch coordinates)
    n_neighbors = min(k, len(df) - 1)
    A = kneighbors_graph(coords, n_neighbors=n_neighbors,
                         mode='connectivity', include_self=False)
    rows, cols = A.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)  # [2, E]

    # Optional: latent KNN graph (similarity in feature space)
    A_latent = kneighbors_graph(features, n_neighbors=n_neighbors,
                                mode='connectivity', include_self=False)
    rows_l, cols_l = A_latent.nonzero()
    edge_latent = torch.tensor(np.array([rows_l, cols_l]), dtype=torch.long)  # [2, E]

    data = Data(
        x=torch.tensor(features),
        edge_index=edge_index,
        edge_latent=edge_latent,
    )
    return data


def csv_to_graph(csv_dir, output_dir, k_neighbors=6, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    print(f"Found {len(csv_files)} CSV files in {csv_dir}")

    for i, fname in enumerate(csv_files):
        slide_id = os.path.splitext(fname)[0]
        csv_path = os.path.join(csv_dir, fname)
        out_path = os.path.join(output_dir, f"{slide_id}.pt")

        if os.path.exists(out_path) and not overwrite:
            print(f"[{i+1}/{len(csv_files)}] Skipping {slide_id} (already exists)")
            continue

        try:
            graph = build_graph(csv_path, k=k_neighbors)
            torch.save(graph, out_path)
            print(f"[{i+1}/{len(csv_files)}] {slide_id}: {graph.x.shape[0]} patches, "
                  f"{graph.edge_index.shape[1]} edges, dim={graph.x.shape[1]}")
        except Exception as e:
            print(f"[{i+1}/{len(csv_files)}] ERROR on {slide_id}: {e}")

    print("Done.")

list_encoder = ["openmidnight", "musk", "virchow2", "gpfm", "hibou_l"]
marker_list = ["BCL2", "BCL6", "CD10", "HE", "MUM1", "MYC"]

for encoder in list_encoder:
    for marker in marker_list:
        csv_dir = os.path.join(BASE_DIR, "data_224_reborn", encoder, marker, ENCODER_CFG[encoder]["tiles_subdir"])
        output_dir = os.path.join(BASE_DIR, "data_224_reborn", encoder, marker, "graphs")
        k_neighbors = 6      # nombre de voisins spatiaux par patch
        overwrite   = False  # True pour re-générer les fichiers existants

        csv_to_graph(csv_dir, output_dir, k_neighbors, overwrite)
