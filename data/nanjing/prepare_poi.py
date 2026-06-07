import os
import numpy as np
import json
import math
import scipy.sparse as sp
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.neighbors import BallTree

def haversine_distance(coord1, coord2):
    """
    Calculate Haversine distance between two points (unit: km)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def check_base_station_poi_match():
    """
    Check the matching between base station coordinates and POI data
    """
    BASE_DIR = '/root/MoELLM/data/nanjing'
    JSON_PATH = os.path.join(BASE_DIR, 'base_station.json')
    POI_PATH = os.path.join(BASE_DIR, 'POI.csv')

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Extract base station coordinates, idx_final already reset to 0~1041
    station_coords = {}  # {idx_final: (lat, lon)}

    for data in json_data.values():
        if 'loc' in data and data['loc'] and 'idx_final' in data:
            idx = data['idx_final']
            loc_array = data['loc']
            if isinstance(loc_array[0], list):
                lon, lat = float(loc_array[0][0]), float(loc_array[0][1])
            else:
                lon, lat = float(loc_array[0]), float(loc_array[1])
            station_coords[idx] = (lat, lon)

    # Load POI data
    poi_df = pd.read_csv(POI_PATH)

    return station_coords, poi_df

def compute_station_poi_features(station_coords, poi_df, search_radius=1.0):

    # Get all unique POI major categories
    all_categories = sorted(poi_df['category'].unique())
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}

    # Build BallTree for POIs (using radians)
    poi_coords = poi_df[['lat', 'lon']].values
    poi_coords_rad = np.radians(poi_coords)  # convert to radians
    tree = BallTree(poi_coords_rad, metric='haversine')

    # Build POI feature vector for each base station
    station_poi_features = {}
    station_index_list = sorted(station_coords.keys())

    # Prepare station coordinate array
    station_coords_arr = np.array([station_coords[idx] for idx in station_index_list])
    station_coords_rad = np.radians(station_coords_arr)

    # Convert km to radians (Earth radius 6371km)
    radius_rad = search_radius / 6371.0

    # Batch query
    indices_list = tree.query_radius(station_coords_rad, r=radius_rad)

    # Pre-encode POI categories as integer array to avoid per-row DataFrame access
    poi_cat_codes = np.array([category_to_index.get(c, -1) for c in poi_df['category']])

    # Build features for each station
    F = np.zeros((len(station_index_list), len(all_categories)), dtype=np.float32)
    for i, idx in enumerate(tqdm(station_index_list, desc="Building POI features")):
        nearby_poi_indices = indices_list[i]
        if len(nearby_poi_indices) > 0:
            cats = poi_cat_codes[nearby_poi_indices]
            valid = cats[cats >= 0]
            np.add.at(F[i], valid, 1)
        station_poi_features[idx] = F[i]

    return station_poi_features, all_categories

def compute_poi_similarity_matrix(station_poi_features, threshold=0.5):
    """
    Compute similarity matrix between stations based on POI features
    Uses cosine similarity
    """

    # Get sorted station index list
    station_index_list = sorted(station_poi_features.keys())
    n = len(station_index_list)

    # Build feature matrix
    feature_matrix = np.array([station_poi_features[idx] for idx in station_index_list])

    # Normalize (L2 norm)
    feature_matrix_norm = feature_matrix / (np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-8)

    # Compute all cosine similarities via matrix multiplication
    W = feature_matrix_norm @ feature_matrix_norm.T
    W = np.clip(W, -1, 1)
    W[W < threshold] = 0
    np.fill_diagonal(W, 1)

    return W

def generate_poi_adj_npz(search_radius=1.0, threshold=0.1):
    """
    Generate POI-based adjacency matrix

    Args:
        search_radius: radius for POI search (km)
        threshold: similarity threshold
    """
    BASE_DIR = '/root/MoELLM/data/nanjing'
    ADJ_PATH = os.path.join(BASE_DIR, 'adj_poi.npz')

    try:
        # Steps 1-3: Check matching
        station_coords, poi_df = check_base_station_poi_match()

        # Step 4: Compute POI features
        station_poi_features, _ = compute_station_poi_features(station_coords, poi_df, search_radius)

        # Step 5: Compute similarity matrix
        W = compute_poi_similarity_matrix(station_poi_features, threshold)

        # Save matrix
        W_sparse = sp.csc_matrix(W)
        sp.save_npz(ADJ_PATH, W_sparse)

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate POI-based adjacency matrix')
    parser.add_argument('--radius', type=float, default=1.0,
                        help='POI search radius (km), default: 1.0')
    parser.add_argument('--threshold', type=float, default=0.995,
                        help='Similarity threshold, between 0-1, default: 0.1')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check matching, do not generate matrix')

    args = parser.parse_args()

    # Generate matrix
    generate_poi_adj_npz(search_radius=args.radius, threshold=args.threshold)
