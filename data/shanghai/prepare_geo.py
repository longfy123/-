import os
import numpy as np
import json
import math

def adj_geo_npy():
    BASE_DIR = '/root/MoELLM/data/shanghai'
    JSON_PATH = os.path.join(BASE_DIR, 'base_station.json')
    ADJ_PATH = os.path.join(BASE_DIR, 'adj_geo.npz')

    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

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

        sorted_indices = sorted(station_coords.keys())
        n = len(sorted_indices)

        # Build adjacency matrix
        W = np.eye(n, dtype=np.float32)
        σ, k = 20.0, 1.0

        edge_count = 0

        # Iterate over all station pairs
        for i, i_idx in enumerate(sorted_indices):
            for j, j_idx in enumerate(sorted_indices[i+1:], start=i+1):
                # Compute Haversine distance
                dist_km = haversine(station_coords[i_idx], station_coords[j_idx])
                if dist_km <= k:
                    weight = math.exp(-(dist_km**2) / (σ**2))
                    weight_f32 = np.float32(weight)
                    W[i, j] = weight_f32
                    W[j, i] = weight_f32
                    edge_count += 2

        # Save as sparse matrix npz
        import scipy.sparse as sp
        sp.save_npz(ADJ_PATH, sp.csr_matrix(W.astype(np.float32)))

        return True     
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

if __name__ == "__main__":
    adj_geo_npy()