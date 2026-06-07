import os
import json
import math
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm

def haversine_distance(coord1, coord2):
    """Calculate Haversine distance between two points (unit: km)"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def generate_poi_json(search_radius=1.0):
    """Generate detailed POI JSON file in Shanghai format"""
    BASE_DIR = '/root/MoELLM/data/nanjing'
    JSON_PATH = os.path.join(BASE_DIR, 'base_station.json')
    POI_PATH = os.path.join(BASE_DIR, 'POI.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'nanjing_poi.json')

    # Load base station data
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        base_data = json.load(f)

    # Load POI data
    poi_df = pd.read_csv(POI_PATH)

    # Build BallTree for POIs
    poi_coords = poi_df[['lat', 'lon']].values
    poi_coords_rad = np.radians(poi_coords)
    tree = BallTree(poi_coords_rad, metric='haversine')

    # Convert search radius to radians
    radius_rad = search_radius / 6371.0

    # Build output JSON
    output_data = {}

    for base_name, base_info in tqdm(base_data.items(), desc="Processing base stations"):
        idx_final = base_info['idx_final']
        loc = base_info['loc']

        # Extract coordinates
        if isinstance(loc[0], list):
            lon, lat = float(loc[0][0]), float(loc[0][1])
        else:
            lon, lat = float(loc[0]), float(loc[1])

        # Query nearby POIs
        station_coord_rad = np.radians([[lat, lon]])
        indices = tree.query_radius(station_coord_rad, r=radius_rad)[0]

        # Build POI list and category stats
        poi_list = []
        category_stats = {}

        for poi_idx in indices:
            poi_row = poi_df.iloc[poi_idx]
            poi_lat = poi_row['lat']
            poi_lon = poi_row['lon']

            # Calculate distance
            distance_km = haversine_distance((lat, lon), (poi_lat, poi_lon))

            # Add to POI list
            poi_list.append({
                "name": str(poi_row['name']),
                "category": str(poi_row['category']),
                "subcategory": str(poi_row['subcategory']),
                "longitude": float(poi_lon),
                "latitude": float(poi_lat),
                "district": str(poi_row['district']),
                "distance_km": round(distance_km, 3)
            })

            # Update category stats
            category = str(poi_row['category'])
            category_stats[category] = category_stats.get(category, 0) + 1

        # Sort POI list by distance
        poi_list.sort(key=lambda x: x['distance_km'])

        # Build output entry
        output_data[base_name] = {
            "base_id": base_name,
            "idx_final": idx_final,
            "longitude": lon,
            "latitude": lat,
            "search_radius_km": search_radius,
            "poi_total": len(poi_list),
            "poi_category_stats": dict(sorted(category_stats.items(), key=lambda x: x[1], reverse=True)),
            "poi_list": poi_list
        }

    # Save to file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {OUTPUT_PATH}")
    print(f"Total base stations: {len(output_data)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate detailed POI JSON')
    parser.add_argument('--radius', type=float, default=1.0,
                        help='POI search radius (km), default: 1.0')
    args = parser.parse_args()

    generate_poi_json(search_radius=args.radius)
