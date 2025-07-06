import os
import json
import numpy as np
import pandas as pd

# --- Configuration ---
PERCEPTION_DIR = "data/perception_output"
OUTPUT_CSV = "data/object_features.csv"

CENTER_LANE_X_MIN = 0.4
CENTER_LANE_X_MAX = 0.6
LARGE_OBJECT_AREA_THRESHOLD = 0.15

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}
VRU_CLASSES = {'person', 'bicycle'}
ANOMALOUS_STATIC_CLASSES = {'stop sign', 'fire hydrant', 'parking meter'}

# --- Main Feature Extraction Function ---
def extract_object_features(frame_data):
    """
    Takes the perception data for a single frame and returns a dictionary of features.
    """
    # --- Create a list of all possible feature names first ---
    # This ensures that even if no objects are detected, the dictionary
    # returned has all the correct keys with a value of 0.
    feature_names = [
        'num_total_objects', 'avg_confidence', 'min_confidence', 'scene_diversity_score',
        'total_object_area_ratio', 'max_box_area', 'num_large_objects',
        'num_vehicles', 'num_vrus', 'log_vehicle_to_vru_ratio', # <-- UPDATED FEATURE NAME
        'is_vru_present', 'num_anomalous_static', 'is_anomalous_static_object_present',
        'num_objects_in_center_lane', 'avg_y_center_of_vehicles'
    ]
    
    # Handle the edge case of no objects detected
    if not frame_data:
        return {name: 0.0 for name in feature_names} # Return 0.0 for float consistency

    # --- Initialize variables ---
    num_total_objects = len(frame_data)
    confidences = [obj['confidence'] for obj in frame_data]
    class_names = {obj['class_name'] for obj in frame_data}
    box_areas = [obj['box_normalized_xywh'][2] * obj['box_normalized_xywh'][3] for obj in frame_data]

    # --- 1. Scene Composition & Diversity ---
    features = {
        'num_total_objects': num_total_objects,
        'avg_confidence': np.mean(confidences) if confidences else 0,
        'min_confidence': np.min(confidences) if confidences else 0,
        'scene_diversity_score': len(class_names),
        'total_object_area_ratio': np.sum(box_areas),
        'max_box_area': np.max(box_areas) if box_areas else 0,
        'num_large_objects': np.sum([1 for area in box_areas if area > LARGE_OBJECT_AREA_THRESHOLD]),
    }

    # --- 2. Anomaly & Risk Indicators ---
    num_vehicles = sum(1 for obj in frame_data if obj['class_name'] in VEHICLE_CLASSES)
    num_vrus = sum(1 for obj in frame_data if obj['class_name'] in VRU_CLASSES)
    num_anomalous_static = sum(1 for obj in frame_data if obj['class_name'] in ANOMALOUS_STATIC_CLASSES)

    # Calculate the raw ratio
    raw_ratio = num_vehicles / (num_vrus + 1e-6)
    # Apply the log1p transform for numerical stability
    log_ratio = np.log1p(raw_ratio)

    features.update({
        'num_vehicles': num_vehicles,
        'num_vrus': num_vrus,
        'log_vehicle_to_vru_ratio': log_ratio, # <-- USE THE NEW LOG-TRANSFORMED FEATURE
        'is_vru_present': 1 if num_vrus > 0 else 0,
        'num_anomalous_static': num_anomalous_static,
        'is_anomalous_static_object_present': 1 if num_anomalous_static > 0 else 0,
    })

    # --- 3. Spatial Layout & Proximity ---
    center_lane_objects = [
        obj for obj in frame_data 
        if CENTER_LANE_X_MIN < obj['box_normalized_xywh'][0] < CENTER_LANE_X_MAX
    ]
    features['num_objects_in_center_lane'] = len(center_lane_objects)

    vehicle_y_centers = [
        obj['box_normalized_xywh'][1] for obj in frame_data 
        if obj['class_name'] in VEHICLE_CLASSES
    ]
    features['avg_y_center_of_vehicles'] = np.mean(vehicle_y_centers) if vehicle_y_centers else 0

    return features

if __name__ == "__main__":
    json_files = sorted([f for f in os.listdir(PERCEPTION_DIR) if f.endswith('.json')])
    all_features_list = []

    for i, json_filename in enumerate(json_files):
        json_path = os.path.join(PERCEPTION_DIR, json_filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
        frame_features = extract_object_features(data)
        frame_features['frame_id'] = os.path.splitext(json_filename)[0]
        all_features_list.append(frame_features)
        print(f"Processed features for frame {i+1}/{len(json_files)}: {json_filename}")

    features_df = pd.DataFrame(all_features_list)
    cols = ['frame_id'] + [col for col in features_df.columns if col != 'frame_id']
    features_df = features_df[cols]
    features_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nSuccessfully created and saved object-level features to '{OUTPUT_CSV}'")
    print("\nDataFrame head:")
    print(features_df.head())

