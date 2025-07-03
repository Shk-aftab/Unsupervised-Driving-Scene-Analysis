import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# --- Configuration ---
FINAL_FEATURES_PATH = "data/final_dataset/final_fused_features.npy"
FINAL_FRAME_IDS_PATH = "data/final_dataset/final_frame_ids.npy"

RESULTS_DIR = "results"
ANOMALY_SCORES_PATH = os.path.join(RESULTS_DIR, "anomaly_scores.csv")

CONTAMINATION_PARAM = 'auto'
RANDOM_STATE = 42

# --- Main Script ---
def train_and_predict_anomalies(features_path, frame_ids_path, output_path):
    """
    Loads the fused feature matrix, scales it, trains an Isolation Forest model,
    predicts anomaly scores, and saves the results.
    """
    print("--- Starting Anomaly Detection Training ---")

    print(f"Loading data from {features_path}...")
    fused_features = np.load(features_path)
    
    # --- FIX IS HERE ---
    # We add allow_pickle=True because frame_ids is an array of strings (objects)
    frame_ids = np.load(frame_ids_path, allow_pickle=True)
    
    if fused_features.shape[0] != len(frame_ids):
        raise ValueError("Mismatch between number of features and frame IDs.")
    
    print(f"Loaded {fused_features.shape[0]} samples with {fused_features.shape[1]} features each.")

    print("Applying StandardScaler to the features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(fused_features)
    print("  -> Features scaled successfully.")
    
    print("Training Isolation Forest model...")
    iso_forest = IsolationForest(
        contamination=CONTAMINATION_PARAM,
        random_state=RANDOM_STATE,
        n_jobs=-1 
    )
    iso_forest.fit(scaled_features)
    print("  -> Model training complete.")

    print("Calculating anomaly scores for each frame...")
    anomaly_scores = iso_forest.decision_function(scaled_features) * -1
    
    print("Saving results to CSV...")
    results_df = pd.DataFrame({
        'frame_id': frame_ids,
        'anomaly_score': anomaly_scores
    })
    
    results_df = results_df.sort_values(by='anomaly_score', ascending=False)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Anomaly Detection Complete ---")
    print(f"Results saved to '{output_path}'")
    print("\nTop 5 most anomalous frames:")
    print(results_df.head())


# --- Run the script ---
if __name__ == "__main__":
    train_and_predict_anomalies(
        FINAL_FEATURES_PATH, 
        FINAL_FRAME_IDS_PATH, 
        ANOMALY_SCORES_PATH
    )