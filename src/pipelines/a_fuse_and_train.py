import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from anomaly_cluster_analysis import anomaly_cluster_analysis

# --- Configuration for Pipeline A ---
VISUAL_FEATURES_PATH = "data/visual_embeddings_vit.npy" # Specific to ViT
OBJECT_FEATURES_PATH = "data/object_features.csv"
MOTION_FEATURES_PATH = "data/motion_features.csv"
FRAMES_DIR = "data/frames"
REPORT_DIR = "report"

def fuse_features():
    print("--- Pipeline A, Step 1: Fusing features (ViT) ---")
    visual_embeddings = np.load(VISUAL_FEATURES_PATH)
    object_df = pd.read_csv(OBJECT_FEATURES_PATH).sort_values('frame_id').reset_index(drop=True)
    motion_df = pd.read_csv(MOTION_FEATURES_PATH).sort_values('frame_id').reset_index(drop=True)
    
    merged_df = pd.merge(object_df, motion_df, on='frame_id', how='inner')
    
    if len(visual_embeddings) != len(merged_df):
        raise ValueError("Frame count mismatch during fusion.")

    other_features_matrix = merged_df.drop(columns=['frame_id']).values
    final_fused_matrix = np.concatenate([visual_embeddings, other_features_matrix], axis=1)
    
    frame_ids = merged_df['frame_id'].values
    visual_feature_names = [f'visual_{i}' for i in range(visual_embeddings.shape[1])]
    other_feature_names = merged_df.drop(columns=['frame_id']).columns.tolist()
    all_feature_names = visual_feature_names + other_feature_names
    
    print(f"Fusion complete. Final matrix shape: {final_fused_matrix.shape}")
    return final_fused_matrix, frame_ids, all_feature_names

def train_model(fused_features):
    print("\n--- Pipeline A, Step 2: Training single Isolation Forest model ---")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(fused_features) # This is what we need
    
    iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
    iso_forest.fit(scaled_features)
    
    scores = iso_forest.decision_function(scaled_features) * -1
    print("Model training and scoring complete.")
    return scores, scaled_features # 

def visualize_results(scores, frame_ids, fused_features, feature_names):
    print("\n--- Pipeline A, Step 3: Visualizing results ---")
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_df = pd.DataFrame({'frame_id': frame_ids, 'anomaly_score': scores}).sort_values('anomaly_score', ascending=False)
    
    # Time-series plot
    plt.figure(figsize=(20, 6))
    plt.plot(results_df.sort_values('frame_id')['anomaly_score'], label='Anomaly Score')
    plt.title('Fused Model (ViT) - Anomaly Score Over Time', fontsize=16)
    plt.xlabel('Frame Number'); plt.ylabel('Anomaly Score'); plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "1_fused_timeseries.png"))
    plt.close()
    
    # Hall of Fame
    features_df = pd.DataFrame(fused_features, columns=feature_names)
    features_df.insert(0, 'frame_id', frame_ids)
    top_anomalies = results_df.head(10)

    for i, row in top_anomalies.iterrows():
        frame_id = row['frame_id']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Top Anomaly #{i+1}: {frame_id} (Score: {row['anomaly_score']:.4f})", fontsize=16)
        
        img_path = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
        axes[0].imshow(mpimg.imread(img_path)); axes[0].set_title("Original Frame"); axes[0].axis('off')

        frame_features = features_df[features_df['frame_id'] == frame_id].iloc[0]
        top_5_features = frame_features.drop('frame_id').sort_values(ascending=False).head(5)
        axes[1].text(0.05, 0.5, "Feature Spotlight:\n\n" + '\n'.join([f"- {k}: {v:.2f}" for k, v in top_5_features.items()]), va='center', fontsize=12, family='monospace')
        axes[1].set_title("Feature Analysis"); axes[1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(REPORT_DIR, f"2_top_anomaly_fused_{i+1}_{frame_id}.png"))
        plt.close()

    print(f"Visualization complete. Reports saved in '{REPORT_DIR}'")

if __name__ == "__main__":
    sns.set_style("whitegrid")
    fused_matrix, frame_ids, feature_names = fuse_features()
    anomaly_scores, scaled_features = train_model(fused_matrix) 
    visualize_results(anomaly_scores, frame_ids, fused_matrix, feature_names)
    
    # MODIFIED CALL to step4
    anomaly_cluster_analysis(
        scaled_features=scaled_features, 
        scores=anomaly_scores, 
        frame_ids_original=frame_ids, # Pass the frame IDs
        report_dir=REPORT_DIR, 
        pipeline_name="Fused-ViT",
        frames_dir=FRAMES_DIR
    )

    print("\nPIPELINE A (FUSED APPROACH) HAS COMPLETED SUCCESSFULLY.")