import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

# --- Configuration ---
# --- Input Data Paths ---
RESULTS_PATH = "results/anomaly_scores.csv"
FEATURES_PATH = "data/final_dataset/final_fused_features.npy"
FRAME_IDS_PATH = "data/final_dataset/final_frame_ids.npy"
FEATURE_NAMES_PATH = "data/final_dataset/final_feature_names.json"
FRAMES_DIR = "data/frames"
ANNOTATED_FRAMES_DIR = "data/validation_images" # We can use these if they exist

# --- Output Directory ---
REPORT_DIR = "report"
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Analysis Parameters ---
NUM_TOP_ANOMALIES = 10  # Number of top anomalies to show in the "Hall of Fame"
ANOMALY_PERCENTILE = 99 # Threshold for defining what is an "anomaly" for clustering

# Set plot style
sns.set_style("whitegrid")

# --- Main Functions ---

def plot_time_series(df):
    """Plots the anomaly score over time."""
    print("Generating time-series plot...")
    df_sorted_by_frame = df.sort_values(by='frame_id').reset_index(drop=True)
    
    plt.figure(figsize=(20, 6))
    plt.plot(df_sorted_by_frame['anomaly_score'], label='Anomaly Score')
    
    threshold = np.percentile(df_sorted_by_frame['anomaly_score'], ANOMALY_PERCENTILE)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'{ANOMALY_PERCENTILE}th Percentile Threshold')
    
    plt.title('Anomaly Score Over Time', fontsize=16)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Anomaly Score (Higher is more anomalous)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "1_anomaly_score_timeseries.png"))
    plt.close()
    print("  -> Saved 1_anomaly_score_timeseries.png")
    return threshold

def generate_hall_of_fame(df, features_df, threshold):
    """Creates a report for the top N most anomalous frames."""
    print("Generating 'Hall of Fame' for top anomalies...")
    top_anomalies = df.head(NUM_TOP_ANOMALIES)
    
    for i, row in top_anomalies.iterrows():
        frame_id = row['frame_id']
        score = row['anomaly_score']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Top Anomaly #{i+1}: {frame_id}\nScore: {score:.4f} (Threshold: {threshold:.4f})", fontsize=16)
        
        # Display original frame
        try:
            img_path = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
            axes[0].imshow(mpimg.imread(img_path))
            axes[0].set_title("Original Frame")
            axes[0].axis('off')
        except FileNotFoundError:
            axes[0].text(0.5, 0.5, "Image not found", ha='center')

        # Display feature spotlight
        frame_features = features_df[features_df['frame_id'] == frame_id].iloc[0]
        top_5_features = frame_features.drop('frame_id').sort_values(ascending=False).head(5)
        
        text_report = "Feature Spotlight (Top 5 Feature Values):\n\n"
        for feature, value in top_5_features.items():
            text_report += f"- {feature}: {value:.2f}\n"

        axes[1].text(0.05, 0.5, text_report, va='center', fontsize=12, family='monospace')
        axes[1].set_title("Feature Analysis")
        axes[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(REPORT_DIR, f"2_top_anomaly_{i+1}_{frame_id}.png"))
        plt.close()
    print(f"  -> Saved {NUM_TOP_ANOMALIES} top anomaly reports.")


def perform_anomaly_clustering(df, features_df, threshold):
    """Performs dimensionality reduction and clustering to find types of anomalies."""
    print("Performing advanced anomaly clustering...")
    
    # 1. Isolate anomalous data
    anomalous_df = df[df['anomaly_score'] >= threshold]
    if len(anomalous_df) < 15: # UMAP/HDBSCAN need a minimum number of points
        print("  -> Not enough anomalies above threshold for clustering. Skipping.")
        return

    print(f"  -> Found {len(anomalous_df)} frames above the {ANOMALY_PERCENTILE}th percentile for clustering.")
    anomalous_features = features_df[features_df['frame_id'].isin(anomalous_df['frame_id'])]
    anomalous_matrix = anomalous_features.drop(columns=['frame_id']).values
    
    # Scale only the anomalous subset
    scaled_anomalous_matrix = StandardScaler().fit_transform(anomalous_matrix)

    # 2. Reduce dimensionality with UMAP
    print("  -> Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_anomalous_matrix)

    # 3. Cluster with HDBSCAN
    print("  -> Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding)
    
    anomalous_df['cluster'] = cluster_labels

    # 4. Visualize the clusters
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=cluster_labels,
        palette=sns.color_palette("hsv", len(set(cluster_labels))),
        s=50,
        alpha=0.7
    )
    plt.title('UMAP Projection of Anomalies, Colored by HDBSCAN Cluster', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Cluster Label (-1=Noise)')
    plt.savefig(os.path.join(REPORT_DIR, "3_anomaly_cluster_map.png"))
    plt.close()
    print("  -> Saved 3_anomaly_cluster_map.png")
    
    # 5. Interpret and report on each cluster
    unique_clusters = sorted(list(set(cluster_labels)))
    for cluster_id in unique_clusters:
        if cluster_id == -1: continue # Skip the noise points for this report
        
        print(f"    -> Analyzing Cluster #{cluster_id}...")
        cluster_frames = anomalous_df[anomalous_df['cluster'] == cluster_id]
        
        # Get 4 example images from the cluster
        sample_frames = cluster_frames.sample(min(4, len(cluster_frames)))
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"Anomaly Cluster #{cluster_id} - Sample Frames", fontsize=16)
        axes = axes.flatten()
        
        for i, (idx, row) in enumerate(sample_frames.iterrows()):
            try:
                img_path = os.path.join(FRAMES_DIR, f"{row['frame_id']}.jpg")
                axes[i].imshow(mpimg.imread(img_path))
                axes[i].set_title(row['frame_id'])
                axes[i].axis('off')
            except FileNotFoundError:
                axes[i].text(0.5, 0.5, "Image not found", ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(REPORT_DIR, f"4_cluster_{cluster_id}_examples.png"))
        plt.close()
    print("  -> Saved example images for each cluster.")


# --- Main Execution ---
if __name__ == "__main__":
    # Load all necessary data
    results_df = pd.read_csv(RESULTS_PATH)
    all_features_matrix = np.load(FEATURES_PATH)
    frame_ids = np.load(FRAME_IDS_PATH, allow_pickle=True)
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = json.load(f)

    # Create a single comprehensive DataFrame for easier analysis
    features_df = pd.DataFrame(all_features_matrix, columns=feature_names)
    features_df.insert(0, 'frame_id', frame_ids)

    # --- Generate Visualizations ---
    anomaly_threshold = plot_time_series(results_df)
    generate_hall_of_fame(results_df, features_df, anomaly_threshold)
    perform_anomaly_clustering(results_df, features_df, anomaly_threshold)
    
    print("\n--- Visualization Report Generation Complete ---")
    print(f"All reports saved in the '{REPORT_DIR}' directory.")