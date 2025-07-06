# Add this function to both a_fuse_and_train.py and b_ensemble_and_train.py
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import pandas as pd



def anomaly_cluster_analysis(scaled_features, scores, frame_ids_original, report_dir, pipeline_name="Fused", frames_dir="data/frames"):
    """
    Performs UMAP projection, HDBSCAN clustering, and visualizes sample frames from each cluster.
    """
    print(f"\n--- Pipeline {pipeline_name}, Step 4: Anomaly Cluster Analysis ---")

    # 1. Identify Anomalies
    threshold = np.percentile(scores, 95)
    is_anomaly = scores >= threshold
    anomaly_indices = np.where(is_anomaly)[0]

    if len(anomaly_indices) < 10:
        print("Not enough anomalies detected for a meaningful cluster analysis. Skipping.")
        return

    anomaly_features = scaled_features[anomaly_indices, :]
    anomaly_frame_ids = frame_ids_original[anomaly_indices]
    print(f"Identified {len(anomaly_indices)} anomalies (top 5%) for clustering.")

    # 2. UMAP Projection
    print("Running UMAP to project features into 2D...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_features)

    # 3. HDBSCAN Clustering
    print("Running HDBSCAN to find anomaly clusters...")
    # Use the high-dimensional features for clustering for more robustness
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=None, gen_min_span_tree=True)
    clusterer.fit(anomaly_features)
    
    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} distinct anomaly clusters and {np.sum(labels == -1)} noise points.")

    # 4. Visualization: UMAP Scatter Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c='lightgrey', s=5, label='Normal Frames')
    
    scatter = plt.scatter(
        embedding[anomaly_indices, 0],
        embedding[anomaly_indices, 1],
        c=labels,
        cmap='Spectral', # Using 'Spectral' as it's good for categorical data
        s=50,
        edgecolor='black',
        linewidth=0.5
    )

    plt.title(f'Anomaly Cluster Analysis ({pipeline_name} Model)', fontsize=16)
    plt.xlabel("UMAP Dimension 1"); plt.ylabel("UMAP Dimension 2")
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(n_clusters)] + ['Noise'])
    
    report_path = os.path.join(report_dir, "3_anomaly_cluster_analysis.png")
    plt.savefig(report_path)
    plt.close()
    print(f"Cluster analysis visualization saved to {report_path}")

    # 5. Visualization: Sample Frames from Each Cluster
    print("Generating sample frames for each identified cluster...")
    results_df = pd.DataFrame({'frame_id': anomaly_frame_ids, 'cluster_id': labels})
    
    for cluster_id in range(n_clusters):
        cluster_frames = results_df[results_df['cluster_id'] == cluster_id]['frame_id'].tolist()
        
        # Select up to 4 sample frames to display
        sample_frames = cluster_frames[:4]
        
        if not sample_frames:
            continue

        num_samples = len(sample_frames)
        fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
        if num_samples == 1: # Matplotlib handles single subplots differently
            axes = [axes]
            
        fig.suptitle(f'Sample Frames from Anomaly Cluster #{cluster_id}', fontsize=16)
        
        for i, frame_id in enumerate(sample_frames):
            img_path = os.path.join(frames_dir, f"{frame_id}.jpg")
            try:
                img = mpimg.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"Frame: {frame_id}")
                axes[i].axis('off')
            except FileNotFoundError:
                print(f"Warning: Image file not found for frame {frame_id}")
                axes[i].text(0.5, 0.5, f"Image not found\n{frame_id}", ha='center')
                axes[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        cluster_report_path = os.path.join(report_dir, f"4_cluster_{cluster_id}_samples.png")
        plt.savefig(cluster_report_path)
        plt.close()

    print("Cluster sample frame generation complete.")