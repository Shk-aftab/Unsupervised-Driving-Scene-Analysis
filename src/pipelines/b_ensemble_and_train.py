import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from anomaly_cluster_analysis import anomaly_cluster_analysis

# --- Configuration for Pipeline B ---
VISUAL_FEATURES_PATH = "data/visual_embeddings_clip.npy" # Specific to CLIP
OBJECT_FEATURES_PATH = "data/object_features.csv"
MOTION_FEATURES_PATH = "data/motion_features.csv"
FRAMES_DIR = "data/frames"
REPORT_DIR = "report_ensemble"

def get_expert_data():
    print("--- Pipeline B, Step 1: Preparing data for expert models ---")
    
    # Create and align context data
    object_df = pd.read_csv(OBJECT_FEATURES_PATH)
    motion_df = pd.read_csv(MOTION_FEATURES_PATH)
    context_df = pd.merge(object_df, motion_df, on='frame_id', how='inner')
    
    # Align visual (CLIP) data
    all_clip_embeddings = np.load(VISUAL_FEATURES_PATH)
    all_frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
    all_frame_ids = [os.path.splitext(f)[0] for f in all_frame_files]
    
    frame_map = {frame_id: i for i, frame_id in enumerate(all_frame_ids)}
    indices = [frame_map[fid] for fid in context_df['frame_id']]
    
    visual_features = all_clip_embeddings[indices]
    context_features = context_df.drop(columns=['frame_id']).values
    frame_ids = context_df['frame_id'].values

    print(f"Data prepared. Shapes: Visual({visual_features.shape}), Context({context_features.shape})")
    return visual_features, context_features, frame_ids

def train_ensemble(visual_features, context_features):
    print("\n--- Pipeline B, Step 2: Training expert models ---")
    
    # Visual Expert
    scaler_viz = StandardScaler(); scaled_viz = scaler_viz.fit_transform(visual_features)
    iforest_viz = IsolationForest(contamination='auto', random_state=42, n_jobs=-1).fit(scaled_viz)
    scores_viz = iforest_viz.decision_function(scaled_viz) * -1
    
    # Context Expert
    scaler_ctx = StandardScaler(); scaled_ctx = scaler_ctx.fit_transform(context_features)
    iforest_ctx = IsolationForest(contamination='auto', random_state=42, n_jobs=-1).fit(scaled_ctx)
    scores_ctx = iforest_ctx.decision_function(scaled_ctx) * -1
    
    print("Fusing scores using MinMaxScaler and Maximum...")
    scaler_scores = MinMaxScaler()
    scores_viz_scaled = scaler_scores.fit_transform(scores_viz.reshape(-1, 1)).flatten()
    scores_ctx_scaled = scaler_scores.fit_transform(scores_ctx.reshape(-1, 1)).flatten()
    final_scores = np.maximum(scores_viz_scaled, scores_ctx_scaled)
    
    return final_scores, scores_viz_scaled, scores_ctx_scaled

def visualize_results(final_scores, visual_scores, context_scores, frame_ids):
    print("\n--- Pipeline B, Step 3: Visualizing ensemble results ---")
    os.makedirs(REPORT_DIR, exist_ok=True)
    results_df = pd.DataFrame({
        'frame_id': frame_ids, 'final_anomaly_score': final_scores,
        'visual_score': visual_scores, 'context_score': context_scores
    }).sort_values('final_anomaly_score', ascending=False)

    # Time-series plot
    df_sorted = results_df.sort_values(by='frame_id').reset_index(drop=True)
    plt.figure(figsize=(20, 8))
    plt.plot(df_sorted['final_anomaly_score'], label='Final Score (Max)', linewidth=2.5, color='black')
    plt.plot(df_sorted['visual_score'], label='Visual Expert (CLIP)', alpha=0.7, linestyle='--')
    plt.plot(df_sorted['context_score'], label='Context Expert (Object+Motion)', alpha=0.7, linestyle='--')
    plt.title('Ensemble Model (CLIP) - Anomaly Scores Over Time', fontsize=16)
    plt.xlabel('Frame Number'); plt.ylabel('Normalized Score'); plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, "1_ensemble_timeseries.png"))
    plt.close()
    
    # Hall of Fame
    top_anomalies = results_df.head(10)
    for i, row in top_anomalies.iterrows():
        frame_id = row['frame_id']
        fig = plt.figure(figsize=(12, 6)); gs = fig.add_gridspec(2, 2)
        ax_img, ax_text = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
        fig.suptitle(f"Top Anomaly #{i+1}: {frame_id}", fontsize=16)
        
        img_path = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
        ax_img.imshow(mpimg.imread(img_path)); ax_img.set_title("Anomalous Frame"); ax_img.axis('off')
        
        primary_cause = "Visual" if row['visual_score'] > row['context_score'] else "Context"
        text_report = f"Final Score : {row['final_anomaly_score']:.3f}\nVisual Score: {row['visual_score']:.3f}\nContext Score: {row['context_score']:.3f}\n\nPrimary Cause: **{primary_cause}**"
        ax_text.text(0.05, 0.5, text_report, va='center', fontsize=12, family='monospace')
        ax_text.set_title("Expert Analysis"); ax_text.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(os.path.join(REPORT_DIR, f"2_top_anomaly_ensemble_{i+1}_{frame_id}.png"))
        plt.close()

    print(f"Visualization complete. Reports saved in '{REPORT_DIR}'")

if __name__ == "__main__":
    sns.set_style("whitegrid")
    visual_features, context_features, frame_ids = get_expert_data()
    final_scores, visual_scores, context_scores = train_ensemble(visual_features, context_features)
    visualize_results(final_scores, visual_scores, context_scores, frame_ids)

    # MODIFIED CALL to step4
    print("\nFusing features for cluster visualization...")
    scaler_viz = StandardScaler()
    scaler_ctx = StandardScaler()
    scaled_viz_for_viz = scaler_viz.fit_transform(visual_features)
    scaled_ctx_for_viz = scaler_ctx.fit_transform(context_features)
    fused_features_for_viz = np.concatenate([scaled_viz_for_viz, scaled_ctx_for_viz], axis=1)

    anomaly_cluster_analysis(
        scaled_features=fused_features_for_viz,
        scores=final_scores,
        frame_ids_original=frame_ids, # Pass the frame IDs
        report_dir=REPORT_DIR,
        pipeline_name="Ensemble-CLIP",
        frames_dir=FRAMES_DIR
    )

    print("\nPIPELINE B (ENSEMBLE APPROACH) HAS COMPLETED SUCCESSFULLY.")