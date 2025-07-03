import pandas as pd
import numpy as np
import os

# --- Configuration ---
VISUAL_EMBEDDINGS_PATH = "data/visual_embeddings.npy"
OBJECT_FEATURES_PATH = "data/object_features.csv"
MOTION_FEATURES_PATH = "data/motion_features.csv"

OUTPUT_DIR = "data/final_dataset"
FINAL_FEATURES_PATH = os.path.join(OUTPUT_DIR, "final_fused_features.npy")
FINAL_FRAME_IDS_PATH = os.path.join(OUTPUT_DIR, "final_frame_ids.npy")
FINAL_FEATURE_NAMES_PATH = os.path.join(OUTPUT_DIR, "final_feature_names.json")


# --- Main Script ---
def fuse_all_features(visual_path, object_path, motion_path):
    """
    Loads visual, object, and motion features, fuses them into a single
    feature matrix, and saves the final dataset.
    """
    print("--- Starting Feature Fusion ---")
    
    # --- 1. Load all feature sets ---
    
    # Load Visual Embeddings (NumPy array)
    print(f"Loading visual embeddings from {visual_path}...")
    visual_embeddings = np.load(visual_path)
    print(f"  -> Loaded visual embeddings with shape: {visual_embeddings.shape}")
    
    # Load Object-level Features (CSV)
    print(f"Loading object features from {object_path}...")
    object_df = pd.read_csv(object_path)
    print(f"  -> Loaded object features with shape: {object_df.shape}")
    
    # Load Motion Features (CSV)
    print(f"Loading motion features from {motion_path}...")
    motion_df = pd.read_csv(motion_path)
    print(f"  -> Loaded motion features with shape: {motion_df.shape}")
    
    # --- 2. Align and Merge DataFrames ---
    
    # The 'frame_id' is the key to join our data correctly.
    # We'll use an 'inner' join to ensure we only keep frames that have all types of features.
    print("Merging object and motion features based on 'frame_id'...")
    
    # Ensure 'frame_id' is sorted consistently before merging
    object_df = object_df.sort_values('frame_id').reset_index(drop=True)
    motion_df = motion_df.sort_values('frame_id').reset_index(drop=True)

    # Merge the two dataframes
    merged_df = pd.merge(object_df, motion_df, on='frame_id', how='inner')
    print(f"  -> Merged DataFrame shape: {merged_df.shape}")

    # --- 3. Align NumPy array with Merged DataFrame ---
    
    # It's crucial that the rows of the visual embeddings NumPy array correspond
    # exactly to the rows of our merged DataFrame. Since we sorted everything
    # by frame_id (e.g., frame_00000, frame_00001, ...), their order
    # should already match perfectly. We'll add a check.
    
    if len(visual_embeddings) != len(merged_df):
        raise ValueError(
            f"Mismatch in number of frames between visual embeddings ({len(visual_embeddings)}) "
            f"and merged features ({len(merged_df)}). Check the input files."
        )
    print("Alignment between visual embeddings and other features confirmed.")

    # --- 4. Create the Final Fused Feature Matrix ---
    
    # Extract the numerical data from the merged dataframe
    # We drop the 'frame_id' column as it's not a feature for the model
    other_features_matrix = merged_df.drop(columns=['frame_id']).values
    
    # Concatenate the visual embeddings and the other features horizontally
    final_fused_matrix = np.concatenate([visual_embeddings, other_features_matrix], axis=1)
    
    print("\n--- Fusion Complete ---")
    print(f"Final fused feature matrix shape: {final_fused_matrix.shape}")
    print(f" (Rows = Frames, Columns = Total Features)")

    # --- 5. Save the Final Datasets for Model Training ---
    
    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save the main feature matrix
    np.save(FINAL_FEATURES_PATH, final_fused_matrix)
    print(f"Saved final fused feature matrix to: {FINAL_FEATURES_PATH}")
    
    # Save the frame_ids for later lookup
    frame_ids = merged_df['frame_id'].values
    np.save(FINAL_FRAME_IDS_PATH, frame_ids)
    print(f"Saved frame IDs for lookup to: {FINAL_FRAME_IDS_PATH}")

    # Save the feature names for interpretability
    # This is extremely useful for understanding our model's results later!
    visual_feature_names = [f'visual_{i}' for i in range(visual_embeddings.shape[1])]
    other_feature_names = merged_df.drop(columns=['frame_id']).columns.tolist()
    all_feature_names = visual_feature_names + other_feature_names
    
    import json
    with open(FINAL_FEATURE_NAMES_PATH, 'w') as f:
        json.dump(all_feature_names, f, indent=4)
    print(f"Saved feature names for interpretability to: {FINAL_FEATURE_NAMES_PATH}")


# --- Run the script ---
if __name__ == "__main__":
    fuse_all_features(
        VISUAL_EMBEDDINGS_PATH, 
        OBJECT_FEATURES_PATH, 
        MOTION_FEATURES_PATH
    )