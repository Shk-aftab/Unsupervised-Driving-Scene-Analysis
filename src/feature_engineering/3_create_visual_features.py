import argparse
import os
import torch
import clip
import timm
from PIL import Image
import numpy as np

def create_embeddings(frames_dir, output_file, model_type):
    """Generates deep feature embeddings using either 'vit' or 'clip'."""
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")

    # --- Model Loading based on argument ---
    if model_type == 'clip':
        print("--- Using CLIP Model (ViT-B/32) ---")
        model, preprocess = clip.load("ViT-B/32", device=device)
        # We need a lambda function to match the expected input for the loop
        encode_func = lambda x: model.encode_image(x)
    elif model_type == 'vit':
        print("--- Using ViT (timm) Model ---")
        model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=0).to(device)
        data_config = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_config, is_training=False)
        encode_func = lambda x: model(x)
    else:
        raise ValueError("model_type must be 'vit' or 'clip'")
        
    model.eval()
    
    # --- Frame Processing Loop ---
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    all_embeddings = []

    with torch.no_grad():
        for i, frame_filename in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_filename)
            img = Image.open(frame_path).convert('RGB')
            
            # Use the generic preprocessor and encoder
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            embedding = encode_func(input_tensor)
            
            embedding_np = embedding.squeeze().cpu().numpy()
            all_embeddings.append(embedding_np)
            
            print(f"Processed frame {i+1}/{len(frame_files)}: {frame_filename}")

    final_embeddings_array = np.array(all_embeddings)
    
    print(f"\nAll frames processed. Final embeddings array shape: {final_embeddings_array.shape}")
    np.save(output_file, final_embeddings_array)
    print(f"Successfully saved visual embeddings to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visual embeddings for frames.")
    parser.add_argument('--model', type=str, required=True, choices=['vit', 'clip'], help="The type of model to use for embeddings.")
    args = parser.parse_args()

    FRAMES_DIR = "data/frames"
    # Generate a unique output file based on the model type
    OUTPUT_FILE = f"data/visual_embeddings_{args.model}.npy"
        
    create_embeddings(FRAMES_DIR, OUTPUT_FILE, args.model)