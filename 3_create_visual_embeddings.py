import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import timm  # PyTorch Image Models library

# --- Configuration ---
FRAMES_DIR = "data/frames"
OUTPUT_FILE = "data/visual_embeddings.npy"

# --- CORRECTED MODEL NAME ---
# This is a robust, well-tested Vision Transformer pre-trained on ImageNet-21k
# and fine-tuned on ImageNet-1k, which is a standard for high-quality features.
MODEL_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'

# --- Main Script ---
def create_visual_embeddings(frames_dir, output_file, model_name):
    """
    Generates deep feature embeddings for each frame using a pre-trained
    Vision Transformer (ViT) model and saves them to a single NumPy file.
    """
    # Device Check for Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")

    # Load the Pre-trained Vision Transformer
    print(f"Loading model: {model_name}...")
    # The 'pretrained=True' flag will now correctly find the weights for this model name.
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    model.eval()
    
    # Get the model's configuration to define the image transformations
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    embedding_dim = model.embed_dim
    print(f"Model loaded. Feature embedding dimension: {embedding_dim}")

    # Process Frames and Generate Embeddings
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    all_embeddings = []

    with torch.no_grad():
        for i, frame_filename in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_filename)
            
            img = Image.open(frame_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            embedding = model(input_tensor)
            embedding_np = embedding.squeeze().cpu().numpy()
            all_embeddings.append(embedding_np)
            
            print(f"Processed frame {i+1}/{len(frame_files)}: {frame_filename}")

    # Convert the list of arrays into a single 2D NumPy array
    final_embeddings_array = np.array(all_embeddings)
    
    # Save the Embeddings
    print(f"\nAll frames processed. Final embeddings array shape: {final_embeddings_array.shape}")
    np.save(output_file, final_embeddings_array)
    print(f"Successfully saved all visual embeddings to '{output_file}'")

# --- Run the script ---
if __name__ == "__main__":
    create_visual_embeddings(FRAMES_DIR, OUTPUT_FILE, MODEL_NAME)