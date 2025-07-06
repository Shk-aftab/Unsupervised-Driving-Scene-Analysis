import os
import json
import random
import cv2
from ultralytics import YOLO
import torch

# --- Configuration ---
FRAMES_DIR = "data/frames"                      # Input directory from Step 1
PERCEPTION_OUTPUT_DIR = "data/perception_output"  # Output for structured JSON data
VALIDATION_DIR = "data/validation_images"       # Output for visual validation images
NUM_VALIDATION_IMAGES = 5                       # How many random images to save with annotations
MODEL_NAME = "yolo11n-seg.pt"                   # Using the robust and official YOLO11 nano segmentation model

# --- Main Script ---
def run_perception_and_validate(frames_dir, perception_dir, validation_dir, model_name):
    """
    Runs Yolo11n detection and segmentation, saves structured data, and
    creates visual validation images for a random subset of frames.
    """
    # Create output directories if they don't exist
    os.makedirs(perception_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # --- Device Check for Apple Silicon (MPS) ---
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device.upper()}")

    # --- Model Loading ---
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)
    print("Model loaded.")

    # --- Select Random Frames for Visual Validation ---
    all_frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    # Ensure we don't try to select more images than we have
    num_to_sample = min(NUM_VALIDATION_IMAGES, len(all_frame_files))
    random_frames_to_visualize = random.sample(all_frame_files, num_to_sample)
    print(f"\nWill save annotated versions of these {num_to_sample} random frames: {random_frames_to_visualize}\n")

    # --- Process Each Frame ---
    for frame_filename in all_frame_files:
        frame_path = os.path.join(frames_dir, frame_filename)
        print(f"Processing {frame_filename}...")

        # Run the model on the current frame
        results = model(frame_path, device=device)
        result = results[0] # We process one image, so we get the first result object

        # --- Visual Validation Step ---
        if frame_filename in random_frames_to_visualize:
            annotated_frame = result.plot()
            validation_path = os.path.join(validation_dir, f"annotated_{frame_filename}")
            cv2.imwrite(validation_path, annotated_frame)
            print(f"  -> Saved validation image to {validation_path}")

        # --- Data Extraction Step (for every frame) ---
        frame_data = []
        num_detections = len(result.boxes)
        has_masks = result.masks is not None

        for i in range(num_detections):
            box = result.boxes[i]
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)
            bounding_box_normalized = box.xywhn.tolist()[0]
            
            segmentation_polygon = []
            if has_masks and result.masks.xy and i < len(result.masks.xy):
                polygon_points_normalized = result.masks.xy[i].tolist()
                segmentation_polygon = polygon_points_normalized

            object_dict = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box_normalized_xywh": bounding_box_normalized,
                "segmentation_normalized_xy": segmentation_polygon
            }
            frame_data.append(object_dict)

        # Save all object data for the current frame to a JSON file
        output_filename = os.path.splitext(frame_filename)[0] + ".json"
        output_path = os.path.join(perception_dir, output_filename)
        with open(output_path, 'w') as f:
            json.dump(frame_data, f, indent=4)
    
    print("\nPerception simulation complete.")
    print(f"Saved all perception data in '{perception_dir}'")
    print(f"Saved {num_to_sample} validation images in '{validation_dir}'")


# --- Run the script ---
if __name__ == "__main__":
    run_perception_and_validate(
        FRAMES_DIR, 
        PERCEPTION_OUTPUT_DIR, 
        VALIDATION_DIR, 
        MODEL_NAME
    )