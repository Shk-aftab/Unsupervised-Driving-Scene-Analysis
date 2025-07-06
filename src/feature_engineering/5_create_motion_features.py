import os
import cv2
import numpy as np
import pandas as pd

# --- Configuration ---
FRAMES_DIR = "data/frames"              # Input directory of frames from Step 1
OUTPUT_CSV = "data/motion_features.csv"   # Output file for our engineered features

# --- Main Script ---
def create_motion_features(frames_dir, output_csv):
    """
    Calculates optical flow between consecutive frames to generate motion features.
    """
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    # We need at least two frames to calculate flow
    if len(frame_files) < 2:
        print("Error: Need at least 2 frames to calculate motion features.")
        return

    all_motion_features = []

    # Load the first frame and convert to grayscale
    prev_frame_path = os.path.join(frames_dir, frame_files[0])
    prev_gray = cv2.cvtColor(cv2.imread(prev_frame_path), cv2.COLOR_BGR2GRAY)

    # The first frame has no preceding frame, so its motion features are zero.
    first_frame_features = {
        'frame_id': os.path.splitext(frame_files[0])[0],
        'avg_flow_magnitude': 0.0,
        'avg_flow_direction': 0.0,
        'flow_magnitude_std_dev': 0.0,
        'flow_angle_std_dev': 0.0,
    }
    all_motion_features.append(first_frame_features)
    print(f"Processed frame 1/{len(frame_files)}: {frame_files[0]} (motion is 0)")


    # Iterate from the second frame to the end
    for i in range(1, len(frame_files)):
        frame_filename = frame_files[i]
        frame_path = os.path.join(frames_dir, frame_filename)
        
        # Load the current frame and convert to grayscale
        current_frame = cv2.imread(frame_path)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # --- Calculate Optical Flow ---
        # cv2.calcOpticalFlowFarneback returns a 2-channel flow vector field,
        # where each element (dx, dy) is the displacement of that pixel.
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=current_gray,
            flow=None,
            pyr_scale=0.5,  # Pyramid scale < 1 to build pyramids for robustness
            levels=3,       # Number of pyramid levels
            winsize=15,     # Averaging window size
            iterations=3,   # Iterations at each pyramid level
            poly_n=5,       # Size of the pixel neighborhood
            poly_sigma=1.2, # Standard deviation for Gaussian
            flags=0
        )

        # --- Calculate Magnitude and Angle ---
        # Convert the (dx, dy) flow vectors to polar coordinates (magnitude, angle)
        # Magnitude represents the speed of pixel movement.
        # Angle represents the direction of pixel movement.
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # --- Engineer Features from Flow ---
        avg_flow_magnitude = np.mean(magnitude)
        avg_flow_direction = np.mean(angle)
        flow_magnitude_std_dev = np.std(magnitude)
        flow_angle_std_dev = np.std(angle) # A new useful feature!

        motion_features = {
            'frame_id': os.path.splitext(frame_filename)[0],
            'avg_flow_magnitude': avg_flow_magnitude,
            'avg_flow_direction': avg_flow_direction,
            'flow_magnitude_std_dev': flow_magnitude_std_dev,
            'flow_angle_std_dev': flow_angle_std_dev,
        }
        all_motion_features.append(motion_features)
        
        print(f"Processed frame {i+1}/{len(frame_files)}: {frame_filename}")

        # Update the previous frame for the next iteration
        prev_gray = current_gray

    # --- Save to CSV ---
    features_df = pd.DataFrame(all_motion_features)
    features_df.to_csv(output_csv, index=False)
    
    print(f"\nSuccessfully created and saved motion features to '{output_csv}'")
    print("\nDataFrame head:")
    print(features_df.head())


# --- Run the script ---
if __name__ == "__main__":
    create_motion_features(FRAMES_DIR, OUTPUT_CSV)


'''
Iterative Process: The script works by always keeping the previous frame in memory (prev_gray) to compare against the current frame.
Handling the First Frame: The very first frame has no preceding frame, so its motion is undefined. We handle this edge case by explicitly setting its motion features to zero.
cv2.calcOpticalFlowFarneback: This is the core function. The parameters are standard defaults that provide a good balance between speed and accuracy. It calculates a displacement vector for every pixel in the image.
cv2.cartToPolar: This is a crucial helper function. The raw flow is in (dx, dy) cartesian coordinates, which is hard to interpret. Converting to polar coordinates gives us magnitude (speed) and angle (direction), which are much more intuitive.
New Feature: flow_angle_std_dev: I've added this feature because it's highly informative.
Low flow_angle_std_dev: When driving straight ahead, most pixels will flow outwards from a central point. Their directions will be varied but organized.
High flow_angle_std_dev: When turning, a large portion of the pixels will move horizontally in the same direction, leading to a low standard deviation in angle. When you are stationary and another car passes you, the flow vectors will be chaotic, leading to a high standard deviation. This feature helps distinguish between ego-motion (turning) and complex scene motion.

'''