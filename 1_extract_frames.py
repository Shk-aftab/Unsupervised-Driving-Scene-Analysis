import cv2
import os

# --- Configuration ---
VIDEO_PATH = "Driving1080p.mp4"  # <--- IMPORTANT: Change this to the path of your downloaded video
OUTPUT_DIR = "data/frames"    # Directory where the frames will be saved
FRAMES_PER_SECOND = 1         # The rate at which to extract frames

# --- Main Script ---
def extract_frames(video_path, output_dir, fps_rate):
    """
    Extracts frames from a video file and saves them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        fps_rate (int): The number of frames to extract per second of video.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get the video's original frames per second (FPS)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_rate) # Calculate how many frames to skip
    
    print(f"Video Info: {video_path}")
    print(f"Original FPS: {video_fps:.2f}, Extracting at {fps_rate} frame(s) per second.")
    print(f"Saving one frame every {frame_interval} frames.")

    frame_count = 0
    saved_frame_count = 0
    
    while True:
        # Read one frame from the video
        ret, frame = cap.read()

        # If 'ret' is False, we've reached the end of the video
        if not ret:
            break

        # Check if the current frame is one we should save
        if frame_count % frame_interval == 0:
            # Generate a clean, sortable filename (e.g., frame_00001.jpg)
            filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
            
            # Save the frame as a JPEG image
            cv2.imwrite(filename, frame)
            
            print(f"Saved {filename}")
            saved_frame_count += 1
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print("\nExtraction complete.")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames saved: {saved_frame_count} in '{output_dir}'")


# --- Run the script ---
if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR, FRAMES_PER_SECOND)