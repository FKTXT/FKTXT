import cv2

def save_frames(video_path, output_folder, name):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0

    # Read and save frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_filename = f"{output_folder}/{name}_frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")

    cap.release()
    print(f"All frames saved to {output_folder}")




if __name__ == "__main__":
    video_path = "W:\Code\FKTXT\FKTXT\datasets\images\testingvid.mp4"  # Replace with your video file path
    output_folder = "output_frames"  # Folder to save frames
    name="jimrecording"

    save_frames(video_path, output_folder, name)
