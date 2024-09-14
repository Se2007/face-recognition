
import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    for _ in tqdm(range(total_frames), desc="Extracting frames"):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"{video_path.split('/')[-1].split('.')[0]}_{frame_count:04d}.jpg")
        
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")



video_path = './vid/sadra.mp4'
output_folder = './dataset/0'
extract_frames(video_path, output_folder)
