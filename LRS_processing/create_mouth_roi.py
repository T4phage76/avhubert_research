from common import *
from tqdm import tqdm 

# Directory to save mouth ROI videos
video_dir = 'LRS_processing/LRS2/mvlrs_v1/main'
mouth_roi_out_dir = 'LRS_processing/LRS2/LRS_pretrain_mouth_roi'
os.makedirs(mouth_roi_out_dir, exist_ok=True)

# Function to create mouth ROI videos
def create_mouth_roi(video_path, mouth_roi_path):
    face_predictor_path = "data/misc/shape_predictor_68_face_landmarks.dat"
    mean_face_path = "data/misc/20words_mean_face.npy"
    preprocess_video(video_path, mouth_roi_path, face_predictor_path, mean_face_path)

# Collect all video file paths
video_files = []
failed_videos = []

for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):
            parent_folder_name = os.path.basename(root)
            video_name = file.split('.')[0]
            video_path = os.path.join(root, file)
            mouth_roi_path = os.path.join(mouth_roi_out_dir, f"{parent_folder_name}_{video_name}_ROI.mp4")
            video_files.append((video_path, mouth_roi_path, parent_folder_name, video_name))


for video_path, mouth_roi_path, parent_folder_name, video_name in tqdm(video_files, desc="Creating Mouth ROI"):
    try:
        create_mouth_roi(video_path, mouth_roi_path)
    except Exception as e:
        print(f"Skipping {parent_folder_name}_{video_name} due to error: {e}")
        failed_videos.append((video_path, str(e)))
        continue

print(f"\nTotal failed: {len(failed_videos)}")

