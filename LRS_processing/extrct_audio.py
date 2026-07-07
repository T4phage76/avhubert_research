from common import *
import os
import subprocess
from tqdm import tqdm  

# Define paths
video_dir = 'LRS_processing/LRS2/mvlrs_v1/main'
audio_out_dir = 'LRS_processing/LRS2/LRS_main_audio'  # Directory to save audio files

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_out_path):
    command = f"ffmpeg -i {video_path} -acodec pcm_s16le -ac 1 -ar 16000 {audio_out_path} -y"
    subprocess.run(command, shell=True, check=True)

# Collect all video files first
video_files = []

for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):
            parent_folder_name = os.path.basename(root)
            video_name = file.split('.')[0]
            video_path = os.path.join(root, file)
            audio_out_path = os.path.join(audio_out_dir, f"{parent_folder_name}_{video_name}.wav")
            video_files.append((video_path, audio_out_path, parent_folder_name, video_name))

# extract audio
for video_path, audio_out_path, parent_folder_name, video_name in tqdm(video_files, desc="Extracting audio"):
    extract_audio_from_video(video_path, audio_out_path)

