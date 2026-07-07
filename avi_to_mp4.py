import os
import subprocess

input_folder = "/Users/haotian/PennNeurosurgery Dropbox/Beauchamp Laboratory/BeauchampLabAtPenn/Haotian/Paper_draft_Dec9/stimuli/McGurkStimuli" 
for file in os.listdir(input_folder):
    if file.endswith(".avi"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(input_folder, os.path.splitext(file)[0] + ".mp4")

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c:v", "libx264",  # Use H.264 codec for compatibility
            "-c:a", "aac",      # Audio codec
            "-strict", "experimental",  # Allows use of aac
            output_path
        ]

        print(f"Converting: {file}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("Conversion complete.")