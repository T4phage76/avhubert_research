from common import *
import re
import subprocess

folder = "visual_benefit/AUDIO_all"
full_file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
clear_file_list = [f for f in full_file_list if not f.endswith("-12dB.wav")]
len(clear_file_list)

out_dir = "visual_benefit/AUDIO_all_16k"
os.makedirs(out_dir, exist_ok=True)

for file_path in clear_file_list:
    file = os.path.basename(file_path)
    out_file = os.path.join(f"visual_benefit/AUDIO_all_16k",file)
    subprocess.run([
        "ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1",out_file
    ])

new_folder = "visual_benefit/AUDIO_all_16k"
clear_file_list_16k = [os.path.join(new_folder, f) for f in os.listdir(new_folder)]


filename = 'Izzie2024'
stim_table = pd.read_csv(f"data/stim_table_{filename}.csv")

# Extract speaeker and word
stim_table['key'] = stim_table['stim_list'].apply(lambda x: re.sub(r'(_-?\d+dB.*|_v.*)$', '', x))

def make_key(path):
    fname = os.path.basename(path)         # e.g. "HOIST_JOHN_a.wav"
    base = fname.split('.')[0]             # remove .wav
    parts = base.split('_')[:2]            # take WORD + SPEAKER
    return "_".join(parts)

file_map = {make_key(f): f for f in clear_file_list_16k}
stim_table['audio_path'] = stim_table['key'].map(file_map)
stim_table = stim_table.drop(columns=['key'])

stim_table.to_csv(f"data/stim_table_{filename}.csv", index=False)

