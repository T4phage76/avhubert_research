import os
import shutil

# Path to your audio directory
audio_dir = 'LRS_processing/LRS2/LRS_pretrain_audio'  # where all the .wav are stored
txt_root_dir = 'LRS_processing/LRS2/mvlrs_v1/pretrain'

# MFA working directories
mfa_text_dir = 'LRS_processing/LRS2/LRS_pretrain_transcripts'
os.makedirs(mfa_text_dir, exist_ok=True)

for wav_file in os.listdir(audio_dir):
    if not wav_file.endswith('.wav'):
        continue
    
    full_wav_path = os.path.join(audio_dir, wav_file)
    
    folder_name, file_id = wav_file.replace('.wav', '').split('_')
    
    txt_path = os.path.join(txt_root_dir, folder_name, f'{file_id}.txt')
    
    if not os.path.exists(txt_path):
        print(f'Warning: Missing text file for {wav_file}')
        continue
    
    # Copy text with same base name
    shutil.copy(txt_path, os.path.join(mfa_text_dir, wav_file.replace('.wav', '.txt')))

print("Finished preparing data for MFA!")

