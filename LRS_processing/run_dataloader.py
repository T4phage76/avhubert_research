from LRS_processing.build_dataloader import AVSRDataset,avsr_collate_fn
from torch.utils.data import DataLoader

dataset = AVSRDataset(
    audio_dir="LRS_processing/LRS2/LRS_main_audio",
    video_dir="LRS_processing/LRS2/LRS_main_mouth_roi",
    label_json_path="LRS_processing/LRS2/LRS_main_labelled_data/main_framewise_labels_cmu.json"
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,  # or 0 ?
    collate_fn=avsr_collate_fn
)


# INSPECT
for batch in dataloader:
    print(batch.keys())
    print(batch['video'].shape)
    print(batch['audio'].shape)
    #print(batch['labels']) 
    break  # just one batch to test




















import os
import json


audio_dir = "LRS_processing/LRS2/LRS_main_audio"
video_dir = "LRS_processing/LRS2/LRS_main_mouth_roi" 
label_json_path = "LRS_processing/LRS2/LRS_main_labelled_data/main_labels_whole_clip_cmu.json"

# Extract stimulus IDs
audio_ids = set(os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.endswith('.wav'))

# Remove both "_ROI" and ".mp4" from video filenames
video_ids = set(f.replace('_ROI.mp4', '') for f in os.listdir(video_dir) if f.endswith('_ROI.mp4'))

# Load label JSON keys
with open(label_json_path, 'r') as f:
    labels = json.load(f)
label_ids = set(labels.keys())

# Compare sets
common_ids = audio_ids & video_ids & label_ids
only_in_audio = audio_ids - (video_ids | label_ids)
only_in_video = video_ids - (audio_ids | label_ids)
only_in_labels = label_ids - (audio_ids | video_ids)

# Output summary
print(f"✅ Common stimulus count: {len(common_ids)}")
print(f"❌ Missing in video/labels (but in audio): {len(only_in_audio)}")
print(f"❌ Missing in audio/labels (but in video): {len(only_in_video)}")
print(f"❌ Missing in audio/video (but in labels): {len(only_in_labels)}")

# Show some examples
print("\nExamples missing in video/labels (audio only):", list(only_in_audio)[:5])
print("Examples missing in audio/labels (video only):", list(only_in_video)[:5])
print("Examples missing in audio/video (labels only):", list(only_in_labels)[:5])
