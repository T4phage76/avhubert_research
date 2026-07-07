import os
from textgrid import TextGrid
import numpy as np
from tqdm import tqdm
import json
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# Settings
textgrid_dir = "LRS_processing/LRS2/MFA_output"
frame_rate = 25  # frames per second (adjust if different)
#frame_shift = 1.0 / frame_rate  # in seconds

def get_sentence_phonemes_transcripts(textgrid_path):

    tg = TextGrid()
    tg.read(textgrid_path)
    
    # Load TextGrid and tier
    phone_tier = None
    for tier in tg.tiers:
            if tier.name.lower() == 'phones':
                phone_tier = tier
                break
    # get all the phonemes in this clip
    clip_phone = []
    for interval in phone_tier.intervals:
                phone = interval.mark.strip()
                if phone == "":
                    phone = "sil"
                clip_phone.append(phone)

    return {
        "labels": clip_phone,                   
        "n_labels": len(clip_phone)               
    }


def assign_framewise_phonemes_by_overlap(textgrid_path, frame_rate=25):
    """
    Assign phoneme labels to each frame by checking overlap between phoneme intervals
    and frame intervals.

    Parameters:
    - textgrid_path: str, path to the TextGrid file
    - frame_rate: int, kept at 25 to match audio and video

    Returns:
    - frame_labels: list of lists (or strings), where each element corresponds to a frame
      and contains the phoneme(s) overlapping with that frame
    """
    # Load TextGrid and tier
    clip_label = {"duration":'', "labels":''}
    tg = TextGrid()
    tg.read(textgrid_path)

    # Calculate number of frames from total duration
    end_time = tg.maxTime
    num_frames = int(np.ceil(end_time * frame_rate))
    frame_duration = 1.0 / frame_rate

    # Precompute time intervals for each frame
    frame_times = [(i * frame_duration, (i + 1) * frame_duration) for i in range(num_frames)]

    # Initialize label container
    frame_labels = [[] for _ in range(num_frames)]

    phone_tier = None
    for tier in tg.tiers:
        if tier.name.lower() == 'phones':
            phone_tier = tier
            break
    if phone_tier is None:
        raise ValueError(f"No phones tier found in {textgrid_path}")
    
    # Assign phonemes to overlapping frames (METHOD1: Assign the frame both phone labels)
    # for interval in phone_tier.intervals:
    #     phone = interval.mark.strip()
    #     if phone == "":
    #         phone = "sil"
    #     for i, (frame_start, frame_end) in enumerate(frame_times):
    #         # Check if phoneme interval overlaps with this frame interval
    #         if interval.minTime < frame_end and interval.maxTime > frame_start:
    #             frame_labels[i].append(phone)
    
    # Assign phonemes to overlapping frames(METHOD2: Assign only the most dominant phoneme to each frame)
    for i, (frame_start, frame_end) in enumerate(frame_times):
        max_overlap = 0

        for interval in phone_tier.intervals:
            phone = interval.mark.strip()
            if phone == "":
                phone = "sil"
            # Calculate overlap between interval and frame
            if interval.minTime < frame_end and interval.maxTime > frame_start:
                overlap_start = max(interval.minTime, frame_start)
                overlap_end = min(interval.maxTime, frame_end)
                overlap_duration = max(0.0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    dominant_phone = phone

        frame_labels[i] = dominant_phone

    clip_label["duration"] = end_time
    clip_label["labels"] = frame_labels
    return clip_label



# Process all TextGrids
# get the phonemes for one clip (for CTC training, more biologically plausible)
all_labels = {}
textgrid_dir = 'LRS_processing/LRS2/MFA_output_main'
textgrid_files = [f for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

for fname in tqdm(textgrid_files, desc="Processing TextGrids"):
    utt_id = os.path.splitext(fname)[0]
    path = os.path.join(textgrid_dir, fname)
    labels = get_sentence_phonemes_transcripts(path)
    all_labels[utt_id] = labels

json_output_path_clip_phone = os.path.join("LRS_processing/LRS2/LRS_main_labelled_data", "main_labels_whole_clip.json")
with open(json_output_path_clip_phone, "w") as f:
    json.dump(all_labels, f, indent=2)
print(f"Labeled dataset saved to {json_output_path_clip_phone}")

# get the frame-aligned phonemes (for frame-wise CE loss)
all_labels = {}
textgrid_dir = 'LRS_processing/LRS2/MFA_output_main'
textgrid_files = [f for f in os.listdir(textgrid_dir) if f.endswith(".TextGrid")]

for fname in tqdm(textgrid_files, desc="Processing TextGrids"):
    utt_id = os.path.splitext(fname)[0]
    path = os.path.join(textgrid_dir, fname)
    labels = assign_framewise_phonemes_by_overlap(path, frame_rate)
    all_labels[utt_id] = labels

json_output_path = os.path.join("LRS_processing/LRS2/LRS_main_labelled_data", "main_framewise_labels(dom_phone).json")
with open(json_output_path, "w") as f:
    json.dump(all_labels, f, indent=2)
print(f"Labeled dataset saved to {json_output_path}")


pkl_output_path = os.path.join("LRS_processing/LRS2/LRS_main_labelled_data", "main_framewise_labels(dom_phone).pkl")
with open(pkl_output_path, "wb") as f:
    pickle.dump(all_labels, f)
print(f"Labeled dataset saved to {pkl_output_path}")





#### CONVERT IPA symbols to CMU symbols (to reduce symbols types, enhancing training)
IPA_TO_CMU = {
    'sil': 'SIL', 'spn': 'SPN','dʒ': 'JH', 'ej': 'EY',
    'aj': 'AY', 'aw': 'AW', 'ow': 'OW','tʃ': 'CH','ɔj': 'OY',
    'aː': 'AA', 'b': 'B', 'bʲ': 'B',
    'c': 'CH', 'cʰ': 'CH', 'cʷ': 'CH', 'd': 'D', 
    'dʲ': 'D', 'd̪': 'D', 'f': 'F', 'fʲ': 'F',
    'h': 'HH', 'i': 'IY', 'iː': 'IY', 'j': 'Y', 'k': 'K',
    'kʰ': 'K', 'kʷ': 'K', 'l': 'L', 'm': 'M', 'mʲ': 'M',
    'm̩': 'M', 'n': 'N', 'n̩': 'N',  'p': 'P',
    'pʰ': 'P', 'pʲ': 'P', 's': 'S', 
    't': 'T', 'tʰ': 'T', 'tʲ': 'T', 'tʷ': 'T',
    't̪': 'T', 'v': 'V', 'vʲ': 'V', 'w': 'W', 'z': 'Z',
    'æ': 'AE', 'ç': 'SH', 'ð': 'DH', 'ŋ': 'NG', 'ɐ': 'AH',
    'ɑ': 'AA', 'ɑː': 'AA', 'ɒ': 'AO', 'ɒː': 'AO', 
    'ə': 'AH', 'ɚ': 'ER', 'ɛ': 'EH', 'ɝ': 'ER', 'ɟ': 'JH',
    'ɟʷ': 'JH', 'ɡ': 'G', 'ɡʷ': 'G', 'ɪ': 'IH', 'ɫ': 'L',
    'ɫ̩': 'L', 'ɱ': 'M', 'ɲ': 'N', 'ɹ': 'R', 'ɾ': 'DX',
    'ɾʲ': 'DX', 'ɾ̃': 'DX', 'ʃ': 'SH', 'ʉ': 'UW', 'ʉː': 'UW',
    'ʊ': 'UH', 'ʎ': 'L', 'ʒ': 'ZH', 'ʔ': 'SIL', 'θ': 'TH'
}


## Function to convert multiple ipa in one frame (labeling method 1) 
def convert_ipa_to_cmu_labels_1(input_json_path, output_json_path, ipa_to_cmu):
    """
    Convert all IPA symbols in the framewise labels to CMU format using a mapping dictionary.

    Parameters:
    - input_json_path: path to the input JSON file with IPA labels
    - output_json_path: path to save the converted JSON file
    - ipa_to_cmu: dictionary mapping IPA symbols to CMU symbols
    """
    with open(input_json_path, "r") as f:
        data = json.load(f)

    updated_data = {}

    for utt_id, info in data.items():
        updated_labels = []
        for frame in info["labels"]:
            cmu_frame = []
            extended_frame = [] # for debug: whether there's any IPA mapped to multiple CMU
            unk_frame = [] # for debug: whether there's any IPA failed mapping to any CMU
            for ipa_symbol in frame:
                if ipa_symbol in ipa_to_cmu:
                    mapped = ipa_to_cmu[ipa_symbol]
                    # If it's a list (e.g., multiple CMU phones), extend, else append
                    if isinstance(mapped, list):
                        extended_frame.extend(mapped)
                    else:
                        cmu_frame.append(mapped)
                else:
                    unk_frame.append(ipa_symbol)  # Keep unknown symbol as is (or handle separately)
            updated_labels.append(cmu_frame)

        updated_data[utt_id] = {
            "duration": info["duration"],
            "labels": updated_labels
        }

    with open(output_json_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Saved CMU-labeled data to {output_json_path}")

    return unk_frame, extended_frame
## Function to convert one ipa in one frame (labeling method 2) 
def convert_ipa_to_cmu_labels_2(input_json_path, output_json_path, ipa_to_cmu):
    """
    Convert all IPA symbols in the framewise labels to CMU format using a mapping dictionary.

    Parameters:
    - input_json_path: path to the input JSON file with IPA labels
    - output_json_path: path to save the converted JSON file
    - ipa_to_cmu: dictionary mapping IPA symbols to CMU symbols
    """
    with open(input_json_path, "r") as f:
        data = json.load(f)

    updated_data = {}

    for utt_id, info in data.items():
        updated_labels = []
        for frame in info["labels"]:
            cmu_frame = []
            extended_frame = [] # for debug: whether there's any IPA mapped to multiple CMU
            unk_frame = [] # for debug: whether there's any IPA failed mapping to any CMU
            if frame in ipa_to_cmu:
                mapped = ipa_to_cmu[frame]
                # If it's a list (e.g., multiple CMU phones), extend, else append
                if isinstance(mapped, list):
                    extended_frame.extend(mapped)
                else:
                    cmu_frame.append(mapped)
            else:
                unk_frame.append(frame)  # Keep unknown symbol as is (or handle separately)
            updated_labels.append(cmu_frame)

        updated_data[utt_id] = {
            "duration": info["duration"],
            "labels": updated_labels
        }

    with open(output_json_path, "w") as f:
        json.dump(updated_data, f, indent=2)

    print(f"Saved CMU-labeled data to {output_json_path}")

    return unk_frame, extended_frame

## Function to convert multiple ipa for whole-clip transformation
def convert_ipa_to_cmu_labels_whole_clip(input_json_path, output_json_path, ipa_to_cmu):
    """
    Convert all IPA symbols in the whole-clip phoneme list to CMU format using a mapping dictionary.

    Parameters:
    - input_json_path: path to the input JSON file with IPA labels
    - output_json_path: path to save the converted JSON file
    - ipa_to_cmu: dictionary mapping IPA symbols to CMU symbols
    """
    with open(input_json_path, "r") as f:
        data = json.load(f)

    updated_data = {}
    unk_frame = [] # for debug: whether there's any IPA failed mapping to any CMU

    for utt_id, info in data.items():
        cmu_frame = []
        for ipa_symbol in info["labels"]:
            if ipa_symbol in ipa_to_cmu:
                mapped = ipa_to_cmu[ipa_symbol]
                cmu_frame.append(mapped)
            else:
                unk_frame.append(ipa_symbol)  # Keep unknown symbol as is (or handle separately)

        updated_data[utt_id] = {
            "n_labels": info["n_labels"],
            "labels": cmu_frame
        }

    with open(output_json_path, "w") as f:
        json.dump(updated_data, f, indent=2)
    print(f"Saved CMU-labeled data to {output_json_path}")

    return unk_frame
# convert the IPA labels to CMU labels and inspect no unmapped or one-mapped-to-many symbols were generated
ipa_tocmu_function = convert_ipa_to_cmu_labels_whole_clip

unk_test =ipa_tocmu_function(
    input_json_path="LRS_processing/LRS2/LRS_main_labelled_data/main_labels_whole_clip.json",
    output_json_path="LRS_processing/LRS2/LRS_main_labelled_data/main_labels_whole_clip_cmu.json",
    ipa_to_cmu=IPA_TO_CMU
)    

## Create fixed index mapping for each CMU phoneme
label_json_path="LRS_processing/LRS2/LRS_main_labelled_data/main_labels_whole_clip_cmu.json"
with open(label_json_path, 'r') as f:
    all_labels = json.load(f)
phoneme_vocab = sorted({p for sample in all_labels.values() for p in sample['labels']})
phoneme_to_idx = {p: i for i, p in enumerate(phoneme_vocab)}

# add blank for training
new_phonemes = ['<BLANK>'] + list(phoneme_to_idx.keys())
phoneme_to_idx = {p: i for i, p in enumerate(new_phonemes)}

phoneme_to_idx_path = os.path.join("LRS_processing/LRS2/LRS_main_labelled_data", "cmu_phoneme_to_idx.json")
with open(phoneme_to_idx_path, "w") as f:
    json.dump(phoneme_to_idx, f, indent=2)




### plot a histogram to inspect how the training data is unbalanced towards certain phonemes (or maybe not)
def count_cmu_phoneme_distribution(json_path):
    # Load CMU-labeled data
    with open(json_path, "r") as f:
        data = json.load(f)

    phoneme_counter = Counter()

    for utt_info in data.values():
        for frame in utt_info["labels"]:
            for phoneme in frame:
                phoneme_counter[phoneme] += 1

    return phoneme_counter

def count_cmu_phoneme_distribution_whole_clip(json_path):
    # Load CMU-labeled data
    with open(json_path, "r") as f:
        data = json.load(f)

    phoneme_counter = Counter()

    for utt_info in data.values():
        for phoneme in utt_info["labels"]:
            phoneme_counter[phoneme] += 1

    return phoneme_counter


def plot_phoneme_histogram(counter, top_n=None, title="CMU Phoneme Distribution", save_to='LRS_processing/LRS2/LRS_main_labelled_data/phoneme_counts_CMU_whole_clip_Aug13.jpg'):
    # Sort phonemes by frequency
    phonemes, counts = zip(*counter.most_common(top_n))

    plt.figure(figsize=(12, 6))
    plt.bar(phonemes, counts, color='skyblue')
    plt.xticks()
    plt.title(title)
    plt.xlabel("CMU Phonemes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_to)

cmu_transcripts_json_path = "LRS_processing/LRS2/LRS_main_labelled_data/main_labels_whole_clip_cmu.json"
phoneme_counts = count_cmu_phoneme_distribution_whole_clip(cmu_transcripts_json_path)
plot_phoneme_histogram(phoneme_counts)























#### INSPECT
with open(json_output_path, "r") as f:
    data = json.load(f)
# Print total number of utterances
print(f"Total utterances: {len(data)}")

# Print the first few entries (change 5 to however many you want)
for i, (utt_id, info) in enumerate(data.items()):
    print(f"Utterance ID: {utt_id}")
    print(f"  Duration: {info['duration']}")
    print(f"  Number of frames: {len(info['labels'])}")
    print(f"  First 5 frame labels: {info['labels'][:5]}")
    print()
    if i >= 4:
        break  

# collect unique IPA symbols
unique_ipa = set()

for utt in data.values():
    frame_labels = utt["labels"]
    for frame in frame_labels:
        for phone in frame:
            unique_ipa.add(phone)

# Display sorted IPA symbols
unique_ipa_list = sorted(unique_ipa)
print(f"Total unique IPA symbols: {len(unique_ipa_list)}")
print(unique_ipa_list)



