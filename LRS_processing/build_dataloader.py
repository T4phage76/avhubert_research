#### BUILD dataset classes
import os
import json
import torch
from torch.utils.data import Dataset
import torchaudio
import cv2
import numpy as np
import torchaudio
from torchaudio.transforms import MFCC
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class AVSRDataset(Dataset):
    def __init__(self, audio_dir, video_dir, label_json_path, phoneme_to_idx,sample_rate=16000, transform=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform
        self.phoneme_to_idx = phoneme_to_idx
        self.mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=26, melkwargs={"n_fft": 400, "hop_length": 640,"n_mels": 32}) # hop size: 640 so 640/16K = 40ms/data point, matching the 25 fps video
        
        # Load labels (somehow dataset has some unmatched audio/video/text transcripts, this way to get the common ones)
        with open(label_json_path, 'r') as f:
            self.labels = json.load(f)

        # Get all IDs from JSON
        label_ids = set(self.labels.keys())

        # Get all IDs from audio and video dirs
        audio_ids = set(os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.endswith('.wav'))
        video_ids = set(f.replace('_ROI.mp4', '') for f in os.listdir(video_dir) if f.endswith('_ROI.mp4'))

        # Keep only IDs that exist in all three
        common_ids = label_ids & audio_ids & video_ids

        # Use sorted list so order is consistent
        self.utt_ids = sorted(list(common_ids))


    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        audio_path = os.path.join(self.audio_dir, f"{utt_id}.wav")
        video_path = os.path.join(self.video_dir, f"{utt_id}_ROI.mp4")

        # Load video as grayscale frames
        video_tensor = self._load_video_gray(video_path)

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)  # sr = 16000 Hz
        mfcc = self.mfcc_transform(waveform).squeeze(0) # (26, T)

        # Check alignment (allow at most 1 frame diff due to rounding)
        assert abs(video_tensor.size(0) - mfcc.size(-1)) <= 2, \
            f"Mismatch too large: video={video_tensor.size(0)}, audio={mfcc.size(-1)}"
        
        # trim last frames to match the length (they may differ in the last frame due to rounding)
        T = min(video_tensor.size(0), mfcc.size(-1))
        video_tensor = video_tensor[:T]
        mfcc = mfcc[..., :T]    
        audio_tensor = mfcc.transpose(0, 1)  # (T, 26)
        # Get label list for the utterance
        label_list = self.labels[utt_id]['labels']

        # Convert phoneme labels to numeric indices
        label_indices = [self.phoneme_to_idx[p] for p in label_list]

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {
            'utt_id': utt_id,
            'audio': audio_tensor,      # [Frame, 26]
            'video': video_tensor,          # [Frame, H, W]
            'labels': torch.tensor(label_indices, dtype=torch.uint8)  # numeric for CTC
        }

    def _load_video_gray(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        video_tensor = torch.tensor(np.stack(frames), dtype=torch.float16)  # [F, H, W]
        return video_tensor



#### COLLATE function for dataloader to batch data

# check whether target length can be aliged to input length for CTC 
# (input length must => 2tgt_length -1 such taht each side of the target can accept a
# blank symbol: helo -> h_e_l_o, input size => 7)
def avsr_collate_fn(batch, ctc_filter=True, verbose=False):
    """
    Collate function for batching AVSRDataset samples with optional CTC validity filtering 
    (input length must => 2*tgt_length -1 such taht each side,except for the first and last frame,
    of the target can accept a blank symbol: helo -> h_e_l_o, input size => 7).
    
    Pads sequences to the longest in the batch (after filtering).

    Each item in batch is a dict:
    {
        'utt_id': str,
        'audio': Tensor (T_audio, n_mfcc)   
        'video': Tensor (T_video, H, W),
        'labels': list[int]  # phoneme ids
    }

    Returns dict or None (if everything is filtered).
    """
    # --- (1) Optional CTC filtering BEFORE padding ---
    if ctc_filter:
        keep_batch = []
        dropped = []
        for s in batch:
            input_len  = s["video"].shape[0]          # your current choice for CTC input length
            target_len = len(s["labels"])
            min_needed = 2 * target_len - 1
            if input_len >= min_needed:
                keep_batch.append(s)
            else:
                dropped.append({
                    "utt_id": s["utt_id"],
                    "input_len": int(input_len),
                    "target_len": int(target_len),
                    "min_needed": int(min_needed),
                })

        if verbose and dropped:
            print(f"[collate] Dropping {len(dropped)} invalid item(s):")
            for d in dropped[:10]:
                print(f"  utt_id={d['utt_id']} in={d['input_len']} "
                      f"tgt={d['target_len']} min_needed={d['min_needed']}")
            if len(dropped) > 10:
                print(f"  ... and {len(dropped)-10} more")

        batch = keep_batch

        # If nothing remains, signal caller to skip this batch
        if len(batch) == 0:
            return None

    # --- (2) Gather fields ---
    utt_ids = [s['utt_id'] for s in batch]
    audios  = [s['audio']  for s in batch]   # Expect (T_audio, 26)
    videos  = [s['video']  for s in batch]   # (T_video, 96, 96)
    labels_list = [torch.as_tensor(s['labels'], dtype=torch.uint8) for s in batch]

    # --- (3) Pad audio time-dimension ---
    padded_audios = pad_sequence(audios, batch_first=True) # padded_audios shape: (B, T_max, 26)

    # --- (4) Pad video to max T ---
    max_len = max([v.shape[0] for v in videos])
    padded_videos = torch.stack([
        F.pad(v, (0,0,0,0,0,max_len - v.shape[0])) for v in videos
    ])  # (B, T_video, 96, 96)

    # --- (5) Concatenate labels (CTC expects 1D concatenated targets) ---
    labels_concat = torch.cat(labels_list, dim=0)

    # --- (6) Length tensors for CTC (use video T as input length, to match your current setup) ---
    input_lengths  = torch.tensor([v.shape[0] for v in videos], dtype=torch.uint8)
    target_lengths = torch.tensor([len(l) for l in labels_list], dtype=torch.uint8)

    return {
        "utt_ids": utt_ids,
        "audio": padded_audios,          # (B, T_a_max, n_mfcc)
        "video": padded_videos,          # (B, T_v_max, H, W)
        "labels": labels_concat,         # (sum(target_lengths),)
        "input lengths": input_lengths,   # (B,)
        "target lengths": target_lengths  # (B,)
    }



# def avsr_collate_fn(batch):
#     """
#     Collate function for batching AVSRDataset samples.
#     Pads sequences to the longest in the batch.

#     Each item in batch is a dict:
#     {
#         'utt_id': str,
#         'audio': Tensor (1, T_audio),
#         'video': Tensor (T_video, H, W),
#         'labels': list of phoneme strings
#     }
#     """
#     utt_ids = [sample['utt_id'] for sample in batch]
#     audios = [sample['audio'] for sample in batch]   # (T, 26)
#     videos = [sample['video'] for sample in batch]   # (T, 96, 96)
#     labels = [torch.as_tensor(sample['labels'], dtype=torch.long) for sample in batch]

#     # Pad audio (time-major for pad_sequence)
#     padded_audios = pad_sequence(audios, batch_first=True)  # padded_audios shape: (B, T_max, 26)

#     # Pad video (T, H, W) -> # (B, T_video, H, W)
#     max_len = max([v.shape[0] for v in videos])
#     padded_videos = torch.stack([
#         torch.nn.functional.pad(v, (0,0,0,0,0,max_len - v.shape[0])) for v in videos
#     ])  # (B, T_video, 96, 96)

#     # Concatenate labels for CTC (needs a 1D tensor)
#     labels_concat = torch.cat(labels)

#     # Lengths for CTC
#     input_lengths = torch.tensor([video.shape[0] for video in videos], dtype=torch.long)
#     label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)


#     return {
#         "utt_ids": utt_ids,
#         "audio": padded_audios,       # (B, T_audio)
#         "video": padded_videos,       # (B, T_video, H, W)
#         "labels": labels_concat,      # (B, T_labels)
#         "input length": input_lengths,
#         "target length": label_lengths
#     }

















class AVSRDataset_framewiseCE(Dataset):
    def __init__(self, audio_dir, video_dir, label_json_path, transform=None):
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.transform = transform

        # Load labels
        with open(label_json_path, 'r') as f:
            self.labels = json.load(f)

        # List of utterance IDs (e.g., '5535415699068794046_00001')
        self.utt_ids = list(self.labels.keys())

    def __len__(self):
        return len(self.utt_ids)


    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        audio_path = os.path.join(self.audio_dir, f"{utt_id}.wav")
        video_path = os.path.join(self.video_dir, f"{utt_id}_ROI.mp4")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path) #sr = 16000 Hz

        # Load video as grayscale frames
        video_tensor = self._load_video_gray(video_path)

        # Get labels
        # Normalize the labels (turn ['B', 'IH'] into 'B+IH' such that no lists in list)
        raw_labels = self.labels[utt_id]['labels']
        frame_labels = [self._normalize_label(l) for l in raw_labels]

        # Apply transform if any (e.g., augmentation)
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return {
            'utt_id': utt_id,
            'audio': waveform.float(),  # shape: [1, T]
            'video': video_tensor,  # shape: [F, H, W]
            'labels': frame_labels  # list of list of CMU phonemes per frame
        }

    def _load_video_gray(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
        video_tensor = torch.tensor(np.stack(frames), dtype=torch.float32)  # shape: [F, H, W]
        return video_tensor

    def _normalize_label(self, label):
        return '+'.join(sorted(label)) if isinstance(label, list) else label



def avsr_collate_fn_framewiseCE(batch):
    """
    Collate function for batching AVSRDataset samples.
    Pads sequences to the longest in the batch.

    Each item in batch is a dict:
    {
        'utt_id': str,
        'audio': Tensor (1, T_audio),
        'video': Tensor (T_video, H, W),
        'labels': list of phoneme strings
    }
    """
    utt_ids = [sample['utt_id'] for sample in batch]
    audios = [sample['audio'].squeeze(0) for sample in batch]  # shape: (T,)
    videos = [sample['video'] for sample in batch]
    label_lists = [sample['labels'] for sample in batch]

    # Pad audio
    padded_audios = pad_sequence(audios, batch_first=True)  # (B, T_audio)

    # Pad video (T, H, W) -> # (B, T_video, H, W)
    max_len = max([v.shape[0] for v in videos])
    padded_videos = torch.stack([
        torch.nn.functional.pad(v, (0,0,0,0,0,max_len - v.shape[0])) for v in videos
    ])  # (B, T_video, H, W)

    # Convert labels to integer indices
    phoneme_vocab = sorted({p for labels in label_lists for p in labels})
    phoneme_to_idx = {p: i for i, p in enumerate(phoneme_vocab)}

    label_tensors = [
        torch.tensor([phoneme_to_idx[p] for p in labels]) for labels in label_lists
    ]

    padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
    
    return {
        "utt_ids": utt_ids,
        "audio": padded_audios,       # (B, T_audio)
        "video": padded_videos,       # (B, T_video, H, W)
        "labels": padded_labels,      # (B, T_labels)
        "phoneme_to_idx": phoneme_to_idx
    }










