from common import *

# CHANGE THE PARAMETERS BEFORE RUNNING THIS CELL!!!
alpha = 0 # noise level
tgt_layers = 0 # top n layers to change weights
n_subjects = 1 # number of subjects for each stimulus
n_heads = 16 # number of attention heads per layer, 16 for AVHuBERT large
feature_dim = 1024 # dimension of feature vector of each of the 24 transformer layers
head_dim = 64 # dimension of value vector before linear layer (together is 64 * 16 = 1024)
n_layers = 24 # number of layers
mfccs = []
frames = []
n_frames = []
models = None
all_output_data = []

# set a local path to save the intermediate outputs (representations, attention wights, transformations) ###### CHANGE!!!!!!!!
local_path = "results"
h5filename_features = "results/May1_Anna_PhonemeGroup_feats.h5" ###### CHANGE!!!!!!!!
h5filename_attn_weights = "results/May1_Anna_PhonemeGroup_attn.h5" ###### CHANGE!!!!!!!!
h5filename_transformation = "results/May1_Anna_PhonemeGroup_trans.h5" ###### CHANGE!!!!!!!!

# Define Directories
# mouth_roi_path, ckpt_path, audio_path = [f"data/stimuli/{stim}_roi.mp4" for stim in stim_list], "./data/finetune-model.pt", stim_table['audio_path'].tolist()
user_dir = "./library"

# Import the the target model for feature extraction
for audio, video in zip(audio_path, mouth_roi_path):
    if models is None:
        mfcc, frame, models = extract_audio_visual_feature(video, audio, ckpt_path, user_dir, is_finetune_ckpt=True)
        #n_frame = mfcc.shape[2] # number of input frames
    else:    
        mfcc, frame, _ = extract_audio_visual_feature(video, audio, ckpt_path, user_dir, is_finetune_ckpt=True)
        #n_frame = mfcc.shape[2] # number of input frames
    mfccs.append(mfcc)
    frames.append(frame)
    #n_frames.append(n_frame) # save the length of each stimulus

n_frames=mfccs[0].shape[2]

# get representation, atten scores
model = models[0]
if hasattr(models[0], 'decoder'):
  print(f"Checkpoint: fine-tuned")
  model = models[0].encoder.w2v_model
else:
  print(f"Checkpoint: pre-trained w/o fine-tuning")
model.cpu().eval()

# tensors to save the output representations
# transformation_a, transformation_v, transformation_av = torch.empty((n_layers, n_heads, n_frames, head_dim)), torch.empty((n_layers, n_heads, n_frames, head_dim)), torch.empty((n_layers, n_heads, n_frames, head_dim))
# features_a, features_v, features_av = torch.empty((n_layers, n_frames, feature_dim)), torch.empty((n_layers, n_frames, feature_dim)), torch.empty((n_layers, n_frames, feature_dim))
# headwise_weights_a, headwise_weights_v, headwise_weights_av = torch.empty((n_layers,n_heads, n_frames, n_frames)), torch.empty((n_layers,n_heads, n_frames, n_frames)), torch.empty((n_layers,n_heads, n_frames, n_frames))

for mfcc, frame in zip(mfccs, frames):
    features_a, features_v, features_av, headwise_weights_a, headwise_weights_v, headwise_weights_av, transformation_a, transformation_v, transformation_av = extract_representations(mfcc, frame, n_subjects, alpha, tgt_layers, model,n_layers,n_heads, n_frames,head_dim, feature_dim)
    # Store everything with explicit copies
    output_entry = {
        "features_a": features_a.clone().detach(),  
        "features_v": features_v.clone().detach(),
        "features_av": features_av.clone().detach(),
        "headwise_weights_a": headwise_weights_a.clone().detach(),
        "headwise_weights_v": headwise_weights_v.clone().detach(),
        "headwise_weights_av": headwise_weights_av.clone().detach(),
        "transformation_a": transformation_a.clone().detach(),
        "transformation_v": transformation_v.clone().detach(),
        "transformation_av": transformation_av.clone().detach(),
    }
    all_output_data.append(output_entry)


with open("results/May1_Anna_PhonemeGroup_all_feats.pkl", "wb") as f:
    pickle.dump(all_output_data, f)










# Save the layer representations to HDF5
all_features = torch.cat([features_a, features_v, features_av], dim=0) # (3*24,n_frames,1024)
with h5py.File(h5filename_features, 'w') as h5f:
    h5f.create_dataset('all_features', data=all_features.numpy())

# Save attention weights to HDF5
all_attention_weights = torch.cat([headwise_weights_a, headwise_weights_v, headwise_weights_av], dim=0) # (3*24,16,n_frames,n_frames)
with h5py.File(h5filename_attn_weights, 'w') as h5f:
    h5f.create_dataset('all_attention_weights', data=all_attention_weights.numpy())

# Save head transformations to HDF5
all_transformation = torch.cat([transformation_a, transformation_v, transformation_av], dim=0) # (3*24,16,n_frames,64)
with h5py.File(h5filename_transformation, 'w') as h5f:
    h5f.create_dataset('all_transformation', data=all_transformation.numpy())