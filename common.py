import torch
import torch.nn as nn

# download stimuli zip
import zipfile
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import csv
import shutil
import urllib.request
from base64 import b64encode
from IPython.display import HTML, display

# mouth ROI preparation
import dlib, cv2, os
import numpy as np
np.float = np.float64
np.int = np.int_
import pandas as pd
import skvideo
import skvideo.io
from tqdm import tqdm
import librosa

os.chdir("library")
from avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
from avhubert.utils import Compose, Normalize, CenterCrop, load_video
os.chdir("../")
# from preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import fairseq

# Inference
import math
import tempfile
from argparse import Namespace
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.configs import GenerationConfig

# Save the layer latent representations
import h5py
import pickle



face_predictor_path = "data/misc/shape_predictor_68_face_landmarks.dat"
mean_face_path = "data/misc/20words_mean_face.npy"

# download zip containing stimulus video (.mp4), unzip, extract audio
def create_stim_from_zip(url, unzip_dir, filename):
    # Unzip directory
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)
    # download file (CHANGE!)
    zip_file_path = os.path.join(unzip_dir, f'{filename}.zip')
    urllib.request.urlretrieve(url, zip_file_path)
    # Unzipping the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    # convert files into .mp4 if necessary
    files_to_convert= []
    files_to_convert = [os.path.join(unzip_dir, file) for file in os.listdir(unzip_dir) if file.endswith('.mov')]
    for file in files_to_convert:
        converted_name = os.path.splitext(file)[0] + ".mp4"
        os.system(f'ffmpeg -i {file} -vcodec copy -acodec copy {converted_name}')
    # create a list containing stimulus paths
    stimulus_path_list = []
    stimulus_path_list = [os.path.join(unzip_dir, file) for file in os.listdir(unzip_dir) if file.endswith('.mp4')]
    length = len(stimulus_path_list)
    # create a list containing stimulus names
    stim_list = []
    for stimulus_path in stimulus_path_list:
        _, tail = os.path.split(stimulus_path)
        stim_name = tail.split('.')[0]
        stim_list.append(f'{stim_name}')
    # extract audio 
    audio_path = []
    for stim, name in zip(stimulus_path_list, stim_list): 
        os.system(f'ffmpeg -i {stim} -acodec pcm_s16le -ac 1 -ar 16000 {unzip_dir}/{name}.wav -y') # sr = 16kHz
        audio_path.append(f'{unzip_dir}/{name}.wav')
    print(f'There are {length} stimuli:')
    print(*stimulus_path_list, sep = '\n')
    print('THE STIMULI USED ARE:', *stim_list, sep='\n')
    print('THE EXTRACTED AUDIO FILES ARE SAVED TO:', *audio_path, sep = '\n')
    stim_table = pd.DataFrame({
        "audio_path" : audio_path,
        "stim_list"  : stim_list,
        "stimulus_path_list" : stimulus_path_list
    })
    stim_tables = [stim_table]
    if os.path.exists(f"data/stim_table_{filename}.csv"):
        old_stim_table = pd.read_csv(f"data/stim_table_{filename}.csv")
        stim_tables.append(old_stim_table)
    stim_table = pd.concat(stim_tables)
    stim_table.to_csv(f"data/stim_table_{filename}.csv", index=False)
    audio_path = stim_table['audio_path'].tolist()
    stim_list = stim_table['stim_list'].tolist()
    stimulus_path_list = stim_table['stimulus_path_list'].tolist()
    return audio_path, stim_list, stimulus_path_list

def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path, ffmpeg = shutil.which("ffmpeg")):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    STD_SIZE = (256, 256)
    mean_face_landmarks = np.load(mean_face_path)
    stablePntsIDs = [33, 36, 39, 42, 45]
    videogen = skvideo.io.vread(input_video_path)
    frames = np.array([frame for frame in videogen])
    landmarks = []
    for frame in tqdm(frames):
        landmark = detect_landmark(frame, detector, predictor)
        landmarks.append(landmark)
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE,
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, ffmpeg)
    pass

def ensure_mouth_roi(stimulus_path_list, mouth_roi_path):
    '''
    crop out the mouth roi
    '''
    for origin_clip, mouth_roi in zip(stimulus_path_list, mouth_roi_path): 
        if not os.path.exists(mouth_roi):
            print(f'\nProcessing {origin_clip}')
            # Cropping the mouth ROI
            preprocess_video(origin_clip, mouth_roi, face_predictor_path, mean_face_path)
            print(f'Saving to {mouth_roi}')
            # Inspect the frame rate
            video_metadata = skvideo.io.ffprobe(mouth_roi)
            frame_rate = eval(video_metadata['video']['@avg_frame_rate'])
            print(f"Frame Rate of {mouth_roi}:", frame_rate)
            # Display video
            # display(play_video(mouth_roi))


def play_audio(audio_path):
    with open(audio_path, 'rb') as audio_file:
        audio_content = audio_file.read()
    audio_data_url = "data:audio/wav;base64," + b64encode(audio_content).decode()
    return HTML(f"""
    <audio controls>
        <source src="{audio_data_url}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """)

# # Inspect the audio files
# for audio in audio_path:
#     print(f'Play {audio}')
#     display(play_audio(audio))


# Functions used in mouth ROI cropping
def play_video(video_path, width=200):
    with open(video_path, 'rb') as video_file:
        mp4 = video_file.read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width="{width}" controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    """)



def to_gpu(x):
    backend_device = torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     backend_device = torch.device("mps")
    return x.to(backend_device)



def permute_weights(model):
    """
    Permutes (scrambles) the weights of Linear and LayerNorm layers in the given model.
    """
    for name, module in model.named_modules():
        # Check if the module is a linear layer (e.g., k_proj, v_proj, q_proj, fc1, fc2)
        if isinstance(module, nn.Linear):
            # Scramble the weight matrix
            permuted_weight = module.weight[torch.randperm(module.weight.size(0)), :]  # Permute rows
            module.weight.data = permuted_weight
            # If there is a bias, also permute the bias vector
            if module.bias is not None:
                permuted_bias = module.bias[torch.randperm(module.bias.size(0))]
                module.bias.data = permuted_bias
        # Check if the module is a LayerNorm layer
        elif isinstance(module, nn.LayerNorm):
            permuted_weight = module.weight[torch.randperm(module.weight.size(0))]
            module.weight.data = permuted_weight
            if module.bias is not None:
                permuted_bias = module.bias[torch.randperm(module.bias.size(0))]
                module.bias.data = permuted_bias
    print("Weights have been permuted successfully!")


def predict(video_path, audio_path, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer=False, num_frames=20, beam_size=20):
    modalities = [
        None if video_path is None else "video",
        None if audio_path is None else "audio"
    ]
    # num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    video_path_str = video_path if modalities.count("video") > 0 else None
    audio_path_str = audio_path if modalities.count("audio") > 0 else None
    # / + data/stimuli/RL_AbbVff_cropped_roi.mp4
    tsv_cont = ["./\n", f"test-0\t{video_path_str}\t{audio_path_str}\t{num_frames}\t{int(16_000*num_frames/25)}\n"]
    label_cont = ["DUMMY\n"]
    data_dir = tempfile.mkdtemp()
    try:
        with open(f"{data_dir}/test.tsv", "w") as fo:
            fo.write("".join(tsv_cont))
        with open(f"{data_dir}/test.wrd", "w") as fo:
            fo.write("".join(label_cont))
        utils.import_user_module(Namespace(user_dir=user_dir))
        gen_subset = "test"
        gen_cfg = GenerationConfig(beam=beam_size, lenpen = 1, unnormalized=False)
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        models = [to_gpu(model.eval()) for model in models]
        # GENERATE RANDOM GAUSSIAN NOISES FOR EACH OF THE Proj_out LAYERS
        noise_shape = models[0].encoder.w2v_model.encoder.layers[0].self_attn.out_proj.weight.data.shape
        # GENERATE RANDOM GAUSSIAN NOISES FOR EACH OF THE fc1 layers
        #noise_shape = models[0].encoder.w2v_model.encoder.layers[0].fc1.weight.data.shape
        if permute_transformer == True:   # Permute the model to see the baseline performance
            permute_weights(models[0].encoder.w2v_model.encoder)
        else:
            for layer in range(tgt_layers):
                # Set a unique seed for each subject-layer combination
                seed = subject_id * 1000 + layer
                torch.manual_seed(seed)
                gaus_noise = to_gpu(torch.randn(noise_shape)) * alpha  # Generate Gaussian noise on GPU mean=0, var=alpha
                # change the out-proj layer
                models[0].encoder.w2v_model.encoder.layers[layer].self_attn.out_proj.weight.data += gaus_noise # out_proj layer
                # change the fc1 layer
                #models[0].encoder.w2v_model.encoder.layers[layer].fc1.weight.data += gaus_noise # fc1 layer
        saved_cfg.task.modalities = modalities
        saved_cfg.task.data = data_dir
        saved_cfg.task.label_dir = data_dir
        saved_cfg.task.max_sample_size = 600  # update max sample size here if the video is too long (default = 500, max =2000, min=5 for pre-training)
        task = tasks.setup_task(saved_cfg.task)
        task.cfg.noise_wav = None
        task.load_dataset(gen_subset)
        generator = task.build_generator(models, gen_cfg)
        def decode_fn(x):
            dictionary = task.target_dictionary
            symbols_ignore = generator.symbols_to_strip_from_output
            symbols_ignore.add(dictionary.pad())
            return task.datasets[gen_subset].label_processors[0].decode(x, symbols_ignore)
        itr = task.get_batch_iterator(dataset=task.dataset(gen_subset)).next_epoch_itr(shuffle=False)
        sample = next(itr)
        sample = utils.move_to_cpu(sample)
        hypos = task.inference_step(generator, models, sample)
        top_hypotheses = []
        for hypo in hypos[0][:200]:
            pass
            # decoded_hypo = decode_fn(hypo['tokens'].int().cpu())
            decoded_hypo = [decode_fn(x['tokens'].int().cpu()) for x in hypo]
            score = np.array([x['score'].cpu() for x in hypo])
            probability = np.exp(score) # Apply exponetial to convert to probability
            top_hypotheses.append((decoded_hypo, probability))
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(data_dir)
    return top_hypotheses


def extract_audio_visual_feature(video_path, audio_path, ckpt_path, user_dir = None, is_finetune_ckpt=True):
    utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    # Extract Viusal features
    transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        Normalize(task.cfg.image_mean, task.cfg.image_std)])
    frames = load_video(video_path)
    video_nframes, _, _ = frames.shape
    print(f"Load video {video_path}: shape {frames.shape}")
    frames = transform(frames)
    print(f"Center crop video to: {frames.shape}")
    frames = to_gpu(torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0))
    # Extract MFCC features
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = np.floor(y.shape[0] / (video_nframes - 1)).astype(int)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=104, hop_length = hop_length)
    mfccs = to_gpu(torch.FloatTensor(mfccs).unsqueeze(dim=0))
    print(f"Load audio {audio_path}: shape {mfccs.shape}")
    return mfccs, frames, models

def extract_representations(mfccs , frames, n_subjects, alpha, tgt_layers, model,n_layers,n_heads, n_frames,head_dim, feature_dim):
    transformation_a, transformation_v, transformation_av = torch.empty((n_layers, n_heads, n_frames, head_dim)), torch.empty((n_layers, n_heads, n_frames, head_dim)), torch.empty((n_layers, n_heads, n_frames, head_dim))
    features_a, features_v, features_av = torch.empty((n_layers, n_frames, feature_dim)), torch.empty((n_layers, n_frames, feature_dim)), torch.empty((n_layers, n_frames, feature_dim))
    headwise_weights_a, headwise_weights_v, headwise_weights_av = torch.empty((n_layers,n_heads, n_frames, n_frames)), torch.empty((n_layers,n_heads, n_frames, n_frames)), torch.empty((n_layers,n_heads, n_frames, n_frames))

    with torch.no_grad():
        for subject_id in range(n_subjects):
            # reset the model vairant to model
            model_variant = model
            # get noise shape
            noise_shape = model_variant.encoder.layers[0].self_attn.out_proj.weight.data.shape
            for layer in range(tgt_layers):
                # Set a unique seed for each subject-layer combination
                seed = subject_id * 1000 + layer
                torch.manual_seed(seed)
                gaus_noise = torch.randn(noise_shape) * alpha  # Generate Gaussian noise on GPU
                # Add noise to the out-proj layer
                model_variant.encoder.layers[layer].self_attn.out_proj.weight.data += gaus_noise
            # Extract auditory features
            for i in range(1, 25):
                feature,_, attn = model_variant.extract_finetune(source={'video': None, 'audio': mfccs}, padding_mask=None, output_layer=i)
                headwise_weights_a[i-1] = attn[i-1][1]
                features_a[i-1] = feature
                transformation_a[i-1] = attn[i-1][2]
                print(f"Shape of auditory features for subject {subject_id + 1} from layer {i}: {np.shape(features_a)}")
            # Extract visual features
            for i in range(1, 25):
                feature,_, attn = model_variant.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=i)
                headwise_weights_v[i-1] = attn[i-1][1]
                features_v[i-1] = feature
                transformation_v[i-1] = attn[i-1][2]
                print(f"Shape of visual features for subject {subject_id + 1} from layer {i}: {np.shape(features_v)}")
            # Extract audiovisual features
            for i in range(1, 25):
                feature,_, attn = model_variant.extract_finetune(source={'video': frames, 'audio': mfccs}, padding_mask=None, output_layer=i)
                headwise_weights_av[i-1] = attn[i-1][1] # multihead attention weights (headwise)
                features_av[i-1] = feature # output latent representation of the layer
                transformation_av[i-1] = attn[i-1][2] # output transformation of the head
                print(f"Shape of audiovisual features for subject {subject_id + 1} from layer {i}: {np.shape(features_av)}")
            print(f'\n***END OF SUBJECT {subject_id}; STARTING NEW***\n')
    
    return features_a, features_v, features_av, headwise_weights_a, headwise_weights_v, headwise_weights_av, transformation_a, transformation_v, transformation_av


def extract_dictionary(ckpt_path, user_dir):
    try:
        # Load the model and configuration
        utils.import_user_module(Namespace(user_dir=user_dir))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

        # Set up the task
        task = tasks.setup_task(saved_cfg.task)

        # Extract and return the target dictionary
        target_dictionary = task.target_dictionary
        return target_dictionary
    except Exception as e:
        raise e