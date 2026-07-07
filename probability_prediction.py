# /Users/beauchamplab/Library/r-rpymat/miniconda/envs/rpymat-conda-env/bin/python
from common import *

unzip_dir = 'data/stimuli/apr132026_visual_benefits/noisy' # data/stimuli/all_stim_paper/noisy_speech or data/stimuli/all_stim_paper/beam_test
url = 'https://www.dropbox.com/scl/fi/rpvovqcc2htbnf1gastjl/298_av_words_noisy.zip?rlkey=za3u4umgjdyf5d6q05grz6f60&st=2383eo2h&dl=1'  
filename = '298_av_words_noisy' # file containing all the stimuli
audio_path, stim_list, stimulus_path_list = create_stim_from_zip(url, unzip_dir, filename) # unzip the file and extract audio, convert stim to .mp4

# ---- Global variables -----------------  ------------------------------------------------------------------------------------
filename_prob_pred = 'Apr27_2026_av_benefit_noisy' # output file to save preds and probs

# ---- Load stimulus --------------------------------------------------------------------------------------------------------
stim_table = pd.read_csv(f"data/stim_table_{filename}.csv")
audio_path = stim_table['audio_path'].tolist() # get the paths to the .wav files
stim_list = stim_table['stim_list'].tolist() # get the stimulus name of each file
stimulus_path_list = stim_table['stimulus_path_list'].tolist() # get the paths to the .mp4 files
mouth_roi_path = [f"{unzip_dir}/{stim}_roi.mp4" for stim in stim_list] 
ensure_mouth_roi(stimulus_path_list, mouth_roi_path) # make mouth ROI stimuli out of original stim

# ---- Inference ------------------------------------------------------------------------------------------------------------
# Define the file pathways
ckpt_path = "./data/finetune-model.pt"
user_dir = "./library"

# Dictionary to save the outputs
raw_output = {}

# CHANGE THE PARAMETERS BEFORE RUNNING THIS CELL!!!
# beam_size = 20
beam_size_list = [5] #range(1,16) # all the beam size to test; default = 5
alpha_list = [0.1]#np.arange(0, 0.45, 0.05) # all the noise level to test;  noise level=0.1 -> randn(0,alpha^2)
tgt_layers = 6 # top n layers to change weights = 6
n_subjects = np.arange(0,10) # [1] #np.arange(0,100) #np.arange(0,60) # subject ID for all the model variants 0-164 were used for experiments; >200 were used for other testings 
permute_transformer = False # scramble the weights

#stim = stim_list
#audio = audio_path
#video = mouth_roi_path
#subject_id = 0
#raw_output[stimulus] = {}
#raw_output[stim][subject_id] = {}
# video_path, audio_path, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer = video, audio, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer
# hypo_av = predict(video, audio, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer)


# Making predictions

# pick one noise level
for alpha in alpha_list:
    # pick one beam size
    for beam_size in beam_size_list:
        # predict
        for stim, audio, video in zip(stim_list, audio_path, mouth_roi_path):
            print(f'\n***PREDICTING FROM {stim}***')
            print(f'\nAUDIO PATH IS: {audio} ; VIDEO PATH IS: {video}\n')
            raw_output[stim] = {}
            num_frames = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT)) # read the video here to capture the frame numbers for prediction
            for subject_id in n_subjects:
                print(f'\nRunning predictions for Subject {subject_id} of {n_subjects[0]} to {n_subjects[-1]}')
                raw_output[stim][subject_id] = {}
                while True:
                    try:
                        # A-only
                        hypo_a = predict(None, audio, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer, num_frames,beam_size)
                        print(f"Auditory-Input-Only Prediction: {hypo_a}")
                        raw_output[stim][subject_id]['auditory-only']= hypo_a
                        print('The sum likelihood of AUDITORY top predictions is:', np.sum([prob_a for pred_a,prob_a in hypo_a]))
                        break
                    except ValueError as e:
                        print(e)
                        print('failed trying again')
                while True:
                    try:
                        # V-only
                        hypo_v = predict(video, None, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer, num_frames, beam_size)
                        print(f"Visual-Input-Only Prediction: {hypo_v}")
                        raw_output[stim][subject_id]['visual-only']= hypo_v
                        print('The sum likelihood of VISUAL top predictions is:', np.sum([prob_v for pred_v,prob_v in hypo_v]))
                        break
                    except ValueError:
                        print('failed trying again')
                while True:
                    try:
                        # AV
                        hypo_av = predict(video, audio, ckpt_path, user_dir, alpha, tgt_layers, subject_id, permute_transformer, num_frames, beam_size)
                        print(f"Audio-Visual Input Prediction: {hypo_av}")
                        raw_output[stim][subject_id]['audiovisual']= hypo_av
                        print('The sum likelihood of AUDIO-VISUAL top predictions is:', np.sum([prob_av for pred_av,prob_av in hypo_av]))
                        break
                    except ValueError:
                        print('failed trying again')
                    
                print(f'\n***END OF SUBJECT {subject_id}; STARTING NEW***\n')
            print(f'\n***END OF STIMULUS {stim}; STARTING NEW***\n')
            # save for each stimulus to prevent losing data due to crashing
            #with open(f"results/{filename_prob_pred}_{beam_size}_{stim}.pkl", "wb") as f:
                #pickle.dump(raw_output, f)

        print(f'\n***END OF PREDICTION')
        # Save all data as pickle
        with open(f"results/{filename_prob_pred}_beam{beam_size}_alpha{alpha}.pkl", "wb") as f:
            pickle.dump(raw_output, f)




# Save the file as .h5
with h5py.File(f"results/{filename_prob_pred}.h5", 'w') as f:
    for stimulus in raw_output:
        stimulus_grp = f.create_group(stimulus)
        for subject_id in raw_output[stimulus]:
            subject_grp = stimulus_grp.create_group(str(subject_id))
            for modality, predictions in raw_output[stimulus][subject_id].items():
                modality_grp = subject_grp.create_group(modality)
                pred_data = modality_grp.create_dataset("predictions",data=[pred[0] for pred in predictions])
                prob_data = modality_grp.create_dataset("probabilities",data=[pred[1] for pred in predictions])

# Save the file as .csv (SUPERRRR SLOW!!!!!)
with open(f"results/{filename_prob_pred}.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Stimulus', 'SubjectID', 'Modality', 'Prediction', 'Probability'])
    for stimulus in raw_output:
        for subject_id in raw_output[stimulus]:
            for modality, predictions in raw_output[stimulus][subject_id].items():
                for prediction, probability in predictions:
                    csvwriter.writerow([stimulus,subject_id, modality, prediction, probability])

