# Prerequisite
# Python = 3.10.13
# Run this script from the root of the cloned repository:
#   bash configure.sh

# Resolve the directory containing this script, regardless of where it is called from
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

pip install --upgrade pip==24.0
pip install omegaconf==2.0.1 hydra-core==1.0.0

git clone --recurse-submodules https://github.com/T4phage76/av_hubert.git library
cd library
git submodule init
git submodule update
conda install pytorch::pytorch torchvision torchaudio -c pytorch
pip install scipy
pip install sentencepiece
pip install python_speech_features
pip install scikit-video



cd fairseq
pip install -e ./
cd "$REPO_ROOT"

# Download face landmark and a mean face for preprocessing
mkdir -p "$REPO_ROOT/data/misc/"
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O "$REPO_ROOT/data/misc/shape_predictor_68_face_landmarks.dat.bz2"
bzip2 -d "$REPO_ROOT/data/misc/shape_predictor_68_face_landmarks.dat.bz2"
wget --content-disposition https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy -O "$REPO_ROOT/data/misc/20words_mean_face.npy"

# create directory for stimuli
mkdir -p "$REPO_ROOT/data/stimuli/"

# download a checkpoint
wget https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/large_noise_pt_noise_ft_433h.pt -O "$REPO_ROOT/data/finetune-model.pt"

# V (some A) base model
# https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt
# AV base
# https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/base_noise_pt_noise_ft_433h.pt
# AV large
# https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/avsr/large_noise_pt_noise_ft_433h.pt
