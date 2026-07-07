# AV-HuBERT Research

Audio-visual speech perception experiments using [AV-HuBERT](https://github.com/facebookresearch/av_hubert).

## Requirements

- macOS or Linux
- [Anaconda](https://www.anaconda.com/download) / Miniconda
- Python 3.10.13
- ~5 GB disk space (model checkpoint ~1.5 GB)

## Setup

Clone this repo and run the setup script:

```bash
git clone <this-repo-url>
cd avhubert
bash configure.sh
```

`configure.sh` will:
1. Clone the [av_hubert](https://github.com/T4phage76/av_hubert.git) library (with the `fairseq` submodule) into `library/`
2. Install Python dependencies (PyTorch, fairseq, etc.)
3. Download the dlib face landmark model and mean face template into `data/misc/`
4. Download the AV-HuBERT fine-tuned checkpoint into `data/finetune-model.pt`

## Key Scripts

| Script | Purpose |
|---|---|
| `Inference.py` | Run AV-HuBERT inference on stimuli |
| `common.py` | Shared utilities |
| `probability_prediction.py` | Phoneme probability predictions |
| `representation_extraction.py` | Extract internal representations |
| `extract_phon_distrib.py` | Extract phoneme distributions |
| `avi_to_mp4.py` | Convert AVI stimuli to MP4 |
| `avhubert_varaints_demo.ipynb` | Demo notebook |

## Stimuli

Stimulus tables (CSV files) are included in `data/`. The raw audio/video stimuli are not included in this repo.

## Model Variants

Other available AV-HuBERT checkpoints (see comments in `configure.sh`):
- AV base: `base_noise_pt_noise_ft_433h.pt`
- V-only base: `base_vox_433h.pt`
