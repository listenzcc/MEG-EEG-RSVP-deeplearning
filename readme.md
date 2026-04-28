# MEG-EEG RSVP Deep Learning

A research codebase for training transformer-based classifiers on RSVP EEG/MEG epoch data. This repository includes a main PyTorch training script, a batch launcher for subjects and modalities, and a results directory for saved models and logs.

## Repository Structure

- `test1.py` - main model training and evaluation script.
- `decoding-step-1/batch.sh` - example batch workflow for running `test1.py` across subjects and modes.
- `decoding-step-1/EEG-SXX/` and `decoding-step-1/MEG-SXX/` - expected data directories for each subject.
- `results/` - target folder for output models, TensorBoard logs, and CSV training logs.

## What the Code Does

`test1.py`:

- reads epoched data from `decoding-step-1/{MODE}-{SUBJ}`
- loads `epochs-1-epo.fif`, `epochs-2-epo.fif`, and `epochs-3-epo.fif`
- concatenates data for binary classification
- normalizes each trial
- trains an `RSVPTransformer` model with convolutional embedding, transformer encoder layers, and attention pooling
- evaluates validation loss and AUC
- saves the best model to `results/{MODE}-{SUBJ}-{timestamp}/best_model.pt`

## Requirements

- Python 3.10
- conda environment named `python3.10` (optional but used in `batch.sh`)

Dependencies likely include:

- numpy
- torch
- mne
- scikit-learn
- pandas
- tqdm
- tensorboard

Install using `pip` or `conda` before running.

## Usage

Run a single subject and mode:

```bash
python test1.py --subj S01 --mode EEG
```

Run the example batch launcher:

```bash
bash decoding-step-1/batch.sh
```

## Notes

- `test1.py` currently hardcodes `device = 4`; update this if your GPU index differs.
- `batch.sh` activates `python3.10` and expects a valid Conda environment.
- The script crops epochs to the first second and uses 200 time points per trial.

## Output

Training output is written to:

```text
results/{MODE}-{SUBJ}-{timestamp}/
```

Output includes:

- `best_model.pt`
- `logs/` for TensorBoard
- `training_log.csv`

## Suggested Improvements

- add `requirements.txt`
- make device selection a CLI argument
- add a config file or training wrapper
- support more dataset formats and structured preprocessing
