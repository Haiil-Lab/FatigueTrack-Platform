# FatigueTrack

Multimodal fatigue assessment using behavioral, EEG, and ECG signals.

## Modalities

- Video: EAR, MAR, blink and yawning dynamics
- EEG: spectral band powers, ratios and spectral entropy
- ECG: heart rate and HRV features

## Model

BiLSTM with Temporal Attention

## Setup

python -m venv .venv
pip install -r requirements.txt