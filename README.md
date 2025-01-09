# Automatic-Speech-Recognition
### Audio Processing and Classification using DTW and CTC
Python code for Automatic Speech Recognition (ASR) tasks, including audio preprocessing, Mel Spectrogram extraction,
Dynamic Time Warping (DTW) for sequence alignment, and Connectionist Temporal Classification (CTC) for modeling.
Features include dataset preparation, sequence evaluation, and performance analysis.

## Overview
This project implements an audio processing pipeline for classifying spoken digits (0-9) using dynamic time warping (DTW) and connectionist temporal classification (CTC). It processes raw audio recordings, extracts features using Mel spectrograms, and uses DTW for classification and CTC for sequence alignment.

## Features
- Audio data preprocessing (resampling, normalization, padding, and Mel spectrogram computation).
- Dynamic Time Warping (DTW) for classification based on distance matrices.
- CTC implementation for probability calculation and force alignment.
- Visualization of results including confusion matrices and heatmaps.

---

## Requirements
Install the required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
#### Notable Dependencies
- Numba: Used to accelerate computational tasks like DTW using Just-In-Time (JIT) compilation.
- CUDA Support: If a CUDA-compatible GPU is available, computations will leverage GPU acceleration to significantly enhance performance.

Project Structure
- main.py: The main script to run the project.
- src/: Contains modules for audio processing, feature extraction, DTW, and CTC.
- config/: Configuration for paths, parameters, and device settings.
- classes/: Contains classes: dataloader for data preparation and mel-spectrogram calculation.
- dataset/: Contains raw recordings and processed data and processed tensor representation of audio and mel-spectrogram.
- results/: All plots and diagrams will be saved here.


## How to Run

### 1. Prepare Data:

- Ensure audio recordings are placed in the dataset/records/ directory. [for this git repo we included the dataset]
- Audio files should be named in the format: speaker_digit_gender.wav.

#### To load or process audio data:
```python
class_repr, training_set, evaluation_set = prepare_audio(update=True)
# To load preprocessed data (if it exists):
class_repr, training_set, evaluation_set = prepare_audio(update=False)
```
#### To load or process mel-spectrogram data:
```python
class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=True)
# To load preprocessed mel-spectrogram (if it exists):
class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=False)
```

### 2. Run the Main Script:
```bash
python main.py
```

Main Script Outputs
1. Data Acquisition: Prints the count of processed audio files and speaker details.
2. DTW Distance Matrix: Visualizes the matrix and prints the optimal threshold for classification.
3. Training Accuracy: Reports classification accuracy for the training set.
4. Validation Accuracy: Reports classification accuracy for the validation set.
5. CTC Results.
6. Prints the sequence probability. 
7. Visualizes forward and backward matrices.

