import os
import torch


class Config:
    # Folders:
    RECORDS_PATH = os.path.join(os.getcwd(), "dataset", "records")
    AUDIO_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "audio")
    MEL_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "mel_spectrogram")
    # Files:
    AUDIO_FILE_PATH = os.path.join(AUDIO_PATH, "audio.pt")
    MEL_PATH_CR = os.path.join(MEL_PATH, "class_repr")
    MEL_PATH_TS = os.path.join(MEL_PATH, "training_set")
    MEL_PATH_ES = os.path.join(MEL_PATH, "evaluation_set")
    # Audio parameters
    SAMPLE_RATE = 16_000
    MAX_LENGTH = 1 * SAMPLE_RATE

    # Mel spectrogram parameters
    WINDOW = int(SAMPLE_RATE * 25 / 1000)
    HOP = int(SAMPLE_RATE * 10 / 1000)
    N_FILTER = 80

    # CUDA
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"