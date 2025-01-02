import os

class Config:
    # Folders:
    RECORDS_PATH = os.path.join(os.getcwd(), "dataset", "records")
    AUDIO_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "audio")
    MEL_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "mel_spectrogram")
    # Files:
    AUDIO_FILE_PATH = os.path.join(AUDIO_PATH, "audio.pt")
    MEL_PATH_CR = os.path.join(AUDIO_PATH, "class_repr")
    MEL_PATH_TS = os.path.join(AUDIO_PATH, "training_set")
    MEL_PATH_ES = os.path.join(AUDIO_PATH, "evaluation_set")
    # Audio parameters
    SAMPLE_RATE = 16_000
    MAX_LENGTH = 1 * SAMPLE_RATE

    # Mel spectrogram parameters
    WINDOW = 25 # ms
    HOP = 10 # ms
    N_FILTER = 80