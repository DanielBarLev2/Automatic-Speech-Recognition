import os

class Config:
    RECORDS_PATH = os.path.join(os.getcwd(), "dataset", "records")
    AUDIO_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "audio")
    MEL_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "mel_spectrogram")

    # Audio parameters
    SAMPLE_RATE = 16_000
    MAX_LENGTH = 1 * SAMPLE_RATE

    # Mel spectrogram parameters
    WINDOW = 25 # ms
    HOP = 10 # ms
    N_FILTER = 80