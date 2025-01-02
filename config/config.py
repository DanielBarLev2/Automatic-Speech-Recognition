import os

class Config:
    RECORDS_PATH = os.path.join(os.getcwd(), "dataset", "records")
    DATA_PATH = os.path.join(os.getcwd(), "dataset", "processed_data", "audio_data.pt")
    # Audio parameters
    SAMPLE_RATE = 16_000
    MAX_LENGTH = 1 * SAMPLE_RATE

    # Labels
    NUM_CLASSES = 10  # 0-9 digits