import torch
from classes.AudioDataProcessor import AudioDataProcessor
from config.config import Config
from classes.MelSpectrogram import MelSpectrogram


def prepare_audio(update: bool = False):
    """
    Prepares audio data by loading, processing, and splitting it into class representative,
    training, and evaluation sets.

    :param update: (bool) If True, removes and reloads processed data. Default is False.
    :return: tuple of dicts (class_repr, training_set, evaluation_set) with 'audio', 'labels',
             and 'gender' tensors.
    """
    dataloader = AudioDataProcessor()

    if update:
        dataloader.remove_data()

    # Loads audio, labels and gender tensors
    try:
        audio_tensor, labels_tensor, gender_tensor = torch.load(Config.AUDIO_FILE_PATH, weights_only=True)
    except FileNotFoundError:
        dataloader.load_data()
        audio_tensor, labels_tensor, gender_tensor = torch.load(Config.AUDIO_FILE_PATH, weights_only=True)


    # Splits data
    class_repr, training_set, evaluation_set = dataloader.split_data(audio_tensor, labels_tensor, gender_tensor)

    return class_repr, training_set, evaluation_set


def prepare_mel_spectrogram(class_repr: dict,
                            training_set: dict,
                            evaluation_set: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares mel spectrograms for class representative, training, and evaluation sets.

    :param class_repr: (dict) Class representative set with 'audio', 'labels', and 'gender'.
    :param training_set: (dict) Training set with 'audio', 'labels', and 'gender'.
    :param evaluation_set: (dict) Evaluation set with 'audio', 'labels', and 'gender'.
    :return: tuple of mel spectrogram tensors (class_repr_ms, training_set_ms, evaluation_set_ms).
    """
    # Initializes
    mel_spec = MelSpectrogram(sample_rate=Config.SAMPLE_RATE,
                              window=Config.WINDOW,
                              hop=Config.HOP,
                              n_filter=Config.N_FILTER)

    MelSpectrogram.remove_mel_folder() # To update data tensors, remove comment

    # Loads class repr mel spectrogram
    try:
        class_repr_ms = torch.load(Config.MEL_PATH_CR, weights_only=True)
    except FileNotFoundError:
        mel_spec.compute_mel_spectrogram(waveforms=class_repr['audio'], file_name=Config.MEL_PATH_CR)
        class_repr_ms = torch.load(Config.MEL_PATH_CR, weights_only=True)

    # Loads training set mel spectrogram
    try:
        training_set_ms = torch.load(Config.MEL_PATH_TS, weights_only=True)
    except FileNotFoundError:
        mel_spec.compute_mel_spectrogram(waveforms=training_set['audio'], file_name=Config.MEL_PATH_TS)
        training_set_ms = torch.load(Config.MEL_PATH_TS, weights_only=True)

    # Loads evaluation set mel spectrogram
    try:
        evaluation_set_ms = torch.load(Config.MEL_PATH_ES, weights_only=True)
    except FileNotFoundError:
        mel_spec.compute_mel_spectrogram(waveforms=evaluation_set['audio'], file_name=Config.MEL_PATH_ES)
        evaluation_set_ms = torch.load(Config.MEL_PATH_ES, weights_only=True)

    return class_repr_ms, training_set_ms, evaluation_set_ms