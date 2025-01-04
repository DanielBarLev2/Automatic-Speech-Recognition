import torch

from classes.AudioDataProcessor import AudioDataProcessor
from config.config import Config
from classes.MelSpectrogram import MelSpectrogram


def prepare_audio(update: bool = True, normalize: bool = False):
    """
    Prepares audio data by loading, processing, and splitting it into class representative,
    training, and evaluation sets.

    i. Class Representative: 1 individual chosen to represent the entire class.
    ii. Training Set: Consisting of 2 males and 2 females selected from the recorded speakers.
    iii. Evaluation Set: Consisting of the remaining speakers, with 2 males and 2 females.

    :param normalize: Apply Automatic Gain Control (AGC) to normalize audio waveforms.
    :param update: (bool) If True, removes and reloads processed data. Default is False.
    :return: tuple of dicts (class_repr, training_set, evaluation_set) with 'audio', 'labels', and 'gender' tensors.
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

    if normalize:
        audio_tensor = audio_tensor / (audio_tensor.abs().max(dim=-1, keepdim=True).values + 1e-8)

    # Splits data
    class_repr, training_set, evaluation_set = dataloader.split_data(audio_tensor, labels_tensor, gender_tensor)

    return class_repr, training_set, evaluation_set


def prepare_mel_spectrogram(class_repr_audio: torch.Tensor,
                            training_set_audio: torch.Tensor,
                            evaluation_set_audio: torch.Tensor,
                            update: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares mel spectrograms for class representative, training, and evaluation audio tensors.

    :param class_repr_audio: (torch.Tensor) Audio tensor for the class representative set. Shape: (N, 1, S).
    :param training_set_audio: (torch.Tensor) Audio tensor for the training set. Shape: (N, 1, S).
    :param evaluation_set_audio: (torch.Tensor) Audio tensor for the evaluation set.  Shape: (N, 1, S).
    :param update: (bool) If True, removes and reloads processed data. Default is True.
    :return: tuple of mel spectrogram tensors (class_repr_ms, training_set_ms, evaluation_set_ms).
    """
    # Initializes
    mel_spec = MelSpectrogram(sample_rate=Config.SAMPLE_RATE,
                              window=Config.WINDOW,
                              hop=Config.HOP,
                              n_filter=Config.N_FILTER,
                              device=Config.DEVICE)

    #mel_spec.display_samples(audio_tensor=training_set_audio[:10], num_samples=2)
    #mel_spec.display_samples(audio_tensor=training_set_audio[20:30], num_samples=2)

    if update:
        MelSpectrogram.remove_mel_folder()

    # Loads class repr mel spectrogram
    try:
        class_repr_ms = torch.load(Config.MEL_PATH_CR, map_location=Config.DEVICE, weights_only=True)
    except FileNotFoundError:
        mel_spec.compute_mel_spectrogram(waveforms=class_repr_audio, file_path=Config.MEL_PATH_CR)
        class_repr_ms = torch.load(Config.MEL_PATH_CR, map_location=Config.DEVICE, weights_only=True)

    # Loads training set mel spectrogram
    try:
        training_set_ms = torch.load(Config.MEL_PATH_TS, map_location=Config.DEVICE, weights_only=True)
    except FileNotFoundError:
        mel_spec.compute_mel_spectrogram(waveforms=training_set_audio, file_path=Config.MEL_PATH_TS)
        training_set_ms = torch.load(Config.MEL_PATH_TS, map_location=Config.DEVICE, weights_only=True)

    # Loads evaluation set mel spectrogram
    try:
        evaluation_set_ms = torch.load(Config.MEL_PATH_ES, map_location=Config.DEVICE, weights_only=True)
    except FileNotFoundError:
        mel_spec.compute_mel_spectrogram(waveforms=evaluation_set_audio, file_path=Config.MEL_PATH_ES)
        evaluation_set_ms = torch.load(Config.MEL_PATH_ES, map_location=Config.DEVICE, weights_only=True)

    return class_repr_ms, training_set_ms, evaluation_set_ms