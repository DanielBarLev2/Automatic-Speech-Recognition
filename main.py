import os

import torch
from config.config import Config
from src import dataloader
from src.mel_spectrogram import MelSpectrogram

if __name__ == "__main__":

    dataloader.remove_data() # To update data tensors, remove comment

    file_path = os.path.join(Config.AUDIO_PATH, 'audio.pt')

    try:
        audio_tensor, labels_tensor, gender_tensor = torch.load(file_path, weights_only=True)
    except FileNotFoundError:
        dataloader.load_data()
        audio_tensor, labels_tensor, gender_tensor = torch.load(file_path, weights_only=True)


    class_repr, training_set, evaluation_set = dataloader.split_data(audio_tensor, labels_tensor, gender_tensor)

    mel_spec = MelSpectrogram(sample_rate=Config.SAMPLE_RATE,
                              window=Config.WINDOW,
                              hop=Config.HOP,
                              n_filter=Config.N_FILTER)

    MelSpectrogram.remove_mel_folder() # To update data tensors, remove comment

    # mel_spec.compute_mel_spectrogram(waveforms=class_repr['audio'], file_name='class_repr_mel')
    # mel_spec.compute_mel_spectrogram(waveforms=training_set['audio'], file_name='training_set_mel')
    # mel_spec.compute_mel_spectrogram(waveforms=evaluation_set['audio'], file_name='evaluation_set_mel')

    # # display 5 samples of a male speaker
    # mel_spec.display_samples(training_set['audio'][10:], num_samples=5)
    # # display 5 samples of a female speaker
    # mel_spec.display_samples(training_set['audio'][30:], num_samples=5)

