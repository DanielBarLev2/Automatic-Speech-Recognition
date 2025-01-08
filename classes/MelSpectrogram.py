import os

import librosa
import numpy as np
import torch
import shutil
import torchaudio.transforms as transforms

from config.config import Config
from matplotlib import pyplot as plt


class MelSpectrogram:
    def __init__(self, sample_rate: int, window: int, hop: int, n_mels: int = 80, device: str = 'cpu'):
        """
        Initialize the MelSpectrogram class.

        :param sample_rate: Sample rate of the audio in Hz.
        :param window: Window size in milliseconds.
        :param hop: Hop size in milliseconds.
        :param n_mels: Number of Mel filter banks (default: 80).
        :param device: Device to use for computations ('cpu' or 'cuda').
        """
        self.sample_rate = sample_rate
        self.window = window
        self.hop = hop
        self.n_mels = n_mels
        self.device = device


    def compute(self, waveform: torch.Tensor):
        """
        Compute the Mel spectrogram for a given waveform.

        :param waveform: Tensor containing the audio waveform.
        :return: Mel spectrogram.
        """
        waveform = waveform.cpu().numpy() if waveform.is_cuda else waveform.numpy()

        mel_spec = librosa.feature.melspectrogram(y=waveform,
                                                  sr=self.sample_rate,
                                                  n_mels=self.n_mels,
                                                  fmax=self.sample_rate // 2,
                                                  n_fft=self.window, hop_length=self.hop)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec = torch.tensor(mel_spec, device=self.device)

        return mel_spec

    def compute_mel_spectrogram(self, waveforms: torch.Tensor, file_path: str):
        """
        Compute and save Mel spectrograms for a batch of audio waveforms.

        If the file already exists, the function skips computation.

        :param waveforms: Tensor of shape (n, sample_size) containing audio waveforms.
        :param file_path: Path of the file to save the spectrograms.
        :return: None
        """
        # Define the save directory, skips if already exists
        if os.path.isfile(file_path):
            return

        if not os.path.exists(Config.MEL_PATH):
            os.makedirs(Config.MEL_PATH, exist_ok=True)

        waveforms = waveforms.to(self.device)
        mel_spectrograms = [self.compute(waveform) for waveform in waveforms]
        mel_spectrograms = torch.stack(mel_spectrograms)

        torch.save(mel_spectrograms, file_path)
        print(f"Mel spectrograms saved to {file_path}")

    @staticmethod
    @staticmethod
    def display(mel_spectrogram: torch.Tensor, title: str = "Mel Spectrogram"):
        """
        Display a single Mel spectrogram using a heatmap.

        :param mel_spectrogram: 2D tensor representing the Mel spectrogram.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.title(title)
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        plt.tight_layout()
        plt.show()

    def display_samples(self, audio_tensor: torch.Tensor, num_samples: int = 5):
        """
        Compute and display Mel spectrograms for several samples.

        :param audio_tensor: Tensor of shape (N, sample_size), where N is the number of samples.
        :param num_samples: Number of samples to display (default: 5).
        """
        for i, waveform in enumerate(audio_tensor[:num_samples]):
            mel_spec = self.compute(waveform)
            self.display(mel_spec, title=f"Mel Spectrogram Sample {i + 1}")



    @staticmethod
    def remove_mel_folder():
        """
        Remove the folder where Mel spectrograms are stored.
        """
        folder_path = Config.MEL_PATH
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
