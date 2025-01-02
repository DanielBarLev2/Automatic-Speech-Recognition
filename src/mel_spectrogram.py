import os
import torch
import shutil
import torchaudio.transforms as transforms

from config.config import Config
from matplotlib import pyplot as plt


class MelSpectrogram:
    def __init__(self, sample_rate: int, window: int = 25, hop: int = 10, n_filter: int = 80):
        """
        Initialize the MelSpectrogram class.

        :param sample_rate: Sample rate of the audio in Hz.
        :param window: Window size in milliseconds (default: 25ms).
        :param hop: Hop size in milliseconds (default: 10ms).
        :param n_filter: Number of Mel filter banks (default: 80).
        """
        self.sample_rate = sample_rate
        self.window = int(sample_rate * window / 1000)
        self.hop = int(sample_rate * hop / 1000)
        self.n_filter = n_filter

        self.mel_transform = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                       n_fft=self.window,
                                                       hop_length=self.hop,
                                                       n_mels=n_filter)

    def compute(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mel spectrogram for a given waveform.

        :param waveform: Tensor of shape (1, L) where L is the length of the waveform.
        :return: Tensor representing the Mel spectrogram.
        """
        return self.mel_transform(waveform)

    @staticmethod
    def display(mel_spectrogram: torch.Tensor, title: str = "Mel Spectrogram"):
        """
        Display the Mel spectrogram using matplotlib with enhanced resolution.

        :param mel_spectrogram: Mel spectrogram tensor of shape (n_mels, time_steps).
        :param title: Title for the plot.
        """
        plt.figure(figsize=(14, 8), dpi=150)  # Larger figure size and higher DPI
        plt.imshow(
            mel_spectrogram.log2().detach().cpu().numpy(),
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(title, fontsize=16)
        plt.xlabel("Time (frames)", fontsize=14)
        plt.ylabel("Frequency (Mel filter banks)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()  # Ensures elements fit well in the plot
        plt.show()

    def display_samples(self, audio_tensor: torch.Tensor, num_samples: int = 5):
        """
        Compute and display Mel spectrogram for several samples.

        :param audio_tensor: Tensor of shape (N, 1, L), where N is the number of samples.
        :param num_samples: Number of samples to display (default: 5).
        """
        for i in range(min(num_samples, audio_tensor.size(0))):
            waveform = audio_tensor[i]
            mel_spectrogram = self.compute(waveform)
            self.display(mel_spectrogram.squeeze(0), title=f"Sample {i + 1} Mel Spectrogram")

    def compute_mel_spectrogram(self, waveforms: torch.Tensor, file_name: str):
        """
        Compute and save Mel spectrogram for a batch of audio waveforms.

        If the file already exists, the function skips computation.

        :param waveforms: Tensor of shape (n, 1, sample_size) containing audio waveforms.
        :param file_name: Name of the file to save the spectrograms.
        :return: None
        """
        # Define the save directory, skips if already exists
        file_path = os.path.join(Config.MEL_PATH, file_name)
        if os.path.isfile(file_path):
            return

        if not os.path.exists(Config.MEL_PATH):
            os.makedirs(Config.MEL_PATH, exist_ok=True)

        # Flatten (n, sample_size) batch and compute Mel spectrograms
        flat_waveforms = waveforms.view(-1, waveforms.size(-1))  # Flatten batch
        mel_spectrograms = torch.stack([self.mel_transform(waveform.unsqueeze(0)) for waveform in flat_waveforms])

        torch.save(mel_spectrograms, file_path)
        print(f"Mel spectrograms saved to {file_path}")

    @staticmethod
    def remove_mel_folder():
        """
        Remove the folder where Mel spectrograms are stored.
        """
        folder_path = Config.MEL_PATH  # This points directly to "mel_spectrogram"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
