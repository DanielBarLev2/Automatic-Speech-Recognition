import os
import torch
import shutil
import torchaudio
from config.config import Config


def load_data():
    """
    Load audio recordings from a specified directory, preprocess them (resample, pad/Cut),
    and save them as tensors along with their labels.

    If the data was already loaded to tensor form that folder, the data will not be overwritten.

    # Data mast be formatted as "name_digit_gender.wav".
    :return: None: The function saves the processed tensors to a file specified in the configuration.
    """

    # Define the save directory, skips if already exists
    save_dir = os.path.join("dataset", "processed_data")
    save_path = os.path.join(save_dir, "audio_data.pt")

    if os.path.exists(save_dir):
        return

    os.makedirs(save_dir, exist_ok=True)

    audio_data = []
    labels = []
    genders = []

    path = Config.RECORDS_PATH

    for file_name in sorted(os.listdir(path)):
        if file_name.endswith(".wav"):
            file_path = os.path.join(path, file_name)

            wave_form, sample_rate = torchaudio.load(file_path)

            # Resample
            if sample_rate != Config.SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=Config.SAMPLE_RATE)
                wave_form = resampler(wave_form)

            # Cut
            wave_form = wave_form[:, :Config.MAX_LENGTH]
            # Pad
            if wave_form.size(1) < Config.MAX_LENGTH:
                padding = Config.MAX_LENGTH - wave_form.size(1)
                wave_form = torch.nn.functional.pad(wave_form, (0, padding))

            # Convert stereo to mono
            if wave_form.size(0) == 2:
                wave_form = torch.mean(wave_form, dim=0, keepdim=True)  # Average the two channels

            audio_data.append(wave_form)

            # Extract label
            label = int(file_name.split("_")[1])
            labels.append(label)

            # Extract gender (0 - male, 1 - female)
            if "m" == file_name.split("_")[2][0]:
                genders.append(0)
            elif "f" == file_name.split("_")[2][0]:
                genders.append(1)

    audio_tensor = torch.stack(audio_data)
    labels_tensor = torch.tensor(labels)
    genders_tensor = torch.tensor(genders)

    torch.save((audio_tensor, labels_tensor, genders_tensor), save_path)


def remove_data():
    """
    Remove the folder where Config.DATA_PATH is located.
    """
    folder_path = os.path.dirname(Config.DATA_PATH)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
