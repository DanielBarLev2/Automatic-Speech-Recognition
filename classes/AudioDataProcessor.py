import os
import torch
import shutil
import numpy as np
from config.config import Config
from scipy.signal import resample
from scipy.io.wavfile import read


class AudioDataProcessor:
    def __init__(self):
        self.config = Config()

    def load_data(self):
        """
        Load audio recordings from a specified directory, preprocess them (resample, pad/Cut),
        and save them as tensors along with their labels.
        """
        # Define the save directory, skips if already exists
        if os.path.exists(self.config.AUDIO_PATH):
            return

        os.makedirs(self.config.AUDIO_PATH, exist_ok=True)

        audio_data = []
        labels = []
        genders = []

        records_path = self.config.RECORDS_PATH

        for file_name in sorted(os.listdir(records_path)):
            if file_name.endswith(".wav"):
                file_path = os.path.join(records_path, file_name)

                sample_rate, wave_form = read(file_path)

                wave_form = np.array(wave_form, dtype=np.float32)

                # Resample
                wave_form = resample(wave_form, int(len(wave_form) * (Config.SAMPLE_RATE / sample_rate)))

                # Cut
                wave_form = wave_form[:Config.SAMPLE_RATE]

                # Convert stereo to mono
                try:
                    if wave_form.shape[1] == 2:
                        wave_form = np.mean(wave_form, axis=1) # Average the two channels
                except IndexError:
                    pass

                # Pad
                if wave_form.shape[0] < Config.SAMPLE_RATE:
                    wave_form = np.pad(wave_form, (0, Config.SAMPLE_RATE), mode='constant', constant_values=0)

                audio_data.append(torch.from_numpy(wave_form).to(Config.DEVICE))

                # Extract label
                label = int(file_name.split("_")[1])
                labels.append(label)

                # Extract gender (0 - male, 1 - female)
                gender = 0 if "m" == file_name.split("_")[2][0] else 1
                genders.append(gender)

        audio_tensor = torch.stack(audio_data).to(Config.DEVICE)
        labels_tensor = torch.tensor(labels).to(Config.DEVICE)
        genders_tensor = torch.tensor(genders).to(Config.DEVICE)

        file_path = os.path.join(self.config.AUDIO_PATH, 'audio.pt')
        torch.save((audio_tensor, labels_tensor, genders_tensor), file_path)
        print(f"audio tensors saved to {file_path}")

    def remove_data(self):
        """
        Remove the folder where AUDIO_PATH is located.
        """
        folder_path = os.path.dirname(self.config.AUDIO_PATH)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    @staticmethod
    def split_data(audio_tensor: torch.Tensor,
                   labels_tensor: torch.Tensor,
                   gender_tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor],
                                                         dict[str, torch.Tensor],
                                                         dict[str, torch.Tensor]]:
        """
        ### The split is HARD CODED to split first specker to class representative, and the next four  ###
        ### (two males two females) to training set. The rest are belong to evaluation set.            ###

        :param audio_tensor: (torch.Tensor) Tensor containing audio data for all samples. Shape: (N, C, SR).
        :param labels_tensor: (torch.Tensor) Tensor containing class labels for each sample. Shape: (N,).
        :param gender_tensor: (torch.Tensor) Tensor indicating gender for each sample. Shape: (N,).
                              Values: 0 for male, 1 for female.

        :return: tuple containing three dictionaries:
            - `class_repr`: Dictionary with 'audio', 'labels', and 'gender' tensors for the Class Representative set.
            - `training_set`: Dictionary with 'audio', 'labels', and 'gender' tensors for the Training Set.
            - `evaluation_set`: Dictionary with 'audio', 'labels', and 'gender' tensors for the Evaluation Set.
        """
        c = labels_tensor.unique().shape[0]  # Number of classes

        class_indices = (gender_tensor == 0)

        # Split the data by male and female
        audio_male_tensor = audio_tensor[class_indices]
        audio_female_tensor = audio_tensor[~class_indices]

        labels_male_tensor = labels_tensor[class_indices]
        labels_female_tensor = labels_tensor[~class_indices]

        gender_male_tensor = gender_tensor[class_indices]
        gender_female_tensor = gender_tensor[~class_indices]

        # Class Representative: one individual is chosen to represent the entire class
        class_repr = {
            "audio": audio_female_tensor[:c],
            "labels": audio_female_tensor[:c],
            "gender": audio_female_tensor[:c]
        }

        # Training set: Clonsists of 2 males and 2 femaes
        training_set = {
            "audio": torch.concat((audio_female_tensor[c:3 * c], audio_male_tensor[:2 * c])),
            "labels": torch.concat((labels_female_tensor[c:3 * c], labels_male_tensor[:2 * c])),
            "gender": torch.concat((gender_female_tensor[c:3 * c], gender_male_tensor[:2 * c]))
        }

        # Evaluation Set: Consists of the remaining speakers
        evaluation_set = {
            "audio": torch.concat((audio_female_tensor[3 * c:], audio_male_tensor[2 * c:])),
            "labels": torch.concat((labels_female_tensor[3 * c:], labels_male_tensor[2 * c:])),
            "gender": torch.concat((gender_female_tensor[3 * c:], gender_male_tensor[2 * c:]))
        }

        return class_repr, training_set, evaluation_set
