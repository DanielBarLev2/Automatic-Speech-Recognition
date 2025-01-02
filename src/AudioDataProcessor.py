import os
import torch
import shutil
import torchaudio
from config.config import Config


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

                wave_form, sample_rate = torchaudio.load(file_path)

                # Resample
                if sample_rate != self.config.SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.config.SAMPLE_RATE)
                    wave_form = resampler(wave_form)

                # Cut
                wave_form = wave_form[:, :self.config.MAX_LENGTH]
                # Pad
                if wave_form.size(1) < self.config.MAX_LENGTH:
                    padding = self.config.MAX_LENGTH - wave_form.size(1)
                    wave_form = torch.nn.functional.pad(wave_form, (0, padding))

                # Convert stereo to mono
                if wave_form.size(0) == 2:
                    wave_form = torch.mean(wave_form, dim=0, keepdim=True)  # Average the two channels

                audio_data.append(wave_form)

                # Extract label
                label = int(file_name.split("_")[1])
                labels.append(label)

                # Extract gender (0 - male, 1 - female)
                gender = 0 if "m" == file_name.split("_")[2][0] else 1
                genders.append(gender)

        audio_tensor = torch.stack(audio_data)
        labels_tensor = torch.tensor(labels)
        genders_tensor = torch.tensor(genders)

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
    def split_data(audio_tensor: torch.Tensor, labels_tensor: torch.Tensor, gender_tensor: torch.Tensor):
        """
        Splits data into Class Representative, Training Set, and Evaluation Set.
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
            "audio": audio_male_tensor[:c],
            "labels": labels_male_tensor[:c],
            "gender": gender_male_tensor[:c]
        }

        # Training set: Consists of 2 males and 2 females
        training_set = {
            "audio": torch.concat((audio_male_tensor[c:3 * c], audio_female_tensor[:2 * c])),
            "labels": torch.concat((labels_male_tensor[c:3 * c], labels_female_tensor[:2 * c])),
            "gender": torch.concat((gender_male_tensor[c:3 * c], gender_female_tensor[:2 * c]))
        }

        # Evaluation Set: Consists of the remaining speakers
        evaluation_set = {
            "audio": torch.concat((audio_male_tensor[3 * c:], audio_female_tensor[2 * c:])),
            "labels": torch.concat((labels_male_tensor[3 * c:], labels_female_tensor[2 * c:])),
            "gender": torch.concat((gender_male_tensor[3 * c:], gender_female_tensor[2 * c:]))
        }

        return class_repr, training_set, evaluation_set
