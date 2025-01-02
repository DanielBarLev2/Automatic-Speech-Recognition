import os
import torch
import shutil
import torchaudio
from config.config import Config


def load_data():
    """
    Load audio recordings from a specified directory, preprocess them (resample, pad/Cut),
    and save them as tensors along with their labels.

    The data is formatted as "name_digit_gender.wav". If the processed data already exists
    in the specified directory, the function will not overwrite it.

    :return: None: The function saves the processed tensors to a file specified in the configuration.
    """

    # Define the save directory, skips if already exists
    if os.path.exists(Config.AUDIO_PATH):
        return

    os.makedirs(Config.AUDIO_PATH, exist_ok=True)

    audio_data = []
    labels = []
    genders = []

    records_path = Config.RECORDS_PATH

    for file_name in sorted(os.listdir(records_path)):
        if file_name.endswith(".wav"):
            file_path = os.path.join(records_path, file_name)

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

    file_path = os.path.join(Config.AUDIO_PATH, 'audio.pt')
    torch.save((audio_tensor, labels_tensor, genders_tensor), file_path)
    print(f"audio tensors saved to {file_path}")


def remove_data():
    """
    Remove the folder where Config.AUDIO_PATH is located.
    """
    folder_path = os.path.dirname(Config.AUDIO_PATH)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def split_data(audio_tensor: torch.Tensor,
               labels_tensor: torch.Tensor,
               gender_tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor],
                                                     dict[str, torch.Tensor],
                                                     dict[str, torch.Tensor]]:
    """
    Splits data into Class Representative, Training Set, and Evaluation Set.

    :param audio_tensor: Audio data tensor (sorted by speaker).
    :param labels_tensor: Digit labels tensor (sorted by speaker).
    :param gender_tensor: Gender tensor (0 for male, 1 for female).

    :return:
        tuple: (class_repr, training_set, evaluation_set)
        - class_repr: dict with keys 'audio', 'labels', 'gender' containing tensors for the class representative.
        - training_set: dict with keys 'audio', 'labels', 'gender' containing tensors for the training set.
        - evaluation_set: dict with keys 'audio', 'labels', 'gender' containing tensors for the evaluation set.
    """
    c = labels_tensor.unique().shape[0]  # num of classes

    class_indices = (gender_tensor == torch.tensor(0))

    # split the data by male and female:
    audio_male_tensor = audio_tensor[class_indices]
    audio_female_tensor = audio_tensor[~class_indices]

    labels_male_tensor = labels_tensor[class_indices]
    labels_female_tensor = labels_tensor[~class_indices]

    gender_male_tensor = gender_tensor[class_indices]
    gender_female_tensor = gender_tensor[~class_indices]

    # Class Representative: one individual is chosen to represent the entire class
    class_repr = {"audio": audio_male_tensor[:c],
                  "labels": labels_male_tensor[:c],
                  "gender": gender_male_tensor[:c]}

    # Training set: Consists 2 males and 2 females
    training_set = {"audio":  torch.concat((audio_male_tensor[c:3 * c],  audio_female_tensor[:2 * c])),
                    "labels": torch.concat((labels_male_tensor[c:3 * c], labels_female_tensor[:2 * c])),
                    "gender": torch.concat((gender_male_tensor[c:3 * c], gender_female_tensor[:2 * c]))}

    # Evaluation Set: Consists the remaining speakers
    evaluation_set = {"audio":   torch.concat((audio_male_tensor[3 * c:],  audio_female_tensor[2 * c:])),
                      "labels": torch.concat((labels_male_tensor[3 * c:], labels_female_tensor[2 * c:])),
                      "gender": torch.concat((gender_male_tensor[3 * c:], gender_female_tensor[2 * c:]))}

    return class_repr, training_set, evaluation_set