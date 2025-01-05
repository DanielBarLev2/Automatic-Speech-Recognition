from classes.DTW import DTW
from classes.CTC import CTC
from config.config import Config
from src.preparation import prepare_audio, prepare_mel_spectrogram
import torch
from scipy.io.wavfile import read, write
import numpy as np
from scipy.signal import resample
from scipy.signal import spectrogram
import librosa

if __name__ == "__main__":

    print(f'Using: {Config.DEVICE}')
    class_repr, training_set, evaluation_set = prepare_audio(update=False)

    class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=False)

    """ctc = CTC()
    print(ctc.pred)
    prob, mat = ctc.word_prob("aba")
    print(prob)
    print(mat)
    prob, seq, mat = ctc.word_prob_for_force_alignment("aba")
    print(prob)
    print(seq)
    print(mat)"""

    rep_data = np.zeros((10, 80, 101))
    for i in range(10):
        sample_rate, audio = read(f'dataset/records/avital_{i}_f.wav')
        audio = np.array(audio, dtype=np.float32)

        target_sample_rate = 16000
        num_samples = int(len(audio) * (target_sample_rate / sample_rate))
        audio = resample(audio, num_samples)
        length = audio.shape[0] / target_sample_rate
        audio = audio[:target_sample_rate]
        # Convert stereo to mono
        try :
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)  # Average the two channels
        except:
            audio=audio
        if audio.shape[0] < target_sample_rate:
            audio = np.pad(audio, (0, target_sample_rate), mode='constant', constant_values=0)
        length = audio.shape[0] / target_sample_rate
        window_size = int(target_sample_rate * 25 / 1000)
        hop_size = int(target_sample_rate * 10 / 1000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sample_rate, n_mels=80, fmax=target_sample_rate // 2,
                                                  n_fft=window_size, hop_length=hop_size)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        rep_data[i] = mel_spec_db

    training_data = np.zeros((40, 80, 101))
    for i in range(10):
        sample_rate, audio = read(f'dataset/records/bar_{i}_m.wav')
        audio = np.array(audio, dtype=np.float32)

        target_sample_rate = 16000
        num_samples = int(len(audio) * (target_sample_rate / sample_rate))
        audio = resample(audio, num_samples)
        length = audio.shape[0] / target_sample_rate
        audio = audio[:target_sample_rate]
        # Convert stereo to mono
        try :
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)  # Average the two channels
        except:
            audio=audio
        if audio.shape[0] < target_sample_rate:
            audio = np.pad(audio, (0, target_sample_rate), mode='constant', constant_values=0)
        length = audio.shape[0] / target_sample_rate
        window_size = int(target_sample_rate * 25 / 1000)
        hop_size = int(target_sample_rate * 10 / 1000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sample_rate, n_mels=80, fmax=target_sample_rate // 2,
                                                  n_fft=window_size, hop_length=hop_size)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        training_data[i] = mel_spec_db

    for i in range(10):
        sample_rate, audio = read(f'dataset/records/guy_{i}_m.wav')
        audio = np.array(audio, dtype=np.float32)

        target_sample_rate = 16000
        num_samples = int(len(audio) * (target_sample_rate / sample_rate))
        audio = resample(audio, num_samples)
        length = audio.shape[0] / target_sample_rate
        audio = audio[:target_sample_rate]
        # Convert stereo to mono
        try :
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)  # Average the two channels
        except:
            audio=audio
        if audio.shape[0] < target_sample_rate:
            audio = np.pad(audio, (0, target_sample_rate), mode='constant', constant_values=0)
        length = audio.shape[0] / target_sample_rate
        window_size = int(target_sample_rate * 25 / 1000)
        hop_size = int(target_sample_rate * 10 / 1000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sample_rate, n_mels=80, fmax=target_sample_rate // 2,
                                                  n_fft=window_size, hop_length=hop_size)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        training_data[10+i] = mel_spec_db

    for i in range(10):
        sample_rate, audio = read(f'dataset/records/neta_{i}_f.wav')
        audio = np.array(audio, dtype=np.float32)

        target_sample_rate = 16000
        num_samples = int(len(audio) * (target_sample_rate / sample_rate))
        audio = resample(audio, num_samples)
        length = audio.shape[0] / target_sample_rate
        audio = audio[:target_sample_rate]
        # Convert stereo to mono
        try :
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)  # Average the two channels
        except:
            audio=audio
        if audio.shape[0] < target_sample_rate:
            audio = np.pad(audio, (0, target_sample_rate), mode='constant', constant_values=0)
        length = audio.shape[0] / target_sample_rate
        window_size = int(target_sample_rate * 25 / 1000)
        hop_size = int(target_sample_rate * 10 / 1000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sample_rate, n_mels=80, fmax=target_sample_rate // 2,
                                                  n_fft=window_size, hop_length=hop_size)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        training_data[20+i] = mel_spec_db

    for i in range(10):
        sample_rate, audio = read(f'dataset/records/nirit_{i}_f.wav')
        audio = np.array(audio, dtype=np.float32)

        target_sample_rate = 16000
        num_samples = int(len(audio) * (target_sample_rate / sample_rate))
        audio = resample(audio, num_samples)
        length = audio.shape[0] / target_sample_rate
        audio = audio[:target_sample_rate]
        # Convert stereo to mono
        try :
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)  # Average the two channels
        except:
            audio=audio
        if audio.shape[0] < target_sample_rate:
            audio = np.pad(audio, (0, target_sample_rate), mode='constant', constant_values=0)
        length = audio.shape[0] / target_sample_rate
        window_size = int(target_sample_rate * 25 / 1000)
        hop_size = int(target_sample_rate * 10 / 1000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sample_rate, n_mels=80, fmax=target_sample_rate // 2,
                                                  n_fft=window_size, hop_length=hop_size)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        training_data[30+i] = mel_spec_db

    # Initialize DTW
    dtw = DTW(torch.from_numpy(rep_data), torch.from_numpy(training_data))

    # Compute DTW distance matrix
    dtw_matrix = dtw.compute_distance_matrix()
    expected_result = torch.FloatTensor(range(10))
    print(dtw_matrix.shape)

    result=torch.argmin(dtw_matrix[0], dim=1)
    print(dtw_matrix[0])
    print(torch.sum((result-expected_result).eq(0)).item())
    print(result)

    result=torch.argmin(dtw_matrix[1], dim=1)
    print(dtw_matrix[1])
    print(torch.sum((result-expected_result).eq(0)).item())
    print(result)

    result=torch.argmin(dtw_matrix[2], dim=1)
    print(dtw_matrix[2])
    print(torch.sum((result-expected_result).eq(0)).item())
    print(result)

    result=torch.argmin(dtw_matrix[3], dim=1)
    print(dtw_matrix[3])
    print(torch.sum((result-expected_result).eq(0)).item())
    print(result)

    # Determine threshold
    threshold = dtw_matrix.mean().item()

    # Classify and plot confusion matrix
    dtw.classify_and_plot_confusion_matrix(
        dtw_matrix=dtw_matrix,
        training_labels=training_set['labels'],
        class_labels=class_repr['labels'],
        threshold=threshold
    )

    print("Done, no errors")




