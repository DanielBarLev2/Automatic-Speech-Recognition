from matplotlib import pyplot as plt

from classes.DTW import DTW
from classes.CTC import CTC
from classes.MelSpectrogram import MelSpectrogram
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
    class_repr, training_set, evaluation_set = prepare_audio(update=True)

    class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=True)

    # ctc = CTC()
    # print(ctc.pred)
    # prob, mat = ctc.word_prob("aba")
    # print(prob)
    # print(mat)
    # prob, seq, mat = ctc.word_prob_for_force_alignment("aba")
    # print(prob)
    # print(seq)
    # print(mat)

    # Initialize DTW
    dtw = DTW(class_repr_ms, training_set_ms)

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
    # dtw.classify_and_plot_confusion_matrix(
    #     dtw_matrix=dtw_matrix,
    #     training_labels=training_set['labels'],
    #     class_labels=class_repr['labels'],
    #     threshold=threshold
    # )

    print("Done, no errors")




