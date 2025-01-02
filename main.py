import torch

from classes.DTW import DTW
from classes.MelSpectrogram import MelSpectrogram
from config.config import Config
from src.preparation import prepare_audio, prepare_mel_spectrogram

if __name__ == "__main__":
    print(f'Using: {Config.DEVICE}')

    class_repr, training_set, evaluation_set = prepare_audio(update=False)

    class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=False)




    # Initialize DTW
    dtw = DTW(class_repr_ms, training_set_ms)

    # Compute DTW distance matrix
    dtw_matrix = dtw.compute_distance_matrix()

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




