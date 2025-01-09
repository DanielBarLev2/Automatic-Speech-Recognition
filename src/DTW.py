"""
The `@njit` decorator from the `numba` library is used to optimize computationally
intensive functions for faster execution by compiling them to machine code. This
is particularly beneficial for operations involving nested loops, such as the
DTW distance computation.
"""
import os

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def frame_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    :param vector1: np.ndarray, First vector.
    :param vector2: np.ndarray, Second vector.
    :return: float, Euclidean distance between vec1 and vec2.
    """
    distance = 0.0
    for i in range(vector1.shape[0]):
        diff = vector1[i] - vector2[i]
        distance += diff * diff
    return np.sqrt(distance)


@njit
def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute the Dynamic Time Warping (DTW) distance between two sequences.

    :param seq1: np.ndarray, First sequence of shape (features, time_steps).
    :param seq2: np.ndarray, Second sequence of shape (features, time_steps).
    :return: float, DTW distance between seq1 and seq2.
    """
    n = seq1.shape[1]
    m = seq2.shape[1]

    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = frame_distance(seq1[:, i - 1], seq2[:, j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])

    distance = dtw_matrix[n, m]
    avg_length = (n + m) / 2.0
    normalized_distance = distance / avg_length

    return float(normalized_distance)


def compute_distance_matrix_dtw(class_repr_ms: np.ndarray, training_set_ms: np.ndarray) -> np.ndarray:
    """
    Compute the DTW distance matrix between reference classes and data samples.

    :param class_repr_ms: np.ndarray, Reference class spectrograms (1, num_ref, features, time_steps).
    :param training_set_ms: np.ndarray, Data spectrograms (num_speakers, num_digits, features, time_steps).
    :return: np.ndarray, Distance matrix of shape (num_speakers, num_digits, num_ref).
    """
    num_speakers = training_set_ms.shape[0]
    num_digits = training_set_ms.shape[1]
    num_ref = class_repr_ms.shape[1]

    distances = np.zeros(shape=(num_speakers, num_digits, num_ref), dtype=np.float32)

    for speaker in range(num_speakers):
        for digit in range(num_digits):
            mel = training_set_ms[speaker, digit]
            for ref_digit in range(num_ref):
                mel_repr = class_repr_ms[0, ref_digit]
                distances[speaker, digit, ref_digit] = dtw_distance(mel, mel_repr)

    return distances


def display_distance_matrix(distance_matrix: np.ndarray) -> None:
    """
    Display the DTW distance matrix as a heatmap.

    :param distance_matrix: Distance matrix of shape (num_speakers, num_digits, num_ref).
    """
    for k in range(distance_matrix.shape[2]):
        rounded_matrix = np.round(distance_matrix[:,:,k])
        num_speakers, num_digits = rounded_matrix.shape
        plt.figure(figsize=(num_digits, num_speakers))

        plt.imshow(rounded_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label="DTW Distance")
        plt.title(f"DTW Distance Matrix for digit {k}")
        plt.xlabel("Digits")
        plt.ylabel("Speakers")

        plt.xticks(ticks=np.arange(num_digits), labels=[f"Digit {i}" for i in range(num_digits)])
        plt.yticks(ticks=np.arange(num_speakers), labels=[f"Speaker {i + 1}" for i in range(num_speakers)])

        # Annotate the cells with the rounded distance values
        for i in range(num_speakers):
            for j in range(num_digits):
                plt.text(j, i, f"{rounded_matrix[i, j]:.0f}",
                         ha='center', va='center',
                         color='white' if rounded_matrix[i, j] > rounded_matrix.max() / 2 else 'black')

        plt.tight_layout()

        # Define the save directory, skips if already exists
        if not os.path.exists("results"):
            os.makedirs("results")

        plt.savefig(f"results/DTW Distance Matrix for digit {k}.png")
        plt.show()


def find_optimal_threshold(dtw_dist_matrix: np.ndarray) -> float:
    """
    Find the optimal classification threshold using correct and incorrect DTW distances.

    :param dtw_dist_matrix: np.ndarray, DTW distance matrix of shape (num_speakers, num_digits, num_ref).
    :return: float, Optimal classification threshold.
    """
    correct_distances = []
    incorrect_distances = []

    num_speakers = dtw_dist_matrix.shape[0]
    num_digits = dtw_dist_matrix.shape[1]

    for speaker in range(num_speakers):
        for digit in range(num_digits):
            dist_correct = dtw_dist_matrix[speaker, digit, digit]
            correct_distances.append(dist_correct)

            for ref_digit in range(num_digits):
                if ref_digit != digit:
                    incorrect_distances.append(dtw_dist_matrix[speaker, digit, ref_digit])

    correct_distances = np.array(correct_distances)
    incorrect_distances = np.array(incorrect_distances)

    mean_correct = np.mean(correct_distances)
    mean_incorrect = np.mean(incorrect_distances)
    threshold = 0.5 * (mean_correct + mean_incorrect)

    return threshold


def classify_with_threshold(dtw_dist_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Classify data samples based on a given threshold.

    :param dtw_dist_matrix: np.ndarray, DTW distance matrix of shape (num_speakers, num_digits, num_ref).
    :param threshold: float, Classification threshold.
    :return: np.ndarray, Predicted labels of shape (num_speakers, num_digits).
    """
    num_speakers = dtw_dist_matrix.shape[0]
    num_digits = dtw_dist_matrix.shape[1]

    predicted_labels = np.full((num_speakers, num_digits), -1, dtype=int)

    for speaker in range(num_speakers):
        for digit in range(num_digits):
            dists = dtw_dist_matrix[speaker, digit, :]
            min_idx = np.argmin(dists)
            if dists[min_idx] < threshold:
                predicted_labels[speaker, digit] = min_idx

    return predicted_labels


def compute_accuracy(predicted_labels: np.ndarray) -> float:
    """
    Compute the classification accuracy.

    :param predicted_labels: np.ndarray, Predicted labels of shape (num_speakers, num_digits).
    :return: float, Classification accuracy.
    """
    num_speakers = predicted_labels.shape[0]
    num_digits = predicted_labels.shape[1]

    correct = 0
    total = num_speakers * num_digits

    for speaker in range(num_speakers):
        for digit in range(num_digits):
            if predicted_labels[speaker, digit] == digit:
                correct += 1

    return correct / total


def confusion_matrix_10x10(predicted_labels: np.ndarray) -> np.ndarray:
    """
    Compute a 10x10 confusion matrix.

    :param predicted_labels: np.ndarray, Predicted labels of shape (num_speakers, num_digits).
    :return: np.ndarray, Confusion matrix of shape (10, 10).
    """
    confusion_matrix = np.zeros((10, 10), dtype=np.int32)
    num_speakers = predicted_labels.shape[0]
    num_digits = predicted_labels.shape[1]

    for speaker in range(num_speakers):
        for true_digit in range(num_digits):
            pred_digit = predicted_labels[speaker, true_digit]
            if 0 <= pred_digit < 10:
                confusion_matrix[true_digit, pred_digit] += 1
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix: np.ndarray, title: str = 'Confusion Matrix') -> None:
    """
    Plot a confusion matrix.

    :param confusion_matrix: np.ndarray, Confusion matrix of shape (10, 10).
    :param title: str, Title for the plot.
    """
    plt.figure(figsize=confusion_matrix.shape)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Annotate the cells with the rounded distance values
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[0]):
            plt.text(j, i, f"{confusion_matrix[i, j]:.0f}",
                     ha='center', va='center',
                     color='white' if confusion_matrix[i, j] > confusion_matrix.max() / 2 else 'black')

    plt.tight_layout()

    # Define the save directory, skips if already exists
    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(f"results/{title}.png")
    plt.show()