import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from config.config import Config


class DTW:
    def __init__(self, class_repr_ms: torch.tensor, training_set_ms: torch.tensor):
        """
        Initialize DTW with class representative and training set Mel spectrograms.

        :param class_repr_ms: Mel spectrograms for class representatives
                                (torch.Tensor of shape (num, 1, n_mels, time_steps)).
        :param training_set_ms: Mel spectrograms for training set
                                (torch.Tensor of shape (num, 1, n_mels, time_steps)).
        """

        self.class_repr_ms = class_repr_ms
        self.training_set_ms = training_set_ms


    def compute_distance_matrix(self) -> torch.Tensor:
        """
        Compute the DTW distance matrix between the training set and class representatives.

        :return: Distance matrix (torch.Tensor of shape (training_set_size, class_repr_size)).
        """
        training_size = self.training_set_ms.size(0)
        class_repr_size = self.class_repr_ms.size(0)

        dtw_matrix = torch.zeros(training_size, class_repr_size, device=Config.DEVICE)

        for i in range(training_size):
            for j in range(class_repr_size):
                train_seq = self.training_set_ms[i, 0]  # Shape: (n_mels, time_steps_train)
                repr_seq = self.class_repr_ms[j, 0]     # Shape: (n_mels, time_steps_repr)
                dtw_matrix[i, j] = self.compute_dtw_distance(train_seq, repr_seq)

        return dtw_matrix

    @staticmethod
    def compute_dtw_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
        """
        Compute the DTW distance between two sequences.

        :param seq1: Mel spectrogram 1 (torch.Tensor of shape (n_mels, time_steps1)).
        :param seq2: Mel spectrogram 2 (torch.Tensor of shape (n_mels, time_steps2)).
        :return: DTW distance (float).
        """
        time_steps1, time_steps2 = seq1.size(1), seq2.size(1)

        # Compute distance matrix
        distance_matrix = torch.cdist(seq1.T, seq2.T)  # Shape: (time_steps1, time_steps2)

        # Initialize cumulative cost matrix
        cumulative_cost = torch.full_like(distance_matrix, float('inf'))
        cumulative_cost[0, 0] = distance_matrix[0, 0]

        # Fill cumulative cost matrix
        for i in range(1, time_steps1):
            cumulative_cost[i, 0] = cumulative_cost[i - 1, 0] + distance_matrix[i, 0]

        for j in range(1, time_steps2):
            cumulative_cost[0, j] = cumulative_cost[0, j - 1] + distance_matrix[0, j]

        for i in range(1, time_steps1):
            cumulative_cost[i, 1:] = (distance_matrix[i, 1:]
                                      + torch.min(torch.stack([cumulative_cost[i - 1, 1:],
                                                               cumulative_cost[i, :-1],
                                                               cumulative_cost[i - 1, :-1]]), dim=0).values)

        # Normalize by path length
        dtw_distance = cumulative_cost[-1, -1]
        path_length = time_steps1 + time_steps2
        return (dtw_distance / path_length).item()

    @staticmethod
    def classify_and_calculate_accuracy(dtw_matrix: torch.Tensor, training_labels: torch.Tensor,
                                        class_labels: torch.Tensor, threshold: float):
        """
        Classify training recordings based on DTW distances and calculate accuracy.

        :param dtw_matrix: DTW distance matrix (torch.Tensor of shape (training_set_size, class_repr_size)).
        :param training_labels: Ground truth labels for the training set (torch.Tensor of shape (training_set_size,)).
        :param class_labels: Ground truth labels for the class representatives (torch.Tensor of shape (class_repr_size,)).
        :param threshold: Distance threshold for classification.
        :return: Accuracy (float) and predicted labels (torch.Tensor of shape (training_set_size,)).
        """
        # Find the closest match for each training recording
        min_distances, closest_indices = torch.min(dtw_matrix, dim=1)  # (training_set_size,)

        # Apply the threshold to determine classification
        predicted_labels = torch.where(
            min_distances <= threshold,
            class_labels[closest_indices],
            torch.tensor(-1, device=dtw_matrix.device)  # Assign -1 for "unknown" classification
        )

        # Calculate accuracy
        correct_predictions = (predicted_labels == training_labels).sum().item()
        accuracy = correct_predictions / training_labels.size(0)

        return accuracy, predicted_labels

    @staticmethod
    def classify_and_plot_confusion_matrix(dtw_matrix: torch.Tensor, training_labels: torch.Tensor, class_labels: torch.Tensor, threshold: float):
        """
        Classify training recordings and plot the confusion matrix.

        :param dtw_matrix: DTW distance matrix (torch.Tensor of shape (training_set_size, class_repr_size)).
        :param training_labels: Ground truth labels for the training set (torch.Tensor).
        :param class_labels: Ground truth labels for the class representatives (torch.Tensor).
        :param threshold: Distance threshold for classification.
        """
        min_distances, closest_indices = torch.min(dtw_matrix, dim=1)
        predicted_labels = torch.where(
            min_distances <= threshold,
            class_labels[closest_indices],
            torch.tensor(-1, device=dtw_matrix.device)  # Assign -1 for "unknown"
        )

        # Convert tensors to numpy for confusion matrix
        true_labels = training_labels.cpu().numpy()
        predicted_labels = predicted_labels.cpu().numpy()
        class_labels = class_labels.cpu().numpy()

        # Compute confusion matrix
        conf_mat = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Classification Accuracy: {accuracy * 100:.2f}%")

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()