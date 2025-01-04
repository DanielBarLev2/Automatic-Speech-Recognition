import torch

class DTW:
    def __init__(self, class_repr_ms, training_set_ms):
        """
        Initialize the DTW class.


        :param class_repr_ms: the representative of each digit.
        :param training_set_ms: the training data.
        """
        self.class_repr_ms = class_repr_ms
        self.training_set_ms = training_set_ms


    def compute_dtw_distance(self, sequence1, sequence2, dist_func=lambda x, y: torch.norm(x - y)):
        """
        Compute the DTW distance between two sequences using dynamic programming.

        :param sequence1: First sequence (e.g., Mel spectrogram)
        :param sequence2: Second sequence (e.g., Mel spectrogram)
        :param dist_func: Distance function to compare individual elements
        :return: DTW distance and the optimal alignment path
        """
        n, m = sequence1.shape[1], sequence2.shape[1]
        dtw_matrix = torch.full((n+1, m+1), float('inf'))

        # Fill the DTW matrix
        dtw_matrix[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = dist_func(sequence1[:, i-1], sequence2[:, j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
        return dtw_matrix[-1, -1]


    def compute_distance_matrix(self):
        """
        Compute the DTW distance between the training set to the representatives set.
        """
        n, m = self.training_set_ms.shape[0]//10, self.class_repr_ms.shape[0]
        dtw_matrix = torch.full((n, m, m), 0.0)
        training_set_squeezed = torch.squeeze(self.training_set_ms)
        class_repr_squeezed = torch.squeeze(self.class_repr_ms)
        for i in range(n):
            for j in range(m):
                for l in range(m):
                    dtw_matrix[i, j, l] = self.compute_dtw_distance(
                        class_repr_squeezed[l],
                        training_set_squeezed[i*10+j]
                    )
        return dtw_matrix
