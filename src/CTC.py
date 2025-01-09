import numpy as np
import matplotlib.pyplot as plt

class CTC:
    def __init__(self, pred=None, dictionary=None):
        """
        Initialize the CTC class.

        :param pred: np.ndarray, the acoustic model out probs.
        :return: dictionary: dictionary, label mapping.

        """
        try:
            if pred == None:
                self.pred = np.zeros(shape=(5, 3), dtype=np.float32)
                self.pred[0][0] = 0.8
                self.pred[0][1] = 0.2
                self.pred[1][0] = 0.2
                self.pred[1][1] = 0.8
                self.pred[2][0] = 0.3
                self.pred[2][1] = 0.7
                self.pred[3][0] = 0.09
                self.pred[3][1] = 0.8
                self.pred[3][2] = 0.11
                self.pred[4][2] = 1.00
            if dictionary == None:
                self.dictionary = {'a': 0, 'b': 1, '^': 2}
        except Exception:
            self.pred = pred
            self.dictionary = dictionary


    def sequence_prob(self, input_sequence):
        """
        The probability to get the sequence

        Args:
            input_sequence (String): the sequence

        Returns:
            Float: The probability to get the sequence
        """
        prob = 1
        for i, char in enumerate(input_sequence):
            prob *= self.pred[i][self.dictionary[char]]
        return float(prob)


    def word_prob(self, input_word: str) -> (float, np.ndarray):
        """
        The probability to get the word

        Args:
            input_word (String): the word

        Returns:
            Float: The probability to get the word
            np.ndarray: The probability matrix
        """
        num_frames = self.pred.shape[0]
        padded_word = f"^{'^'.join(input_word)}^"
        prob_mat = np.zeros(shape=(len(padded_word), num_frames), dtype=np.float32)
        prob_mat[0, 0] = self.pred[0][self.dictionary[padded_word[0]]]
        prob_mat[1, 0] = self.pred[0][self.dictionary[padded_word[1]]]
        for i in range(len(padded_word)):
            for j in range(1, num_frames):
                if i == 0:
                    char = padded_word[i]
                    prob_char = self.pred[j][self.dictionary[char]]
                    prob_mat[i, j] = (prob_mat[i, j-1])*prob_char
                elif i == 1 or j == num_frames-1 or padded_word[i] == "^":
                    char = padded_word[i]
                    prob_char = self.pred[j][self.dictionary[char]]
                    prob_mat[i, j] = (prob_mat[i, j-1] + prob_mat[i-1, j-1])*prob_char
                else:
                    char = padded_word[i]
                    prob_char = self.pred[j][self.dictionary[char]]
                    prob_mat[i, j] = (prob_mat[i, j-1] + prob_mat[i-1, j-1] + prob_mat[i-2, j-1])*prob_char
        return float(prob_mat[-1][-1]+prob_mat[-2][-1]), prob_mat

    def word_prob_for_force_alignment(self, input_word: str) -> (float, str, np.ndarray):
        """
        The probability to get the word

        Args:
            input_word (String): the word

        Returns:
            Float: The probability to get the word by force alignment
            String: The sequence that achieve the most probability
            np.ndarray: The probability matrix
            np.ndarray: The backtrace matrix
        """
        num_frames = self.pred.shape[0]
        padded_word = f"^{'^'.join(input_word)}^"
        prob_mat = np.zeros(shape=(len(padded_word), num_frames), dtype=np.float32).tolist()
        back_mat = np.zeros(shape=(len(padded_word), num_frames), dtype=np.float32)
        for i in range(len(padded_word)):
            for j in range(num_frames):
                prob_mat[i][j] = {"path": "", "prob": 0}
        prob_mat[0][0] = {
            "path": padded_word[0],
            "prob": float(self.pred[0][self.dictionary[padded_word[0]]])
        }
        prob_mat[1][0] = {
            "path": padded_word[1],
            "prob": float(self.pred[0][self.dictionary[padded_word[1]]])
        }

        for i in range(len(padded_word)):
            for j in range(1, num_frames):
                if i == 0:
                    char = padded_word[i]
                    prob_char = float(self.pred[j][self.dictionary[char]])
                    prob_mat[i][j]["path"] = prob_mat[i][j - 1]["path"] + char
                    prob_mat[i][j]["prob"] = prob_mat[i][j - 1]["prob"] * prob_char
                elif i == 1 or j == num_frames - 1 or padded_word[i] == "^":
                    char = padded_word[i]
                    prob_char = float(self.pred[j][self.dictionary[char]])
                    if prob_mat[i][j - 1]["prob"] > prob_mat[i-1][j - 1]["prob"]:
                        prob_mat[i][j]["path"] = prob_mat[i][j - 1]["path"] + char
                        prob_mat[i][j]["prob"] = prob_mat[i][j - 1]["prob"] * prob_char
                    else:
                        prob_mat[i][j]["path"] = prob_mat[i-1][j - 1]["path"] + char
                        prob_mat[i][j]["prob"] = prob_mat[i-1][j - 1]["prob"] * prob_char
                else:
                    char = padded_word[i]
                    prob_char = float(self.pred[j][self.dictionary[char]])
                    if prob_mat[i][j - 1]["prob"] > prob_mat[i-1][j - 1]["prob"] and prob_mat[i][j - 1]["prob"] > prob_mat[i-2][j - 1]["prob"]:
                        prob_mat[i][j]["path"] = prob_mat[i][j - 1]["path"] + char
                        prob_mat[i][j]["prob"] = prob_mat[i][j - 1]["prob"] * prob_char
                    elif prob_mat[i-1][j - 1]["prob"] > prob_mat[i][j - 1]["prob"] and prob_mat[i-1][j - 1]["prob"] > prob_mat[i-2][j - 1]["prob"]:
                        prob_mat[i][j]["path"] = prob_mat[i-1][j - 1]["path"] + char
                        prob_mat[i][j]["prob"] = prob_mat[i-1][j - 1]["prob"] * prob_char
                    else:
                        prob_mat[i][j]["path"] = prob_mat[i-2][j - 1]["path"] + char
                        prob_mat[i][j]["prob"] = prob_mat[i-2][j - 1]["prob"] * prob_char
        if prob_mat[-1][-1]["prob"] > prob_mat[-2][-1]["prob"]:
            max_prob =prob_mat[-1][-1]["prob"]
            max_prob_seq =prob_mat[-1][-1]["path"]
        else:
            max_prob =prob_mat[-2][-1]["prob"]
            max_prob_seq =prob_mat[-3][-1]["path"]
        for i in range(len(padded_word)):
            for j in range(num_frames):
                prob_mat[i][j] = prob_mat[i][j]["prob"]
        possible_indxs = [0, 1, 2]
        for i in range(len(max_prob_seq)):
            for j in possible_indxs:
                if j < len(padded_word):
                    if padded_word[j] == max_prob_seq[i]:
                        back_mat[j][i] = prob_mat[j][i]
                        k = j+1
                        l = j+2
                        possible_indxs = [j, k, l]
        prob_mat = np.array(prob_mat, dtype=np.float32)
        return max_prob, max_prob_seq, prob_mat, back_mat

    def display_pred_matrix(self) -> None:
        """
        Display the pred matrix as a heatmap.

        """
        n,m = self.pred.shape
        plt.figure(figsize=(n, m))
        plt.imshow(self.pred, cmap='viridis', interpolation='nearest')
        plt.colorbar(label="probability")
        plt.title("The pred matrix")
        plt.xlabel("letter")
        plt.ylabel("timestep")

        plt.xticks(ticks=np.arange(m), labels=["a", "b", "^"])
        plt.yticks(ticks=np.arange(n), labels=[f"Timestep {i}" for i in range(n)])

        # Annotate the cells with the rounded distance values
        for i in range(n):
            for j in range(m):
                plt.text(j, i, f"{self.pred[i, j]:.2f}",
                     ha='center', va='center',
                     color='black' if self.pred[i, j] > self.pred.max() / 2 else 'white')

        plt.tight_layout()
        plt.savefig(f"results/The pred matrix.png")
        plt.show()


    def display_ctc_matrix(self, mat: np.ndarray, seq: str = "", backtrace: str = "forward", text: list = ["^", "a", "^", "b", "^", "a", "^"]) -> None:
        """
        Display the matrix as a heatmap.

        :param forward_matrix: dorward matrix.
        :param seq: the most probable sequance for the word if given, else none.
        :param backtrace: set to backtrace if you want to print the backtrace matrix
        :param text: the word to check
        """


        n, m = mat.shape
        plt.figure(figsize=(n, m))
        plt.imshow(mat, cmap='viridis', interpolation='nearest')
        plt.colorbar(label="probability")
        plt.title(f"The {backtrace} matrix {seq}")
        plt.xlabel("timestep")
        plt.ylabel("letter")

        plt.yticks(ticks=np.arange(n), labels=text)
        plt.xticks(ticks=np.arange(m), labels=[f"TS {i}" for i in range(m)])

        # Annotate the cells with the rounded distance values
        for i in range(n):
            for j in range(m):
                plt.text(j, i, f"{mat[i, j]:.2f}",
                     ha='center', va='center',
                     color='black' if mat[i, j] > mat.max() / 2 else 'white')

        plt.tight_layout()
        plt.savefig(f"results/The {backtrace} matrix {seq}.png")
        plt.show()


    @staticmethod
    def B(input_sequence: str) -> str:
        """
        Collapses repeated characters in the sequence and removes blank tokens.

        Args:
            input_sequence (String): A String of letters

        Returns:
            String: A collapsed sequence without unwanted repeated characters or blank tokens.
        """
        collapsed_sequence_tmp = ""
        prev_char = None
        blank_symbol = '^'
        for char in input_sequence:
            if char != blank_symbol:
                if char != prev_char:
                    collapsed_sequence_tmp += char
            else:
                collapsed_sequence_tmp += char
            prev_char = char

        collapsed_sequence = ""
        for char in collapsed_sequence_tmp:
            if char != blank_symbol:
                collapsed_sequence += char

        return collapsed_sequence
