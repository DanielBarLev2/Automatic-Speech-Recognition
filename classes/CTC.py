import torch
from itertools import product

class CTC:
    def __init__(self):
        """
        Initialize the CTC class.


        """
        self.pred = torch.zeros(size=(5, 3), dtype=torch.float32)
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
        self.dictionary = {'a': 0, 'b': 1, '^': 2}


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


    def word_prob(self, input_word):
        """
        The probability to get the word

        Args:
            input_word (String): the word

        Returns:
            Float: The probability to get the word
            Torch: The probability matrix
        """
        num_frames = 5
        padded_word = f"^{'^'.join(input_word)}^"
        prob_mat = torch.zeros(size=(len(padded_word), num_frames), dtype=torch.float32)
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

    def word_prob_for_force_alignment(self, input_word):
        """
        The probability to get the word

        Args:
            input_word (String): the word

        Returns:
            Float: The probability to get the word by force alignment
            String: The sequence that achieve the most probability
            list: The probability matrix
        """
        num_frames = 5
        padded_word = f"^{'^'.join(input_word)}^"
        prob_mat = torch.zeros(size=(len(padded_word), num_frames), dtype=torch.float32).tolist()
        for i in range(len(padded_word)):
            for j in range(num_frames):
                prob_mat[i][j] = {"paths": [], "probs": []}
        prob_mat[0][0] = {
            "paths": [padded_word[0]],
            "probs": [float(self.pred[0][self.dictionary[padded_word[0]]])]
        }
        prob_mat[1][0] = {
            "paths": [padded_word[1]],
            "probs": [float(self.pred[0][self.dictionary[padded_word[1]]])]
        }

        for i in range(len(padded_word)):
            for j in range(1, num_frames):
                if i == 0:
                    char = padded_word[i]
                    prob_char = float(self.pred[j][self.dictionary[char]])
                    paths = []
                    probs = []
                    for l, path in enumerate(prob_mat[i][j-1]["paths"]):
                        paths.append(path+char)
                        probs.append(prob_mat[i][j-1]["probs"][l]*prob_char)
                    prob_mat[i][j] = {"paths": paths, "probs": probs}
                elif i == 1 or j == num_frames-1 or padded_word[i] == "^":
                    char = padded_word[i]
                    prob_char = float(self.pred[j][self.dictionary[char]])
                    paths = []
                    probs = []
                    for l, path in enumerate(prob_mat[i][j-1]["paths"]):
                        paths.append(path+char)
                        probs.append(prob_mat[i][j-1]["probs"][l]*prob_char)
                    for l, path in enumerate(prob_mat[i-1][j-1]["paths"]):
                        paths.append(path+char)
                        probs.append(prob_mat[i-1][j-1]["probs"][l]*prob_char)
                    prob_mat[i][j] = {"paths": paths, "probs": probs}
                else:
                    char = padded_word[i]
                    prob_char = float(self.pred[j][self.dictionary[char]])
                    paths = []
                    probs = []
                    for l, path in enumerate(prob_mat[i][j-1]["paths"]):
                        paths.append(path+char)
                        probs.append(prob_mat[i][j-1]["probs"][l]*prob_char)
                    for l, path in enumerate(prob_mat[i-1][j-1]["paths"]):
                        paths.append(path+char)
                        probs.append(prob_mat[i-1][j-1]["probs"][l]*prob_char)
                    for l, path in enumerate(prob_mat[i-2][j-1]["paths"]):
                        paths.append(path+char)
                        probs.append(prob_mat[i-2][j-1]["probs"][l]*prob_char)
                    prob_mat[i][j] = {"paths": paths, "probs": probs}


        max_prob = 0
        max_prob_seq = ""
        for i, path in enumerate(prob_mat[-1][-1]["paths"]):
            if prob_mat[-1][-1]["probs"][i] > max_prob:
                max_prob = prob_mat[-1][-1]["probs"][i]
                max_prob_seq = path
        return max_prob, max_prob_seq, prob_mat



    def BInverse(self, word):
        """
        The inverse of B

        Args:
            word (String): the desired word

        Returns:
            List: All the sequence that will return the word after applying B
        """
        # Define the characters and the length of the string
        chars = ['a', 'b', '^']
        length = 5

        # Use itertools.product to generate all combinations
        all_strings = [''.join(comb) for comb in product(chars, repeat=length)]
        sequences = []
        for sequence in all_strings:
            if self.B(sequence) == word:
                sequences.append(sequence)
        return sequences

    def B(self, input_sequence):
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