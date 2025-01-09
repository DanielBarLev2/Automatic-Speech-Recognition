from src.DTW import *
from src.CTC import *
from src.preparation import prepare_audio, prepare_mel_spectrogram, preprocess_tensors
import pickle as pkl

def main():
    # 1. Data Acquisition:
    class_repr, training_set, evaluation_set = prepare_audio(update=False)

    # 2. Mel Spectrogram:
    class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=False)

    class_repr_ms, training_set_ms, evaluation_set_ms = preprocess_tensors(class_repr_ms=class_repr_ms,
                                                                           training_set_ms=training_set_ms,
                                                                           evaluation_set_ms=evaluation_set_ms)

    # 3. Dynamic time warping
    # 3.a Select the class representative recordings as the reference database: class_repr_ms

    # 3.b Implement the DTW algorithm that was described in the lecture.
    # 3.c Compare each audio recording in the training set with each of the audios in the DB using DTW algorithm.
    # 3.d Construct a distance matrix:
    dtw_dist_matrix_train = compute_distance_matrix_dtw(class_repr_ms, training_set_ms)

    # 3.e Show the distance matrix:
    display_distance_matrix(dtw_dist_matrix_train)

    # 3.f Set Threshold and Determine Classification:
    threshold = find_optimal_threshold(dtw_dist_matrix_train)
    print("Optimal Threshold:", threshold)

    # Calculate the accuracy over the training set.
    pred_labels_train = classify_with_threshold(dtw_dist_matrix_train, threshold)
    train_accuracy = compute_accuracy(pred_labels_train)
    print("Training Set Accuracy:", train_accuracy)

    # 3.g Apply Threshold on Validation Set:
    dtw_dist_matrix_eval = compute_distance_matrix_dtw(class_repr_ms, evaluation_set_ms)

    pred_labels_eval = classify_with_threshold(dtw_dist_matrix_eval, threshold)
    eval_accuracy = compute_accuracy(pred_labels_eval)
    print("Validation Set Accuracy:", eval_accuracy)

    # 3.h Plot the confusion matrix:
    confusion_matrix = confusion_matrix_10x10(pred_labels_eval)
    plot_confusion_matrix(confusion_matrix, title='Confusion Matrix (Validation Set)')


    # 4. implement the collapse function B
    sequence_example = "^ab^a"
    collapse_func_example = CTC.B(sequence_example)
    print(f"for example {sequence_example} collapse after B to {collapse_func_example}")

    # 5.  Implement the forward pass of the CTC algorithm
    # 5.a set the pred matrix
    # 5.b set the mapping
    ctc = CTC()

    # 5.c calculate the probability for "aba"
    prob, forward_mat = ctc.word_prob("aba")
    print(f"the probability for aba is: {round(prob, 3)}")

    # 5.d plot the pred and forward matrix
    ctc.display_pred_matrix()
    ctc.display_ctc_matrix(mat=forward_mat)

    # 6.  Implement the forward pass of the CTC algorithm for force alignment
    # 6.a replace to sum operator
    prob, most_prob_path, prob_mat, back_mat = ctc.word_prob_for_force_alignment("aba")

    # 6.b the most probable path
    print(f"the most probable path for aba is: {most_prob_path}")

    # 6.c the probability of the path
    print(f"the probability of {most_prob_path} is: {prob}")

    # 6.d plot the forward matrix
    ctc.display_ctc_matrix(mat=prob_mat, seq=most_prob_path)

    # 6.e plot the backward matrix
    ctc.display_ctc_matrix(mat=back_mat, seq=most_prob_path, backtrace="backtrace")

    # 7.  repeat q.6 for given data
    data = pkl.load(open('force_align.pkl', 'rb'))
    label_mapping = data["label_mapping"]
    label_mapping = {value: key for key, value in label_mapping.items()}
    audio = data["audio"]
    acoustic_model_out_probs = data["acoustic_model_out_probs"]
    gt_text = data["gt_text"]
    text_to_align = data["text_to_align"]
    padded_word = list(f"^{'^'.join(text_to_align)}^")
    ctc = CTC(acoustic_model_out_probs, label_mapping)


    prob, most_prob_path, prob_mat, back_mat = ctc.word_prob_for_force_alignment(text_to_align)
    print(f"the most probable path for {text_to_align} is: {most_prob_path}")
    print(f"the probability of {most_prob_path} is: {prob}")
    ctc.display_ctc_matrix(mat=prob_mat, seq=most_prob_path, text=padded_word)
    ctc.display_ctc_matrix(mat=back_mat, seq=most_prob_path, backtrace="backtrace", text=padded_word)

if __name__ == '__main__':
    main()



