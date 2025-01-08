from src.DTW import *
from src.preparation import prepare_audio, prepare_mel_spectrogram, preprocess_tensors


def main():
    # 1. Data Acquisition:
    class_repr, training_set, evaluation_set = prepare_audio(update=True)

    # 2. Mel Spectrogram:
    class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr['audio'],
                                                                                training_set['audio'],
                                                                                evaluation_set['audio'],
                                                                                update=True)

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

if __name__ == '__main__':
    main()

    # ctc = CTC()
    # print(ctc.pred)
    # prob, mat = ctc.word_prob("aba")
    # print(prob)
    # print(mat)
    # prob, seq, mat = ctc.word_prob_for_force_alignment("aba")
    # print(prob)
    # print(seq)
    # print(mat)


