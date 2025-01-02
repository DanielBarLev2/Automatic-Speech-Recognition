from config.config import Config
from src.preparation import prepare_audio, prepare_mel_spectrogram

if __name__ == "__main__":
    print(f'Using: {Config.DEVICE}')

    class_repr, training_set, evaluation_set = prepare_audio()

    class_repr_ms, training_set_ms, evaluation_set_ms = prepare_mel_spectrogram(class_repr,
                                                                                training_set,
                                                                                evaluation_set,
                                                                                update=True)
    print("Done, no errors")


    # # display 5 samples of a male speaker
    # mel_spec.display_samples(training_set['audio'][10:], num_samples=5)
    # # display 5 samples of a female speaker
    # mel_spec.display_samples(training_set['audio'][30:], num_samples=5)

