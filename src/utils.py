from gammatone.gtgram import gtgram
import numpy as np
import librosa


def get_fft_gram(signal,  time_window=0.08, channels=128, freq_min=120):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    gammatone subband filters.
    """
    assert signal.shape[1] == 2
    right_channel = signal[:, 0]
    left_channel = signal[:, 1]

    thop = time_window / 2

    fft_gram_right = fft_gtgram(right_channel, 16000, time_window, thop, channels, freq_min)
    fft_gram_left = fft_gtgram(left_channel, 16000, time_window, thop, channels, freq_min)

    fft_gram_right = np.flipud(20 * np.log10(fft_gram_right))
    fft_gram_left = np.flipud(20 * np.log10(fft_gram_left))

    return fft_gram_right, fft_gram_left


def format_signal(audio_list_samples):
    """
    Format an audio given a list of samples
    :param audio_list_samples:
    :return: numpy array
    """
    np_audio = np.concatenate(audio_list_samples, axis=1)
    np_audio = librosa.util.normalize(np_audio, axis=1)
    np_audio = np.squeeze(np_audio)
    signal = np.transpose(np_audio, (1, 0))

    return signal