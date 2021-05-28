from gammatone.gtgram import gtgram
from gammatone.fftweight import fft_gtgram

import numpy as np
import librosa
import scipy
import tensorflow as tf
import apkit

def resample_signal(stereo_signal, nb_samples=16000):
    signal1 = stereo_signal[:, 0]
    signal2 = stereo_signal[:, 1]

    signal1 = np.array(scipy.signal.resample(signal1, nb_samples))
    signal2 = np.array(scipy.signal.resample(signal2, nb_samples))
    signal = np.stack((signal1, signal2), axis=-1)

    return signal


def get_fbanks_gcc(signal, fs, win_size=2048, hop_size=1024, nfbank=40, zoom=25, eps=1e-8):
    signal = np.transpose(signal)

    _FREQ_MAX = 8000
    _FREQ_MIN = 100

    tf = apkit.stft(signal, apkit.cola_hamming, win_size, hop_size)
    nch, nframe, _ = tf.shape

    # trim freq bins
    nfbin = int(_FREQ_MAX * win_size / fs)  # 0-8kHz
    freq = np.fft.fftfreq(2048)
    freq = freq[:nfbin]
    tf = tf[:, :, :nfbin]

    # compute pairwise gcc on f-banks
    ecov = apkit.empirical_cov_mat(tf, fw=1, tw=1)
    fbw = apkit.mel_freq_fbank_weight(nfbank, freq, fs, fmax=_FREQ_MAX,
                                      fmin=_FREQ_MIN)
    fbcc = apkit.gcc_phat_fbanks(ecov, fbw, zoom, freq, eps=eps)

    # merge to a single numpy array, indexed by 'tpbd'
    #                                           (time, pair, bank, delay)
    feature = np.asarray([fbcc[(i, j)] for i in range(nch)
                          for j in range(nch)
                          if i < j])

    feature = np.squeeze(feature, axis=0)
    feature = np.moveaxis(feature, 2, 0)

    # and map [-1.0, 1.0] to 16-bit integer, to save storage space
    dtype = np.int16
    vmax = np.iinfo(dtype).max
    feature = (feature * vmax).astype(dtype)

    feature = np.expand_dims(feature, axis=0)
    return feature




def get_fft_gram(signal,  time_window=0.01, channels=128, freq_min=20, fs=16000):
    """
    Calculate a spectrogram-like time frequency magnitude array based on
    gammatone subband filters.
    """
    assert signal.shape[1] == 2
    right_channel = signal[:, 0]
    left_channel = signal[:, 1]


    thop = time_window / 2

    fft_gram_right = fft_gtgram(right_channel, fs, time_window, thop, channels, freq_min)
    fft_gram_left = fft_gtgram(left_channel, fs, time_window, thop, channels, freq_min)

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

def format_input_channels(gammagram_right, gammagram_left):
    input_x = np.stack((gammagram_right[:, :128], gammagram_left[:, :128]), axis=-1)
    input_x = np.expand_dims(input_x, axis=0)

    return input_x


def format_input_side(gammagram_right, gammagram_left):
    input_x = np.hstack((gammagram_right[:, :144], gammagram_left[:, :144]))
    input_x = np.expand_dims(input_x, axis=-1)
    input_x = np.expand_dims(input_x, axis=0)

    return input_x


def get_model(input_shape=(128, 288, 1), output_shape=4, regression=False, L1_REGULARIZATION=1e-5,
L2_REGULARIZATION=1e-5):
    activation_output = "softmax"

    if regression:
        output_shape = 1
        activation_output = "linear"

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),


        tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2,),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),


        tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(5, 5), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.MaxPooling2D((1,1)),
        tf.keras.layers.Dropout(0.25),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512 , activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256 , activation="relu", kernel_regularizer=tf.keras.regularizers.l1_l2(L1_REGULARIZATION, L2_REGULARIZATION)),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(output_shape, activation=activation_output)


    ])
    return model