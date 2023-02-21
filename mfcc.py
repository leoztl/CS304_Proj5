import librosa
import numpy as np


def cepstra_normalization(mel_ceptrum):
    ## mean normalization
    (nframes, ncoeff) = mel_ceptrum.shape
    total_cep = np.sum(mel_ceptrum, axis=0)
    mean_cep = total_cep / nframes
    cepstra_mean_normal = mel_ceptrum - mean_cep

    ## variance normalization
    std = np.sqrt(np.sum(np.square(cepstra_mean_normal), axis=0) / nframes)
    cepstra_normal = 1 / std * cepstra_mean_normal
    return cepstra_normal


def velocity(cepstra_normal):
    n_frames = len(cepstra_normal)
    denominator = 2 * sum([i ** 2 for i in range(1, 2)])
    vel_feat = np.empty_like(cepstra_normal)
    padded = np.pad(cepstra_normal, ((1, 1), (0, 0)), mode='edge')
    for t in range(n_frames):
        vel_feat[t] = np.dot(np.arange(-1, 2), padded[t : t + 3]) / denominator

    return vel_feat


def acceleration(cepstra_normal):
    n_frames = len(cepstra_normal)
    denominator = 2 * sum([i ** 2 for i in range(1, 3)])
    acce_feat = np.empty_like(cepstra_normal)
    padded = np.pad(cepstra_normal, ((2, 2), (0, 0)), mode='edge')
    for t in range(n_frames):
        acce_feat[t] = np.dot(np.arange(-2, 3), padded[t : t + 5]) / denominator
    return acce_feat


def mfcc_features(filename, num_filter):
    y, sr = librosa.load(filename, sr=16000)
    melcep = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, n_mels=num_filter).T
    cepstra_normal = cepstra_normalization(melcep)
    vel_feat = velocity(cepstra_normal)
    acce_feat = acceleration(cepstra_normal)
    mfcc = np.hstack((cepstra_normal, vel_feat, acce_feat))
    return mfcc
