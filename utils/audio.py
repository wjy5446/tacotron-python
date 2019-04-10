import numpy as np
import scipy
from copy import deepcopy
import librosa

def load_audio(path, sample_rate):
    audio, sample_rate = librosa.load(path, sample_rate)
    return audio, sample_rate

def save_audio(path, audio, sample_rate):
    librosa.output.write_wav(path, audio, sample_rate)


def get_spectrogram(audio, sampling_rate):
    y, _ = librosa.effects.trim(audio)
    y = _pre_emphasis(y, 0.97)

    # parameter
    n_fft = 2048
    sampling_rate = 22050
    frame_hop = 0.0125
    frame_len = 0.05
    hop_length = int(sampling_rate * frame_hop)
    win_length = int(sampling_rate * frame_len)
    ref_db = 20
    max_db = 100
    n_mels = 80

    # Linear spectrogram
    linear = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(linear)
    mel = _mag2mel(mag, sampling_rate, n_fft, n_mels)

    mag = _amp_to_db(mag)
    mel = _amp_to_db(mel)

    mag = _normalize(mag, ref_db, max_db)
    mel = _normalize(mel, ref_db, max_db)

    mag = mag.T.astype(np.float32)
    mel = mel.T.astype(np.float32)

    return mag, mel

def mag2wav(mag):
    # parameter
    n_fft = 2048
    sampling_rate = 22050
    frame_hop = 0.0125
    frame_len = 0.05
    hop_length = int(sampling_rate * frame_hop)
    win_length = int(sampling_rate * frame_len)

    ref_db = 20
    max_db = 100

    mag = mag.T
    mag = _de_normalize(mag, ref_db, max_db)
    mag = _db_to_amp(mag)

    wav = _griffin_lim(mag, n_fft, hop_length, win_length)
    wav = _de_emphasis(wav, 0.97)
    wav, _ = librosa.effects.trim(wav)
    return wav

######
# simple code
######

def _pre_emphasis(signal, coeff):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def _de_emphasis(signal, coeff):
    return scipy.signal.lfilter([1], [1, -coeff], signal)

def _mag2mel(mag, sample_rate, n_fft, n_mels):
    mel_matrix = librosa.filters.mel(sample_rate, n_fft, n_mels)
    mel = np.dot(mel_matrix, mag)
    return mel

def _amp_to_db(signal, maximum=1e-5):
    return 20 * np.log10(np.maximum(maximum, signal))

def _db_to_amp(signal_db):
    return np.power(10.0, signal_db / 20.0)

def _normalize(signal_db, ref_db, max_db):
    return np.clip((signal_db - ref_db + max_db) / max_db, a_min=1e-8, a_max=1)

def _de_normalize(signal, ref_db, max_db):
    return (np.clip(signal, a_min=0, a_max=1) * max_db) - max_db + ref_db


def _griffin_lim(mag, n_fft, hop_length, win_length):
    X_best = deepcopy(mag)

    for _ in range(200):
        X_t = librosa.istft(X_best, hop_length=hop_length,
                            win_length=win_length, window='hann')
        est = librosa.stft(X_t, n_fft, hop_length, win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = mag * phase

    X_t = librosa.istft(X_best, hop_length=hop_length,
                        win_length=win_length, window='hann')

    y = np.real(X_t)

    return y