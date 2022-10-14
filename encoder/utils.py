import librosa
from scipy.signal import lfilter
import numpy as np

from encoder.sigproc import preemphasis, framesig


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)
    end_frame = int(max_sec * frames_per_sec)
    step_frame = int(step_sec * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets


def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m

def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout


def normalize_frames(m, epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


def get_fft_spectrum(filename, buckets, sample_rate, n_fft, frame_len, frame_step, preemphasis_alpha):
    signal = load_wav(filename, sample_rate)
    signal *= 2 ** 15

    min_len = int(frame_step * sample_rate * list(buckets.keys())[0] + frame_len * sample_rate)
    if signal.size < min_len:
        signal = np.pad(signal, (0, min_len - signal.size), 'constant', constant_values=0)

        # get FFT spectrum
    signal = remove_dc_and_dither(signal, sample_rate)
    # signal = sigproc.preemphasis(signal, coeff=preemphasis_alpha)
    # frames = sigproc.framesig(signal, frame_len=frame_len*sample_rate, frame_step=frame_step*sample_rate, winfunc=np.hamming)
    signal = preemphasis(signal, coeff=preemphasis_alpha)
    frames = framesig(signal, frame_len=frame_len * sample_rate, frame_step=frame_step * sample_rate,
                      winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=n_fft))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1] - rsize) / 2)
    out = fft_norm[:, rstart:rstart + rsize]

    return out


def get_fft_spectrum_from_array(signal, buckets, sample_rate, n_fft, frame_len, frame_step, preemphasis_alpha):
    # signal = load_wav(filename,sample_rate)
    signal *= 2 ** 15

    min_len = int(frame_step * sample_rate * list(buckets.keys())[0] + frame_len * sample_rate)
    if signal.size < min_len:
        signal = np.pad(signal, (0, min_len - signal.size), 'constant', constant_values=0)

        # get FFT spectrum
    signal = remove_dc_and_dither(signal, sample_rate)
    # signal = sigproc.preemphasis(signal, coeff=preemphasis_alpha)
    # frames = sigproc.framesig(signal, frame_len=frame_len*sample_rate, frame_step=frame_step*sample_rate, winfunc=np.hamming)
    signal = preemphasis(signal, coeff=preemphasis_alpha)
    frames = framesig(signal, frame_len=frame_len * sample_rate, frame_step=frame_step * sample_rate,
                      winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=n_fft))
    fft_norm = normalize_frames(fft.T)

    # truncate to max bucket sizes
    # print(fft_norm.shape)
    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1] - rsize) / 2)
    out = fft_norm[:, rstart:rstart + rsize]

    return out