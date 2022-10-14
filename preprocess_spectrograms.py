import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
from nlpaug.util.audio.loader import AudioLoader
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
import gc

"""
This script contains the implementation that allows you to generate the spectrograms associated 
with the audio tracks of the datasets and perform data augmentation.

Params:
    --datasets_dir          path of the datasets
    --overwrite             set whether already generated spectrograms should be generated again
    --spectrogram           set if spectrogram must be generated without color map
    --colored_spectrogram   set if spectrogram must be generated with jet color map
    --augmentations         define the types of data augmentation to perform (as a list of type name space-separated)
    --window                define the type of window of the DFT to apply ("hamm" or "hann")
    
Output:
    --spectrograms          spectrogram saved in each datasets directory
"""

parser = argparse.ArgumentParser(description='Vit-for-SER Preprocess Spectrograms')

# Dataset / Model parameters
parser.add_argument('--datasets_dir', metavar='DIR', default='./datasets',
                    help='path to datasets')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='define if overwrite existing augmentated spectrograms')
parser.add_argument('--spectrogram', type=bool, default=True,
                    help='define if generate also raw spectrograms or only augmentated spectrogram')
parser.add_argument('--colored_spectrogram', action='store_true', default=False,
                    help='define if generate also raw spectrograms colored with cmap')
parser.add_argument('--augmentations', nargs='+',
                    default=["loud", "noise", "speed", "pitch", "shift", "norm", "mask", "maskfreq"],
                    help='type of data augmentations to perform '
                         '(from the types "loud", "noise", "speed", "pitch", "shift", "norm", "mask" and "maskfreq"')
parser.add_argument('--window', type=str, default='hamm',
                    help='define type of window of the DFT to apply ("hamm" or "hann")')

def generate_spectrogram(y, sr, save_path, filename, window='hamm'):
    """
    Function that generate the spectrogram
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :return:
    """
    path_save = save_path + '/' + filename + '.jpeg'
    #mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(0.025*sr), hop_length=int(0.010*sr), window=window)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr/2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)
    plt.close()
    gc.collect()


def generate_colored_spectrogram(y, sr, save_path, filename, window='hamm'):
    """
    Function that generate spectrogram colored with jet color map
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :return:
    """
    path_save = save_path + '/' + filename + '.jpeg'
    #mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(0.025*sr), hop_length=int(0.010*sr), window=window)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr/2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    fig.clf()
    plt.clf()
    plt.cla()
    plt.close(fig)
    plt.close()
    gc.collect()


def generate_loudness(y, sr, save_path, filename, window='hamm', colored=False):
    """
    Function that generate loudness augmentation (in the time domain)
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + "/" + filename + '_loud.jpeg'
    aug = naa.LoudnessAug()
    augmented_data = aug.augment(y)[0]
    #mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=int(0.025*sr), hop_length=int(0.010*sr), window=window)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)

def generate_mask(wav_path, save_path, filename, coverage, colored=False):
    """
    Function that generate time mask augmentation (in the frequency domain)
    :param wav_path: input wav path
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param coverage: set percent of the augmentation coverage
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_mask.jpeg'
    data = AudioLoader.load_mel_spectrogram(wav_path, n_mels=128)
    aug = nas.TimeMaskingAug(zone=(0.25, 0.75), coverage=coverage)
    mel_spect_aug = aug.augment(data)[0]
    mel_spect_aug = librosa.power_to_db(mel_spect_aug, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect_aug, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect_aug, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)

def generate_maskfreq(wav_path, save_path, filename, colored=False):
    """
    Function that generate frequency mask augmentation (in the frequency domain)
    :param wav_path: input wav path
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_maskfreq.jpeg'
    data = AudioLoader.load_mel_spectrogram(wav_path, n_mels=128)
    aug = nas.FrequencyMaskingAug(zone=(0, 1), factor=(0, 30))
    mel_spect_aug = aug.augment(data)[0]
    mel_spect_aug = librosa.power_to_db(mel_spect_aug, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect_aug, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect_aug, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)

def generate_noise(y, sr, save_path, filename, window='hamm', colored=False):
    """
    Function that generate noise augmentation (in the time domain)
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_noise.jpeg'
    aug = naa.NoiseAug()
    augmented_data = aug.augment(y)[0]
    mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=int(0.025*sr), hop_length=int(0.010*sr), window=window)
    #mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)


def generate_pitch(y, sr, save_path, filename, window='hamm', colored=False):
    """
    Function that generate pitch augmentation (in the time domain)
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_pitch.jpeg'
    aug = naa.PitchAug(sampling_rate=sr, factor=(2, 3))
    augmented_data = aug.augment(y)[0]
    mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=int(0.025*sr), hop_length=int(0.010*sr), window=window)
    #mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)


def generate_shift(y, sr, save_path, filename, duration=2, window='hamm', colored=False):
    """
    Function that generate shift augmentation (in the time domain)
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_shift.jpeg'
    aug = naa.ShiftAug(sampling_rate=sr, duration=duration)
    augmented_data = aug.augment(y)[0]
    mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=int(0.025 * sr),
                                               hop_length=int(0.010 * sr), window=window)
    # mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)


def generate_speed(y, sr, save_path, filename, window='hamm', colored=False):
    """
    Function that generate speed augmentation (in the time domain)
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_speed.jpeg'
    aug = naa.SpeedAug()
    augmented_data = aug.augment(y)[0]
    mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=int(0.025 * sr),
                                               hop_length=int(0.010 * sr), window=window)
    # mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)


def generate_normalization(y, sr, save_path, filename, window='hamm', colored=False):
    """
    Function that generate normalization augmentation (in the time domain)
    :param y: audio data
    :param sr: sampling rate
    :param save_path: path to save the output
    :param filename: spectrogram filename
    :param window: DFT window
    :param colored: set if the spectrogram must be colored or not
    :return:
    """
    path_save = save_path + '/' + filename + '_norm.jpeg'
    aug = naa.NormalizeAug(zone=(0.15, 0.85))
    augmented_data = aug.augment(y)[0]
    mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=int(0.025 * sr),
                                               hop_length=int(0.010 * sr), window=window)
    # mel_spect = librosa.feature.melspectrogram(y=augmented_data, sr=sr, n_fft=1024, hop_length=128)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    # fig.set_size_inches(3.46, 3.49)
    fig.set_size_inches(2.88, 2.88)
    # fig.set_size_inches(0.32, 0.32)
    #fig.set_size_inches(3.46, 3.49)
    if colored:
        librosa.display.specshow(mel_spect, cmap='jet', y_axis='mel', fmax=sr / 2, x_axis='time')
    else:
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr / 2, x_axis='time')
    fig.savefig(path_save, dpi=100, pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    args = parser.parse_args()

    all_augmentations = ["loud", "noise", "speed", "pitch", "shift", "norm", "mask", "maskfreq"]
    augmentations = args.augmentations
    window = args.window
    spectrogram = args.spectrogram
    colored_spectrogram = args.colored_spectrogram

    assert len(list(set(all_augmentations) - set(augmentations))) == 0
    assert window == "hamm" or window == "hann"
    assert not spectrogram or not colored_spectrogram

    datasets_list = [f.path for f in os.scandir(args.datasets_dir) if f.is_dir()]
    for dataset_folder in datasets_list:

        dataset = os.path.basename(dataset_folder)
        output_dir = dataset_folder
        wav_dir = os.path.join(dataset_folder, '_'.join([dataset, 'wav']))

        if not os.path.isdir(wav_dir):
            continue

        output_spectrogram_dir = os.path.join(dataset_folder, '_'.join([dataset, 'img']))
        output_spectrogram_colored_dir = os.path.join(dataset_folder, '_'.join([dataset, 'imgcolored']))
        output_spectrogram_augmented_dir = os.path.join(dataset_folder, '_'.join([dataset, 'augmented']))
        MASK_COVERAGE_PARAM = 0.1

        if spectrogram and not os.path.isdir(output_spectrogram_dir):
            os.makedirs(output_spectrogram_dir)

        if colored_spectrogram and not os.path.isdir(output_spectrogram_colored_dir):
            os.makedirs(output_spectrogram_colored_dir)

        if len(augmentations) and not os.path.isdir(output_spectrogram_augmented_dir):
            os.makedirs(output_spectrogram_augmented_dir)

        file_list = os.listdir(wav_dir)

        start = 0
        step = len(file_list)

        print("Processing", start, (start + step))

        cnt = -1
        num_processed = 0
        # end = len(file_list)
        end = (start+step) if (start+step) <= len(file_list) else len(file_list)
        for clip in tqdm(file_list, desc="FILE"):
            try:
                cnt += 1
                if cnt < start:
                    continue
                elif cnt >= end:
                    break

                filepath = wav_dir + "/" + clip
                filename = clip[0:-4]

                y, sr = None, None
                base_output_file = output_spectrogram_dir + "/" + filename + ".jpeg"
                if colored_spectrogram or not os.path.exists(base_output_file):
                    y, sr = librosa.load(filepath, sr=16000)
                    yt, _ = librosa.effects.trim(y)
                    y = yt
                    if spectrogram:
                        generate_spectrogram(y, sr, output_spectrogram_dir, filename, window)
                    if colored_spectrogram:
                        generate_colored_spectrogram(y, sr, output_spectrogram_colored_dir, filename, window)

                all_exists = True

                if not args.overwrite:
                    for aug in augmentations:
                        path_save = output_spectrogram_augmented_dir + '/' + filename + '_' + aug + '.jpeg'
                        if not os.path.exists(path_save):
                            all_exists = False
                            break

                if args.overwrite or not all_exists:
                    if y is None or sr is None:
                        y, sr = librosa.load(filepath, sr=16000)
                        yt, _ = librosa.effects.trim(y)
                        y = yt

                    for aug in augmentations:
                        path_save = output_spectrogram_augmented_dir + '/' + filename + '_' + aug + '.jpeg'
                        if args.overwrite or not os.path.exists(path_save):
                            if aug == "loud":
                                generate_loudness(y, sr, output_spectrogram_augmented_dir, filename,
                                                  window, colored_spectrogram)
                            elif aug == "mask":
                                generate_mask(filepath, output_spectrogram_augmented_dir, filename,
                                              MASK_COVERAGE_PARAM, colored_spectrogram)
                            elif aug == "maskfreq":
                                generate_maskfreq(filepath, output_spectrogram_augmented_dir,
                                                  filename, colored_spectrogram)
                            elif aug == "noise":
                                generate_noise(y, sr, output_spectrogram_augmented_dir, filename,
                                               window, colored_spectrogram)
                            elif aug == "pitch":
                                generate_pitch(y, sr, output_spectrogram_augmented_dir, filename,
                                               window, colored_spectrogram)
                            elif aug == "shift":
                                total_duration = len(y) / sr
                                duration_max_shift = (total_duration / 2)
                                generate_shift(y, sr, output_spectrogram_augmented_dir, filename,
                                               duration_max_shift, window, colored_spectrogram)
                            elif aug == "speed":
                                generate_speed(y, sr, output_spectrogram_augmented_dir, filename,
                                               window, colored_spectrogram)
                            elif aug == "norm":
                                generate_normalization(y, sr, output_spectrogram_augmented_dir, filename,
                                                       window, colored_spectrogram)
                            else:
                                exit("error")

            except Exception as e:
                print("Error in clip:", clip, e)

            num_processed += 1
