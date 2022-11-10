import pickle

from timm.data.parsers.parser import Parser
import torchaudio
import csv
import os

class ParserCSV(Parser):
    def __init__(self, csv_dataset_path, class_to_idx, read_wav=False, read_vgg_features=False):
        super().__init__()
        self.samples, self.classes = self.__read_all_dataset_files(csv_dataset_path)


        self.wav_data = []
        if read_wav:
            self.wav_data = self.__read_all_wav_files(csv_dataset_path)
        self.vgg_features = {}
        if read_vgg_features:
            self.vgg_features = self.__read_vgg_vox_features()
            self.samples, self.classes = self.__clean_files_without_embedding()

        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.class_to_idx[self.classes[index]]
        if len(self.wav_data):
            waveform, _ = torchaudio.load(self.wav_data[index])
            return (open(path, "rb"), self.wav_data[index]), target
        if len(self.vgg_features):
            sample_dataset = self.__get_dataset_from_filename(path)
            filename_without_augmentations = self.__remove_augmentation_from_filename(os.path.basename(path))
            """ With scripts outside there are extension .wav in vgg_features_dict (first row)
                With scripts in project there are not extension .wav in vgg_features_dict (second row)
            """
            #current_vgg_features = self.vgg_features[sample_dataset][os.path.basename(filename_without_augmentations).split('.')[0]]
            current_vgg_features = self.vgg_features[sample_dataset][os.path.basename(filename_without_augmentations)]
            return (open(path, "rb"), current_vgg_features), target
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def __get_dataset_from_filename(self, filename):
        file_dataset = os.path.basename(filename).split('_')[1]
        return file_dataset

    def __read_all_dataset_files(self, csv_dataset_path):
        dataset_csv = open(csv_dataset_path)
        csvreader = csv.reader(dataset_csv)
        all_dataset_files = []
        classes_files = []
        for idx, row in enumerate(csvreader):
            if idx > 0 and row[0] != 'path_file':
                all_dataset_files.append(row[0])
                classes_files.append(row[1])
        return all_dataset_files, classes_files

    """ Ci sono alcuni file di IEMOCAP che non hanno embedding (errori durante la generazione dell'embedding) 
        Si eliminano dal dataset con questa funzione. """
    def __clean_files_without_embedding(self):
        cleaned_dataset_files = []
        cleaned_classes_files = []

        for idx, sample in enumerate(self.samples):
            sample_dataset = self.__get_dataset_from_filename(sample)
            filename_without_augmentations = self.__remove_augmentation_from_filename(os.path.basename(sample))
            filename_without_augmentations = '.'.join([filename_without_augmentations.split('.')[0], 'wav'])
            if filename_without_augmentations in self.vgg_features[sample_dataset]:
                cleaned_dataset_files.append(sample)
                cleaned_classes_files.append(self.classes[idx])
            else:
                print('File ' + os.path.basename(sample) + ' ignored: not found embedding of this wav file')

        return cleaned_dataset_files, cleaned_classes_files

    def __read_all_wav_files(self, csv_dataset_path):
        dataset_csv = open(csv_dataset_path)
        csvreader = csv.reader(dataset_csv)
        all_wav_files = []
        for idx, row in enumerate(csvreader):
            if idx > 0:
                spectrogram_path = row[0]
                spectrogram_filename = os.path.basename(spectrogram_path)
                original_wav_filename = self.__remove_augmentation_from_filename(spectrogram_filename)
                root_spectrogram_path = os.path.split(os.path.split(spectrogram_path)[0])[0]
                dataset = os.path.basename(root_spectrogram_path)
                wav_folder = '_'.join([dataset, 'wav'])
                wav_path = os.path.join(root_spectrogram_path, wav_folder, original_wav_filename)
                all_wav_files.append(wav_path)
        return all_wav_files

    """
    Method that return dictionary with associated for each dataset of training one of his samples
    """
    def __get_datasets_in_training(self):
        datasets = {}
        for sample in self.samples:
            sample_dataset = self.__get_dataset_from_filename(sample)
            if sample_dataset not in datasets:
                datasets[sample_dataset] = sample
        return datasets

    def __read_vgg_vox_features(self):
        training_datasets = self.__get_datasets_in_training()
        vgg_features = {}
        for dataset in training_datasets:
            first_spectrogram_path = training_datasets[dataset]
            root_path = os.path.split(os.path.split(first_spectrogram_path)[0])[0]
            dataset_upper_case = dataset.upper()
            vgg_features_folder = '_'.join([dataset_upper_case, 'vgg_vox_embeddings'])
            vgg_features_path = os.path.join(root_path, vgg_features_folder, "PCA_vggvox_embeddings.pkl")
            with open(vgg_features_path, 'rb') as handle:
                vgg_features[dataset] = pickle.load(handle)
        return vgg_features

    def __remove_augmentation_from_filename(self, filename):
        augmentations_methods = ["loud", "noise", "speed", "pitch", "shift", "norm", "mask", "maskfreq"]
        name, _ = os.path.splitext(filename)
        splitted_filename = name.split("_")

        last_part_filename = splitted_filename[len(splitted_filename) - 1]
        if last_part_filename in augmentations_methods:
            name = '_'.join(name.split('_')[:-1])

        filename_without_last_part = '.'.join([name, 'wav'])

        return filename_without_last_part