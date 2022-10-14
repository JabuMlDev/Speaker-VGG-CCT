import argparse
import csv
import os
from os import listdir
from os.path import isfile, join
from random import shuffle

"""
This script contains the implementation that allows you to split the datasets into train, validation and test sets.
Dataset to use as test set must be indicated as a param.
The other datasets are used for the train. 
The validation set is also composed by data of speakers not in training set. You can set the percent of speakers 
to use for the validation set from the script param --percent_subject_validation.

!! Filename of the audio files must be of the type: emotion_dataset_subjectId_subjectGender_filename 
    (see README for more details)


Params:
    --datasets_dir                  path of the datasets 
    --eval_dataset                  dataset to use as test set
    --num_classes                   number of classes: 3 (positive, negative, neutral) / 4 (anger, sad, happy, neutral) 
                                    or all ('happy', 'sad', 'neutral', 'anger', 'disgust'', 'fearful', 'surprised')
    --augmentation                  type of augmentation that must be considered (no_aug, time_aug, freq_aug, time_freq_aug)
    --balanced                      type of balancing that must be considered (no_balanced, undersampling, balanced_aug)
    --percent_subject_validation    percent of speakers to use for the validation set
    --train_datasets                datasets to use as training set (and validation). 
                                    (By default script use all datasets excluding the eval dataset')
    
Output:
    --train, validation and test    splitting data are saved in the local ./data folder as csv files 
                                    (train.csv, validation.csv, test.csv)
    
"""

parser = argparse.ArgumentParser(description='Vit-for-SER Preprocess Splitted Datasets into Train,'
                                             ' Validation and Test csv')

# Dataset / Model parameters
parser.add_argument('--datasets_dir', metavar='DIR', default='./datasets',
                    help='path to datasets')
parser.add_argument('--eval_dataset', type=str, default='',
                    help='dataset to use for the test model (choose one of the datasets in datasets_dir)')
parser.add_argument('--num_classes', type=str, default='4',
                    help='define task to compute ("3", "4" or "all" classes)')
parser.add_argument('--augmentation', type=str, default='no_aug',
                    help='set the type of augmentation to use: '
                         'no_aug | time_aug | freq_aug | time_freq_aug. Default is no_aug')
parser.add_argument('--balanced', type=str, default='undersampling',
                    help='set the type of balancing technique to use: '
                         'no_balanced | undersampling | balanced_aug '
                         'default is undersampling')
parser.add_argument('--percent_subject_validation', type=float, default=0.25,
                    help='percent of subjects to use for the validation set. Default is 0.1 (10%)')
parser.add_argument('--train_datasets', nargs='+',
                    default=["all"],
                    help='dataset to use for the training set (By default script use all '
                         'datasets excluding the eval dataset')


def get_emotion(filename):
    """
    Function that return emotion of a spectrogram from filename
    :param filename: filename of the spectrogram
    :return: emotion
    """
    head, tail = os.path.split(filename)
    return tail.split("_")[0]


def get_augmentation_type(filename):
    """
    Function that return augmentation type of a spectrogram from filename
    :param filename: filename of the spectrogram
    :return: augmentation type
    """
    head, tail = os.path.split(filename)
    tail_split = tail.split("_")
    return tail_split[len(tail_split) - 1].split(".")[0]


def get_dataset_name(filename):
    """
    Function that return dataset name of a spectrogram from filename
    :param filename: filename of the spectrogram
    :return: dataset name
    """
    head, tail = os.path.split(filename)
    return tail.split("_")[1].upper()


def get_subject_id(filename):
    """
    Function that return subject id (speaker id in the respective dataset) of a spectrogram from filename
    :param filename: filename of the spectrogram
    :return: subject id
    """
    head, tail = os.path.split(filename)
    return tail.split("_")[2].upper()


def save_on_csv(files, output_file_path):
    """
    Function that save files into csv
    :param files: list of data to save on the csv file (in the format filename, class)
    :param output_file_path: path of the csv
    :return:
    """
    header = ["path_file", "class"]
    with open(output_file_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for file_info in files:
            writer.writerow(file_info)


def get_train_val_files(datasets_dir, dataset_lists, augmentation, classes2labels,
                        subject_for_val_percent=0.1):
    """
    Function that generate lists of spectrogram to use as trainng set and validation set
    :param datasets_dir: path of the datasets
    :param dataset_lists: list of the dataset to use as train (and validation)
    :param augmentation: type of augmentation that must be considered
    :classes2labels: mapping of the classes to labels (in case of num classes is set to 3 or 4)
    :subject_for_val_percent: percent of speakers to use for the validation set
    :return: lists containing files that compose training set and validation set respectively
    """
    print("READ TRAIN AND VALIDATION DATA...")
    all_files = []
    all_subjects = []

    for dataset in dataset_lists:
        print("---Add " + dataset + " files...")
        current_dataset_path = os.path.join(datasets_dir, dataset)
        current_dataset_img_path = os.path.join(current_dataset_path, dataset + "_img")
        current_dataset_aug_path = os.path.join(current_dataset_path, dataset + "_augmented")
        current_imgs = [f for f in listdir(current_dataset_img_path) if isfile(join(current_dataset_img_path, f))]
        current_aug = [f for f in listdir(current_dataset_aug_path) if isfile(join(current_dataset_aug_path, f))]
        for img_file in current_imgs:
            all_files.append(os.path.join(current_dataset_img_path, img_file))
            dataset_filename = get_dataset_name(img_file)
            subjectId_filename = get_subject_id(img_file)
            subject = '-'.join([dataset_filename, subjectId_filename])
            if subject not in all_subjects:
                all_subjects.append(subject)
        if augmentation != "no_aug":
            for img_file in current_aug:
                aug_type = get_augmentation_type(img_file)
                if augmentation == "time_freq_aug" or (
                        (aug_type == "mask" or aug_type == "maskfreq") and augmentation == "freq_aug") \
                        or ((aug_type != "mask" and aug_type != "maskfreq") and augmentation == "time_aug"):
                    all_files.append(os.path.join(current_dataset_aug_path, img_file))

    print("SPLIT TRAIN AND VALIDATION DATA...")

    train_files = []
    validation_files = []
    num_validation_subjects = int(subject_for_val_percent * len(all_subjects))
    print("---Use " + str(num_validation_subjects) + " subjects for validation and the others for train---")
    shuffle(all_subjects)
    eval_subjects = []
    augmentations = ["loud", "noise", "speed", "pitch", "shift", "norm", "mask", "maskfreq"]
    for subject in all_subjects:
        eval_subjects.append(subject)
        if len(eval_subjects) == num_validation_subjects:
            break
    for file in all_files:
        dataset_file = get_dataset_name(os.path.basename(file))
        subject_id_file = get_subject_id(os.path.basename(file))
        subject_file = '-'.join([dataset_file, subject_id_file])
        try:
            emotion = classes2labels[get_emotion(file)]
            if subject_file in eval_subjects:
                if get_augmentation_type(os.path.basename(file)) not in augmentations:
                    validation_files.append([file, emotion])
            else:
                train_files.append([file, emotion])
        except:
            continue

    return train_files, validation_files


def get_test_files(dataset_dir, dataset_test, classes2labels):
    """
    Function that generate lists of spectrogram to use as test set
    :param datasets_dir: path of the datasets
    :param dataset_test: dataset to use as test set
    :classes2labels: mapping of the classes to labels (in case of num classes is set to 3 or 4)
    :return: list that contains files that compose the test set
    """
    print("READ TEST FILES...")
    test_files = []
    test_img_path = os.path.join(dataset_dir, dataset_test, dataset_test + "_img")
    test_imgs = [f for f in listdir(test_img_path) if isfile(join(test_img_path, f))]
    for file in test_imgs:
        try:
            emotion = classes2labels[get_emotion(file)]
            test_files.append([os.path.join(dataset_dir, dataset_test, '_'.join([dataset_test, 'img']), file),
                               emotion])
        except:
            continue

    return test_files


def balance_with_augmented_samples(files, datasets_dir, dataset_lists):
    """
    Function that compute balancing. Increase number of samples for the minority classes adding augmented spectrograms.
    :param files: data (filename, emotion) to compute undersampling
    :param datasets_dir: path of the datasets
    :param dataset_lists: list of the dataset to use as train (and validation)
    :return: list that contains files after balancing
    """
    # Read emotion classes info
    balanced_files = files
    num_samples_per_class = {}
    for file_info in files:
        try:
            emotion = file_info[1]
            if emotion not in list(num_samples_per_class.keys()):
                num_samples_per_class[emotion] = 0
            num_samples_per_class[emotion] += 1
        except KeyError:
            continue
    print('Distribution before undersampling: ' + str(num_samples_per_class))
    max_num_samples_per_class = max(num_samples_per_class.values())

    # read all augmented files
    all_augmented_files = []
    for dataset in dataset_lists:
        current_dataset_path = os.path.join(datasets_dir, dataset)
        current_dataset_aug_path = os.path.join(current_dataset_path, dataset + "_augmented")
        current_aug = [f for f in listdir(current_dataset_aug_path) if isfile(join(current_dataset_aug_path, f))]
        for img_file in current_aug:
            all_augmented_files.append(
                [os.path.join(current_dataset_aug_path, img_file), get_emotion(os.path.basename(img_file))])

    # read dataset balanced
    shuffle(all_augmented_files)
    for emotion in num_samples_per_class:
        count = max_num_samples_per_class - num_samples_per_class[emotion]
        for file_info in all_augmented_files:
            if count == 0:
                break
            if file_info[1] == emotion:
                balanced_files.append(file_info)
                count -= 1

    # read new emotion classes distribution
    num_samples_per_class = {}
    for file_info in balanced_files:
        try:
            emotion = file_info[1]
            if emotion not in list(num_samples_per_class.keys()):
                num_samples_per_class[emotion] = 0
            num_samples_per_class[emotion] += 1
        except KeyError:
            continue
    print('Distribution after undersampling: ' + str(num_samples_per_class))

    return balanced_files


def compute_undersampling(files, set='Training set'):
    """
    Function that compute undersampling. Reduce number of samples for the majority classes.
    :param files: data (filename, emotion) to compute undersampling
    :param type: type of the set (Training set or Validation set). Info used for print.
    :return: list that contains files after undersampling
    """
    undersampling_files = []
    num_samples_per_class = {}
    for file_info in files:
        try:
            emotion = file_info[1]
            if emotion not in list(num_samples_per_class.keys()):
                num_samples_per_class[emotion] = 0
            num_samples_per_class[emotion] += 1
        except KeyError:
            continue
    print(set + ': distribution before undersampling: ' + str(num_samples_per_class))

    min_num_samples_per_class = min(num_samples_per_class.values())
    shuffle(files)
    for emotion in num_samples_per_class:
        count = 0
        for file_info in files:
            if file_info[1] == emotion:
                undersampling_files.append(file_info)
                count += 1
            if count == min_num_samples_per_class:
                break

    num_samples_per_class = {}
    for file_info in undersampling_files:
        try:
            emotion = file_info[1]
            if emotion not in list(num_samples_per_class.keys()):
                num_samples_per_class[emotion] = 0
            num_samples_per_class[emotion] += 1
        except KeyError:
            continue
    print('Distribution after undersampling: ' + str(num_samples_per_class))

    return undersampling_files


if __name__ == '__main__':

    # Paths parameters
    args = parser.parse_args()
    # Check value params
    assert os.path.exists(args.datasets_dir)
    assert args.num_classes == "3" or args.num_classes == "4" or args.num_classes == "all"
    assert args.augmentation == "no_aug" or args.augmentation == "freq_aug" or args.augmentation == "time_aug" \
           or args.augmentation == "time_freq_aug"
    assert args.balanced == "no_balanced" or args.balanced == "undersampling" or args.balanced == "balanced_aug"
    # force balanced aug balancing to have an augmentation of type no_aug
    if args.balanced == 'balanced_aug':
        args.augmentation = 'no_aug'

    output_dir = os.path.join('./data', args.eval_dataset, args.num_classes + '_classes', args.augmentation,
                              args.balanced)
    output_train_dir = os.path.join(output_dir, 'train_data')
    output_eval_dir = os.path.join(output_dir, 'eval_data')
    if not os.path.isdir(output_train_dir):
        os.makedirs(output_train_dir)
    if not os.path.isdir(output_eval_dir):
        os.makedirs(output_eval_dir)

    if len(args.train_datasets) == 1 and args.train_datasets[0] == 'all':
        datasets_list = [os.path.basename(f) for f in os.scandir(args.datasets_dir) if f.is_dir()]
        try:
            datasets_list.remove(args.eval_dataset)
        except:
            print("ERROR: Dataset to test not in datasets dir")
            exit(1)
    else:
        datasets_list = args.train_datasets

    if args.num_classes == "3":
        classes2labels = {'happy': 'positive', 'sad': 'negative', 'neutral': 'neutral', 'anger': 'negative',
                          'disgust': 'negative', 'fearful': 'negative', 'surprised': 'positive',
                          'excited': 'positive',
                          'frustration': 'negative'}

    elif args.num_classes == "4":
        classes2labels = {'happy': 'happy', 'sad': 'sad', 'neutral': 'neutral', 'anger': 'anger'}

    else:
        classes2labels = {'happy': 'happy', 'sad': 'sad', 'neutral': 'neutral', 'anger': 'anger',
                          'disgust': 'disgust',
                          'fearful': 'fearful', 'surprised': 'surprised'}

    train_files, validation_files = get_train_val_files(datasets_dir=args.datasets_dir,
                                                        dataset_lists=datasets_list, augmentation=args.augmentation,
                                                        classes2labels=classes2labels,
                                                        subject_for_val_percent=args.percent_subject_validation)
    if args.balanced == "undersampling":
        print("COMPUTE UNDERSAMPLING...")
        train_files = compute_undersampling(train_files)
    elif args.balanced == "balanced_aug":
        print("COMPUTE BALANCED AUGMENTATION...")
        train_files = balance_with_augmented_samples(files=train_files, datasets_dir=args.datasets_dir,
                                                     dataset_lists=datasets_list)

    validation_files = compute_undersampling(validation_files, set='Validation set')
    test_files = get_test_files(args.datasets_dir, args.eval_dataset, classes2labels)

    print("SAVE DATA INTO CSV FILES...")
    save_on_csv(train_files, os.path.join(output_train_dir, 'train.csv'))
    save_on_csv(validation_files, os.path.join(output_train_dir, 'validation.csv'))
    save_on_csv(test_files, os.path.join(output_eval_dir, 'test.csv'))
