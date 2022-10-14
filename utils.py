import os
import yaml
from matplotlib import pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sn


def update_graph(train_loss_history, val_loss_history, val_acc_history, path):
    """
    Function that update graphs with the trend of loss and accuracy on the train and validation data
    :param train_loss_history, val_loss_history: list that contains loss value for train and validation detected
                                                 at each epoch
    :param train_acc_history, val_loss_hystory: list that contains mean accuracy for train and validation
                                                detected at each epoch
    :param path: path of the directory where to save the graphs
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)

    losses_img_file = os.path.join(path, "training_losses.png")
    acc_img_file = os.path.join(path, "training_accuracy.png")
    epochs = np.arange(1, len(train_loss_history) + 1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Plot Training/Validation Losses")
    plt.ylim(0, max(max(train_loss_history), max(val_loss_history)))
    plt.plot(epochs, train_loss_history, label="average train loss")
    plt.plot(epochs, val_loss_history, label="average validation loss")
    plt.legend()
    plt.savefig(losses_img_file)
    plt.close()
    plt.title("Plot Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.ylim(0, 100)
    # plt.plot(epochs, train_acc_history, label="average train accuracy")
    plt.plot(epochs, val_acc_history, label="average validation accuracy")
    plt.legend()
    plt.savefig(acc_img_file)
    plt.close()


def read_history_from_csv(path):
    """
    Function that read precedent training and validation loss
    :param path: path of the directory where to read losses
    :return: training accuracy, validation accuray and validation loss
    """
    csv_path = os.path.join(path, "summary.csv")
    train_loss_history, val_loss_history, val_acc_history = [], [], []
    if not os.path.exists(csv_path):
        print("ERROR: not find csv in path ", csv_path)
        return
    with open(csv_path, mode='r') as file:
        for idx, line in enumerate(csv.reader(file)):
            if line[0] != "epoch":
                train_loss_history.append(float(line[1]))
                val_loss_history.append(float(line[2]))
                val_acc_history.append(float(line[3]))
    return train_loss_history, val_loss_history, val_acc_history


def save_confusion_matrix(cm, labels, fname):
    """
    Function that save confusion matrix on image
    :param cm: confusion matrix
    :param labels: labels of the confusion matrix
    :param fname: path of the output image
    :return:
    """
    df_cm = pd.DataFrame(cm, index=[str(i) for i in labels],
                         columns=[str(i) for i in labels])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.savefig(fname, dpi=240)
    plt.close()


def save_performance(performance, path):
    """
    Function that save model performance
    :param performance: model performance
    :param path: path of the directory where to save the graphs
    :return:
    """
    with open(path, 'w') as outfile:
        yaml.dump(performance, outfile, default_flow_style=False)


def get_class_to_idx(dataset, num_classes):
    """
    Function that return mapping classes to respective index target
    :param dataset: dataset
    :param num_classes: number of classes of the experiment
    :return: dictionary that mapping classes to respective index target

    !!! If you want train on all classes of your datasets and these are a subset of 'happy', 'sad', 'anger',
        'disgust', 'fearful', 'neutral', 'surprised' you have to change this method.
        Currently, this method already manages cases of EMODB and DEMOS.
    """
    class_to_idx = {}
    if num_classes == "3_classes":
        class_to_idx = {'neutral': 0, 'positive': 1, 'negative': 2}
    elif num_classes == "4_classes":
        class_to_idx = {'anger': 0, 'happy': 1, 'neutral': 2, "sad": 3}
    elif num_classes == "all_classes":
        if dataset == "EMODB":
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'neutral': 5}
        elif dataset == "DEMOS":
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'surprised': 5}
        else:
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'neutral': 5, 'surprised': 6}
        """
        elif dataset == "IEMOCAP":
            class_to_idx = {'happy': 0, 'sad': 1, 'anger': 2,
                            'disgust': 3, 'fearful': 4, 'surprised': 6, 'neutral': 5,
                            'excited': 7, 'frustration': 8}
        """
    return class_to_idx

def get_class_labels(train_dataset_path):
    """
    Function that return classes labels from dataset
    :param train_dataset_path: dataset
    :return: list that contains the classes labels
    """
    classes_label = []
    for root, subdirs, files in os.walk(train_dataset_path):
        for file in files:
            if os.path.splitext(file)[-1].lower() == ".csv":
                dataset_csv = open(os.path.join(root, file))
                csvreader = csv.reader(dataset_csv)
                for idx, row in enumerate(csvreader):
                    if idx > 0:
                        current_class = row[1]
                        if current_class not in classes_label and current_class != "class":
                            classes_label.append(current_class)
    return classes_label

def get_dataset_name_from_path(path):
    """
    Function that return the eval dataset name from path.
    It is used to set output paths automatically from the input paths
    :param path: input path (of the dataset in train / of the model in test)
    :return: eval dataset name of the experiment
    """
    datasets_dir = './datasets'
    all_datasets = [f.name for f in os.scandir(datasets_dir) if f.is_dir()]
    path_tails = path.split('/')
    for dataset in all_datasets:
        if dataset in path_tails:
            return dataset
    return None

def get_classes_name_from_path(path):
    """
    Function that return the number of classes. It is used to set output paths automatically from the input paths
    :param path: input path (of the dataset in train / of the model in test)
    :return: number of classes of the experiment
    """
    all_classes_name = ["3_classes", "4_classes", "all_classes"]
    path_tails = path.split('/')
    for classes_name in all_classes_name:
        if classes_name in path_tails:
            return classes_name
    return None

def get_aug_type_from_path(path):
    """
    Function that return the augmentation type used for the experiment.
    It is used to set output paths automatically from the input paths
    :param path: input path (of the dataset in train / of the model in test)
    :return: augmentation type used for the experiment
    """
    all_aug = ["no_aug", "time_aug", "freq_aug", "time_freq_aug"]
    path_tails = path.split('/')
    for aug in all_aug:
        if aug in path_tails:
            return aug
    return None

def get_balanced_type_from_path(path):
    """
    Function that return the balanced type used for the experiment.
    It is used to set output paths automatically from the input paths
    :param path: input path (of the dataset in train / of the model in test)
    :return: balanced type used for the experiment
    """
    all_balanced_technique = ["no_balanced", "undersampling", "balanced_aug", "smart_balanced"]
    path_tails = path.split('/')

    # to cover also smart_balanced (future implementation on preprocess_split_datasets.py)
    for tail in path_tails:
        if 'smart' in tail.split('_') and 'balanced' in tail.split('_'):
            return tail
    for balance_technique in all_balanced_technique:
        if balance_technique in path_tails:
            return balance_technique
    return None