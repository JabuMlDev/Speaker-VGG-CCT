import argparse
import os
import glob
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from sklearn import manifold
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from encoder.utils import build_buckets, get_fft_spectrum
from encoder.vggvox_model import vggvox_model

"""
This script contains the implementation that allows you to generate the speaker embeddings associated 
with the audio tracks of the datasets.

!! Filename of the audio files must be of the type: emotion_dataset_subjectId_subjectGender_filename 
    (see README for more details)

Params:
    --datasets_dir              path of the datasets
    --model_path                path of the vggvox_weights.h5 (weights of the VGG-Vox Nnetwork used for the speaker
                                embeddings extraction)
    --n_components              define number of components of the resulted pca speaker embedding '
                                '(must be the same of the used ViT hidden dimension)
    --output_pca_model          path where must be saved the fitted pca model
    --save_2D_embed_compare    define if must be generated the 2D graphs that compare the speaker embeddings 
                                for each datasets (using manifold to reduce the speaker embedding dim)
                                
Output:
    --speaker embeddings        speaker embeddings in each datasets directory
    --pca model                 pca model that is fitted during computation
    --speaker embedding in 2D   graph that shows the speaker embeddings of each datasets in a 2D graph 
                                (if saved_2D_embed_compare is True)
"""

parser = argparse.ArgumentParser(description='Vit-for-SER Preprocess Speaker Embedding')

# Dataset / Model parameters
parser.add_argument('--datasets_dir', metavar='DIR', default='./datasets',
                    help='path to datasets')
parser.add_argument('--model_path', type=str, default='./encoder/models/vggvox_weights.h5',
                    help='path to vgg-vox weights')
parser.add_argument('--n_components', type=int, default=384,
                    help='define number of components of the resulted speaker embedding '
                         '(must be the same of the used ViT hidden dimension)')
parser.add_argument('--output_pca_model', type=str, default='./encoder/models/',
                    help='path to dump pca fitted model')
parser.add_argument('--save_2D_embed_compare', action='store_true', default=False,
                    help='define if must be generated the 2D graphs that compare the speaker embeddings for each datasets')

###PARAMETERS
# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.005
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

if __name__ == '__main__':
    args = parser.parse_args()

    datasets_list = [f.path for f in os.scandir(args.datasets_dir) if f.is_dir()]
    embedding_data_for_PCA = {'embeddings': [], 'dataset': []}
    print('LOAD VGG-VOX MODEL... ')
    model = vggvox_model(NUM_FFT)
    model.load_weights(args.model_path)
    model.summary()
    buckets = build_buckets(MAX_SEC, BUCKET_STEP, FRAME_STEP)

    print('GENERATE EMBEDDINGS... ')
    all_files = []
    for dataset_folder in datasets_list:
        dataset = os.path.basename(dataset_folder)
        print('----GENERATE EMBEDDINGS FROM ' + dataset + ' ...----')
        wav_dir = os.path.join(dataset_folder, '_'.join((dataset, 'wav')))
        embeddings_dir = os.path.join(dataset_folder, '_'.join((dataset, 'vgg', 'vox', 'embeddings')))
        embs = {}
        for wav_file in sorted(glob.glob(wav_dir + '/*.wav')):
            try:
                signal = get_fft_spectrum(wav_file, buckets, SAMPLE_RATE, NUM_FFT, FRAME_LEN, FRAME_STEP, PREEMPHASIS_ALPHA)
                features = model.predict(signal.reshape(1, *signal.shape, 1), verbose=False)
                current_embedding = np.squeeze(features)
                all_files.append(os.path.basename(wav_file))
                embedding_data_for_PCA['embeddings'].append(current_embedding)
                embedding_data_for_PCA['dataset'].append(dataset)
                embs[os.path.basename(wav_file)] = current_embedding
            except:
                continue
        print('Dumping original vggvox embeddings...')

        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        fdump = open(os.path.join(embeddings_dir, 'vggvox_embeddings.pkl'), 'wb')
        pkl.dump(embs, fdump, protocol=pkl.HIGHEST_PROTOCOL)
        fdump.close()

    print('FIT PCA AND SAVE MODEL...')

    df = pd.DataFrame(embedding_data_for_PCA)
    pca = PCA(n_components=args.n_components)
    pca_result = pca.fit_transform(embedding_data_for_PCA['embeddings'])
    torch.save(pca, os.path.join(args.output_pca_model, 'pca.pkl'))

    print("SAVE REDUCED FEATURES AND PLOT 2D GRAPH...")

    for dataset_folder in datasets_list:
        dataset = os.path.basename(dataset_folder)
        print("---" + dataset + "...---")
        pca_embeddings_dir = os.path.join(dataset_folder, '_'.join((dataset, 'vgg', 'vox', 'embeddings')))
        pca_embs = {}
        all_subjects = []
        pca_dataset_features = []
        subjects_data = []
        for idx, d in enumerate(embedding_data_for_PCA['dataset']):
            if d == dataset:
                pca_embs[all_files[idx]] = pca_result[idx]
                pca_dataset_features.append(pca_result[idx])
                if args.save_2D_embed_compare:
                    subject = all_files[idx].split('_')[2]
                    if subject not in all_subjects:
                        all_subjects.append(subject)
                    subjects_data.append(subject)

        fdump = open(os.path.join(pca_embeddings_dir, 'PCA_vggvox_embeddings.pkl' ), 'wb')
        pkl.dump(pca_embs, fdump, protocol=pkl.HIGHEST_PROTOCOL)
        fdump.close()
        if args.save_2D_embed_compare:
            model = manifold.MDS(n_components=2, metric=True, n_init=4, random_state=1, max_iter=200,
                                 dissimilarity='euclidean')
            data_transformed = model.fit_transform(pca_dataset_features)
            for i in all_subjects:
                indexes = []
                for idx, sub in enumerate(subjects_data):
                    if sub == i:
                        indexes.append(idx)
                plt.scatter(data_transformed[indexes, 0], data_transformed[indexes, 1], label=i)

            plt.legend()
            fig = plt.gcf()
            plt.show()
            fig.savefig(os.path.join(pca_embeddings_dir, 'PCA_subjects_embedding_plot.png'))
            plt.close()




