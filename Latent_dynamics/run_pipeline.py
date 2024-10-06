"""
To train the autoencoder neural networks, please run codes in Part 1.
To calculate the latent features of the images extracted by the autoencoder model, please run codes in Part 2.

We provide the 5-dimensional latent features and physical features for all the 3,697 embryoids, which are stored in the files './test_data/latent_fts.npy', and './test_data/physical_fts.npy'
To analyze the latent features and predict the continuous evolution of latent features:
      first, run codes in Part 3.1, which fits mixed Gaussian distributions at the discreted time points
      then, run codes in Part 3.2, which constructs continuous stochastic evolution of latent features
To generate artificial images in a finer time resolution, please run Part 3.1-->Part 3.2-->Part 4
"""
import numpy as np

import model
import argparse
from utils import *
import consts

from sklearn.decomposition import PCA
from GMM_funcs import fit_Guassian_processes
from stochastic_process_funcs import fit_mean_reverting_process, lat_fts_propation


import gc
gc.collect()

import pdb


# Autoencoder parameters
parser = argparse.ArgumentParser(description='AAE', formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--epochs', '-e', default=10000, type=int)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch-size', '--bs', dest='batch_size', default=64, type=int)
parser.add_argument('--weight-decay', '--wd', dest='weight_decay', default=1e-5, type=float)
parser.add_argument('--learning-rate', '--lr', dest='learning_rate', default=2e-4, type=float)
parser.add_argument('--b1', '-b', dest='b1', default=0.5, type=float)
parser.add_argument('--b2', '-B', dest='b2', default=0.999, type=float)

parser.add_argument('--load', '-l', default='./model')
parser.add_argument('--input_data_path', '-i', default='./test_data')
parser.add_argument('--output_path', '-o', default='./saved_model')
parser.add_argument('-z', dest='z_channels', default=5, type=int)
args = parser.parse_args()

consts.NUM_Z_CHANNELS = args.z_channels
net = model.Net()


##### Part 1: train autoencoder model (adversarial training) #####
# print('Part 1: start training Autoencoder model...')
# betas = (args.b1, args.b2) if args.load is None else None
# weight_decay = args.weight_decay if args.load is None else None
# lr = args.learning_rate if args.load is None else None
#
# if args.load is not None:
#     net.load(path=args.load, slim=False)
#     print("Loading pre-trained models from {}".format(args.load))
#
# print("Training data path is {}".format(args.input_data_path))
# os.makedirs(args.output_path, exist_ok=True)
# print("Results saved in {}".format(args.output_path))
#
# net.train_model(
#     embryoid_path=args.input_data_path,
#     batch_size=args.batch_size,
#     betas=betas,
#     epochs=args.epochs,
#     weight_decay=weight_decay,
#     lr=lr,
#     valid_size=50,
#     where_to_save=args.output_path,
#     args=args
# )


##### Part 2:  calculate latent features #####
# print('Part 2: start calculate latent features...')
# net.load(path=args.load, slim=True)
# lat_fts = net.calculate_latent_features(args=args)
# np.save('./test_data/latent_fts.npy', lat_fts)         # save latent features


##### Part 3.1: fit a mixed Gaussian process at discrete time points #####
# print('Part 3.1: start to calculate parameters of mixed Gaussian distributions...')
# pca_ = PCA(n_components=5)
# lat_fts = pca_.fit_transform(np.load('./test_data/w1405_dim5_fts.npy',allow_pickle=True))   # 'latent_fts.npy' contains 5-dim features for all embryoids
# phy_fts = np.load('./test_data/physical_fts.npy',allow_pickle=True).item()       # 'physical_fts.npy' contains physical features for all embryoids (e.g., time, group, filename, morphologies)
# tps, wts, mus, vars, dists = fit_Guassian_processes(lat_fts, phy_fts)        # time_points, (wts, mus, vars) are for Gaussian processes, distribution of lat_fts at discrete tps
# np.save('./test_data/mixed_Gaussian_params.npy', {'tps':tps, 'wts':wts, 'mus':mus, 'vars':vars, 'dists':dists})


##### Part 3.2: construct a mean-reverting stochastic process (for continous latent feature evolution) #####
# print('Part 3.2: start to construct the mean-reverting stochastic process...')
# lat_ft_evol = fit_mean_reverting_process(tps, wts, mus, vars, dists, n_comp=2, dt=0.01)


##### Part 4: generate images, visualize developmental evolution #####
# print('Part 4: Generate a possible embryoid morphogenesis process...')
# mixed_Gaussian_params = np.load('./test_data/mixed_Gaussian_params.npy', allow_pickle=True).item()
# emb_fts = lat_fts[phy_fts['file_name'].index('JC080420_t021_g01_n1_4ch_s1'),:].copy()   # initial state
# emb_ft_evol = lat_fts_propation(emb_fts, mixed_Gaussian_params['tps'], mixed_Gaussian_params['wts'], mixed_Gaussian_params['mus'], mixed_Gaussian_params['vars'], n_comp=2, dt=0.01)
# net.load(path=args.load, slim=True)
# for ti in range(int(np.shape(emb_ft_evol)[0]/10)):
#     net.generate_img(pca_.inverse_transform(np.mean(emb_ft_evol[ti*10,:np.shape(emb_ft_evol)[1],:], axis=0))[np.newaxis,:], ti*10)

