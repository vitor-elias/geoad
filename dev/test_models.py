import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import matplotlib
from matplotlib.colors import ListedColormap

import os
import gc
import argparse
import torch
import optuna
import joblib
import pickle
import warnings
import time

from tqdm.notebook import tqdm

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.nn.recurrent import TGCN

from sklearn.cluster import KMeans, BisectingKMeans, SpectralClustering

import geoad.nn.models as models
import geoad.utils.utils as utils
import geoad.utils.fault_detection as fd

from geoad.utils.utils import roc_params, compute_auc

from importlib import reload
models = reload(models)
utils = reload(utils)

from pyprojroot import here
root_dir = str(here())

data_dir = '~/data/interim/'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

warnings.simplefilter("ignore")

def train_model(model, X, edge_index, edge_weight, N_epochs, lr, use_index, use_weight):
 
    training_loss=torch.nn.MSELoss()

    model.train()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(N_epochs):

        optimizer.zero_grad()

        if use_index and use_weight:
            output = model(X, edge_index, edge_weight)
        elif use_index:
            output = model(X, edge_index)
        else:
            output = model(X)

        loss = training_loss(X, output)
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, data, labels, edge_index, edge_weight, N_epochs, lr, use_index, use_weight):

    auc_list = []
    elapsed_time_list = []
    fpr_list = []
    tpr_list = []
    for i in range(data.shape[0]):

        X = data[i,:,:]
        label = labels[i,:]

        start_time = time.time()

        model = train_model(model, X, edge_index, edge_weight, N_epochs, lr, use_index, use_weight)

        end_time = time.time()  # End time of the iteration
        
        model.eval()

        if use_index and use_weight:
            Y = model(X, edge_index, edge_weight)
        elif use_index:
            Y = model(X, edge_index)
        else:
            Y = model(X)

        score_function = torch.nn.MSELoss(reduction='none')
        score = torch.mean(score_function(X,Y), axis=1).cpu().detach().numpy()

        tpr, fpr, _ = utils.roc_params(metric=score, label=label, interp=True)
        auc = utils.compute_auc(tpr,fpr)

        auc_list.append(auc)

        elapsed_time = end_time - start_time
        elapsed_time_list.append(elapsed_time)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

        print(f'    {i} of {data.shape[0]} - time: {elapsed_time}', flush=True)
    
    return auc_list, elapsed_time_list, tpr_list, fpr_list

def main(args):

    AEmodels = ['AE', 'AEconv1D', 'GCN2MLP', 'GCNAE', 'GALA', 'GUNet', 'RAELSTM', 'RAEGRU', 'REGAE']
    sizes = [2, 5, 10, 20, 30]   

    df_partial = []

    for s in sizes:
        print(f'Clustering: {s}', flush=True)

        rng_seed = 0
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed(rng_seed)
        np.random.seed(rng_seed)

        # OBTAINING DATA
        device = args.device
        dataset = 'df_StOlavs_D1L2B'
        df_orig = pd.read_parquet(data_dir + f'{dataset}.parq')

        df_ds = df_orig[df_orig.timestamp<'2022-06'].copy()
        df_ds = df_ds.groupby('pid').resample('30d', on='timestamp').mean().reset_index()

        data, labels, data_dfs, G, nodes = utils.generate_data(df_ds, s, select_cluster=1, samples=args.data_size,
                                                               anomalous_nodes=5, noise=args.noise)
        data = torch.tensor(data).float()
        data = data.to(device)

        A = torch.tensor(G.W.toarray()).float() #Using W as a float() tensor
        edge_index, edge_attr = dense_to_sparse(A)
        edge_index = edge_index.to(device)

        conv1d_n_feats = 3
        conv1d_kernel_size = 10
        conv1d_stride = 5
        n_timestamps = data.shape[2]     

        for AEmodel in AEmodels:
            print(f'  model: {AEmodel}', flush=True)

            A = torch.tensor(G.W.toarray()).float() #Using W as a float() tensor
            edge_index, edge_weight = dense_to_sparse(A)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)

            use_index=False
            use_weight=False

            study = joblib.load(root_dir + f'/outputs/HP_training/{AEmodel}_{s}.pkl')
            params = study.best_params

            N_epochs = params['N_epochs']
            lr = params['lr']

            if AEmodel=='AE':
                model = models.AE(n_timestamps=n_timestamps, 
                                n_encoding_layers=params['n_layers'], reduction=params['reduction'])
                usegraph=False

            elif AEmodel=='AEconv1D':
                model = models.AEconv1D(conv1d_n_feats=conv1d_n_feats, conv1d_kernel_size=conv1d_kernel_size,
                                        conv1d_stride=conv1d_stride, n_timestamps=n_timestamps,
                                        n_encoding_layers=params['n_layers'], reduction=params['reduction'])
                
            elif AEmodel=='GCN2MLP':
                model = models.GCN2MLP(n_timestamps=n_timestamps,
                                    n_encoding_layers=params['n_layers'], reduction=params['reduction'])
                use_index = True

            elif AEmodel=='GCNAE':
                model = models.GCNAE(n_timestamps=n_timestamps,
                                    n_encoding_layers=params['n_layers'], reduction=params['reduction'])
                use_index = True

            elif AEmodel=='GALA':
                model = models.GALA(n_timestamps=n_timestamps,
                                    n_encoding_layers=params['n_layers'], reduction=params['reduction'])
                use_index = True
                use_weight= True
                if params['use_weight']=='False':
                    edge_weight=None
                
            elif AEmodel=='GUNet':
                model = models.GUNet(in_channels=n_timestamps, hidden_channels=params['hidden_channels'],
                                    out_channels=n_timestamps, depth=params['depth'], pool_ratios=params['pool_ratio'])
                use_index = True
                
            elif AEmodel=='RAELSTM':
                model = models.RAE(n_features=1, latent_dim=params['latent_dim'],
                                rnn_type='LSTM', rnn_act='relu', device=device)
                
            elif AEmodel=='RAEGRU':
                model = models.RAE(n_features=1, latent_dim=params['latent_dim'],
                                rnn_type='GRU', rnn_act='relu', device=device)
                
            elif AEmodel=='REGAE':
                model = models.RecurrentGAE(G.N, latent_dim=params['latent_dim'], rnn=TGCN,
                                            conv_params=None, device=device)
                use_index=True
                use_weight=True
                
            model = model.to(device)
            auc_list, time_list, tpr_list, fpr_list = evaluate_model(model, data, labels, edge_index, edge_weight,
                                                 N_epochs, lr, use_index, use_weight)

            df_partial.append({'model':AEmodel, 'size':s,
                               'auc_list':auc_list,
                               'time_list':time_list,
                               'tpr_list':tpr_list,
                               'fpr_list':fpr_list})

    df_results = pd.DataFrame(df_partial)
    df_results.to_parquet(f'{args.filename}.parq')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=root_dir+'/outputs/results_testing/')
  
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--noise', type=float, default=0.1)

    parser.add_argument('--device', type=str, default='cuda')


    args = parser.parse_args()

    print(f"Initializing tests", flush=True)
    main(args)

