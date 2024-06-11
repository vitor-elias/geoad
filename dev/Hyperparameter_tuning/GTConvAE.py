import numpy as np
import pandas as pd

import ast
import os
import gc
import argparse
import torch
import optuna
import joblib
import warnings

from optuna.samplers import TPESampler
from sklearn.cluster import KMeans
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

import geoad.utils.utils as utils
import geoad.utils.fault_detection as fd
import geoad.nn.models as models

from torch_geometric.utils import to_dense_adj
from torch_geometric_temporal.nn.recurrent import TGCN

from torch_geometric.utils import dense_to_sparse

from pyprojroot import here
root_dir = str(here())

data_dir = '~/data/interim/'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

warnings.simplefilter("ignore")

def train_GTConvAE(model, X, S, reg, N_epochs, lr, training_loss):
 
    model.train()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(N_epochs):

        optimizer.zero_grad()
        output = model(X, S)

        loss = training_loss(X, output) + reg*model.s.norm(p=1)
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, data, labels, S, reg, N_epochs, lr=1e-3, training_loss=torch.nn.MSELoss()):

    auc_list = []
    for i in range(data.shape[0]):

        X = data[i,:,:]
        label = labels[i,:]
        model = train_GTConvAE(model, X, S, reg, N_epochs, lr, training_loss)

        model.eval()
        Y = model(X, S)

        score_function = torch.nn.MSELoss(reduction='none')
        score = torch.mean(score_function(X,Y), axis=1).cpu().detach().numpy()

        tpr, fpr, _ = utils.roc_params(metric=score, label=label, interp=True)
        auc = utils.compute_auc(tpr,fpr)

        auc_list.append(auc)
    
    return auc_list

def main(args):

    def objective(trial):

        gc.collect()

        # Parameters
        n_timestamps = data.shape[2]     
        N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
        hidden_features = trial.suggest_categorical('hidden_features', args.hidden_features)
        filter_order = trial.suggest_categorical('filter_order', args.filter_order)
        reg = trial.suggest_categorical('reg', args.reg)
        lr = trial.suggest_categorical('lr', args.lr)

        training_loss = torch.nn.MSELoss()
        ###

        print(f"Trial: {trial.number}", flush=True)
        print(f"- N Epochs: {N_epochs}", flush=True)
        print(f"- Hidden features: {hidden_features}", flush=True)
        print(f"- Filter order: {filter_order}", flush=True)
        print(f"- Reg: {reg}", flush=True)
        print(f"- Learing rate: {lr}", flush=True)
        print(f"- Training loss: {training_loss}", flush=True)

        ###

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()


        model = models.GTConvAE(hidden_features=hidden_features, K=filter_order, r=reg,
                                temp_graph='cyclic_directed', device=device)
        model = model.to(device)

        auc_list = evaluate_model(model, data, labels, S, reg, N_epochs, lr, training_loss)
        
        trial.set_user_attr("std_auc", np.std(auc_list))
        trial.set_user_attr("min_auc", np.min(auc_list)) 
        trial.set_user_attr("auc_list", [round(elem, 2) for elem in auc_list])

        return np.mean(auc_list).round(3)

    if args.device=='auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    # OBTAINING DATA
    dataset = 'df_StOlavs_D1L2B'
    df_orig = pd.read_parquet(data_dir + f'{dataset}.parq')

    df_ds = df_orig[df_orig.timestamp<'2022-06'].copy()
    df_ds = df_ds.groupby('pid').resample('30d', on='timestamp').mean().reset_index()

    data, labels, data_dfs, G, nodes = utils.generate_data(df_ds, args.graph_clusters, args.select_cluster,
                                                           samples=args.data_size, anomalous_nodes=args.anomalous_nodes)
    data = torch.tensor(data).float()
    data = data.to(device)
    
    G.compute_laplacian('normalized')
    S = torch.tensor(G.L.toarray()).float().to(device) # symmetric normalized Laplacian

    study = optuna.create_study(sampler=TPESampler(), direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=24,
                                                                   interval_steps=6))
    
    study.set_metric_names(['auc'])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)

    if args.log_mod == '':
        log_file = args.log_dir + args.model + '_' + str(args.graph_clusters) + '.pkl'
    else:
        log_file = args.log_dir + args.model + '_' + str(args.graph_clusters) + '_' + args.log_mod + '.pkl'

    if args.reuse:
        if os.path.isfile(log_file):
            print('Reusing previous study', flush=True)
            study = joblib.load(log_file)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    joblib.dump(study, log_file)

    print('____ END OF STUDY ___\n\n')


if __name__ == "__main__":

    for n_clusters in [30, 20]:
        print(f'Clusters: {n_clusters}', flush=True)

        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default='GTConvAE')


        parser.add_argument('--n_trials', type=int, default=1)
        parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/')
        parser.add_argument('--log_mod', type=str, default='')


        parser.add_argument('--data_size', type=int, default=10)
        parser.add_argument('--anomalous_nodes', type=int, default=20)
        parser.add_argument('--select_cluster', type=int, default=0)

        parser.add_argument('--N_epochs', type=int, nargs='+', default=[50, 100, 250, 200, 500, 1000])
        parser.add_argument('--hidden_features', type=ast.literal_eval, default=[[4, 2], [8, 4]])
        parser.add_argument('--filter_order', type=int, nargs='+', default=[3, 4, 6])
        parser.add_argument('--reg', type=float, nargs='+', default=[0.15, 0.20, 0.25])

        parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4])
        
        parser.add_argument('--device', type=str, default='cuda')

        parser.add_argument('--reuse', action='store_true', default=True)
        parser.add_argument('--no-reuse', dest='reuse', action='store_false')

        parser.add_argument('--graph_clusters', type=int, default=n_clusters)

        args = parser.parse_args()

        main(args)

