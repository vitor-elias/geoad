# Imports

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import ruptures as rpt

from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans


plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Times New Roman'})

# ______________________________  ONSET DETECTION _____________________________________

def center_column(df, column, on, name):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Group by ID and perform the centering
    for _, group in df_copy.groupby('pid'):
        max_y_index = group[on].idxmax()
        max_x_value = group.at[max_y_index, column]
        df_copy.loc[group.index, name] = df_copy.loc[group.index, column] - max_x_value

    return df_copy

def center_column_v2(df, column, on, name):
    # Faster if large memory available
    
    df_copy = df.copy()
    def center_on_gradmax(group):
        max_grad_abs_idx = group[on].idxmax()  # Index of max 'grad_abs'
        max_smoothed = group.at[max_grad_abs_idx, 'smoothed']  # Value of 'smoothed' at max 'grad_abs'
        group[name] -= max_smoothed  # Normalize 'smoothed'
        return group

    df_copy[name] = df[column]
    df_copy = df_copy.groupby('pid', group_keys=False).apply(center_on_gradmax)
    return df_copy


def get_df_onset(df, threshold, clustering_length, prediction_range=30, pelt=None):
    # Receives a partial df for a single pixel

    if pelt is None:
        pelt = rpt.Pelt(model='rbf', min_size=60)

    ts = df.smoothed.values
    gs = df.grad_abs.values

    change = np.array(pelt.fit_predict(ts, pen=1))
    change_tuples = [(0, change[0])]+[(change[i]+1,change[i+1]) for i in range(len(change)-1)] # interval tuples
    
    id_max_gs = np.argmax(gs)
    id_max_pelt_segment = np.where(change>id_max_gs)[0][0]


    # If the max gradient is too early, the onset is probably not available
    if id_max_gs < prediction_range:
        onset=0
        case=0

    # If the onset happens at the beggining of the data. There is space before max_grad, but no prediction samples
    elif id_max_pelt_segment==0:
        onset = 0
        case = 1

    else:
        onset = change[id_max_pelt_segment-1]
        
        # check if previous interval has grad > threshold
        # For example, it goes down quickly before going up even more quickly. Pelt might separate two segments.
        prev_start = change_tuples[id_max_pelt_segment-1][0]
        prev_end = change_tuples[id_max_pelt_segment-1][1]
        if (gs[prev_start:prev_end]>threshold).any():
            onset = prev_start
        if onset == 0:
            case = 1
        else:
            # If there are enough samples for clustering after onset (2) or not (3)
            case = 2 if (onset+clustering_length)<=len(ts) else 3

    df_onset = df.iloc[onset:min(len(ts),onset+clustering_length)].copy()
    df_onset['onset'] = onset
    df_onset['onset_case'] = case
    return df_onset
##============================== END OF ONSET DETECTION  =====================================##



# ____________________________________ CLUSTERING ___________________________________________ 

def tskmeans(df, cluster_by='smoothed', cluster_to='kmeans_clusters', n_clusters=6, metric='dtw',
             n_init=5, plot=True, savefig=None):

    data_ts = df[cluster_by].values.reshape((-1, df.pid.value_counts().iloc[0],1))
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, n_init=n_init)
    y_pred = model.fit_predict(data_ts)

    df_cluster = df.copy()
    df_cluster[cluster_to] = np.repeat(y_pred, df_cluster.pid.value_counts().iloc[0])

    if plot:
        plot_clusters(df_cluster, plot_column='smoothed', clusters=cluster_to, savefig=savefig)

    return df_cluster, y_pred



def plot_clusters(df, plot_column='smoothed', clusters='kmeans_clusters', savefig=None):

    data_ts = df[plot_column].values.reshape((-1, df.pid.value_counts().iloc[0],1))
    y_pred = df.drop_duplicates('pid')[clusters].values

    n_clusters = df[clusters].nunique()

    # Plotting
    nrows = int(np.ceil(n_clusters/3))
    ncols = 3
    ratios = [3]
    ratios.extend([0.4,3]*(nrows-1))

    if n_clusters == 4:
        ncols = 2

    fig, ax = plt.subplots(nrows=2*nrows -1, ncols=ncols, figsize=(30,12), gridspec_kw={'height_ratios': ratios})

    for cluster in range(n_clusters):

        if df.timestamp.nunique()==data_ts.shape[1]:
            timestamp_list = df.timestamp.unique()
        else:
            timestamp_list = range(data_ts.shape[1])
            
        for timeseries in data_ts[y_pred == cluster]:
            ax[(cluster//ncols)*2,cluster%ncols].plot(timestamp_list, timeseries.ravel(), alpha=0.1)
        
        ax[(cluster//ncols)*2,cluster%ncols].text(0.05, 0.95, 
                                            str(sum(y_pred==cluster)),
                                            transform=ax[(cluster//ncols)*2,cluster%ncols].transAxes)

    for i in range(nrows):
        if ~i%2:
            ax[i,0].set_ylabel(plot_column)
            ax[i,0].set_ylabel(plot_column)
        else:
            for j in range(ncols):
                ax[i,j].set_visible(False)

    ax[-1,1].set_xlabel('Time index')
    if ncols==2:
        ax[-1,0].set_xlabel('Time index')

    fig.suptitle(clusters)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    if savefig is not None:
        plt.savefig(savefig)

    plt.show()