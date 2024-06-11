import pandas as pd
import geopandas as gpd
import numpy as np
import scipy as sp
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.io as pio
import statsmodels.api as sm

import matplotlib
import pygsp
import sys
import yaml
import warnings
import os

from sklearn.neighbors import KDTree
from collections import Counter

from pyprojroot import here

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'font.family': 'Times New Roman'})

ROOT_DIR = str(here())
insar_path = ROOT_DIR + "/data/raw/insar/"

def get_mismatch(df_orig, radius=6, data='smoothed'):

    # Obtaining nodes and treating duplicated positions
    df, nodes = treat_nodes(df_orig)

    # Obtaining subgraphs
    # 6 meters is slightly more than swath resolution, for small overlap. Reasonably smaller than amizuth resolution.
    G, nodes['subgraph'] = NNGraph(nodes, radius=radius, subgraphs=True)
    df = df.merge(nodes[['pid','subgraph']], how='left', on='pid')

    # Obtaining mismatch
    nodes['mac'] = calculate_mismatch(df, nodes, data)

    return nodes


def treat_nodes(df_orig):
    # Creating dataframe with nodes info only
    columns = ['pid', 'easting','northing']
    nodes_orig = df_orig.drop_duplicates(['pid'])[columns].copy()
    nodes_orig['ID'] = nodes_orig.northing.astype(str) + "N" + nodes_orig.easting.astype(str) + "E"

    # Testing for nodes in the same 2D position
    nodes_to_treat = nodes_orig[nodes_orig.duplicated('ID', keep='last')]

    # Treating duplicated coordinates by adding 0.005 to northing.
    df = df_orig.copy()
    df.loc[df.pid.isin(nodes_to_treat.pid), 'northing'] = df[df.pid.isin(nodes_to_treat.pid)].northing + 0.005
    nodes = df.drop_duplicates(['pid'])[columns].reset_index()

    return df, nodes

def NNGraph(nodes, radius, plotting_params = None, subgraphs=False):
    # Creating nearest neighbor graph with the nodes and gets the connected subgraphs
    # Nodes are connected if within <radius> m of each other. This yields many disconnected subgraphs
    # Later, we analyse each of these subgraphs

    sigma = radius**2  #w = np.exp(-(dist**2)/sigma) > w = 0.36 at max radius
    G = pygsp.graphs.NNGraph(nodes[['easting','northing']].values,
                            NNtype='radius',
                            epsilon = radius, sigma = sigma,
                            center=False, rescale=False)

    # Plotting
    if plotting_params is None:
        plotting_params = {'edge_color':'darkgray', 'edge_width':1.5,'vertex_color':'black', 'vertex_size':150}
    G.plotting.update(plotting_params)

    if subgraphs:
        subgraph_labels = sp.sparse.csgraph.connected_components(G.A.toarray(), directed=False)[1]
        subgraph_labels[G.d==0] = -1
        return G, pd.factorize(subgraph_labels, sort=True)[0]
    else:
        return G


def calculate_mismatch(df, nodes, data):
    subgraph_result = (df.query('subgraph!=0')
                        .groupby('subgraph')
                        .apply(max_anticorrelation(data))
                        .reset_index(name='max_decorrelation').set_index('subgraph')
                        )
    subgraph_result.loc[0] = 0
    return subgraph_result.max_decorrelation[nodes.subgraph.values].values

def graph_filter(df_orig, cut=2, radius=15, cluster=True):

    df_input = df_orig.copy()
    df_metrics = []

    if cluster==False:
        df, nodes = treat_nodes(df_input)
        G, nodes['subgraph'] = NNGraph(nodes, radius=radius, subgraphs=True)
        
        for sub_index in sorted(nodes.subgraph.unique())[1:]:

            subnodes = nodes.query('subgraph==@sub_index').copy()
            subdf = df[df.pid.isin(subnodes.pid)].copy()

            G = NNGraph(subnodes, radius=radius)

            w, V = np.linalg.eigh(G.L.toarray())
            wh = np.ones(G.N)
            wh[w<cut] = 0
            Hh = V @ np.diag(wh) @ V.T

            smoothed = subdf[['pid', 'timestamp', 'smoothed' ]].pivot(index='pid', columns='timestamp')

            subdf['hf'] = np.abs((Hh @ smoothed.values).reshape((-1,), order='C'))

            df_metrics.append(subdf)
        df_metrics = pd.concat(df_metrics)
    
    elif cluster==True:

        for cluster in sorted(df_input.cluster.unique()):

            df, nodes = treat_nodes(df_input.query('cluster==@cluster'))
            G, nodes['subgraph'] = NNGraph(nodes, radius=radius, subgraphs=True)

            df_metrics_cluster = []
            for sub_index in sorted(nodes.subgraph.unique())[1:]:

                subnodes = nodes.query('subgraph==@sub_index').copy()
                subdf = df[df.pid.isin(subnodes.pid)].copy()

                G = NNGraph(subnodes, radius=radius)

                w, V = np.linalg.eigh(G.L.toarray())
                wh = np.ones(G.N)
                wh[w<cut] = 0
                Hh = V @ np.diag(wh) @ V.T

                smoothed = subdf[['pid', 'timestamp', 'smoothed' ]].pivot(index='pid', columns='timestamp')

                subdf['hf'] = np.abs((Hh @ smoothed.values).reshape((-1,), order='C'))

                df_metrics_cluster.append(subdf)

            df_metrics_cluster = pd.concat(df_metrics_cluster)
            df_metrics.append(df_metrics_cluster)

        df_metrics = pd.concat(df_metrics)

    return df_metrics

def detection(df_metrics, column_name='hf', quantile=0.995, detect_ratio = 0.5):

    
    detection_param_threshold = np.ceil(detect_ratio*df_metrics.timestamp.nunique())

    selected_pixels=[]

    for ts in df_metrics.timestamp.unique():
        df = df_metrics[df_metrics.timestamp==ts].copy()
        th = df_metrics[column_name].quantile(quantile)

        selected_pixels.append(df[df[column_name] >= th].pid.unique())

    flat_list = [item for sublist in selected_pixels for item in sublist]
    id_counts = Counter(flat_list)
    df_anomaly_count = pd.DataFrame({'pid':id_counts.keys(), 'count':id_counts.values()})

    faulty_pixels = df_anomaly_count.query('count>@detection_param_threshold').pid.unique()

    return faulty_pixels


######### Visualisation #############
def visualise_subgraphs_data(df):
    # Plots smoothed data for all pixels in all subgraphs, animated by subgraph.
    df_toplot = (
                df.query('subgraph != 0')[['timestamp','pid','subgraph','smoothed']]
                  .pivot(index=['timestamp','subgraph'], columns='pid', values='smoothed')
                  .reset_index()
                  .sort_values(['subgraph','timestamp'])
    )
    columns = [c for c in df_toplot.columns if '3' in c]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = px.line(df_toplot, x='timestamp', y=columns, animation_frame='subgraph', width = 850)
        fig.update_layout(showlegend=False)
        fig.show()


def visualise_mismatch_map(nodes_input, min_value=-np.inf, max_value=np.inf, single_nodes=False, color='smoothed',
                           colormap = px.colors.diverging.oxy, zoom=15, range_color = None, opacity=1, recenter = None,
                           size=None, size_max = None, animation_frame=None, hover_data=[None], title=None,
                           transparent=False,
                           figsize=(1200,800), renderer=None, mapbox_style='carto-positron', savehtml=None):
    
    
    if not single_nodes:
        if 'subgraph' in nodes_input.columns:
            nodes_input = nodes_input[nodes_input.subgraph>0]

    columns = [col for col in ['pid', 'easting','northing',color, size, animation_frame]+hover_data if col is not None]
    columns = list(set(columns))

    nodes = nodes_input[columns].copy()

    if recenter is not None:
        nodes.easting = nodes.easting + recenter[0]
        nodes.northing = nodes.northing + recenter[1]

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.easting,nodes.northing), crs='3035')
    nodes_gdf = nodes_gdf.to_crs('4326')

    nodes_gdf = nodes_gdf[nodes_gdf[color]>=min_value]
    nodes_gdf = nodes_gdf[nodes_gdf[color]<=max_value]

    if animation_frame is not None:
        nodes_gdf[animation_frame] = nodes_gdf[animation_frame].astype(str)
        nodes_gdf = nodes_gdf.sort_values([animation_frame, 'pid'])

    if range_color is None:
        range_color = (nodes_gdf[color].min(), nodes_gdf[color].max())

    fig = px.scatter_mapbox(nodes_gdf, lat=nodes_gdf.geometry.y, lon=nodes_gdf.geometry.x,
                            hover_name = 'pid', hover_data = hover_data, opacity=opacity,
                            color=color, size=size, size_max = size_max,
                            mapbox_style=mapbox_style, animation_frame=animation_frame,
                            width=figsize[0], height=figsize[1], zoom=zoom,
                            color_continuous_scale=colormap, range_color=range_color,
                            )
    
    cbar_y = 0.775 if animation_frame is not None else 0.9
    fig.update_layout(coloraxis={'colorbar': {'title': {'text': ''},
                                            'len':0.5,
                                            'y':cbar_y,
                                            'thickness':5
                                            }})
    fig.update_layout(title=title)
    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    if savehtml is not None:
        fig.write_html(savehtml)

    fig.show(renderer=renderer)

def save_mismatch_map(nodes_input, min_value=-np.inf, max_value=np.inf, single_nodes=False, color='mismatch',
                           colormap = px.colors.diverging.oxy, zoom=15, range_color = None, size=None,
                           size_max = None, animation_frame=None, hover_data=[None],
                           figsize=(1200,800), renderer=None, mapbox_style='carto-positron', folder=None):
    if folder is None:
        folder = 'frames'
    output_folder = ROOT_DIR+f'/models/outputs/figs/{folder}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not single_nodes:
        if 'subgraph' in nodes_input.columns:
            nodes_input = nodes_input[nodes_input.subgraph>0]

    columns = [col for col in ['pid', 'easting','northing',color, size, animation_frame]+hover_data if col is not None]
    columns = list(set(columns))

    nodes = nodes_input[columns].copy()
   
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.easting,nodes.northing), crs='3035')
    nodes_gdf = nodes_gdf.to_crs('4326')

    nodes_gdf = nodes_gdf[nodes_gdf[color]>=min_value]
    nodes_gdf = nodes_gdf[nodes_gdf[color]<=max_value]

    if animation_frame is not None:
        nodes_gdf[animation_frame] = nodes_gdf[animation_frame].astype(str)
        nodes_gdf = nodes_gdf.sort_values([animation_frame, 'pid'])

    if range_color is None:
        range_color = (nodes_gdf[color].min(), nodes_gdf[color].max())

    center = {'lon':np.mean(nodes_gdf.geometry.x),
              'lat':np.mean(nodes_gdf.geometry.y)} 
    
    if size_max is None:
        size_max = 20.0

    max_value = nodes_gdf[size].max()
    number_of_frames = nodes_gdf[animation_frame].nunique()
    for i, value in enumerate(sorted(nodes_gdf[animation_frame].unique())):

        frame_gdf = nodes_gdf[nodes_gdf[animation_frame]==value]
        size_max_frame = frame_gdf[size].max()*size_max/max_value
        fig = px.scatter_mapbox(frame_gdf,
                                lat=frame_gdf.geometry.y, lon=frame_gdf.geometry.x,
                                hover_name = 'pid', hover_data = hover_data,
                                color=color, size=size, size_max = size_max_frame,
                                mapbox_style=mapbox_style, center=center,
                                width=figsize[0], height=figsize[1], zoom=zoom,
                                color_continuous_scale=colormap, range_color=range_color,
                                )
        
        cbar_y = 0.775 if animation_frame is not None else 0.9
        fig.update_layout(coloraxis={'colorbar': {'title': {'text': ''},
                                                'len':0.5,
                                                'y':cbar_y,
                                                'thickness':5
                                    }
                          },
                          title={'text': f'o {value} >>',
                                 'x': 0.05+0.95*i/number_of_frames
                          },
                        #   paper_bgcolor='rgb(34,33,34)'
                        paper_bgcolor='rgb(38,38,38)'
        )
        
        fig.write_image(output_folder+f'/frame_{i:03d}.png')


def relevant_neighborhood(df_orig, column_name, range_meters=15, lower=0, upper = np.inf, color='smoothed',
                           colormap = px.colors.diverging.oxy, zoom=15, range_color = None, size=None, return_df=False,
                           size_max = None, hover_data=None, figsize=(1200,800), recenter=None, transparent=False,
                           filter_dates=True, only_relevant=False, title=None, by_max=False,
                           animation_frame='timestamp', renderer=None, mapbox_style='carto-positron', plot=True):
    
    df = df_orig.copy()

    # Getting pids of relevant pixels (inside [lower,upper])

    if by_max:
        dfmax = df[['pid',column_name]].groupby('pid', as_index=False).max()
        selected_pixels = dfmax[(dfmax[column_name] >= lower) & (dfmax[column_name] <= upper)].pid.unique()
    else:
        selected_pixels = df[(df[column_name] >= lower) & (df[column_name] <= upper)].pid.unique()


    if only_relevant:
        nodes_to_plot = selected_pixels

    else:
        all_nodes = df[['pid','easting', 'northing']].drop_duplicates('pid').reset_index(drop=True).copy()
        all_positions = all_nodes[['easting', 'northing']].values

        tree = KDTree(all_positions)
        all_nodes['neighbors'] = tree.query_radius(all_positions, r=range_meters).tolist()
        all_nodes['neighbors'] = all_nodes.neighbors.apply(lambda x: all_nodes.pid[x].to_list())

        selected_nodes = all_nodes[all_nodes.pid.isin(selected_pixels)].copy()
        nodes_to_plot = list(set([item for sublist in selected_nodes.neighbors for item in sublist]))
    
    df = df[df.pid.isin(nodes_to_plot)]
    nodes = df[['pid','easting', 'northing']].drop_duplicates('pid').reset_index(drop=True).copy()
    G, nodes['group'] = NNGraph(nodes, radius=range_meters, subgraphs=True)
    df_relevant = df.merge(nodes[['pid','group']], how='left', on='pid')
    
    if filter_dates:
        df_filtered = []
        for group in sorted(df_relevant.group.unique()):
            df_group = df_relevant.query('group==@group')
            dates = df_group[(df_group[column_name]>=lower) & (df_group[column_name]<=upper)].timestamp.unique()
            df_filtered.append(df_group[df_group.timestamp.isin(dates)])

        df_relevant = pd.concat(df_filtered)

    if size is None:
        size = column_name
    
    if hover_data is None:
        hover_data = [column_name, 'group']

    if plot:
        visualise_mismatch_map(df_relevant, range_color=range_color, color=color, colormap=colormap, size=size,
                            size_max=size_max, hover_data = hover_data, recenter=recenter, title=title,
                            animation_frame=animation_frame, zoom=zoom, figsize=figsize, transparent=transparent,
                            renderer=renderer, mapbox_style=mapbox_style)
    
    if return_df:
        return df_relevant


def plot_selected_map(df_orig, column_name, selected_pixels, range_meters=15, color='smoothed', size=None,title=None,
                           colormap = px.colors.diverging.oxy, zoom=15, range_color = None,only_relevant=False, 
                           size_max = None, hover_data=None, figsize=(1200,800), recenter=None, transparent=False,
                           animation_frame='timestamp', renderer=None, mapbox_style='carto-positron'):
    
    df = df_orig.copy()

    # Getting pids of relevant pixels (inside [lower,upper])

    if only_relevant:
        nodes_to_plot = selected_pixels

    else:
        all_nodes = df[['pid','easting', 'northing']].drop_duplicates('pid').reset_index(drop=True).copy()
        all_positions = all_nodes[['easting', 'northing']].values

        tree = KDTree(all_positions)
        all_nodes['neighbors'] = tree.query_radius(all_positions, r=range_meters).tolist()
        all_nodes['neighbors'] = all_nodes.neighbors.apply(lambda x: all_nodes.pid[x].to_list())

        selected_nodes = all_nodes[all_nodes.pid.isin(selected_pixels)].copy()
        nodes_to_plot = list(set([item for sublist in selected_nodes.neighbors for item in sublist]))
    
    df = df[df.pid.isin(nodes_to_plot)]
    nodes = df[['pid','easting', 'northing']].drop_duplicates('pid').reset_index(drop=True).copy()
    G, nodes['group'] = NNGraph(nodes, radius=range_meters, subgraphs=True)
    df_relevant = df.merge(nodes[['pid','group']], how='left', on='pid')
    
    if size is None:
        size = column_name
    
    if hover_data is None:
        hover_data = [column_name, 'group']

 
    visualise_mismatch_map(df_relevant, range_color=range_color, color=color, colormap=colormap, size=size,
                        size_max=size_max, hover_data = hover_data, recenter=recenter, title=title,
                        animation_frame=animation_frame, zoom=zoom, figsize=figsize, transparent=transparent,
                        renderer=renderer, mapbox_style=mapbox_style)



def plot_graph(G, plotting_params=None, plot_disconnected=True, figsize=(16,8), name=None):

    if plotting_params is not None:
        G.plotting.update(plotting_params)

    # # Obtaining some attributes
    disconnected_nodes = G.d==0
    n_subgraphs = sp.sparse.csgraph.connected_components(G.A.toarray(), directed=False)[0]
    n_subgraphs = n_subgraphs - disconnected_nodes.sum()
    
    ax = plt.subplots(figsize=figsize)[1]

    if name is None:
        name = f'{G.N} nodes, {disconnected_nodes.sum()} disconnected nodes, {n_subgraphs} connected subgraphs'
    else:
        name = name

    G.plot(ax=ax, show_edges=True, plot_name=name)

    ax.spines[['top','bottom','left','right']].set_alpha(0.2)

    # Showing also disconnected nodes
    
    if plot_disconnected:
        for pos in G.coords[G.d==0]:
            ax.annotate('x', xy=(pos[0], pos[1]), fontsize=10, color='blue', weight='bold')

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.box(False)

    plt.xlim([G.coords[:,0].min()-5,G.coords[:,0].max()+5])
    plt.ylim([G.coords[:,1].min()-5,G.coords[:,1].max()+5])

    plt.tight_layout()


def plot_selected_pixels(df, center=None, y='smoothed', range=10, coord_type=None, timestamp='timestamp', id='pid',
                id_list=None, animate=False, figsize=(1200,800), range_y=None):
    
    # center = (x,y) = (easting, northing) = (lon, lat)
    # range in meters
    
    if id_list is not None:
        df = df.loc[df[id].isin(id_list), [id, timestamp, y, 'easting', 'northing', 'longitude', 'latitude']]
    
    if coord_type=='meters':
        df = df[ (df.easting > center[0] - range) & (df.easting < center[0] + range)  ]
        df = df[ (df.northing > center[1] - range) & (df.northing < center[1] + range)]  

    if coord_type=='degrees':
        center = gpd.points_from_xy([center[0]], [center[1]], crs=4036).to_crs(3035)
        center = (center.x[0], center.y[0])
        df = df[ (df.easting > center[0] - range) & (df.easting < center[0] + range)  ]
        df = df[ (df.northing > center[1] - range) & (df.northing < center[1] + range)]  


    if animate:
        px.line(data_frame = df,
                x=timestamp, y = y, animation_frame=id, range_y = range_y, markers=True,
                width = figsize[0], height = figsize[1]).show()
    else:
        px.line(data_frame = df.pivot(index=timestamp, columns=id, values=y),
                y = id_list, markers=True,
                width = figsize[0], height = figsize[1]).show()
        
def plot_pixel_data(df, y='smoothed', animation_frame=None, range_y=None, figsize=(1200,800)):
    
    if animation_frame:
        fig = px.line(data_frame = df, y='smoothed', color='pid', animation_group='pid',
                x='timestamp',animation_frame=animation_frame, range_y = range_y, markers=True,
                width = figsize[0], height = figsize[1])
        fig.update_layout(font_family="Times New Roman", font_size=14)
        return fig
    else:
        id_list = df['pid'].unique()
        fig = px.line(data_frame = df.pivot(index='timestamp', columns='pid', values=y),
                y = id_list, markers=True, labels={'variable':'ID', 'value':'Movement'},
                width = figsize[0], height = figsize[1])
        fig.update_layout(font_family="Times New Roman", font_size=14)
        return fig


############# Windows ###############

def spectral_window(w, V, decay, L, norm='energy' ):

    h_hat = np.exp(-w*decay) # creating spectral window
    H_hat = h_hat.reshape((-1,1)).repeat(len(w), axis=1) # repeating the window N times to create shifted versions
    Hm_hat = H_hat*(V.T) # creating spectral shifted versions by hadamard product with the m-th row of V
    Hm = V.dot(Hm_hat) # obtaining windows in the vertex domain with the IGFT

    # Original normalization by norm preserves energy of the windowing
    if norm == 'energy':
        Hm = Hm/np.linalg.norm(Hm,axis=0)


    # Normalization to unitary smoothness, makes different nodes have the same baseline.
    if norm == 'smoothness':
        # w[w<0] = 0
        # L = V.dot(np.diag(w).dot(V.T))
        # Hm = Hm/np.sqrt(( (Hm_hat**2).T.dot(w) )).reshape((1,-1))
        smoothness = np.diag(Hm.T.dot(L.dot(Hm)))
        Hm = Hm/np.sqrt(smoothness).reshape((1,-1)) # sqrt(hTLh)

    return Hm

############# Metrics ###############
def max_anticorrelation(data):
    def max_anticorrelation_(x):
        corr = x.pivot(index='timestamp', columns='pid', values=data).corr().values
        anticorr = 0.5 - 0.5*corr
        return anticorr.max()
    return max_anticorrelation_

def WGFT(nodes, x, radius=15, decay=5, window_norm='energy'):
    # nodes is a dataframe with easting and northing coordinates
    # x is an array of dim = (number_of_nodes, number_of_samples), with at least 1 column

    # Applies window hm to signal x, for all nodes, to obtain the power spectral density S

    # x = x/np.linalg.norm(x)
    # x = x - x.mean(axis=0)
     
    G = NNGraph(nodes, radius=radius)
    w, V = np.linalg.eigh(G.L.toarray())

    # creating shapes for single matrix processing of several samples at once
    nsamples = x.shape[1] # samples of an entire graph signal. Probably the number of timestamps
    X = np.tile(x, reps=(1, G.N)) # For each node, its samples. N times

    # Creating window
    Hm = spectral_window(w, V, decay, L=G.L.toarray(), norm=window_norm)
    Hm = np.repeat(Hm,repeats=nsamples, axis=1) # For each node, its window. nsamples times.

    HmX = Hm*X

    # Power spectral density S
    S = V.T.dot(HmX)
    return S, w, V

def get_psc(S, w, cut=2):
    PSD = S**2
    w[w<cut] = 0
    psc = np.sum(PSD*w.reshape(-1,1)/(PSD.sum(axis=0)), axis=0)
    return psc

def psc(nodes, x, radius=15, decay=5, window_norm='energy', cut=2):
    S, w, V = WGFT(nodes, x, radius=radius, decay=decay, window_norm=window_norm)
    return get_psc(S, w, cut)

def get_wse(S,w, cut=2):
    PSD = S**2
    w[w<cut] = 0
    return np.sum(PSD*w.reshape(-1,1), axis=0) # Normalization by w.sum() affects stuff. How?
    # If you normalize by w.sum(), the weighted spectral energy is bounded by the signal energy, since it is a convex
    # combination of the terms that compose the original energy. This would imply that two graphs with different
    # structures would yield an equally bounded wse. In practice, a graph that has higher frequencies should be able
    # to yield a higher metric  if the original energy is the same.

def wse(nodes, x, radius=15, decay=5, window_norm='energy', cut=2):
    S, w, V = WGFT(nodes, x, radius, decay, window_norm)
    return get_wse(S,w,cut)

def wse_(nodes, x, radius=15, decay=15, window_norm='energy', centralize=False, normalize=False):
    G = NNGraph(nodes, radius=radius)
    L = G.L.toarray()
    w, V = np.linalg.eigh(G.L.toarray())

    # creating shapes for single-matrix processing of several samples at once
    nsamples = x.shape[1] # samples of an entire graph signal. Probably the number of timestamps
    X = np.tile(x, reps=(1, G.N)) # For each node, its samples. N times

    # Centralizing X
    if centralize:
        central_idx = list( zip( np.arange(G.N).repeat(nsamples), np.arange(G.N*nsamples) ) )
        central_values = np.array([X[idx] for idx in central_idx])
        X = X - central_values

    # Creating window
    Hm = spectral_window(w, V, decay, L=G.L.toarray(), norm=window_norm)
    Hm = np.repeat(Hm,repeats=nsamples, axis=1) # For each node, its window. nsamples times.

    HmX = Hm*X

    if normalize:
        wse_ = np.diag(HmX.T.dot(L.dot(HmX)))/np.linalg.norm(HmX,axis=0)**2
    else:
        wse_ = np.diag(HmX.T.dot(L.dot(HmX)))

    return wse_

def lgv(nodes, x, radius=15, threshold=0.5 ):
    G = NNGraph(nodes, radius=radius)
    L = G.L.toarray()

    y = x
    zeroes = np.where(np.abs(x)<threshold)
    y[zeroes] = 1
    
    lgv = L.dot(x)/y
    lgv[zeroes] = 0
    return np.abs(lgv.flatten())
   

######################## UTILS #######################

# DATA CREATION ---------------------------------------------
def spatial_ramp(p, slope, start=-np.inf, end= np.inf):
    if (p[0] >= start) and (p[0]< end):
        return slope*(p[0] - start)
    else:
        return 0
    
def elevation(p, mean=0, var=0.1, xmin=-np.inf, xmax=np.inf):
    if (p[0] >= xmin) and (p[0] < xmax):
        return mean + np.sqrt(var)*np.random.randn()
    else:
        return 0
    
def step_anomaly(ts, start=0, end=10, var=0.1, slope=0, onset=None):
    
    if onset is None:
        onset = np.random.randint(low=len(ts)//2, high=4*len(ts)//5)

    # Adding anomaly transient
    transition = np.linspace(start,end,slope)
    ts[onset:onset+slope]=ts[onset:onset+slope] + transition

    # Adding anomaly steady-state portion of the anomaly
    ts[onset+slope:] = ts[onset+slope:] + end

    # Adding transient variation
    ts[onset:] += np.sqrt(var)*np.random.randn(len(ts)-onset)

    return ts, onset, np.abs(start-end)

def ramp_to_plateou(pos, slope, start=-np.inf, end= np.inf):
    mask =  (pos[:,0] >= start) * (pos[:,0]< end)
    ramp = slope*(pos[:,0] - start)
    ramp = ramp*mask
    ramp[pos[:,0]>=end] = ramp.max()
    return ramp


def create_data(G, anomaly=0.1, size=20, noise_var = 1e-2, signal_power = 1, seed=None, max_slope=1, eigs=1):

    np.random.seed(seed)

    pos = G.coords

    w, V = np.linalg.eigh(G.L.toarray())
    w[eigs:] = 0 # Frequency filter

    displacement = np.random.randn(G.N, size)
    displacement = V @ np.diag(w) @ V.T @ displacement # filtering
    # normalizing for desired power
    displacement = np.sqrt(signal_power*G.N)*displacement/(np.linalg.norm(displacement,axis=0))

    noise = np.sqrt(noise_var)*np.random.randn(G.N,size)

    # terrain corresponds to a ramp to a plateou in the horizontal direction
    slope = max_slope*np.random.rand() # makes small difference given proportional anomaly and scaler
    start = pos[:,0].max()*np.random.rand()*0.5 # Slope always start in the first half
    # end = start + (pos[:,0].max()-start)*np.random.rand()

    min_slope_dist = 100
    end = start + min_slope_dist + (pos[:,0].max()-start-min_slope_dist)*np.random.rand() #At least 50m of slope

    terrain = ramp_to_plateou(pos, slope=slope, start=start, end=end).reshape((-1,1))
    terrain = np.tile(terrain, (1,size)) # Matching number of samples (size)

    ptp = terrain.ptp()

    signal = displacement + noise + terrain
    label = np.zeros(signal.shape)

    for timestamp in range(size):

        anomalous_sensors = np.vstack([
                             np.random.choice(np.arange(0, G.N//3), size=(2,1), replace=False),
                             np.random.choice(np.arange(G.N//3, 2*G.N//3), size=(2,1), replace=False),
                             np.random.choice(np.arange(2*G.N//3, G.N), size=(2,1), replace=False)]
                            ).flatten()
        signal[anomalous_sensors, timestamp] += np.random.choice([anomaly*ptp, -anomaly*ptp], size=(6,))
        label[anomalous_sensors, timestamp] = 1

    return signal, label

def synth_data(anomalous_nodes, pos, Nsamples, plot=True, model_var = 1e-1, anomaly_var=1e-2):
    np.random.seed(0)

    # Define the ARMA model parameters

    ar1 = np.max((1, 2+0.1*np.random.randn())) #AR root 1
    ar2 = np.max((1, 1.5+0.1*np.random.randn())) #AR root 2
    ar_params = [1, -(ar1+ar2)/(ar1*ar2), 1/(ar1*ar2)] #AR coefficients from roots

    ma1 = np.min((-1, -2+0.1*np.random.randn())) #MA root 1
    ma2 = np.max((1, 1.5+0.1*np.random.randn())) #MA root 2
    ma_params = [1, -(ma1+ma2)/(ma1*ma2), 1/(ma1*ma2)] #MA coefficients from roots

    # Create the ARMA model
    arma_model = sm.tsa.ArmaProcess(ar_params, ma_params)

    noise_scale = model_var  # Scale of the white noise

    date_range = pd.date_range(start='01/01/2000', periods=Nsamples, freq='6D')

    ts_data = []

    for node, p in enumerate(pos):

        trend = 0
        drift = spatial_ramp(p, 0.1, 100, 200) + elevation(p, mean=10, var=model_var, xmin=200)

        # Generate the time series
        ts = arma_model.generate_sample(Nsamples, scale=noise_scale) + trend*np.linspace(0,Nsamples,Nsamples) + drift
        
        if node in anomalous_nodes:
            ts, onset, offset = step_anomaly(ts,
                                            var=anomaly_var,
                                            start=anomalous_nodes[node]['start'],
                                            end=anomalous_nodes[node]['end'],
                                            slope=anomalous_nodes[node]['slope'],
                                            onset=anomalous_nodes[node]['onset'])
            

        df_pos = pd.DataFrame({'pid':node, 'easting':p[0], 'northing':p[1],
                            'timestamp': date_range,
                            'data':ts,
                            'anomaly': date_range[onset] if node in anomalous_nodes else False})
        ts_data.append(df_pos)

    ts_data = pd.concat(ts_data).reset_index(drop=True)

    if plot:
        visualise_mismatch_map(ts_data, animation_frame='timestamp', color='data', zoom=16.5,
                                   recenter=[4341000,4479550], range_color=None, colormap='magma', figsize=(1000,600))
    return ts_data

# -------------------------------------------------- / DATA CREATION

def synth_graph(seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Define the length and position of x
    lenx = 300
    posx = np.arange(0, lenx, 5)

    # Define the length and position of y
    leny = 3
    posy = np.linspace(-15,15,leny)

    # Create a meshgrid of x and y
    X, Y = np.meshgrid(posx, posy)

    # Combine x and y into a single array of positions
    pos = np.c_[X.ravel(),Y.ravel()]

    # Add random noise to the positions
    pos = pos + np.random.randn(pos.shape[0], pos.shape[1])

    # Add extra random nodes to the positions
    extra_nodes = np.random.uniform(low=[0, -15], high=[lenx, 15], size=(len(pos)//10, 2))
    pos = np.r_[pos, extra_nodes]

    # Sort the positions by x-coordinate
    pos = pos[np.argsort(pos[:,0]),:]

    # Create a graph using the positions
    G = pygsp.graphs.NNGraph(pos,
                            NNtype='radius',
                            epsilon = 15, sigma = 100,
                            center=False, rescale=False)

    plotting_params = {'edge_color':'lightblue', 'edge_width':2,'vertex_color':'black', 'vertex_size':150}
    G.plotting.update(plotting_params)
    return G


def combine_sets(lst):
    sets = [set(s) for s in lst]
    combined_sets = []

    while sets:
        current_set = sets.pop(0)
        i = 0
        while i < len(sets):
            if current_set.intersection(sets[i]):
                current_set |= sets.pop(i)
            else:
                i += 1
        combined_sets.append(current_set)

    output_list = [next((b for b in combined_sets if any(a in b for a in lst[i])), None) for i in range(len(lst))]
    output_list = [sorted(c) for c in output_list]

    return output_list

def consecutive_ones(arr):
    max_count = 0  # Initialize the maximum count
    current_count = 0  # Initialize the current count
    
    for num in arr:
        if num == 1:
            current_count += 1  # Increment the current count
            max_count = max(max_count, current_count)  # Update the maximum count
        else:
            current_count = 0  # Reset the current count
    
    return max_count


####################################
def main():

    if len(sys.argv)<1:
        df_path = ROOT_DIR + f"/data/interim/df_smoothed_2016_01_clusters.parq"
        filename_output = ROOT_DIR + f"/data/processed/mismatch_df_smoothed_2016_01_clusters.parq"
    else:
        df_path = ROOT_DIR + sys.argv[1]
        filename_output = ROOT_DIR + f"/data/processed/mismatch_{sys.argv[1]}.parq"

    df_orig = pd.read_parquet(df_path)
    nodes = get_mismatch(df_orig)

    nodes.to_parquet(filename_output)



if __name__ == "__main__":
    main()




# def visualise_mismatch_map_OLD(nodes):
#     nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.easting,nodes.northing), crs='3035')
#     nodes_gdf = nodes_gdf.to_crs('4326')
#     nodes_gdf = nodes_gdf[nodes_gdf.subgraph>0]

#     sweden_gdf = gpd.read_file(ROOT_DIR + "/data/raw/shapefiles/sweden/SWE_adm1.shp")
#     map_gdf = sweden_gdf.query('NAME_1=="Skåne"').rename( {"NAME_1":"County"} , axis=1 ).set_index('County')

#     bgcolor = 'black'
#     terraincolor = 'darkolivegreen'
#     fig = px.choropleth(map_gdf,
#                         geojson=map_gdf.geometry,
#                         locations=map_gdf.index,
#                         fitbounds='locations',
#                         width=1000,
#                         height=600,
#                         basemap_visible=False, color_discrete_sequence=[terraincolor]
#                         )
#     fig.update_layout(showlegend=False, geo_bgcolor=bgcolor)

#     scatter_fig = px.scatter_geo(nodes_gdf,
#                         color='mismatch',
#                         lat=nodes_gdf.geometry.y,
#                         lon=nodes_gdf.geometry.x,
#                         hover_name="pid",
#                         hover_data={'pid':False,
#                                     'mismatch':':.2f',
#                                 }
#                         )

#     fig.add_trace(list(scatter_fig.select_traces())[0])

#     fig.update_layout(coloraxis={'colorbar': {'title': {'text': 'Mismatch level'},
#                                             'len':0.8,
#                                             },
#                                 'colorscale':px.colors.sequential.gray_r},
#                                 margin={"r":0,"t":0,"l":0,"b":0},
#                                 paper_bgcolor=bgcolor)
#     fig.show()