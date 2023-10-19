import os
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import bct as bct
import pandas as pd
import logging
from datetime import datetime
import nibabel as nib
from tqdm import tqdm
import seaborn as sns
from joblib import Parallel, delayed
import matplotlib.patheffects as path_effects

def add_median_labels(ax, fmt='.3f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def read_subj_list(file_):
    with open(file_, 'r') as txt:
        data = txt.read()
        data = data.split('\n')
        data = list(filter(None, data)) # remove empty str
    return data


def comp_density(mat, thresholds=None):
    # graph = nx.from_numpy_array(mat.astype([("weight", float)]))
    # density = nx.density(graph)

    triu_only = mat[np.triu_indices_from(mat, k=1)]  # k=1 because we dont want diag

    if thresholds is None:
        threshold_vals = np.percentile(triu_only, [range(100)]).squeeze()
    if isinstance(thresholds, (int, float)):
        threshold_vals = [np.percentile(triu_only, thresholds).squeeze()]
    else:
        threshold_vals = np.percentile(triu_only, thresholds).squeeze()

    ret = np.zeros_like(threshold_vals)

    for idx, th in enumerate(threshold_vals):
        graph = nx.from_numpy_array((mat > th).astype(int))
        density = nx.density(graph)  # TODO: unefficient, could replace with simple matrix calc
        ret[idx] = density
        logging.debug(f'[I] {idx} with threshold {th} and density {density}')

    if isinstance(thresholds, (int, float)):
        return density
    else:
        return ret


def comp_small_world(mat, coeff_type='sigma', asym=True):
    graph = nx.from_numpy_array(mat.astype([("weight", float)]))

    if asym:
        asym_data = np.triu(mat)
        graph = nx.from_numpy_array(asym_data)

    if coeff_type == 'sigma':
        sigma = nx.sigma(graph, niter=2, nrand=1)#, seed=0)
        small_world_coeff = sigma

    if coeff_type == 'omega':
        omega = nx.omega(graph, niter=2, nrand=1)#, seed=0)
        small_world_coeff = omega

    return small_world_coeff


def comp_degree(mat, use_weights=True): # for weighted matrices returns node strength

    if use_weights:
        graph = nx.from_numpy_array(mat.astype([("weight", float)]))
        degrees = [k[1] for k in graph.degree(weight='weight')]
    else:
        graph = nx.from_numpy_array(mat)
        degrees = [k[1] for k in nx.degree_centrality(graph).items()]

    return np.array(degrees)


def comp_connectivity_ratio(tractogram_path, mat):
    # Load the tractogram file
    tractogram = nib.streamlines.load(tractogram_path)
    streamlines = tractogram.streamlines
    all_streamlines_num = len(streamlines)
    connected_streamlines_num = np.sum(mat)
    connectivity_ratio = connected_streamlines_num/all_streamlines_num

    return connectivity_ratio


def comp_clustering_coeff(mat, use_weights=True):
    if use_weights:
        graph = nx.from_numpy_array(mat.astype([("weight", float)]))
        cc = nx.clustering(graph, weight='weight')
    else:
        graph = nx.from_numpy_array(mat)
        cc = nx.clustering(graph)

    cc = [cc[idx] for idx in range(mat.shape[-1])]

    return np.array(cc)

def comp_betweenness_centrality(mat, k = 2, use_weights=True, normalized=True):

    if use_weights:
        distance_mat = np.divide(1, mat,
                                 out=np.zeros_like(mat),
                                 where=mat!=0)
        graph = nx.from_numpy_array(distance_mat.astype([("weight", float)]))
        btwc = nx.algorithms.centrality.betweenness_centrality(graph,
                                                               k=k,
                                                               normalized=normalized,
                                                               weight='weight')
    else:
        graph = nx.from_numpy_array(mat)
        btwc = nx.algorithms.centrality.betweenness_centrality(graph,
                                                               k=k,
                                                               normalized=normalized)

    btwc = [btwc[idx] for idx in range(mat.shape[-1])]

    return np.array(btwc)

def get_metrics(filtered='', metric='', root_path=''):
    if metric:
        base_connectome_path = f'_{metric}'
    else:
        base_connectome_path = ''

    if filtered:
        tractogram_file = 'tractogram/tck2dwi_tracks.tck'
        connectome_file = f'connectivity_finta/connectome{base_connectome_path}.csv'
    else:
        tractogram_file = 'tractogram/compressed_.tck'
        connectome_file = f'connectivity/connectome{base_connectome_path}.csv'

    tractogram_paths = []
    connectome_paths = []
    connectivity_ratios = []
    graph_densities = []
    clustering_coeffs = []
    nodestrengths = []
    centralities = []

    for i, data in tqdm(subject_df.iterrows()):  # .loc[(subject_df['Group'] == 'MCI') & (subject_df['Subject'] != 19)]:
        if data['Subject'] != 19:
            tractogram_paths.append(os.path.join(root_path, f'{data["Subject"]:02d}', tractogram_file))
            connectome_paths.append(os.path.join(root_path, f'{data["Subject"]:02d}', connectome_file))
            connectome = np.loadtxt(connectome_paths[-1], delimiter=' ')
            connectome = np.nan_to_num(connectome)

            # connectivity_ratios.append(comp_connectivity_ratio(tractogram_paths[-1], connectome))
            # graph_densities.append(comp_density(connectome, thresholds=0))

            clustering_coeffs.append(comp_clustering_coeff(connectome, use_weights=True))
            if metric:
                nodestrengths.append(comp_degree(connectome, use_weights=True))
            else:
                nodestrengths.append(comp_degree(connectome, use_weights=False))
            centralities.append(comp_betweenness_centrality(connectome, k=50, use_weights=True, normalized=True))

    return clustering_coeffs, nodestrengths, centralities


# out_path = '/home/teodorps/results_analysis/graphs'
results_path = '/home/teodorps/filtering_results/finta_test2/'  # '/home/Data/marvin_data/results/datasink/'

clustering_coeffs_ufilt, nodestrengths_ufilt, centralities_ufilt = get_metrics(filtered='', metric='fa',
                                                                               root_path=results_path)
clustering_coeffs_filt, nodestrengths_filt, centralities_filt = get_metrics(filtered='filtered', metric='fa',
                                                                            root_path=results_path)

def print_avg(x):
    print('mean:', np.mean(np.array(x)))
    print('std:', np.std(np.array(x)))
    return






