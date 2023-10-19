import os
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import bct as bct
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


def plot_metrics(clustering_coeffs, nodestrengths, centralities, filtered='', metric='', out_path=''):
    # Define the path to the connectivity matrices
    if metric:
        title_ending_metric = f'- {metric.upper()} Weighted'
    else:
        title_ending_metric = ''

    if filtered:
        title_ending = '- Filtered'
    else:
        title_ending = '- Unfiltered'

    # https://www.geeksforgeeks.org/sort-boxplot-by-mean-with-seaborn-in-python/
    parcellation_labels_path = '/home/teodorps/home_backup/teodorps/HOA_labelID_110.txt'
    labels_parcel = pd.read_csv(parcellation_labels_path, sep='\t', index_col=[0], header=None, names=['Region'])
    subject_df = pd.read_csv('/home/teodorps/home_backup/teodorps/home_folders/results_analysis/group_labels.csv', index_col=False)
    num_ctl = len(subject_df['Subject'].loc[(subject_df['Group'] == 'Control')])
    num_mci = len(subject_df['Subject'].loc[(subject_df['Group'] == 'MCI')])


    df_mean_clustering = pd.DataFrame(clustering_coeffs)
    df_mean_clustering.columns = labels_parcel['Region'].tolist()

    index_sort = df_mean_clustering.mean().sort_values(ascending=False).index
    df_clustering_sorted = df_mean_clustering[index_sort]
    df_clustering_sorted_top_ten = df_clustering_sorted.iloc[:, :10]
    df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()
    df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()

    stacked_clustering = df_clustering_sorted_top_ten.stack()
    stacked_clustering_frame = stacked_clustering.to_frame()
    stacked_clustering_frame = stacked_clustering_frame.reset_index()
    stacked_clustering_frame.columns = ['Subject', 'Node', 'Value']

    clust_ctl = df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()
    clust_mci = df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()

    m = clust_mci.loc['mean'].shape[0]
    cell_text_mean = np.zeros(m * 2, dtype=clust_mci.loc['mean'].dtype)
    cell_text_mean[::2] = clust_mci.loc['mean']
    cell_text_mean[1::2] = clust_ctl.loc['mean']

    cell_text_std = np.zeros(m * 2, dtype=clust_mci.loc['std'].dtype)
    cell_text_std[::2] = clust_mci.loc['std']
    cell_text_std[1::2] = clust_ctl.loc['std']

    cell_text = np.array([cell_text_mean, cell_text_std])

    col_labels = df_clustering_sorted_top_ten.columns.tolist()
    for i in range(0, len(stacked_clustering_frame)):
        stacked_clustering_frame.loc[i, "Subject"] = stacked_clustering_frame.loc[i, "Subject"] + 1

    stacked_clustering_frame['Group'] = ['MCI' if x < num_mci + 1 else 'Control' for x in
                                         stacked_clustering_frame['Subject']]

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(data=stacked_clustering_frame, x=stacked_clustering_frame["Node"],
                     y=stacked_clustering_frame["Value"], hue=stacked_clustering_frame["Group"],
                     showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                                                      markeredgecolor='black'))
    ax.set(xlabel=None)
    plt.xticks(rotation=45, ha="right", fontsize=12, wrap=True)
    plt.yticks(fontsize=11)
    ####################################
    # plt.ylim(0, 1)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ####################################
    plt.ylabel('Clustering Coefficient', fontsize=12)
    plt.legend(loc='lower left', borderpad=1, prop={'size': 15}, labelspacing=1)
    add_median_labels(ax)
    the_table = plt.table(cellText=np.round(cell_text, decimals=3),
                          rowLabels=['mean', 'std'], cellLoc='center',  # colLabels=col_labels,
                          loc='top', fontsize=18)
    the_table.scale(1, 1.4)
    plt.subplots_adjust(left=0.2, bottom=0.1)
    ax.set_title(f'Clustering Coefficient of MCI and CTL {title_ending} {title_ending_metric}', fontsize=18, pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'boxplot_clustering_coeff_{metric}_{filtered}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(out_path, f'boxplot_clustering_coeff_{metric}_{filtered}.pdf'), bbox_inches='tight')
    plt.close()

    df_mean_nodestrengths = pd.DataFrame(nodestrengths)
    df_mean_nodestrengths.columns = labels_parcel['Region'].tolist()

    index_sort = df_mean_nodestrengths.mean().sort_values(ascending=False).index
    df_nodestrengths_sorted = df_mean_nodestrengths[index_sort]
    df_nodestrengths_sorted_top_ten = df_nodestrengths_sorted.iloc[:, :10]
    df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()
    df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()

    stacked_nodestrengths = df_nodestrengths_sorted_top_ten.stack()
    stacked_nodestrengths_frame = stacked_nodestrengths.to_frame()
    stacked_nodestrengths_frame = stacked_nodestrengths_frame.reset_index()
    stacked_nodestrengths_frame.columns = ['Subject', 'Node', 'Value']

    nodestrengths_ctl = df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()
    nodestrengths_mci = df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()

    m = clust_mci.loc['mean'].shape[0]
    cell_text_mean = np.zeros(m * 2, dtype=nodestrengths_mci.loc['mean'].dtype)
    cell_text_mean[::2] = nodestrengths_mci.loc['mean']
    cell_text_mean[1::2] = nodestrengths_ctl.loc['mean']

    cell_text_std = np.zeros(m * 2, dtype=nodestrengths_mci.loc['std'].dtype)
    cell_text_std[::2] = nodestrengths_mci.loc['std']
    cell_text_std[1::2] = nodestrengths_ctl.loc['std']

    cell_text = np.array([cell_text_mean, cell_text_std])

    col_labels = df_nodestrengths_sorted_top_ten.columns.tolist()
    for i in range(0, len(stacked_nodestrengths_frame)):
        stacked_nodestrengths_frame.loc[i, "Subject"] = stacked_nodestrengths_frame.loc[i, "Subject"] + 1

    stacked_nodestrengths_frame['Group'] = ['MCI' if x < num_mci + 1 else 'Control' for x in
                                            stacked_nodestrengths_frame['Subject']]

    plt.figure(figsize=(12, 8))
    # plt.figure()
    ax = sns.boxplot(data=stacked_nodestrengths_frame, x=stacked_nodestrengths_frame["Node"],
                     y=stacked_nodestrengths_frame["Value"], hue=stacked_nodestrengths_frame["Group"],
                     showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                                                      markeredgecolor='black'))
    ax.set(xlabel=None)
    plt.xticks(rotation=45, ha="right", fontsize=12, wrap=True)
    plt.yticks(fontsize=11)
    ####################################
    # plt.ylim(0, 1)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ####################################
    if metric:
        plt.ylabel('Nodestrength', fontsize=12)
    else:
        plt.ylabel('Degree Centrality', fontsize=12)
    plt.legend(loc='lower left', borderpad=1, prop={'size': 15}, labelspacing=1)
    add_median_labels(ax)
    the_table = plt.table(cellText=np.round(cell_text, decimals=3),
                          rowLabels=['mean', 'std'], cellLoc='center',  # colLabels=col_labels,
                          loc='top', fontsize=18)
    the_table.scale(1, 1.4)
    if metric:
        ax.set_title(f'Nodestrength of MCI and CTL {title_ending} {title_ending_metric}', fontsize=18, pad=40)
    else:
        ax.set_title(f'Degree Centrality of MCI and CTL {title_ending}', fontsize=18, pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'boxplot_nodestrength_{metric}_{filtered}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(out_path, f'boxplot_nodestrength_{metric}_{filtered}.pdf'), bbox_inches='tight')
    plt.close()

    df_mean_centralities = pd.DataFrame(centralities)
    df_mean_centralities.columns = labels_parcel['Region'].tolist()

    index_sort = df_mean_centralities.mean().sort_values(ascending=False).index
    df_centralities_sorted = df_mean_centralities[index_sort]
    df_centralities_sorted_top_ten = df_centralities_sorted.iloc[:, :10]
    df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()
    df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()

    stacked_centralities = df_centralities_sorted_top_ten.stack()
    stacked_centralities_frame = stacked_centralities.to_frame()
    stacked_centralities_frame = stacked_centralities_frame.reset_index()
    stacked_centralities_frame.columns = ['Subject', 'Node', 'Value']

    centralities_ctl = df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()
    centralities_mci = df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()

    m = clust_mci.loc['mean'].shape[0]
    cell_text_mean = np.zeros(m * 2, dtype=centralities_mci.loc['mean'].dtype)
    cell_text_mean[::2] = centralities_mci.loc['mean']
    cell_text_mean[1::2] = centralities_ctl.loc['mean']

    cell_text_std = np.zeros(m * 2, dtype=centralities_mci.loc['std'].dtype)
    cell_text_std[::2] = centralities_mci.loc['std']
    cell_text_std[1::2] = centralities_ctl.loc['std']

    cell_text = np.array([cell_text_mean, cell_text_std])

    col_labels = df_centralities_sorted_top_ten.columns.tolist()
    for i in range(0, len(stacked_centralities_frame)):
        stacked_centralities_frame.loc[i, "Subject"] = stacked_centralities_frame.loc[i, "Subject"] + 1

    stacked_centralities_frame['Group'] = ['MCI' if x < num_mci + 1 else 'Control' for x in
                                           stacked_centralities_frame['Subject']]
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(data=stacked_centralities_frame, x=stacked_centralities_frame["Node"],
                     y=stacked_centralities_frame["Value"], hue=stacked_centralities_frame["Group"],
                     showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                                                      markeredgecolor='black'))
    ax.set(xlabel=None)
    plt.xticks(rotation=45, ha="right", fontsize=12, wrap=True)
    plt.yticks(fontsize=11)
    ####################################
    # plt.ylim(0, 1)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ####################################
    plt.ylabel('Betweenness Centrality', fontsize=12)
    plt.legend(loc='lower left', borderpad=1, prop={'size': 15}, labelspacing=1)
    add_median_labels(ax)
    the_table = plt.table(cellText=np.round(cell_text, decimals=3),
                          rowLabels=['mean', 'std'], cellLoc='center',  # colLabels=col_labels,
                          loc='top', fontsize=18)
    the_table.scale(1, 1.4)
    ax.set_title(f'Betweenness Centrality of MCI and CTL {title_ending} {title_ending_metric}', fontsize=18, pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'boxplot_centrality_{metric}_{filtered}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(out_path, f'boxplot_centrality_{metric}_{filtered}.pdf'), bbox_inches='tight')

    plt.close()

    return


def plot_metrics_diffs(clustering_coeffs, nodestrengths, centralities, metric='', out_path='', outliers=False):
    # Define the path to the connectivity matrices
    if metric:
        title_ending_metric = f'- {metric.upper()} Weighted'
    else:
        title_ending_metric = ''

    title_ending = '- ' r'$\Delta V$'


    # https://www.geeksforgeeks.org/sort-boxplot-by-mean-with-seaborn-in-python/
    parcellation_labels_path = '/home/teodorps/home_backup/teodorps/HOA_labelID_110.txt'
    labels_parcel = pd.read_csv(parcellation_labels_path, sep='\t', index_col=[0], header=None, names=['Region'])
    subject_df = pd.read_csv('/home/teodorps/home_backup/teodorps/home_folders/results_analysis/group_labels.csv', index_col=False)
    num_ctl = len(subject_df['Subject'].loc[(subject_df['Group'] == 'Control')])
    num_mci = len(subject_df['Subject'].loc[(subject_df['Group'] == 'MCI')])


    df_mean_clustering = pd.DataFrame(clustering_coeffs)
    df_mean_clustering.columns = labels_parcel['Region'].tolist()

    index_sort = df_mean_clustering.mean().abs().sort_values(ascending=False).index
    df_clustering_sorted = df_mean_clustering[index_sort]
    df_clustering_sorted_top_ten = df_clustering_sorted.iloc[:, :10]
    df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()
    df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()

    stacked_clustering = df_clustering_sorted_top_ten.stack()
    stacked_clustering_frame = stacked_clustering.to_frame()
    stacked_clustering_frame = stacked_clustering_frame.reset_index()
    stacked_clustering_frame.columns = ['Subject', 'Node', 'Value']

    clust_ctl = df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()
    clust_mci = df_clustering_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()

    m = clust_mci.loc['mean'].shape[0]
    cell_text_mean = np.zeros(m * 2, dtype=clust_mci.loc['mean'].dtype)
    cell_text_mean[::2] = clust_mci.loc['mean']
    cell_text_mean[1::2] = clust_ctl.loc['mean']

    cell_text_std = np.zeros(m * 2, dtype=clust_mci.loc['std'].dtype)
    cell_text_std[::2] = clust_mci.loc['std']
    cell_text_std[1::2] = clust_ctl.loc['std']

    cell_text = np.array([cell_text_mean, cell_text_std])

    col_labels = df_clustering_sorted_top_ten.columns.tolist()
    for i in range(0, len(stacked_clustering_frame)):
        stacked_clustering_frame.loc[i, "Subject"] = stacked_clustering_frame.loc[i, "Subject"] + 1

    stacked_clustering_frame['Group'] = ['MCI' if x < num_mci + 1 else 'Control' for x in
                                         stacked_clustering_frame['Subject']]

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(data=stacked_clustering_frame, x=stacked_clustering_frame["Node"],
                     y=stacked_clustering_frame["Value"], hue=stacked_clustering_frame["Group"],
                     showfliers=outliers, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                                                      markeredgecolor='black'),
                     medianprops=dict(linestyle='-', color='navy', linewidth=2.5), width=0.65)
    ax.set(xlabel=None)
    plt.xticks(rotation=45, ha="right", fontsize=12, wrap=True)
    plt.yticks(fontsize=11)
    ax.set_axisbelow(True) # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
    ax.grid(axis='y')
    plt.axhline(y=0, color="purple", linestyle=(0, (5, 5)), linewidth=2)
    ####################################
    # plt.ylim(0, 1)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ####################################
    plt.ylabel(r'$\Delta V$'' of Clustering Coefficient', fontsize=12)
    plt.legend(loc='lower left', borderpad=1, prop={'size': 15}, labelspacing=1)
    # add_median_labels(ax)
    the_table = plt.table(cellText=np.round(cell_text, decimals=3),
                          rowLabels=['mean', 'std'], cellLoc='center',  # colLabels=col_labels,
                          loc='top', fontsize=18.5)
    the_table.scale(1, 1.4)
    plt.subplots_adjust(left=0.2, bottom=0.1)
    ax.set_title(f'Clustering Coefficient of MCI and CTL {title_ending} {title_ending_metric}', fontsize=18, pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'boxplot_clustering_coeff_{metric}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(out_path, f'boxplot_clustering_coeff_{metric}.pdf'), bbox_inches='tight')
    plt.close()

    df_mean_nodestrengths = pd.DataFrame(nodestrengths)
    df_mean_nodestrengths.columns = labels_parcel['Region'].tolist()

    index_sort = df_mean_nodestrengths.mean().abs().sort_values(ascending=False).index
    df_nodestrengths_sorted = df_mean_nodestrengths[index_sort]
    df_nodestrengths_sorted_top_ten = df_nodestrengths_sorted.iloc[:, :10]
    df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()
    df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()

    stacked_nodestrengths = df_nodestrengths_sorted_top_ten.stack()
    stacked_nodestrengths_frame = stacked_nodestrengths.to_frame()
    stacked_nodestrengths_frame = stacked_nodestrengths_frame.reset_index()
    stacked_nodestrengths_frame.columns = ['Subject', 'Node', 'Value']

    nodestrengths_ctl = df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()
    nodestrengths_mci = df_nodestrengths_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()

    m = clust_mci.loc['mean'].shape[0]
    cell_text_mean = np.zeros(m * 2, dtype=nodestrengths_mci.loc['mean'].dtype)
    cell_text_mean[::2] = nodestrengths_mci.loc['mean']
    cell_text_mean[1::2] = nodestrengths_ctl.loc['mean']

    cell_text_std = np.zeros(m * 2, dtype=nodestrengths_mci.loc['std'].dtype)
    cell_text_std[::2] = nodestrengths_mci.loc['std']
    cell_text_std[1::2] = nodestrengths_ctl.loc['std']

    cell_text = np.array([cell_text_mean, cell_text_std])

    col_labels = df_nodestrengths_sorted_top_ten.columns.tolist()
    for i in range(0, len(stacked_nodestrengths_frame)):
        stacked_nodestrengths_frame.loc[i, "Subject"] = stacked_nodestrengths_frame.loc[i, "Subject"] + 1

    stacked_nodestrengths_frame['Group'] = ['MCI' if x < num_mci + 1 else 'Control' for x in
                                            stacked_nodestrengths_frame['Subject']]

    plt.figure(figsize=(12, 8))
    # plt.figure()
    ax = sns.boxplot(data=stacked_nodestrengths_frame, x=stacked_nodestrengths_frame["Node"],
                     y=stacked_nodestrengths_frame["Value"], hue=stacked_nodestrengths_frame["Group"],
                     showfliers=outliers, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                                                      markeredgecolor='black'),
                     medianprops=dict(linestyle='-', color='navy', linewidth=2.5), width=0.65)
    ax.set(xlabel=None)
    plt.xticks(rotation=45, ha="right", fontsize=12, wrap=True)
    plt.yticks(fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    plt.axhline(y=0, color="purple", linestyle=(0, (5, 5)), linewidth=2)
    ####################################
    # plt.ylim(0, 1)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ####################################
    if metric:
        plt.ylabel(r'$\Delta V$'' of Nodestrength', fontsize=12)
    else:
        plt.ylabel(r'$\Delta V$'' of Degree Centrality', fontsize=12)
    plt.legend(loc='lower left', borderpad=1, prop={'size': 15}, labelspacing=1)
    # add_median_labels(ax)
    the_table = plt.table(cellText=np.round(cell_text, decimals=3),
                          rowLabels=['mean', 'std'], cellLoc='center',  # colLabels=col_labels,
                          loc='top', fontsize=18.5)
    the_table.scale(1, 1.4)
    if metric:
        ax.set_title(f'Nodestrength of MCI and CTL {title_ending} {title_ending_metric}', fontsize=18, pad=40)
    else:
        ax.set_title(f'Degree Centrality of MCI and CTL {title_ending}', fontsize=18, pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'boxplot_nodestrength_{metric}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(out_path, f'boxplot_nodestrength_{metric}.pdf'), bbox_inches='tight')
    plt.close()

    df_mean_centralities = pd.DataFrame(centralities)
    df_mean_centralities.columns = labels_parcel['Region'].tolist()

    index_sort = df_mean_centralities.mean().abs().sort_values(ascending=False).index
    df_centralities_sorted = df_mean_centralities[index_sort]
    df_centralities_sorted_top_ten = df_centralities_sorted.iloc[:, :10]
    df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()
    df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()

    stacked_centralities = df_centralities_sorted_top_ten.stack()
    stacked_centralities_frame = stacked_centralities.to_frame()
    stacked_centralities_frame = stacked_centralities_frame.reset_index()
    stacked_centralities_frame.columns = ['Subject', 'Node', 'Value']

    centralities_ctl = df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'Control')].describe()
    centralities_mci = df_centralities_sorted_top_ten.loc[(subject_df['Group'] == 'MCI')].describe()

    m = clust_mci.loc['mean'].shape[0]
    cell_text_mean = np.zeros(m * 2, dtype=centralities_mci.loc['mean'].dtype)
    cell_text_mean[::2] = centralities_mci.loc['mean']
    cell_text_mean[1::2] = centralities_ctl.loc['mean']

    cell_text_std = np.zeros(m * 2, dtype=centralities_mci.loc['std'].dtype)
    cell_text_std[::2] = centralities_mci.loc['std']
    cell_text_std[1::2] = centralities_ctl.loc['std']

    cell_text = np.array([cell_text_mean, cell_text_std])

    col_labels = df_centralities_sorted_top_ten.columns.tolist()
    for i in range(0, len(stacked_centralities_frame)):
        stacked_centralities_frame.loc[i, "Subject"] = stacked_centralities_frame.loc[i, "Subject"] + 1

    stacked_centralities_frame['Group'] = ['MCI' if x < num_mci + 1 else 'Control' for x in
                                           stacked_centralities_frame['Subject']]
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(data=stacked_centralities_frame, x=stacked_centralities_frame["Node"],
                     y=stacked_centralities_frame["Value"], hue=stacked_centralities_frame["Group"],
                     showfliers=outliers, showmeans=True, meanprops=dict(marker='o', markerfacecolor='red', markersize=6,
                                                                      markeredgecolor='black'),
                     medianprops=dict(linestyle='-', color='navy', linewidth=2.5), width=0.65)
    ax.set(xlabel=None)
    plt.xticks(rotation=45, ha="right", fontsize=12, wrap=True)
    plt.yticks(fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    plt.axhline(y=0, color="purple", linestyle=(0, (5, 5)), linewidth=2)
    ####################################
    # plt.ylim(0, 1)
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    ####################################
    plt.ylabel(r'$\Delta V$'' of Betweenness Centrality', fontsize=12)
    plt.legend(loc='lower left', borderpad=1, prop={'size': 15}, labelspacing=1)
    # add_median_labels(ax)
    the_table = plt.table(cellText=np.round(cell_text, decimals=3),
                          rowLabels=['mean', 'std'], cellLoc='center',  # colLabels=col_labels,
                          loc='top', fontsize=18.5)
    the_table.scale(1, 1.4)
    ax.set_title(f'Betweenness Centrality of MCI and CTL {title_ending} {title_ending_metric}', fontsize=18, pad=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f'boxplot_centrality_{metric}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(out_path, f'boxplot_centrality_{metric}.pdf'), bbox_inches='tight')

    plt.close()

    return


def get_metrics(filtered=True, metric='', root_path='', grp_labels_path=''):
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

    subject_df = pd.read_csv(grp_labels_path, index_col=False)

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
results_path = '/home/teodorps/home_backup/teodorps/filtering_results/finta_test2/'  # '/home/Data/marvin_data/results/datasink/'
group_labels_path = '/home/teodorps/home_backup/teodorps/home_folders/results_analysis/group_labels.csv'

# filtered = True
# metric = 'md'
clustering_coeffs_ufilt, nodestrengths_ufilt, centralities_ufilt = get_metrics(filtered=False, metric='rd',
                                                                               root_path=results_path,
                                                                               grp_labels_path=group_labels_path)

clustering_coeffs_filt, nodestrengths_filt, centralities_filt = get_metrics(filtered=True, metric='rd',
                                                                            root_path=results_path,
                                                                            grp_labels_path=group_labels_path)


def get_diffs(ufilt_met, filt_met):
    diff_met = [i - j for i, j in zip(ufilt_met, filt_met)]
    return diff_met

clustering_coeffs_diffs = get_diffs(ufilt_met=clustering_coeffs_ufilt, filt_met=clustering_coeffs_filt)
nodestrengths_diffs = get_diffs(ufilt_met=nodestrengths_ufilt, filt_met=nodestrengths_filt)
centralities_diffs = get_diffs(ufilt_met=centralities_ufilt, filt_met=centralities_filt)



out_path = '/home/teodorps/home_backup/teodorps/home_folders/results_analysis/diff_graphs_outliers'


plot_metrics_diffs(clustering_coeffs_diffs, nodestrengths_diffs, centralities_diffs, metric='rd',
             out_path=out_path, outliers=True)






def norm_filt_ufilt(ufilt_met, filt_met):
    max_ufilt = max([np.max(i) for i in ufilt_met])
    max_filt = max([np.max(i) for i in filt_met])
    min_ufilt = min([np.min(i) for i in ufilt_met])
    min_filt = min([np.min(i) for i in filt_met])

    max_both = max(max_filt, max_ufilt)
    min_both = min(min_filt, min_ufilt)

    ufilt_met_norm = [(i-min_both)/(max_both-min_both) for i in ufilt_met]
    filt_met_norm = [(i-min_both)/(max_both-min_both) for i in filt_met]

    return ufilt_met_norm, filt_met_norm

norm_clustering_ufilt, norm_clustering_filt = norm_filt_ufilt(clustering_coeffs_ufilt, clustering_coeffs_filt)
norm_nodestrengths_ufilt, norm_nodestrengths_filt = norm_filt_ufilt(nodestrengths_ufilt, nodestrengths_filt)
norm_centralities_ufilt, norm_centralities_filt = norm_filt_ufilt(centralities_ufilt, centralities_filt)





out_path = '/home/teodorps/results_analysis/graphs_adjusted_2'
# plot_metrics(norm_clustering_ufilt,norm_nodestrengths_ufilt,norm_centralities_ufilt, filtered='', metric='rd',
#              out_path=out_path)
#
# plot_metrics(norm_clustering_filt,norm_nodestrengths_filt,norm_centralities_filt, filtered='filtered', metric='rd',
#              out_path=out_path)

plot_metrics(clustering_coeffs_ufilt, nodestrengths_ufilt, centralities_ufilt, filtered='', metric='fa',
             out_path=out_path)

plot_metrics(clustering_coeffs_filt, nodestrengths_filt, centralities_filt, filtered='filtered', metric='fa',
             out_path=out_path)






