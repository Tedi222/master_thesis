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

def comp_streamlines(tractogram_path):
    # Load the tractogram file
    tractogram = nib.streamlines.load(tractogram_path)
    streamlines = tractogram.streamlines
    all_streamlines_num = len(streamlines)
    filtered_num = all_streamlines_num
    filtered_ratio = (all_streamlines_num/1000000)*100

    return [filtered_ratio, filtered_num]

subject_df = pd.read_csv('/home/teodorps/results_analysis/group_labels.csv', index_col=False)

out_path = '/home/teodorps/results_analysis/graphs_adjusted_2'

filtered = ''
metric = ''

filtered_tractogram_file = 'tractogram/tck2dwi_tracks.tck'

# Define the path to the tractogram files
results_path = '/home/teodorps/filtering_results/finta_test2/'  # '/home/Data/marvin_data/results/datasink/'

if filtered:
    tractogram_file = 'tractogram/tck2dwi_tracks.tck'
else:
    tractogram_file = 'finta_cluster/sft_implausible.trk'#'tractogram/compressed_.tck'

# Define the path to the connectivity matrices

if metric:
    base_connectome_path = f'_{metric}'
else:
    base_connectome_path = ''

if filtered:
    connectome_file = f'connectivity_finta/connectome{base_connectome_path}.csv'
else:
    connectome_file = f'connectivity/connectome{base_connectome_path}.csv'

tractogram_paths = []
connectome_paths = []
connectivity_ratios = []
graph_densities = []

streamline_nums = []
streamline_ratios = []


for i, data in tqdm(subject_df.iterrows()):  #.loc[(subject_df['Group'] == 'MCI') & (subject_df['Subject'] != 19)]:
    if data['Subject'] != 19:
        tractogram_paths.append(os.path.join(results_path, f'{data["Subject"]:02d}', tractogram_file))
        connectome_paths.append(os.path.join(results_path, f'{data["Subject"]:02d}', connectome_file))
        connectome = np.loadtxt(connectome_paths[-1], delimiter=' ')
        connectome = np.nan_to_num(connectome)
        streamline_tmp = comp_streamlines(tractogram_paths[-1])
        streamline_nums.append(streamline_tmp[1])
        streamline_ratios.append(streamline_tmp[0])

        # connectivity_ratios.append(comp_connectivity_ratio(tractogram_paths[-1], connectome))
        # graph_densities.append(comp_density(connectome, thresholds=0))



dens_ = pd.DataFrame({"Density": graph_densities})
dens_["Group"] = subject_df['Group'].loc[(subject_df['Subject'] != 19)]


# plt.figure(figsize=(4, 4))
# plt.subplot(121)
# plt.suptitle('Density')
# sns.boxplot(data=dens_, y=dens_["Density"], x=dens_["Group"], notch=True)
# if filtered:
#     plt.savefig(os.path.join(out_path,f'boxplot_densities_{filtered}.png'))
# else:
#     plt.savefig(os.path.join(out_path, 'boxplot_densities.png'))


# conn_ = pd.DataFrame({"Connectivity Ratio": connectivity_ratios})
# conn_["Group"] = subject_df['Group'].loc[(subject_df['Subject'] != 19)]
#
# plt.subplot(122)
# plt.suptitle('Density')
# sns.boxplot(data=conn_, y=conn_["Connectivity Ratio"], x=conn_["Group"], notch=True)
# plt.tight_layout()
# if filtered:
#     plt.savefig(os.path.join(out_path,f'boxplot_connectivity_{filtered}.png'))
# else:
#     plt.savefig(os.path.join(out_path, 'boxplot_connectivity.png'))
# plt.close()




filtered = ''
metric = ''

filtered_tractogram_file = 'tractogram/tck2dwi_tracks.tck'

# Define the path to the tractogram files
results_path = '/home/teodorps/filtering_results/finta_test2/'  # '/home/Data/marvin_data/results/datasink/'

if filtered:
    tractogram_file = 'tractogram/tck2dwi_tracks.tck'
else:
    tractogram_file = 'tractogram/compressed_.tck'

if metric:
    base_connectome_path = f'_{metric}'
else:
    base_connectome_path = ''

if filtered:
    connectome_file = f'connectivity_finta/connectome{base_connectome_path}.csv'
else:
    connectome_file = f'connectivity/connectome{base_connectome_path}.csv'


tractogram_paths_filtered = []
connectome_paths_filtered = []
connectivity_ratios_filtered = []
graph_densities_filtered = []


for i, data in tqdm(subject_df.iterrows()):  #.loc[(subject_df['Group'] == 'MCI') & (subject_df['Subject'] != 19)]:
    if data['Subject'] != 19:
        tractogram_paths_filtered.append(os.path.join(results_path, f'{data["Subject"]:02d}', tractogram_file))
        connectome_paths_filtered.append(os.path.join(results_path, f'{data["Subject"]:02d}', connectome_file))
        connectome_filtered = np.loadtxt(connectome_paths_filtered[-1], delimiter=' ')
        connectome_filtered = np.nan_to_num(connectome_filtered)

        connectivity_ratios_filtered.append(comp_connectivity_ratio(tractogram_paths_filtered[-1], connectome_filtered))
        graph_densities_filtered.append(comp_density(connectome_filtered, thresholds=0))


dens_filt = pd.DataFrame({"Density": graph_densities_filtered})
dens_filt["Group"] = subject_df['Group'].loc[(subject_df['Subject'] != 19)]


# plt.figure(figsize=(4, 4))
# plt.subplot(121)
# plt.suptitle('Density')
# sns.boxplot(data=dens_filt, y=dens_filt["Density"], x=dens_filt["Group"], notch=True)
# if filtered:
#     plt.savefig(os.path.join(out_path,f'boxplot_densities_{filtered}.png'))
# else:
#     plt.savefig(os.path.join(out_path, 'boxplot_densities.png'))


conn_filt = pd.DataFrame({"Connectivity Ratio": connectivity_ratios_filtered})
conn_filt["Group"] = subject_df['Group'].loc[(subject_df['Subject'] != 19)]

# plt.subplot(122)
# plt.suptitle('Density')
# sns.boxplot(data=conn_filt, y=conn_filt["Connectivity Ratio"], x=conn_filt["Group"], notch=True)
# plt.tight_layout()
# if filtered:
#     plt.savefig(os.path.join(out_path,f'boxplot_connectivity_{filtered}.png'))
# else:
#     plt.savefig(os.path.join(out_path, 'boxplot_connectivity.png'))
# plt.close()


dens_fil_tmp = dens_filt
dens_fil_tmp['Status'] = ['Filtered' for i in range(0, len(dens_fil_tmp))]
dens_unfil_tmp = dens_
dens_unfil_tmp['Status'] = ['Unfiltered' for i in range(0, len(dens_unfil_tmp))]
dens_concat = pd.concat([dens_unfil_tmp, dens_fil_tmp])
plt.figure(figsize=(4,4))
ax = sns.boxplot(data=dens_concat, y=dens_concat["Density"], x=dens_concat["Group"], hue =dens_concat['Status'],  notch=True)
sns.despine(offset=10)
plt.legend(bbox_to_anchor=(-.3, -0.3), loc='lower left')#, borderaxespad=0)

cell_text_dens = np.zeros((2,4))
for i, t in enumerate(['mean', 'std']):
    cell_text_dens[i, 0] = dens_concat['Density'].loc[
        (dens_concat['Status'] == 'Unfiltered') & (dens_concat['Group']=='MCI')].describe()[t]
    cell_text_dens[i, 1] = dens_concat['Density'].loc[
        (dens_concat['Status'] == 'Filtered') & (dens_concat['Group'] == 'MCI')].describe()[t]
    cell_text_dens[i, 2] = dens_concat['Density'].loc[
        (dens_concat['Status'] == 'Unfiltered') & (dens_concat['Group'] == 'Control')].describe()[t]
    cell_text_dens[i, 3] = dens_concat['Density'].loc[
        (dens_concat['Status'] == 'Filtered') & (dens_concat['Group'] == 'Control')].describe()[t]

add_median_labels(ax)
the_table = plt.table(cellText=np.round(cell_text_dens, decimals=3),
                      rowLabels=['mean', 'std'], cellLoc='center', #colLabels=col_labels,
                      loc='top', fontsize=18)
the_table.scale(1,1.3)
plt.tight_layout()
plt.savefig(os.path.join(out_path,f'boxplot_density.png'), dpi = 300)
plt.savefig(os.path.join(out_path,f'boxplot_density.pdf'))
plt.close()

conn_fil_tmp = conn_filt
conn_fil_tmp['Status'] = ['Filtered' for i in range(0, len(conn_fil_tmp))]
conn_unfil_tmp = conn_
conn_unfil_tmp['Status'] = ['Unfiltered' for i in range(0, len(conn_unfil_tmp))]
conn_concat = pd.concat([conn_unfil_tmp, conn_fil_tmp])
plt.figure(figsize=(4,4))
ax = sns.boxplot(data=conn_concat, y=conn_concat["Connectivity Ratio"], x=conn_concat["Group"], hue =conn_concat['Status'],  notch=True)
sns.despine(offset=10)
plt.legend(bbox_to_anchor=(-.3, -0.3), loc='lower left')#, borderaxespad=0)

cell_text_conn = np.zeros((2,4))
for i, t in enumerate(['mean', 'std']):
    cell_text_conn[i, 0] = conn_concat['Connectivity Ratio'].loc[
        (conn_concat['Status'] == 'Unfiltered') & (conn_concat['Group']=='MCI')].describe()[t]
    cell_text_conn[i, 1] = conn_concat['Connectivity Ratio'].loc[
        (conn_concat['Status'] == 'Filtered') & (conn_concat['Group'] == 'MCI')].describe()[t]
    cell_text_conn[i, 2] = conn_concat['Connectivity Ratio'].loc[
        (conn_concat['Status'] == 'Unfiltered') & (conn_concat['Group'] == 'Control')].describe()[t]
    cell_text_conn[i, 3] = conn_concat['Connectivity Ratio'].loc[
        (conn_concat['Status'] == 'Filtered') & (conn_concat['Group'] == 'Control')].describe()[t]

add_median_labels(ax)
the_table = plt.table(cellText=np.round(cell_text_conn, decimals=3),
                      rowLabels=['mean', 'std'], cellLoc='center', #colLabels=col_labels,
                      loc='top', fontsize=18)
the_table.scale(1,1.3)
plt.tight_layout()
plt.savefig(os.path.join(out_path,f'boxplot_connectivity.png'), dpi=300)
plt.savefig(os.path.join(out_path,f'boxplot_connectivity.pdf'))
plt.close()
