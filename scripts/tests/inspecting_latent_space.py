import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking import metrics as tm


def get_streamlines_at_indices(sft, idx_list, invert_indices=False):
    """
    Parameters
    ----------

    sft: StatefulTractogram (DIPY)
        Tractogram to filter

    idx_list: list of integers
        Indices of streamlines we want to keep

    invert_indices: Bool
        If the list is the indices of streamlines we want to remove

    Returns
    -------
    new_sft: StatefulTractogram

    """
    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    if invert_indices:
        line_based_indices = np.setdiff1d(
            range(len(stf_nib.streamlines)), np.unique(idx_list)
        )

    line_based_indices = np.asarray(idx_list, dtype=np.int32)

    streamlines = sft.streamlines[line_based_indices]
    data_per_streamline = sft.data_per_streamline[line_based_indices]
    data_per_point = sft.data_per_point[line_based_indices]

    new_sft = StatefulTractogram.from_sft(
        streamlines,
        sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point,
    )

    new_sft.to_space(orig_space)

    return new_sft


def get_streamlines_masked(sft, binary_mask):
    """
    Parameters
    ----------

    sft: StatefulTractogram (DIPY)
        Tractogram to filter

    idx_list: Bool, numpy array
        Mask of streamlines we want to keep

    Returns
    -------
    new_sft: StatefulTractogram

    """
    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    streamlines = sft.streamlines[binary_mask]
    data_per_streamline = sft.data_per_streamline[binary_mask]
    data_per_point = sft.data_per_point[binary_mask]

    new_sft = StatefulTractogram.from_sft(
        streamlines,
        sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point,
    )

    new_sft.to_space(orig_space)

    return new_sft


def filter_streamlines_by_length(sft, min_length=0., max_length=np.inf):
    """
    Filter streamlines using minimum and max length.

    Parameters
    ----------
    sft: StatefulTractogram
        SFT containing the streamlines to filter.
    min_length: float
        Minimum length of streamlines, in mm.
    max_length: float
        Maximum length of streamlines, in mm.

    Return
    ------
    filtered_sft : StatefulTractogram
        A tractogram without short streamlines.
    """

    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    if sft.streamlines:
        # Compute streamlines lengths
        lengths = length(sft.streamlines)

        # Filter lengths
        filter_stream = np.logical_and(lengths >= min_length,
                                       lengths <= max_length)
    else:
        filter_stream = []

    filtered_sft = sft[filter_stream]
    ids = np.where(filter_stream == True)[0]
    ids.tolist()
    # Return to original space
    filtered_sft.to_space(orig_space)

    return filtered_sft, ids, filter_stream


def remove_loops_and_sharp_turns(streamlines,
                                 max_angle):
    """
    Remove loops and sharp turns from a list of streamlines.
    Parameters
    ----------
    streamlines: list of ndarray
        The list of streamlines from which to remove loops and sharp turns.
    max_angle: float
        Maximal winding angle a streamline can have before
        being classified as a loop.

    Returns
    -------
    list: the ids of clean streamlines
        Only the ids are returned so proper filtering can be done afterwards
    """

    streamlines_clean = []
    ids = []
    mask = []
    for i, s in tqdm(enumerate(streamlines), total=len(streamlines), ascii=True, mininterval=2, miniters=700):
        if tm.winding(s) < max_angle:
            ids.append(i)
            mask.append(True)
            # streamlines_clean.append(s)
        else:
            mask.append(False)

    return ids, np.squeeze(mask)


#############################################################


# loading tractogram as StatefulTractogram
# tracogram_path = '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final/tck2mni/tck2mni_tracks.trk'
tracogram_path = '/home/teodorps/filtering_results/finta_test2/__id_02/tck2trk/tck2mni_tracks.trk'
sft = load_tractogram(tracogram_path, 'same')

#############################################################


# Filtering tractogram based on streamline length
sft_filtered_by_length, ids_by_length, mask_by_length = filter_streamlines_by_length(sft, 20, 200)

#############################################################


# Filtering tractogram based on streamline turn angle/winding
ids_by_winding, mask_by_winding = remove_loops_and_sharp_turns(sft.streamlines, max_angle=335)

#############################################################


# I'm using boolean masks where True is a preserved streamline and False is a removed streamline
# i want to make 0 - valid streamline , 1 - streamline removed by length, 2 - streamline removed by winding, 3 streamline removed by both
both = ~mask_by_length & ~mask_by_winding
both_ids = []

for idx in range(len(both)):
    if both[idx] == True: both_ids.append(idx)

classes = np.zeros(len(sft.streamlines), dtype=int)
classes[~mask_by_length] = 1
classes[~mask_by_winding] = 2
classes[both] = 3

print(sum(~mask_by_winding))

#############################################################


# loading model for embedding
import torch
from tractolearn.models.model_pool import get_model

torch.set_flush_denormal(True)

device = 'cpu'
model = '/home/teodorps/tractolearn_data/best_model_contrastive_tractoinferno_hcp.pt'

# Loading the model
checkpoint = torch.load(model, map_location=device, )
state_dict = checkpoint["state_dict"]
model = get_model("IncrFeatStridedConvFCUpsampReflectPadAE", 32, device)
model.load_state_dict(state_dict)
model.eval();

#############################################################

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger = logging.getLogger("root")

logger.info("Loading tractogram ...")

from dipy.io.stateful_tractogram import Space
from tractolearn.tractoio.utils import load_ref_anat_image

# Loading common space = T1 in MNI
common_space_reference_path = '/home/teodorps/data_full/subj02/mni_icbm152_t1_tal_nlin_asym_09c.nii'
mni_img = load_ref_anat_image(common_space_reference_path)

# Loading tractogram
common_space_tractogram = load_tractogram(
    tracogram_path,
    mni_img.header,
    to_space=Space.RASMM,
    trk_header_check=False,
    bbox_valid_check=False,
)

#############################################################


num_points = 256
from scilpy.tracking.tools import resample_streamlines_num_points

# Resampling streamlines to 256 points
streamlines = resample_streamlines_num_points(
    common_space_tractogram, num_points
).streamlines

#############################################################


# Dump streamline data to array

# unfiltered streamlines labels
X = np.vstack(
    [streamlines[i][np.newaxis,] for i in range(len(streamlines))]
)
# Create a vector with unique class labels (created earlier name 'classes')


#############################################################


from tractolearn.learning.dataset import OnTheFlyDataset

# Create dataloaders for more efficient plotting and encoding

dataset = OnTheFlyDataset(X, classes)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=128, shuffle=False
)

#############################################################


from tractolearn.models.autoencoding_utils import encode_data

# Encode tractograms into latent space

latent_f, y_f = encode_data(dataloader, device, model)  # , limit_batch=128)

#############################################################
from sklearn.manifold import TSNE
import time

start = time.time()
tsne = TSNE(n_components=2, random_state=0, verbose=True)
projections = tsne.fit_transform(latent_f)
end = time.time()
print(f"generating projections with T-SNE took: {(end - start):.2f} sec")

#############################################################


# import plotly.express as px
# import plotly.graph_objects as go

# unique_labels = np.unique(classes)
# print(classes)
# traces = []
# track_indices = np.arange(0, len(projections))
# for unique_label in tqdm(unique_labels):
#     mask = classes == unique_label
#     print(type(mask))
#     customdata_masked = track_indices[mask]
#     trace = go.Scatter(
#         x=projections[mask][:, 0],
#         y=projections[mask][:, 1],
#         mode='markers',
#         text=classes[mask],
#         customdata=customdata_masked,
#         name=str(unique_label),
#         marker=dict(size=4),
#         hovertemplate="<b>class: %{text}</b><br>index: %{customdata}<extra></extra>"
#     )
#     traces.append(trace)

# fig = go.Figure(data=traces)

# fig.update_layout(
#     scene=dict(xaxis_title='X', yaxis_title='Y'),
#     width=1000,
#     height=1000,
#     showlegend=True
# )

#############################################################


import matplotlib.pyplot as plt
from matplotlib import colors as cs
import numpy as np
from matplotlib import cm

# colors = cm.get_cmap('Set1')
colors = [(0.1, 0.1, 0.1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
color_map = cs.ListedColormap(colors, name='from_list')
plt.figure(figsize=(15, 15))
plt.scatter(projections[:, 0], projections[:, 1], s=8, c=classes, cmap=color_map)
plt.show()

#############################################################


print(np.__version__) # 1.23.5

#############################################################


import matplotlib.pyplot as plt
from matplotlib import colors as cs
import numpy as np
from matplotlib import cm

# colors = [(0.1,0.1,0.1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
# color_map = cs.ListedColormap(colors, name='from_list')
# plt.figure(figsize=(15,15))
# plt.scatter(projections[:,0], projections[:,1], s=8, c=classes, cmap=color_map)
# plt.show()
plt.figure(figsize=(10, 10))
unique_labels = np.unique(classes)
label_names = ['plausible', 'removed by length', 'removed by winding', 'removed by both']
for unique_label in unique_labels:
    mask = classes == unique_label
    plt.plot(projections[mask][:, 0], projections[mask][:, 1], marker='o', linestyle='', markersize=4,
             label=label_names[unique_label], color=colors[unique_label])

plt.legend()
plt.title('T-SNE projection of 1000000 streamlines')
# plt.savefig('/home/teodorps/t_sne_1M_subj02.png', dpi=200)
plt.savefig('/home/teodorps/t_sne_1M_subj02.pdf', dpi=200)
# plt.savefig('/home/teodorps/t_sne_1M_subj02.svg')
plt.show()

#############################################################


import umap


def fit_umap_metric(latent_vec, result_dimensionality):
    reducer = umap.UMAP(random_state=42,
                        n_components=result_dimensionality, metric='minkowski', verbose=True)
    umap_fitted = reducer.fit(latent_vec)

    return umap_fitted


#############################################################


start = time.time()
mapper_UMAP = fit_umap_metric(latent_f, 2)
projections_UMAP = mapper_UMAP.transform(latent_f)
end = time.time()
print(f"generating projections with UMAP took: {(end - start):.2f} sec")

#############################################################

plt.figure(figsize=(15, 15))
unique_labels = np.unique(classes)
label_names = ['valid', 'removed by length', 'removed by winding', 'removed by both']
for unique_label in unique_labels:
    mask = classes == unique_label
    plt.plot(projections_UMAP[mask][:, 0], projections_UMAP[mask][:, 1], marker='o', linestyle='', markersize=4,
             label=label_names[unique_label], color=colors[unique_label])

plt.legend()
plt.title('UMAP projection of 100000 streamlines')
plt.show()

#############################################################


from sklearn.cluster import DBSCAN
from sklearn import metrics

db = DBSCAN(eps=1, min_samples=10).fit(projections)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

#############################################################


unique_labels = np.unique(labels)[[3, 5]]
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.gist_ncar(each) for each in np.linspace(0.1, 1, len(unique_labels))]
plt.figure(figsize=(10, 10))
for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k

    xy = projections[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=4,
        label=k
    )
plt.legend()
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()

#############################################################

print(set(labels))
print(np.unique(labels))

#############################################################


unique_labels = np.unique(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.gist_ncar(each) for each in np.linspace(0.1, 1, len(unique_labels))]
plt.figure(figsize=(10, 10))
for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k

    xy = projections[class_member_mask & core_samples_mask]
    print(f'label: {k} // {len(xy)}')
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=4,
        label=k
    )
plt.legend()
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()

#############################################################


print(core_samples_mask)
len(np.where(class_member_mask & core_samples_mask == True)[0])

#############################################################

cluster_ids = np.array((2, 4), dtype=int)
all_mask = np.full((len(sft.streamlines)), True, dtype=bool)

core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

cluster_mask = np.zeros_like(labels, dtype=bool)

for k in cluster_ids:
    class_member_mask = labels == k
    cluster_mask[class_member_mask & core_samples_mask] = True

sft_length_cluster = get_streamlines_masked(sft, cluster_mask & ~mask_by_length & ~both)
save_tractogram(sft_length_cluster, 'sft_length_cluster.tck', bbox_valid_check=True)

sft_winding_cluster = get_streamlines_masked(sft, cluster_mask & ~mask_by_winding & ~both)
save_tractogram(sft_winding_cluster, 'sft_winding_cluster.tck', bbox_valid_check=True)

sft_both_cluster = get_streamlines_masked(sft, cluster_mask & both)
save_tractogram(sft_both_cluster, 'sft_both_cluster.tck', bbox_valid_check=True)

sft_valid_cluster = get_streamlines_masked(sft, cluster_mask & mask_by_length & mask_by_winding)
save_tractogram(sft_valid_cluster, 'sft_valid_cluster.tck', bbox_valid_check=True)

#############################################################

print(len(sft_winding_cluster.streamlines))
print(len(sft_length_cluster.streamlines))
print(f'entire cluster {sum(cluster_mask)}')
print(f'valid {sum(cluster_mask & mask_by_length & mask_by_winding)}')
print(f'by winding {sum(cluster_mask & ~mask_by_winding & ~both)}')
print(f'by length {sum(cluster_mask & ~mask_by_length & ~both)}')
print(f'by both {sum(cluster_mask & both)}')

#############################################################


x = np.array((0, 1, 2, 3, 4, 5))
mx_arr = np.array((0, 4, 5))
mx_bool = np.array((True, False, False, False, True, True))

#############################################################

print(x)
print(mx_arr)
print(mx_bool)

#############################################################
print(x[mx_arr])
print(x[mx_bool])

#############################################################

mx = np.array((0, 0))
mx[np.array((True, False))] = False

#############################################################


print(f' removed by winding: {sum(~mask_by_winding & ~both)}')
print(f' removed by length: {sum(~mask_by_length & ~both)}')
print(f' removed by both: {sum(both)}')

#############################################################
print(sum(~mask_by_length))

#############################################################

print(sum(both))
print(100000 - len(ids_by_length))
print(100000 - len(ids_by_winding) - len(both_ids))

#############################################################
sum(core_samples_mask & mask_by_winding & mask_by_length)

#############################################################

print(100000 - 95187)
#############################################################

print(2204 + 1820 + 657)
#############################################################
import dipy

print(dipy.__version__) #1.5.0


##########################
# Perform quickbundles on the original tractogram and get centroids
from dipy.segment.clustering import QuickBundles, QuickBundlesX
from dipy.segment.metric import IdentityFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.metric import ResampleFeature

# make sure that quickbundles output streamlines are 256 points because of the autoencoder input
feature = ResampleFeature(nb_points=256)
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb = QuickBundles(threshold=10., metric=metric)
clusters = qb.cluster(sft.streamlines)

print(len(clusters[0].centroid))

print(clusters[0].centroid)

centroids_sft = StatefulTractogram.from_sft(
    clusters.centroids, sft)

type(centroids_sft)
print(centroids_sft.streamlines[0])

save_tractogram(centroids_sft, 'centroids_sft.tck', bbox_valid_check=True)

#quickbundles vs quickbundlesx
# trying quickbundlesx
thresholds = [30., 20., 10.]
feature = ResampleFeature(nb_points=256)
metric = AveragePointwiseEuclideanMetric(feature)
qbx = QuickBundlesX(thresholds, metric=metric)
clusters_qbx = qbx.cluster(sft.streamlines)
clusters_10 = clusters_qbx.get_clusters(3)

#performance is waaay faster
#after qbX perform this: https://github.com/dipy/dipy/blob/1.3.0/dipy/segment/clustering.py
# function qbx_and_merge will end up with clustering that gets qbX than quickbunldes
from dipy.segment.clustering import qbx_and_merge

thresholds = [30., 20., 10.]
qbx_merged = qbx_and_merge(sft.streamlines, thresholds, nb_pts=256, verbose=True)
centroids_qbx_sft = StatefulTractogram.from_sft(
    qbx_merged.centroids, sft)
save_tractogram(centroids_qbx_sft, 'centroids_qbx_sft.tck', bbox_valid_check=True)
