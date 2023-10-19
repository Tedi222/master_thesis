import numpy as np
from tqdm import tqdm, trange
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
import time
import os
import importlib
moduleName = input('streamline_space_filtering')
importlib.import_module(moduleName)
import json
import matplotlib.pyplot as plt

from scripts.streamline_space_filtering import get_streamlines_masked, get_streamlines_at_indices
from dipy.segment.clustering import qbx_and_merge


# valid_bundles_list = os.listdir()

sft = load_tractogram(tractogram_path, 'same',bbox_valid_check=True)
tractogram_path = '../tck2mni_tracks.trk'
qbx2 = qbx_and_merge(sft.streamlines, [60., 50., 40., 35.], nb_pts=256, verbose=True)



centroids_qbx_sft = StatefulTractogram.from_sft(
    qbx.clusters[3].centroid[np.newaxis, :], sft)
    #qbx_merged.centroids[0:3][np.newaxis, :], sft)
# save_tractogram(centroids_qbx_sft, 'centroids_qbx_sft.tck', bbox_valid_check=True)

clusters_qbx_sft = StatefulTractogram.from_sft(
    sft.streamlines[qbx.clusters[3].indices],
    sft,
    data_per_streamline=sft.data_per_streamline[qbx.clusters[3].indices],
    data_per_point=sft.data_per_point[qbx.clusters[3].indices])
# save_tractogram(clusters_qbx_sft, 'clusters_qbx_sft.tck', bbox_valid_check=True)


def qbx_to_tractogram(qbx, save_atlas=True, extension='trk', test_name=''):
    class_config = {}
    for i in tqdm(range(0, len(qbx))):
        cluster = qbx[i]
        if len(cluster) > 1:
            bundle = get_streamlines_at_indices(sft, cluster.indices)
            centroid = StatefulTractogram.from_sft(cluster.centroid[np.newaxis, :], sft)

            clusters_path = f'./qbx_atlas/clusters{test_name}'
            centroids_path = f'./qbx_atlas/centroids{test_name}'

            if not os.path.exists(clusters_path):
                os.makedirs(clusters_path)

            if not os.path.exists(centroids_path):
                os.makedirs(centroids_path)

            save_tractogram(bundle, os.path.join(clusters_path, f'cluster_{i}.{extension}'), bbox_valid_check=True)
            save_tractogram(centroid, os.path.join(centroids_path, f'centroid_{i}.{extension}'), bbox_valid_check=True)

            class_config[f'cluster_{i}'] = i
        else:
            continue

    if save_atlas:
        serialised = json.dumps(class_config, indent=4)
        with open("./qbx_atlas/streamline_classes.json", "w") as outfile:
            outfile.write(serialised)


def qbx_to_tractogram_pl_impl(sft, qbx, ids_valid, ids_invalid, save_atlas=True, test_name='qbx', extension='trk'):
    class_config = {}
    for i in tqdm(range(0, len(qbx))):
        cluster = qbx[i]
        if len(cluster) > 1:
            valid_intersect = np.intersect1d(cluster.indices, ids_valid).tolist()
            invalid_intersect = np.intersect1d(cluster.indices, ids_invalid).tolist()
            bundle_valid = get_streamlines_at_indices(sft, valid_intersect)
            bundle_invalid = get_streamlines_at_indices(sft, invalid_intersect)
            centroid = StatefulTractogram.from_sft(cluster.centroid[np.newaxis, :], sft)

            clusters_valid_path = f'./{test_name}/clusters_valid'
            clusters_invalid_path = f'./{test_name}/clusters_invalid'
            centroids_path = f'./{test_name}/centroids'

            if not os.path.exists(clusters_valid_path):
                os.makedirs(clusters_valid_path)

            if not os.path.exists(clusters_invalid_path):
                os.makedirs(clusters_invalid_path)

            if not os.path.exists(centroids_path):
                os.makedirs(centroids_path)

            save_tractogram(bundle_valid, os.path.join(clusters_valid_path, f'cluster_{i}.{extension}'),
                            bbox_valid_check=True)
            save_tractogram(bundle_invalid, os.path.join(clusters_invalid_path, f'cluster_{i}.{extension}'),
                            bbox_valid_check=True)
            save_tractogram(centroid, os.path.join(centroids_path, f'cluster_{i}.{extension}'), bbox_valid_check=True)

            class_config[f'cluster_{i}'] = i
        else:
            continue

    if save_atlas:
        serialised = json.dumps(class_config, indent=4)
        with open(f'./{test_name}/streamline_classes.json', "w") as outfile:
            outfile.write(serialised)

both = np.arange(0, len(sft))
both_invalid = both[~mask_by_length | ~mask_by_winding]
both_valid = both[mask_by_length & mask_by_winding]

from dipy.segment.clustering import qbx_and_merge
qbx = qbx_and_merge(sft.streamlines, [30., 25., 20., 15.], nb_pts=256, verbose=True)
qbx_large = qbx.get_large_clusters(500)
qbx_small = qbx.get_small_clusters(499)

invalid_ids_large = []
valid_ids_large = []
ratio_large = []
cl_sizes_large = []
for i, cluster in enumerate(qbx_large):
    invalid = np.intersect1d(cluster.indices, both_invalid).tolist()
    valid = np.intersect1d(cluster.indices, both_valid).tolist()
    invalid_ids_large[len(invalid_ids_large):] = invalid
    valid_ids_large[len(valid_ids_large):] = valid
    ratio_large.append(len(invalid) / len(cluster))
    cl_sizes_large.append(len(cluster))
    # if len(valid) != 0:
    #     ratio_large.append(len(invalid) / len(valid))
    # else:
    #     ratio_large.append(1)

id_of_max_ratio_cluster = np.where(np.array(ratio_large) == np.array(ratio_large).max())

invalid_ids_small = []
valid_ids_small = []
ratio_small = []
cl_sizes_small = []
for i, cluster in enumerate(qbx_small):
    invalid = np.intersect1d(cluster.indices, both_invalid).tolist()
    valid = np.intersect1d(cluster.indices, both_valid).tolist()
    invalid_ids_small[len(invalid_ids_small):] = invalid
    valid_ids_small[len(valid_ids_small):] = valid
    ratio_small.append(len(invalid) / len(cluster))
    cl_sizes_small.append(len(cluster))
    # if len(valid) != 0:
    #     ratio_small.append(len(invalid) / len(valid))
    # else:
    #     ratio_small.append(1)

ratio_whole = (len(sft) - sum(mask_by_winding & mask_by_length)) / len(sft)
ratio_qbx_preserved = len(invalid_ids_large) / (len(valid_ids_large)+len(invalid_ids_large))
ratio_qbx_removed = len(invalid_ids_small) / (len(valid_ids_small)+len(invalid_ids_small))

sft_implausible = get_streamlines_at_indices(sft_resampled, both_invalid)
sft_plausible = get_streamlines_at_indices(sft_resampled, both_valid)
save_tractogram(sft_implausible, 'sft_implausible.trk', bbox_valid_check=True)
save_tractogram(sft_plausible, 'sft_plausible.trk', bbox_valid_check=True)


# resampling
num_points = 256
from scilpy.tracking.tools import resample_streamlines_num_points

# Resampling streamlines to 256 points
sft_resampled = resample_streamlines_num_points(
    sft, num_points
)

qbx_to_tractogram_pl_impl(sft_resampled, qbx_large, both_valid, both_invalid, save_atlas=True, test_name='qbx_pl_impl_trk')

counts, bins = np.histogram(ratio_large)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()


qbx_large_ids = [len(i.indices) for i in qbx_large]
sum(qbx_large_ids)
