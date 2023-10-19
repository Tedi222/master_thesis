import numpy as np
from tqdm import tqdm
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import json
import os
from scripts.streamline_space_filtering import get_streamlines_masked, get_streamlines_at_indices


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
