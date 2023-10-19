import os
import json
import numpy as np
from tqdm import tqdm, trange
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from dipy.tracking import metrics as tm
from dipy.io.streamline import load_tractogram, save_tractogram

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


def filter_streamlines_by_length(sft, min_length=0., max_length=np.inf, return_sft=True):
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
    save_tracks: bool
        Saving tracks to .trk
    Return
    ------
    filtered_sft : StatefulTractogram
        A tractogram without short streamlines.
    ids:
    """

    # Make sure we are in world space
    orig_space = sft.space
    sft.to_rasmm()

    if sft.streamlines:
        # Compute streamlines lengths
        lengths = length(sft.streamlines)

        # Filter lengths
        mask = np.logical_and(lengths >= min_length,
                              lengths <= max_length)
    else:
        mask = []

    filtered_sft = StatefulTractogram.from_sft(
        sft.streamlines[mask],
        sft,
        data_per_streamline=sft.data_per_streamline[mask],
        data_per_point=sft.data_per_point[mask],
    )
    ids = np.where(mask == True)[0]
    ids.tolist()
    # Return to original space
    filtered_sft.to_space(orig_space)

    if return_sft:
        return filtered_sft, ids, mask
    else:
        return ids, mask



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


def qbx_to_tractogram_pl_impl(sft,
                              qbx,
                              ids_valid,
                              ids_invalid,
                              save_atlas=True,
                              test_name='qbx',
                              extension='trk',
                              save_invalid=False):
    class_config = {}
    for i in tqdm(range(0, len(qbx))):
        cluster = qbx[i]
        if len(cluster) > 1:
            valid_intersect = np.intersect1d(cluster.indices, ids_valid).tolist()
            invalid_intersect = np.intersect1d(cluster.indices, ids_invalid).tolist()
            bundle_valid = get_streamlines_at_indices(sft, valid_intersect)
            bundle_invalid = get_streamlines_at_indices(sft, invalid_intersect)
            centroid = StatefulTractogram.from_sft(cluster.centroid[np.newaxis, :], sft)

            clusters_valid_path = f'{test_name}/clusters_valid'
            clusters_invalid_path = f'{test_name}/clusters_invalid'
            centroids_path = f'{test_name}/centroids'

            if save_invalid:
                if not os.path.exists(clusters_invalid_path):
                    os.makedirs(clusters_invalid_path)

            if not os.path.exists(clusters_valid_path):
                os.makedirs(clusters_valid_path)

            if not os.path.exists(centroids_path):
                os.makedirs(centroids_path)

            if save_invalid:
                save_tractogram(bundle_invalid, os.path.join(clusters_invalid_path, f'cluster_{i}.{extension}'),
                                bbox_valid_check=True)

            save_tractogram(bundle_valid, os.path.join(clusters_valid_path, f'cluster_{i}.{extension}'),
                            bbox_valid_check=True)
            save_tractogram(centroid, os.path.join(centroids_path, f'cluster_{i}.{extension}'),
                            bbox_valid_check=True)

            class_config[f'cluster_{i}'] = i
        else:
            continue

    if save_atlas:
        serialised = json.dumps(class_config, indent=4)
        with open(f'{test_name}/streamline_classes.json', "w") as outfile:
            outfile.write(serialised)


def test_fun(direc='umm'):
    os.makedirs(direc)


