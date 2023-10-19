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


def main():

    tractogram_path = '../tck2mni_tracks.trk'
    sft = load_tractogram(tractogram_path, 'same', bbox_valid_check=True)

    _, ids_by_length, mask_by_length = filter_streamlines_by_length(sft, 20, 200)
    ids_by_winding, mask_by_winding = remove_loops_and_sharp_turns(sft.streamlines, max_angle=335)


if __name__ == "__main__":
    main()
