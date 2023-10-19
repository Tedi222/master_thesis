import numpy as np
from tqdm import tqdm, trange
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import Tractogram

import os


def get_track_paths(tractogram_directory):
    base_path = os.path.abspath(tractogram_directory)
    files = os.listdir(tractogram_directory)
    abs_paths = [os.path.join(base_path, filename) for filename in files if filename.endswith(".trk")]

    return abs_paths


def join_tractograms(track_path_list):
    assert len(track_path_list) > 1, f'You must provide at least 2 tractograms, provided {len(track_file_list)}'

    new_sft = load_tractogram(track_path_list[0], 'same', bbox_valid_check=True)
    for tracks in tqdm(track_path_list[1:]):
        other_sft = load_tractogram(tracks, 'same', bbox_valid_check=True)
        # assert StatefulTractogram.are_compatible(new_sft,other_sft),
        # "Tractograms are not compatible, check ensure space, origin are compatible"

        new_sft = new_sft.__add__(other_sft)

    return new_sft
