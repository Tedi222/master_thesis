import os
import sys
sys.path.append(".")
import numpy as np
from scripts.custom_clustering_filtering_utils import (
    get_streamlines_at_indices,
    get_streamlines_masked,
    filter_streamlines_by_length,
    remove_loops_and_sharp_turns,
    qbx_to_tractogram_pl_impl
)

from dipy.io.streamline import (
    load_tractogram,
    save_tractogram
)

from dipy.segment.clustering import qbx_and_merge


# specify path and load tractogram
track_path = '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final/tck2mni/tck2mni_tracks.trk'
track_path1M = '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final_1M/tck2mni/tck2mni_tracks.trk'

# load tractogram
tractogram = load_tractogram(track_path,
                             'same',
                             bbox_valid_check=True,
                             trk_header_check=True)

# filtering streamlines
ids_by_length, mask_by_length = filter_streamlines_by_length(tractogram, 20, 200, return_sft=False)
ids_by_winding, mask_by_winding = remove_loops_and_sharp_turns(tractogram.streamlines, max_angle=335)

both = np.arange(0, len(tractogram))
both_invalid = both[~mask_by_length | ~mask_by_winding]
both_valid = both[mask_by_length & mask_by_winding]


# perform clustering with QuickBundlesX and separate big and small clusters
qbx = qbx_and_merge(streamlines=tractogram.streamlines,
                    thresholds=[30., 25., 20., 15.],
                    nb_pts=256,
                    verbose=True)

qbx_large = qbx.get_large_clusters(500)

qbx_to_tractogram_pl_impl(sft=tractogram,
                          qbx=qbx_large,
                          ids_valid=both_valid,
                          ids_invalid=both_invalid,
                          save_atlas=True,
                          test_name='qbx_pl_impl_trk',
                          extension='trk',
                          save_invalid=False)

