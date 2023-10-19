import os

from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine,
    traits,
    TraitedSpec,
    File,
    InputMultiPath,
    isdefined,
)
import os.path as op

from nipype.interfaces.mrtrix3.base import MRTrix3BaseInputSpec, MRTrix3Base
from nipype import config
from graphviz import dot
from nipype import Node, Workflow, SelectFiles, IdentityInterface, DataSink
from nipype.interfaces import mrtrix3, fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.base import TraitedSpec, TraitError
from nipype.interfaces.base.traits_extension import Directory, BasePath, File, Undefined
import os

def get_track_paths(tractogram_directory):
    base_path = os.path.abspath(tractogram_directory)
    files = os.listdir(tractogram_directory)
    abs_paths = [os.path.join(base_path, filename) for filename in files]

    return abs_paths

def iterate_outs(track_path):

    def get_full_paths(directory):
        base_path = os.path.abspath(directory)
        files = os.listdir(directory)
        abs_paths = [os.path.join(base_path, filename) for filename in files]

        return abs_paths

    import os
    import sys
    sys.path.append("/home/teodorps/tests/scripts")
    import numpy as np
    from custom_clustering_filtering_utils import (
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
    class_path = os.path.abspath('qbx_pl_impl_trk/streamline_classes.json')
    centroids_path = os.path.abspath('qbx_pl_impl_trk/centroids')
    clusters_valid_path = os.path.abspath('qbx_pl_impl_trk/clusters_valid')

    return [class_path, centroids_path, clusters_valid_path]


def iterate_ins(in_path, in_path2, in_path3):
    import os
    import json
    print(f'passed path: {in_path}')
    print(f'passed path2: {in_path2}')
    print(f'passed path2: {in_path3}')
    new = {'path1': in_path, 'path2': os.listdir(in_path2), 'path3': os.listdir(in_path3)}
    serialised = json.dumps(new, indent=4)
    with open(f'test.json', "w") as outfile:
        outfile.write(serialised)
    return os.path.abspath(f'test.json')


pipeline = Workflow(name='multiout_test_zzzzz',
                     base_dir='/home/teodorps/tests/scripts/nipype_scripts')

iterate_node2 = Node(Function(input_names=['track_path'],
                              output_names=['out_path', 'out_path2', 'out_path3'],
                              function=iterate_outs), name='clustering')

iterate_node2.inputs.track_path = '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final/tck2mni/tck2mni_tracks.trk'

iterate_node3 = Node(Function(input_names=['in_path', 'in_path2', 'in_path3'], function=iterate_ins), name='check_outs')

pipeline.connect([(iterate_node2, iterate_node3, [('out_path', 'in_path'),
                                                  ('out_path2', 'in_path2'),
                                                  ('out_path3', 'in_path3')])
                  ])

res = pipeline.run()