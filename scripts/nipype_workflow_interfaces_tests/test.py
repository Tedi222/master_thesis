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

def iterate_outs(number, in_path):

    def get_track_paths(tractogram_directory):
        base_path = os.path.abspath(tractogram_directory)
        files = os.listdir(tractogram_directory)
        abs_paths = [os.path.join(base_path, filename) for filename in files]

        return abs_paths

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

    # specify path and load tractogram
    track_path = '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final/tck2mni/tck2mni_tracks.trk'

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


    import json
    import os
    if not os.path.exists(in_path):
        os.mkdir(in_path)
    class_config = {}
    for i in range(0, number):
        class_config[f'cluster_{i}'] = i
        serialised = json.dumps(class_config, indent=4)
        with open(os.path.join(in_path, f'test_{i}.json'), "w") as outfile:
            outfile.write(serialised)

    return [get_track_paths(in_path), os.getcwd()]


def iterate_ins(in_path, in_path2):
    import os
    import json
    print(f'passed path: {in_path}')
    print(f'passed path2: {in_path2}')
    new = {'path': in_path}
    serialised = json.dumps(new, indent=4)
    with open(f'akjhsgkjasj.json', "w") as outfile:
        outfile.write(serialised)
    return os.path.abspath(f'akjhsgkjasj.json')


from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec

from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine,
    traits,
    TraitedSpec,
    File,
    Str,
    InputMultiPath,
    isdefined,
)
#
#
# class CustomFuncInputSpec(BaseInterfaceInputSpec):
#     # in_file = File(exists=True, mandatory=True, desc='the input image')
#     # out_file = File(mandatory=True, desc='the output image') # Do not set exists=True !!
#     number = traits.Int()
#     out_path = Directory()
#
# from nipype.interfaces.base import OutputMultiObject
# from nipype.utils.filemanip import ensure_list
# class CustomFuncOutputSpec(TraitedSpec):
#     out_file = Str()
#
#
# class CustomFunc(BaseInterface):
#     input_spec = CustomFuncInputSpec
#     output_spec = CustomFuncOutputSpec
#
#     def __init__(
#             self,
#             output_names="out",
#             **inputs
#     ):
#         """
#
#         Parameters
#         ----------
#
#         output_names: single str or list
#             names corresponding to function outputs (default: 'out') """
#
#         super().__init__(**inputs)
#         self._output_names = ensure_list(output_names)
#         self._out = {}
#         for name in self._output_names:
#             self._out[name] = None
#
#     def _add_output_traits(self, base):
#         undefined_traits = {}
#         for key in self._output_names:
#             base.add_trait(key, traits.Any)
#             undefined_traits[key] = Undefined
#         base.trait_set(trait_change_notify=False, **undefined_traits)
#         return base
#
#     def _run_interface(self, runtime):
#         # Call our python code here:
#         iterate_outs(
#             self.inputs.number,
#             self.inputs.out_path
#         )
#         # And we are done
#         return runtime
#
#     def _list_outputs(self):
#         outputs = self._outputs().get()
#         for key in self._output_names:
#             outputs[key] = self._out[key]
#         return os.listdir(self.inputs.out_path)
#
#
# # pipeline = Workflow(name='multiout_test',
# #                     base_dir='/home/teodorps/tests/scripts/nipype_scripts')
#
# iterate_node = Node(CustomFunc(number=5, out_path='files'), name='func')
# iterate_node.base_dir = '/home/teodorps/tests/scripts/nipype_scripts'
# # iterate_node.inputs.number = 5
#
# res = iterate_node.run()

pipeline = Workflow(name='multiout_test_xxxx',
                     base_dir='/home/teodorps/tests/scripts/nipype_scripts')
# pipeline.base_dir = '/home/teodorps/tests/scripts/nipype_scripts'

iterate_node2 = Node(Function(input_names=['number', 'in_path'],
                              output_names=['out_path', 'mmm'],
                              function=iterate_outs), name='func1')

iterate_node2.inputs.number = 5
iterate_node2.inputs.in_path = 'lmao'

iterate_node3 = Node(Function(input_names=['in_path', 'in_path2'], function=iterate_ins), name='func2')

pipeline.connect([(iterate_node2, iterate_node3, [('out_path', 'in_path'),
                                                  ('mmm', 'in_path2')])
                  ])

res = pipeline.run()