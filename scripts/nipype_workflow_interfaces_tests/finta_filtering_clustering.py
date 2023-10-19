from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine
)
from nipype import config, logging
from nipype import Node, Workflow, SelectFiles
from nipype.interfaces.utility import Function
from nipype.interfaces.base import TraitedSpec
from nipype.interfaces.base.traits_extension import Directory, File, Str, OutputMultiPath
import os
import os.path as op
import sys
sys.path.append("/tests/scripts")



class FindThresholdsInputSpec(CommandLineInputSpec):
    config_file = File(desc="ae_find_thresholds_config",
                       exists=True, mandatory=True, argstr='%s', position=0)

    model = File(desc="AutoEncoder model file (AE) [ *.pt ]",
                 exists=True, mandatory=True, argstr='%s', position=1)

    valid_bundle_path = Directory(desc="Path to folder containing valid bundle files [ *.trk ]",
                             exists=True, mandatory=True, argstr='%s', position=2)

    invalid_bundle_file = File(desc="Path to invalid streamline file [ *.trk ]",
                               exists=True, mandatory=True, argstr='%s', position=3)

    atlas_path = Directory(desc="""Path containing all atlas bundles [ *.trk ] used to bundle the tractogram.
                                    Bundles must be in the same space as the common_space_tractogram.""",
                      exists=True, mandatory=True, argstr='%s', position=4)

    reference = File(desc="Reference anatomical filename (usually a t1.nii.gz or wm.nii.gz) [ *.nii/.nii.gz ]",
                     exists=True, mandatory=True, argstr='%s', position=5)

    output = Directory(desc="Output path to save experiment.",
                     exists=False, mandatory=True, argstr='%s', position=6)

    streamline_classes = File(desc="""Config file [ *.json ]. JSON file containing bundle names with
                                   corresponding class label.""",
                     exists=True, mandatory=True, argstr='%s', position=7)
    options = Str(argstr="%s", position=8)


class FindThresholdsOutputSpec(TraitedSpec):
    out_thresholds = File(argstr="%s", desc="output thresholds .json file")
    out_log = File(argstr="%s", desc="output log file file")
    out_all = OutputMultiPath(Directory(argstr="%s", desc="output all files"))


class FindThresholds(CommandLine):
    _cmd = "python3.10 /home/teodorps/tractolearn/scripts/ae_find_thresholds.py" # is that cool, calling python from python (?!)

    input_spec = FindThresholdsInputSpec
    output_spec = FindThresholdsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_log"] = op.join(op.abspath(self.inputs.output), 'logfile.log')
        outputs["out_thresholds"] = op.join(op.abspath(self.inputs.output), 'thresholds.json')

        def get_full_paths(directory):
            base_path = op.abspath(directory)
            files = os.listdir(directory)
            abs_paths = [op.join(base_path, filename) for filename in files if op.isdir(filename)]

            return abs_paths

        outputs["out_all"] = get_full_paths(self.inputs.output)

        return outputs


class FINTA_ModifiedInputSpec(CommandLineInputSpec):
    common_space_tractogram = File(desc="""Tractogram to bundle [ *.trk ]. 
                        Must be in the same as the one used to train the AE""",
                                   exists=True, mandatory=True, argstr='%s', position=0)

    atlas_path = Directory(desc="""Path containing all atlas bundles [ *.trk ] used to bundle the tractogram.
                        Bundles must be in the same space as the common_space_tractogram.""",
                           exists=True, mandatory=True, argstr='%s', position=1)

    model = File(desc="AutoEncoder model file (AE) [ *.pt ]",
                      exists=True, mandatory=True, argstr='%s', position=2)

    common_space_reference = File(desc="Reference T1 file [ *.nii/.nii.gz ]",
                                  exists=True, mandatory=True, argstr='%s', position=3)

    thresholds_file = File(desc="""Config file [ *.json ] containing
                                     the per-bundle latent space distance thresholds.""",
                                exists=True, mandatory=True, argstr='%s', position=4)

    streamline_classes = File(desc="""Anatomy file [ *.json ]. JSON file containing 
                            bundle names with corresponding class label.""",
                        exists=True, mandatory=True, argstr='%s', position=5)

    output = Directory(desc="Output path",
                       exists=False, mandatory=True, argstr='%s', position=6)

    original_tractogram = File(argstr="--original_tractogram %s",
                               desc="""Tractogram in the original space.
                               If a file is passed, output bundles will be in its space.
                               Else, it will be in common_space_tractogram space [ *.trk ]""",
                               exists=True, mandatory=False, position=7)

    original_reference = File(argstr="--original_reference %s",
                              desc="""Reference T1 file in the native space [ *.nii/.nii.gz ]""",
                              exists=True, mandatory=False, position=8)

    device = Str(argstr="--device %s",
                 desc="Device to use for inference [ cpu | cuda ], default: 'cpu' ",
                 mandatory=False, position=9)

    batch_loading = Str(argstr="--batch_loading %s",
                        desc="""If the size of the tractogram is too big,
                        you can filter it by batches. Will produce many files for one bundles""",
                        mandatory=False, position=10)

    num_neighbors = Str(argstr="--num_neighbors %s",
                        desc="""Number of neighbors to consider for classification.Maximum allowed (30)""",
                        mandatory=False, position=11)


class FINTA_Modified_OutputSpec(TraitedSpec):
    out_filtered_trk = File(argstr="%s", desc="output filtered tractogram", output_name='filtered.trk')
    out_log = File(argstr="%s", desc="output log file file")
    out_all = File(argstr="%s", desc="output all files")


class FINTA_Modified(CommandLine):
    _cmd = "python3.10 /home/teodorps/tractolearn/scripts/ae_bundle_streamlines.py" # is that cool, calling python from python (?!)

    input_spec = FINTA_ModifiedInputSpec
    output_spec = FINTA_Modified_OutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        print(self.output_spec().get())
        outputs["out_log"] = op.join(op.abspath(self.inputs.output), 'logfile.log')
        outputs["out_filtered_trk"] = op.join(op.abspath(self.inputs.output), 'merged.trk')
        outputs["out_all"] = op.abspath(self.inputs.output)

        return outputs


def streamline_space_filtering_clustering(mni_tractogram_path):

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
    tractogram = load_tractogram(mni_tractogram_path,
                                 'same',
                                 bbox_valid_check=True,
                                 trk_header_check=True)

    # filtering streamlines
    ids_by_length, mask_by_length = filter_streamlines_by_length(tractogram, 20, 200, return_sft=False)
    ids_by_winding, mask_by_winding = remove_loops_and_sharp_turns(tractogram.streamlines, max_angle=335)

    both = np.arange(0, len(tractogram))
    both_invalid = both[~mask_by_length | ~mask_by_winding]
    both_valid = both[mask_by_length & mask_by_winding]

    # get invalid
    sft_implausible = get_streamlines_at_indices(tractogram, both_invalid)
    save_tractogram(sft_implausible, 'sft_implausible.trk', bbox_valid_check=True)
    implausible_streamlines_path = os.path.abspath('sft_implausible.trk')

    # perform clustering with QuickBundlesX and separate big and small clusters
    qbx = qbx_and_merge(streamlines=tractogram.streamlines,
                        thresholds=[30., 25., 20., 15.],
                        nb_pts=256,
                        verbose=True)

    qbx_large = qbx.get_large_clusters(500)

    # create classes, get atlas/centroids and valid clusters
    qbx_to_tractogram_pl_impl(sft=tractogram,
                              qbx=qbx_large,
                              ids_valid=both_valid,
                              ids_invalid=both_invalid,
                              save_atlas=True,
                              test_name='qbx_pl_impl_trk',
                              extension='trk',
                              save_invalid=False)

    streamline_classes_path = os.path.abspath('qbx_pl_impl_trk/streamline_classes.json')
    centroids_path = os.path.abspath('qbx_pl_impl_trk/centroids')
    clusters_valid_path = os.path.abspath('qbx_pl_impl_trk/clusters_valid')

    return [streamline_classes_path, centroids_path, clusters_valid_path, implausible_streamlines_path]




cfg = dict(logging=dict(workflow_level = 'INFO', interface_level= 'DEBUG'))
           # execution={'keep_inputs': True,
           #            'remove_unneccesary_outputs': False})
config.update_config(cfg)
logging.update_logging(config)

templates = {'config_file': '/home/teodorps/tractolearn_data/configs/find_threshold_config.yaml',
             'model': '/home/teodorps/tractolearn_data/best_model_contrastive_tractoinferno_hcp.pt',
             'valid_bundle_path': '/home/teodorps/qbx_clustering_filtering_outputs/qbx_pl_impl_trk2/clusters_valid',
             'invalid_bundle_file': '/home/teodorps/qbx_clustering_filtering_outputs/sft_implausible.trk',
             'atlas_path': '/home/teodorps/qbx_clustering_filtering_outputs/qbx_pl_impl_trk2/centroids',
             'reference': '/home/teodorps/atlases/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii',
             'streamline_classes': '/home/teodorps/qbx_clustering_filtering_outputs/qbx_pl_impl_trk2/streamline_classes.json',
             'tractogram_mni': '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final/tck2mni/tck2mni_tracks.trk'}

sf = Node(SelectFiles(templates), name='fileselector')

cluster_streamlines = Node(Function(input_names=['mni_tractogram_path'],
                                    output_names=['streamline_classes_path', 'centroids_path',
                                                  'clusters_valid_path', 'implausible_streamlines_path'],
                                    function=streamline_space_filtering_clustering), name='cluster_streamlines')

find_threshold = Node(FindThresholds(output='thresholds', options='-v'), name='find_thresholds')
# find_threshold.config = {'execution': {'keep_unnecessary_outputs': 'true'}}

finta_filter = Node(FINTA_Modified(output='filtered', device="cuda"), name='finta_filter')

pipeline = Workflow(name='cluster_threshold_filter_test_config_plots',
                    base_dir='/home/teodorps/tests/scripts/nipype_scripts')

pipeline.connect([(sf, cluster_streamlines, [('tractogram_mni', 'mni_tractogram_path')])
                  ])

pipeline.connect([(sf, find_threshold, [('config_file', 'config_file'),
                                        ('model', 'model'),
                                        # ('valid_bundle_path', 'valid_bundle_path'),
                                        # ('invalid_bundle_file', 'invalid_bundle_file'),
                                        # ('atlas_path', 'atlas_path'),
                                        ('reference', 'reference'),
                                        # ('streamline_classes', 'streamline_classes')
                                        ]),

                  (cluster_streamlines, find_threshold, [('clusters_valid_path', 'valid_bundle_path'),
                                                         ('implausible_streamlines_path', 'invalid_bundle_file'),
                                                         ('streamline_classes_path', 'streamline_classes'),
                                                         ('centroids_path', 'atlas_path')
                                                         ])
                  ])

pipeline.connect([(sf, finta_filter, [('model', 'model'),
                                      # ('atlas_path', 'atlas_path'),
                                      ('reference', 'common_space_reference'),
                                      ('tractogram_mni', 'common_space_tractogram'),
                                      ('reference', 'original_reference'),
                                      ('tractogram_mni', 'original_tractogram'),
                                      # ('streamline_classes', 'streamline_classes')
                                      ]),

                  (cluster_streamlines, finta_filter, [('streamline_classes_path', 'streamline_classes'),
                                                       ('centroids_path', 'atlas_path')]),

                  (find_threshold, finta_filter, [('out_thresholds', 'thresholds_file')])
                  ])
FindThresholds.help()
print(find_threshold.outputs)
pipeline.write_graph(graph2use='orig')

res = pipeline.run()
print(find_threshold.outputs)