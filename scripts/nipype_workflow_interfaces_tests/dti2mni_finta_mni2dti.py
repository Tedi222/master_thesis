
from nipype.interfaces.base import (
    TraitedSpec,
    File,
    isdefined,
)

import os.path as op

from nipype.interfaces.mrtrix3.base import MRTrix3BaseInputSpec, MRTrix3Base


from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine
)
from nipype import config, logging
from nipype import Node, Workflow, SelectFiles
from nipype.interfaces.utility import Function
from nipype.interfaces.base import TraitedSpec, isdefined
from nipype.interfaces.base.traits_extension import Directory, File, Str, OutputMultiPath
import os
import os.path as op
import sys
sys.path.append("/tests/scripts")


cfg = dict(logging=dict(workflow_level = 'INFO', interface_level= 'DEBUG'),
           execution={'keep_inputs': True, 'remove_unneccesary_outputs': False})
config.update_config(cfg)
logging.update_logging(config)

# DWI to MNI and MNI to DWI tractogram registration
class WarpInitInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s',
                   mandatory=True, position=-2,
                   desc='image to be used as a template for identity warp')
    out_file = File("inv_identity_warp_no.nii", argstr='%s', mandatory=True, position=-1, usedefault=True,
                    desc='output identity warp')


class WarpInitOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output identity warp')


class WarpInit(MRTrix3Base):
    """Use warpinit to create an initial warp image, representing an identity transformation
    (https://mrtrix.readthedocs.io/en/dev/reference/commands/warpinit.html)
    """

    _cmd = 'warpinit'
    input_spec = WarpInitInputSpec
    output_spec = WarpInitOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)

        return outputs


class WarpMRTransformInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s',
                   mandatory=True, position=-3,
                   desc='Image to transform')

    warp_file = File(exists=True, argstr="-warp %s", position=-2,
                     desc="""apply a non-linear 4D deformation field to warp the input image. 
            Each voxel in the deformation field must define the scanner space 
            position that will be used to interpolate the input image during warping
            (i.e. pull-back/reverse warp convention). 
            If the -template image is also supplied the deformation field 
            will be resliced first to the template image grid.)""")

    out_file = File("dwi2mni_warpcheck.nii.gz", argstr='%s', mandatory=True, position=-1, usedefault=True,
                    desc='output transformed image')


class WarpMRTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output transformed image')


class WarpMRTransform(MRTrix3Base):
    """Use mrtransform -warp to apply non-linear transformation with mrtrix
    (-warp option not included in nipype MRTransform)
    (https://mrtrix.readthedocs.io/en/dev/reference/commands/mrtransform.html)
    """

    _cmd = 'mrtransform'
    input_spec = WarpMRTransformInputSpec
    output_spec = WarpMRTransformOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self.inputs.out_file
        if not isdefined(outputs["out_file"]):
            outputs["out_file"] = op.abspath(self._gen_outfilename())
        else:
            outputs["out_file"] = op.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_files[0])
        return name + "_mrtransform.nii.gz"


class TCKTransformInputSpec(MRTrix3BaseInputSpec):
    tracks = File(exists=True, argstr='%s',
                  mandatory=True, position=-3,
                  desc='The input track file')

    transform = File(exists=True, argstr="%s", position=-2,
                     desc='The image containing the transform')

    out_file = File("tck2mni_tracks.tck", argstr='%s', mandatory=True, position=-1, usedefault=True,
                    desc='output transformed track file')


class TCKTransformOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output transformed track file')


class TCKTransform(MRTrix3Base):
    """Apply a spatial transformation to a tracks file
    (https://mrtrix.readthedocs.io/en/dev/reference/commands/tcktransform.html)
    """

    _cmd = 'tcktransform'
    input_spec = TCKTransformInputSpec
    output_spec = TCKTransformOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = self.inputs.out_file
        if not isdefined(outputs["out_file"]):
            outputs["out_file"] = op.abspath(self._gen_outfilename())
        else:
            outputs["out_file"] = op.abspath(outputs["out_file"])
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_files[0])
        return name + "_tcktransform.tck"

# FILTERING AND CLUSTERING - FINTA
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

#####################################################

from nipype.interfaces.utility import IdentityInterface
from nipype import config
from graphviz import dot
from nipype import Node, Workflow, SelectFiles, IdentityInterface, DataSink
from nipype.interfaces import mrtrix3, fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.base import TraitedSpec, TraitError
from nipype.interfaces.base.traits_extension import Directory, BasePath, File

# templates = {'dwi':         'sub_{_id}/eddy_corrected_data.nii.gz',
#              'standard':    'mni_icbm152_nlin_asym_09c/'+\
#                             'mni_icbm152_t1_tal_nlin_asym_09c.nii',
#              'fieldcoeff_file_dwi2t1':       'dwi2t1_nl/dtifit__FA_fieldwarp.nii.gz',
#              'fieldcoeff_file_t12mni':       't12mni_nl/T1_fieldwarp.nii.gz'}

templates = {'dwi': '/home/teodorps/data/marvin_testdata/sub_02/eddy_corrected_data.nii.gz',
             'standard': '/home/teodorps/data/marvin_testdata/mni_icbm152_t1_tal_nlin_asym_09c.nii',
             'fieldcoeff_file_dwi2t1': '/home/teodorps/tests/pipeline_tests/diff_pipeline_test_28_04_2023/__id_02/dwi2t1_nl/dtifit__FA_fieldwarp.nii.gz',
             'fieldcoeff_file_t12mni': '/home/teodorps/tests/pipeline_tests/diff_pipeline_test_28_04_2023/__id_02/t12mni_nl/T1_fieldwarp.nii.gz',
             'tracks': '/home/teodorps/tests/pipeline_tests/diff_pipeline_test_28_04_2023/__id_02/tckgen/tracked.tck',

             'config_file': '/home/teodorps/tractolearn_data/configs/find_threshold_config.yaml',
             'model': '/home/teodorps/tractolearn_data/best_model_contrastive_tractoinferno_hcp.pt',
             'valid_bundle_path': '/home/teodorps/qbx_clustering_filtering_outputs/qbx_pl_impl_trk2/clusters_valid',
             'invalid_bundle_file': '/home/teodorps/qbx_clustering_filtering_outputs/sft_implausible.trk',
             'atlas_path': '/home/teodorps/qbx_clustering_filtering_outputs/qbx_pl_impl_trk2/centroids',
             # 'reference': '/home/teodorps/atlases/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii',
             'streamline_classes': '/home/teodorps/qbx_clustering_filtering_outputs/qbx_pl_impl_trk2/streamline_classes.json',
             #'tractogram_mni': '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test_final/tck2mni/tck2mni_tracks.trk'
             }

sf = Node(SelectFiles(templates), name='fileselector')
# combine fsl warps
convertwarp_dti2mni = Node(fsl.utils.ConvertWarp(
    output_type='NIFTI_GZ'
), name='convertwarp_dti2mni')

# invert combined warp
invwarp_mni2dti = Node(fsl.utils.InvWarp(
    output_type='NIFTI_GZ'
), name='invwarp_mni2dti')


# generate the MNI2DWI transformation warp with mrtrix (combined with the fsl warps: warps_MNI2dwi.nii.gz)
# warpinit is a MRTrix3 command
init_inv_identity_warp = Node(WarpInit(
), name='init_inv_identity_warp')

mrtrix_mni2dwi_warp = Node(fsl.preprocess.ApplyWarp(
    output_type='NIFTI_GZ'
), name='mrtrix_mni2dwi_warp')

# check the warping results by applying it to MNI_3mm_brain
standard_to_dwi_check = Node(WarpMRTransform(
), name='standard_to_dwi_check')

tck2mni = Node(TCKTransform(
), name='tck2mni')


# convert MRTrix tracks (.tck) to TrackVis (.trk)
def mrtrix_to_trackvis(anatomy, tractogram):
    import nibabel as nib
    from nibabel.orientations import aff2axcodes
    from nibabel.streamlines import Field
    import os
    nii = nib.load(anatomy)

    tractogram_format = nib.streamlines.detect_format(tractogram)

    filename, _ = os.path.splitext(os.path.basename(tractogram))
    output_filename = filename + '.trk'

    # Build header using infos from the anatomical image.
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = ''.join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(tractogram)
    nib.streamlines.save(tck.tractogram, output_filename, header=header)

    return os.path.abspath(output_filename)


tck2trk = Node(Function(input_names=['anatomy', 'tractogram'],
                        output_names=['out_file'],
                        function=mrtrix_to_trackvis),
               name='tck2trk')

# map mrtrix streamlines on top of standard space to check if they were regstered to MNI correctly
tckmap = Node(mrtrix3.utils.ComputeTDI(
    out_file='tck2mni_mapped.nii.gz'
), name='tckmap')


# convert TrackVis (.trk) to MRTrix tracks (.tck)
def trackvis_to_mrtrix(tractogram):
    import nibabel as nib
    import os

    tractogram_format = nib.streamlines.detect_format(tractogram)
    if tractogram_format is not nib.streamlines.TrkFile:
        print(f'Skipping non TRK file')

    filename, _ = os.path.splitext(os.path.basename(tractogram))
    output_filename = filename + '.tck'

    trk = nib.streamlines.load(tractogram)
    nib.streamlines.save(trk.tractogram, output_filename)

    return os.path.abspath(output_filename)

dti2mni_wf = Workflow(name='dti2mni_wf',
                      base_dir='/home/teodorps/tests/pipeline_tests/finta_whole')

dti2mni_wf.connect([(sf, convertwarp_dti2mni, [('standard', 'reference'),  # merge warps dti2t1 + t12mni
                                             ('fieldcoeff_file_dwi2t1', 'warp1'),
                                             ('fieldcoeff_file_t12mni', 'warp2')]),
                  (sf, invwarp_mni2dti, [('dwi', 'reference')]),  # invert dt12mni warp
                  (convertwarp_dti2mni, invwarp_mni2dti, [('out_file', 'warp')]),
                  (sf, init_inv_identity_warp, [('standard', 'in_file')]),  # initialize identity warp
                  (sf, mrtrix_mni2dwi_warp, [('dwi', 'ref_file')]),
                  (init_inv_identity_warp, mrtrix_mni2dwi_warp, [('out_file', 'in_file')]),  # create mrtrix warp
                  (invwarp_mni2dti, mrtrix_mni2dwi_warp, [('inverse_warp', 'field_file')]),
                  (sf, standard_to_dwi_check, [('standard', 'in_file')]),
                  # check if mrtrix warp works well by applying warp to mni image
                  (mrtrix_mni2dwi_warp, standard_to_dwi_check, [('out_file', 'warp_file')]),
                  (sf, tck2mni, [('tracks', 'tracks')]),  # transform tracks from dwi to mni space
                  (mrtrix_mni2dwi_warp, tck2mni, [('out_file', 'transform')])
                  ])

# convert mrtrix streamlines format(.tck) to trackvis format(.trk)
dti2mni_wf.connect([(tck2mni, tck2trk, [('out_file', 'tractogram')]),
                  (sf, tck2trk, [('standard', 'anatomy')])
                  ])

# map mrtrix streamlines on top of standard space to check if they were regstered to MNI correctly
dti2mni_wf.connect([(tck2mni, tckmap, [('out_file', 'in_file')]),
                  (sf, tckmap, [('standard', 'reference')])
                  ])

# dti2mni_wf.write_graph(graph2use='orig')
# dti2mni_wf.run()
#################################################################################################




cluster_streamlines = Node(Function(input_names=['mni_tractogram_path'],
                                    output_names=['streamline_classes_path', 'centroids_path',
                                                  'clusters_valid_path', 'implausible_streamlines_path'],
                                    function=streamline_space_filtering_clustering), name='cluster_streamlines')

find_threshold = Node(FindThresholds(output='thresholds', options='-v'), name='find_thresholds')
# find_threshold.config = {'execution': {'keep_unnecessary_outputs': 'true'}}

finta_filter = Node(FINTA_Modified(output='filtered', device="cuda"), name='finta_filter')

finta_wf = Workflow(name='clustering_finta_wf',
                    base_dir='/home/teodorps/tests/pipeline_tests/finta_whole')

finta_wf.connect([(dti2mni_wf, cluster_streamlines, [('tck2trk.out_file', 'mni_tractogram_path')])
                  ])

finta_wf.connect([(sf, find_threshold, [('config_file', 'config_file'),
                                        ('model', 'model'),
                                        # ('valid_bundle_path', 'valid_bundle_path'),
                                        # ('invalid_bundle_file', 'invalid_bundle_file'),
                                        # ('atlas_path', 'atlas_path'),
                                        ('standard', 'reference'),
                                        # ('streamline_classes', 'streamline_classes')
                                        ]),

                  (cluster_streamlines, find_threshold, [('clusters_valid_path', 'valid_bundle_path'),
                                                         ('implausible_streamlines_path', 'invalid_bundle_file'),
                                                         ('streamline_classes_path', 'streamline_classes'),
                                                         ('centroids_path', 'atlas_path')
                                                         ])
                  ])

finta_wf.connect([(sf, finta_filter, [('model', 'model'),
                                      # ('atlas_path', 'atlas_path'),
                                      ('standard', 'common_space_reference'),
                                      # ('tractogram_mni', 'common_space_tractogram'),
                                      ('standard', 'original_reference'),
                                      # ('tractogram_mni', 'original_tractogram'),
                                      # ('streamline_classes', 'streamline_classes')
                                      ]),
                  (dti2mni_wf, finta_filter, [('tck2trk.out_file', 'common_space_tractogram'),
                                              ('tck2trk.out_file', 'original_tractogram')]),

                  (cluster_streamlines, finta_filter, [('streamline_classes_path', 'streamline_classes'),
                                                       ('centroids_path', 'atlas_path')]),

                  (find_threshold, finta_filter, [('out_thresholds', 'thresholds_file')])
                  ])




#######################################################################################
mni2dti_wf = Workflow(name='mni2dti2_wf',
                       base_dir='/home/teodorps/tests/pipeline_tests/finta_whole')


mrtrix_dwi2mni_warp = Node(fsl.preprocess.ApplyWarp(
    output_type='NIFTI_GZ'
), name='mrtrix_dwi2mni_warp')

# check the warping results by applying it to MNI_3mm_brain
dwi_to_standard_check = Node(WarpMRTransform(
), name='dwi_to_standard_check')

tck2dwi = Node(TCKTransform(
), name='tck2dwi')

init_identity_warp = Node(WarpInit(
), name='init_identity_warp')

tckmap_mni2dti = Node(mrtrix3.utils.ComputeTDI(
    out_file='tck2dwi_mapped.nii.gz'
), name='tckmap_mni2dti')

trk2tck = Node(Function(input_names=['tractogram'],
                        output_names=['out_file'],
                        function=trackvis_to_mrtrix),
               name='trk2tck')


mni2dti_wf.connect([(sf, init_identity_warp, [('dwi', 'in_file')]),  # initialize identity warp
                  (sf, mrtrix_dwi2mni_warp, [('standard', 'ref_file')]),
                  (init_identity_warp, mrtrix_dwi2mni_warp, [('out_file', 'in_file')]),  # create mrtrix warp
                  (dti2mni_wf, mrtrix_dwi2mni_warp, [('convertwarp_dti2mni.out_file', 'field_file')]),
                  (sf, dwi_to_standard_check, [('dwi', 'in_file')]),
                  # check if mrtrix warp works well by applying warp to mni image
                  (mrtrix_dwi2mni_warp, dwi_to_standard_check, [('out_file', 'warp_file')]),
                  (trk2tck, tck2dwi, [('out_file', 'tracks')]),  # transform tracks from dwi to mni space
                  (mrtrix_dwi2mni_warp, tck2dwi, [('out_file', 'transform')])
                  ])

# map mrtrix streamlines on top of standard space to check if they were regstered to MNI correctly
mni2dti_wf.connect([(tck2dwi, tckmap_mni2dti, [('out_file', 'in_file')]),
                  (sf, tckmap_mni2dti, [('dwi', 'reference')])
                  ])

# convert trk back to tck
mni2dti_wf.connect([(finta_wf, trk2tck, [('finta_filter.out_filtered_trk', 'tractogram')])
                     ])

mni2dti_wf.write_graph(dotfilename='graph_orig.dot', graph2use='orig', format='svg')
mni2dti_wf.write_graph(dotfilename='graph_orig.dot', graph2use='orig', format='png')

mni2dti_wf.write_graph(dotfilename='graph_hierarchical.dot', graph2use='hierarchical', format='svg')
mni2dti_wf.write_graph(dotfilename='graph_hierarchical.dot', graph2use='hierarchical', format='png')

mni2dti_wf.write_graph(dotfilename='graph_exec.dot', graph2use='exec', format='svg')
mni2dti_wf.write_graph(dotfilename='graph_exec.dot', graph2use='exec', format='png')

mni2dti_wf.write_graph(dotfilename='graph_flat.dot', graph2use='flat', format='svg')
mni2dti_wf.write_graph(dotfilename='graph_flat.dot', graph2use='flat', format='png')

mni2dti_wf.write_graph(dotfilename='graph_colored.dot', graph2use='colored', format='svg')
mni2dti_wf.write_graph(dotfilename='graph_colored.dot', graph2use='colored', format='png')
mni2dti_wf.run()