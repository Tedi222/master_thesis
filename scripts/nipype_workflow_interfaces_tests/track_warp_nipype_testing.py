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


#####################################################


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
             'tracks': '/home/teodorps/tests/pipeline_tests/diff_pipeline_test_28_04_2023/__id_02/tckgen/tracked.tck'}
             #'tracks': '/home/teodorps/data_full/subj02/tracts1M.tck'}

sf = Node(SelectFiles(templates), name='fileselector')
# sf.inputs.base_directory = '/home/teodorps/tests/pipeline_tests/diff_pipeline_test_28_04_2023/__id_02'

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

    filename, _ = os.path.splitext(tractogram)
    output_filename = filename + '.trk'

    # Build header using infos from the anatomical image.
    header = {}
    header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
    header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = nii.shape[:3]
    header[Field.VOXEL_ORDER] = ''.join(aff2axcodes(nii.affine))

    tck = nib.streamlines.load(tractogram)
    nib.streamlines.save(tck.tractogram, output_filename, header=header)


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

    filename, _ = os.path.splitext(tractogram)
    output_filename = filename + '.trk'

    trk = nib.streamlines.load(tractogram)
    nib.streamlines.save(trk.tractogram, output_filename)


trk2tck = Node(Function(input_names=['tractogram'],
                        output_names=['out_file'],
                        function=trackvis_to_mrtrix),
               name='trk2tck')

pipeline = Workflow(name='convertwarp_test_final_aaaaa',
                    base_dir='/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing')

pipeline.connect([(sf, convertwarp_dti2mni, [('standard', 'reference'),  # merge warps dti2t1 + t12mni
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
pipeline.connect([(tck2mni, tck2trk, [('out_file', 'tractogram')]),
                  (sf, tck2trk, [('standard', 'anatomy')])
                  ])

# map mrtrix streamlines on top of standard space to check if they were regstered to MNI correctly
pipeline.connect([(tck2mni, tckmap, [('out_file', 'in_file')]),
                  (sf, tckmap, [('standard', 'reference')])
                  ])

pipeline.write_graph(graph2use='orig')
pipeline.run()

##############

# from IPython.display import Image
#
# import networkx as nx
#
# G = nx.nx_agraph.read_dot(
#     '/home/teodorps/tests/pipeline_tests/track_warp_nipype_testing/convertwarp_test/graph_detailed.dot')
# A = nx.nx_agraph.to_agraph(G)
# A.layout('dot')
# A.draw('abcd.png')

##############################

# import dipy
# import nipype
# import nibabel
#
# print(dipy.__version__)
# #1.3.0
# print(nipype.__version__)
# #1.8.5
# print(nibabel.__version__)
# #3.0.0