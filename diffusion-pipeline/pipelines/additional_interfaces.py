from nipype.interfaces.mrtrix3.reconst import (EstimateFODInputSpec, 
                                               EstimateFODOutputSpec,
                                               EstimateFOD)

from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine,
    traits,
    TraitedSpec,
    File,
    InputMultiPath,
    isdefined,
)

from nipype.interfaces.base.traits_extension import Directory, Str, OutputMultiPath

from nipype.interfaces.mrtrix3.base import MRTrix3BaseInputSpec, MRTrix3Base
from nipype.interfaces.mrtrix3.utils import TensorMetricsInputSpec, TensorMetricsOutputSpec
from nipype.interfaces.mrtrix3.connectivity import (BuildConnectomeInputSpec,
                                                    BuildConnectomeOutputSpec,
                                                    BuildConnectome)
from nipype.utils.filemanip import split_filename
import os
import os.path as op


class TckCompressInputSpec(TraitedSpec):
   
    error_rate = traits.Float(
        argstr="-e %f",
        desc="Maximum compression distance in mm [0.1]",
    )

    in_file = File(
        exists=True, argstr="%s", position=-2, mandatory=True, desc="input tractogram"
    )
    
    out_file = File(
        "compressed_.tck",
        argstr="%s",
        position=-1,
        usedefault=True,
        mandatory=True,
        desc="output tractogram",
    )


class TckCompressOutputSpec(TraitedSpec):
    out_file = File(argstr="%s", desc="output tractogram")


class TckCompress(MRTrix3Base):
    _cmd = "scil_compress_streamlines.py" # is that cool, calling python from python (?!)

    input_spec = TckCompressInputSpec
    output_spec = TckCompressOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)

        return outputs


class BuildConnectomeSmplInputSpec(BuildConnectomeInputSpec):
    stat_edge = traits.String(argstr="-stat_edge %s", 
                             desc="statistic for edges [mean/median/min/max]")
    
    # scale_file = File(exists=True, argstr="-scale_file %s", position=-4, desc="scale edge contribution .txt file")
    scale_file = File(argstr="-scale_file %s", position=-4, mandatory=True, desc="scale edge contribution .txt file")

class BuildConnectomeSmpl(MRTrix3Base):
    """
    see https://github.com/nipy/nipype/interfaces/mrtrix3/connectivity.py
    """

    _cmd = 'tck2connectome'
    input_spec = BuildConnectomeSmplInputSpec
    output_spec = BuildConnectomeOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)

        return outputs


class TckSampleInputSpec(MRTrix3BaseInputSpec):
   
    in_file = File(
        exists=True, argstr="%s", position=-3, mandatory=True, desc="input tractogram"
    )

    in_image = File(
        argstr="%s", position=-2, mandatory=True, desc="input image (FA/MD/RD)"
    )
    
    out_file = File(
        "sampled_.txt",
        argstr="%s",
        position=-1,
        usedefault=True,
        mandatory=True,
        desc="output sampled values",
    )
    
    stat_tck = traits.String(argstr="-stat_tck %s", 
                             desc="statistic from values [mean/median/min/max]")


class TckSampleOutputSpec(TraitedSpec):
    out_file = File(argstr="%s", desc="output sampled values")


class TckSample(MRTrix3Base):
    _cmd = "tcksample"

    input_spec = TckSampleInputSpec
    output_spec = TckSampleOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)

        # return outputs
        return outputs


class TensorMetricsRDInputSpec(TensorMetricsInputSpec):
    out_rd = File(argstr='-rd %s', desc='output RD file')


class TensorMetricsRDOutputSpec(TensorMetricsOutputSpec):
    out_rd = File(argstr='-rd %s', desc='output RD file')


class TensorMetricsRD(CommandLine):
    _cmd = 'tensor2metric'
    input_spec = TensorMetricsRDInputSpec
    output_spec = TensorMetricsRDOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()

        for k in list(outputs.keys()):
            if isdefined(getattr(self.inputs, k)):
                outputs[k] = op.abspath(getattr(self.inputs, k))

        return outputs


class FilterTckSIFTInputSpec(MRTrix3BaseInputSpec):
   
    in_file = File(
        exists=True, argstr="%s", position=-3, mandatory=True, desc="input tractogram"
    )

    in_fod = File(
        argstr="%s", position=-2, mandatory=True, desc="input fod (ss3t.mif)"
    )
    
    out_file = File(
        "filtered_tck.tck",
        argstr="%s",
        position=-1,
        usedefault=True,
        mandatory=True,
        desc="output tractogram",
    )
    
    output_counts = traits.Int(argstr="-output_at_counts %d", 
        desc='output filtered track files (one per integer), provide list of int')
    
    output_sift_indices = traits.File(argstr="-out_selection %s", 
        desc='output a text file containing the binary selection of streamlines')


class FilterTckSIFTOutputSpec(TraitedSpec):

    out_file = File(argstr="%s", desc="output tractogram")



class FilterTckSIFT(MRTrix3Base):
    """
    interface for tcksift https://mrtrix.readthedocs.io/en/dev/reference/commands/tcksift.html
    """

    _cmd = "tcksift"

    input_spec = FilterTckSIFTInputSpec
    output_spec = FilterTckSIFTOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)

        return outputs



class EstimateFODSS3TInputSpec(MRTrix3BaseInputSpec):
   
    in_file = File(
        exists=True, argstr="%s", position=-7, mandatory=True, desc="input DWI image"
    )
    
    wm_txt = File(
        argstr="%s", position=-6, mandatory=True, desc="WM response text file"
    )
    
    wm_odf = File(
        "wm.mif",
        argstr="%s",
        position=-5,
        usedefault=True,
        mandatory=True,
        desc="output WM ODF",
    )
    
    gm_txt = File(argstr="%s", position=-4, desc="GM response text file")
    
    gm_odf = File(
        "gm.mif", usedefault=True, argstr="%s", position=-3, desc="output GM ODF"
    )
    
    csf_txt = File(argstr="%s", position=-2, desc="CSF response text file")
    
    csf_odf = File(
        "csf.mif", usedefault=True, argstr="%s", position=-1, desc="output CSF ODF"
    )
    
    mask_file = File(exists=True, argstr="-mask %s", desc="mask image")


class EstimateFODSS3TDOutputSpec(TraitedSpec):

    wm_odf = File(argstr="%s", desc="output WM ODF")
    gm_odf = File(argstr="%s", desc="output GM ODF")
    csf_odf = File(argstr="%s", desc="output CSF ODF")


class EstimateFODSS3T(EstimateFOD):
    """
    this is an adaptation from EstimateFOD to use 3 tissue type fork
    """

    _cmd = "ss3t_csd_beta1"

    input_spec = EstimateFODSS3TInputSpec
    output_spec = EstimateFODSS3TDOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["wm_odf"] = op.abspath(self.inputs.wm_odf)
        if isdefined(self.inputs.gm_odf):
            outputs["gm_odf"] = op.abspath(self.inputs.gm_odf)
        if isdefined(self.inputs.csf_odf):
            outputs["csf_odf"] = op.abspath(self.inputs.csf_odf)

        return outputs


class Generate5ttMaskInputSpec(MRTrix3BaseInputSpec):
    algorithm = traits.Enum(
        "fsl",
        "gif",
        "freesurfer",
        argstr="%s",
        position=-5,
        mandatory=True,
        desc="tissue segmentation algorithm",
    )
    in_file = File(
        exists=True, argstr="%s", mandatory=True, position=-4, desc="input image"
    )
    out_file = File(argstr="%s", mandatory=True, position=-3, desc="output image")

    mask = File(argstr="-mask %s", mandatory=True, position=-2, desc="precomputed mask")
    options = traits.String(argstr=" %s", position =-1)


class Generate5ttMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output image")


class Generate5ttMask(MRTrix3Base):

    _cmd = "5ttgen"
    input_spec = Generate5ttMaskInputSpec
    output_spec = Generate5ttMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        # outputs = self.output_spec().trait_get()
        outputs["out_file"] = op.abspath(self.inputs.out_file)
        return outputs
                                                
                                                
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
        outputs["out_filtered_trk"] = op.join(op.abspath(self.inputs.output), 'filtered_merged.trk')
        outputs["out_all"] = op.abspath(self.inputs.output)

        return outputs


def streamline_space_filtering_clustering(mni_tractogram_path):

    import os
    import sys
    sys.path.append("/home/teodorps/tests/scripts") # add path to scripts folder containing custom_clustering_filtering_utils.py
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