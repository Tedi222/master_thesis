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

from nipype.interfaces.mrtrix3.base import MRTrix3BaseInputSpec, MRTrix3Base
from nipype.interfaces.mrtrix3.utils import TensorMetricsInputSpec, TensorMetricsOutputSpec
from nipype.interfaces.mrtrix3.connectivity import (BuildConnectomeInputSpec,
                                                    BuildConnectomeOutputSpec,
                                                    BuildConnectome)
import os.path as op


class TckCompressInputSpec(TraitedSpec):
   
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
                             desc="statistic ffor edges [mean/median/min/max]")
    
    scale_file = File(exists=True, argstr="-scale_file %s", desc="scale edge contribution .txt file")

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
    output_sift_indices = File(argstr="%s", desc="output a text file containing the binary selection of streamlines")



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
        if isdefined(self.inputs.output_sift_indices):
            outputs["output_sift_indices"] = op.abspath(self.inputs.output_sift_indices)

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
                                                
                                                
