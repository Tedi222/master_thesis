#![usr/local/lib/python3.8]
from nipype import config
from graphviz import dot
from nipype import Node, Workflow, SelectFiles, IdentityInterface, DataSink
from nipype.interfaces import mrtrix3, fsl
from nipype.interfaces.utility import Function
from nipype.interfaces.base import TraitedSpec, TraitError
from nipype.interfaces.base.traits_extension import Directory, BasePath, File

from additional_interfaces import (EstimateFODSS3T, 
                                   Generate5ttMask, 
                                   FilterTckSIFT,
                                   TensorMetricsRD,
                                   TckSample,
                                   BuildConnectomeSmpl,
                                   TckCompress)

from nipype.interfaces.utility import Merge
from nipype.interfaces.mrtrix.tensors import FSL2MRTrix
from nipype.interfaces.mrtrix.preprocess import DWI2Tensor

# from nipype.utils.draw_gantt_chart import generate_gantt_chart


# (-1) fractional anisotropy
# dtifit 
#    -k ${entry}/eddy_corrected_data.nii.gz 
#    -o ${entry}/dti 
#    -m ${entry}/nodif_brain.nii.gz 
#    -r ${entry}/eddy_corrected_data.eddy_rotated_bvecs 
#    -b ${entry}/bvals 
#    --kurtdir -V  # NOTE --kurtdir only when kurtosis output required, -V: vebose
dwi2fa = Node(fsl.DTIFit(
    output_type = 'NIFTI_GZ',
    ), name='dwi2fa')

fsl2mrtrix = Node(FSL2MRTrix(
    invert_x = True,
    invert_y = True,
    invert_z = False
    ), name='fsl2mrtrix')

dwi2dti = Node(DWI2Tensor(
    ), name='dwi2dti')

dti2met = Node(TensorMetricsRD(
    out_fa = 'fa.nii.gz',
    out_adc = 'md.nii.gz',
    out_rd = 'rd.nii.gz'
    ), name='dti2met')

# (0) registration nodes
dwi2t1_aff = Node(fsl.FLIRT(
    output_type = 'NIFTI_GZ',
    cost = 'normmi',
    rigid2D = True
    ), name='dwi2t1_aff')

dwi2t1_aff_mask = Node(fsl.FLIRT(
    output_type = 'NIFTI_GZ',
    cost = 'normmi',
    rigid2D = True,
    interp = 'nearestneighbour'
    ), name='dwi2t1_aff_mask')

dwi2t1_nl = Node(fsl.FNIRT(
    output_type = 'NIFTI_GZ',
    fieldcoeff_file=True
    ), name='dwi2t1_nl')

t12mni_aff = Node(fsl.FLIRT(
    output_type = 'NIFTI_GZ'
    ), name='t12mni_aff')

# previous: dof = 6

t12mni_nl = Node(fsl.FNIRT(
    output_type = 'NIFTI_GZ',
    fieldcoeff_file=True
    ), name='t12mni_nl')

# (1) inverse transformation nodes
t12dwi_inv_aff = Node(fsl.ConvertXFM(
    invert_xfm = True,
    ), name='t12dwi_inv_aff')

t12dwi_inv_warp = Node(fsl.InvWarp(
    output_type = 'NIFTI_GZ',
    ), name='t12dwi_inv_warp')

mni2t1_inv_aff = Node(fsl.ConvertXFM(
    invert_xfm = True,
    ), name='mni2t1_inv_aff')

mni2t1_inv_warp = Node(fsl.InvWarp(
    output_type = 'NIFTI_GZ',
    ), name='mni2t1_inv_warp')

# (2) warp (inverse registration) nodes
mni2t1_nl = Node(fsl.ApplyWarp(
    interp = 'nn',
    ), name='mni2t1_nl')

# [!] no orig code here, fabistuff TODO test
mni2t1_aff = Node(fsl.ApplyXFM(
    apply_xfm = True
    ), name='mni2t1_aff')

t12dwi_nl = Node(fsl.ApplyWarp(
    interp = 'nn'
    ), name='t12dwi_nl')

# [!] no orig code here, fabistuff TODO test
t12dwi_aff = Node(fsl.ApplyXFM(
    apply_xfm = True
    ), name='t12dwi_aff')

# (3) TRACTOGRAPHY nodes
def connect_two_files(file1, file2):
    return (file1, file2)

merge_bvec_bval = Node(Function(input_names=['file1', 'file2'],
                                output_names=['out_file'], 
                                function=connect_two_files),
                                name='merge_bvec_bval')

nii2mif = Node(mrtrix3.MRConvert(
    out_file = 'dwi.mif'
    ), name='nii2mif')

five_tt = Node(Generate5ttMask(
    algorithm = 'fsl',
    out_file = '5tt.mif',
    options='-debug'
    ), name='five_tt')

# '5ttgen fsl '+ path + filename + ' ' 
#             + path + '5TT.mif -mask ' 
#             + path + 'nodif_brain_mask.nii.gz 
#             -force')# 

dwi2response = Node(mrtrix3.preprocess.ResponseSD(
    algorithm = 'dhollander',
    wm_file = 'wm.txt',
    gm_file = 'gm.txt',
    csf_file = 'csf.txt',
    ), name='dwi2response')

dwi2fod_3tt = Node(EstimateFODSS3T(
    ), name='dwi2fod_3tt')

tckgen = Node(mrtrix3.tracking.Tractography(
    backtrack = True,
    crop_at_gmwmi = True,
    min_length = 25,
    max_length = 350,
    power = 3, # int? 3? orig: 0.33
    select = 100000, # orig:  10000000,
    cutoff = 0.08,
    ), name='tckgen')

# -act ' + path + '5TT.mif 
# -seed_dynamic ' + path + 'wmfod_ss3t.mif 
# -nthreads 16 
# (#-step 0.4)

# (4) Filtering 
# (4.1) Compression 1
tckcompress = Node(TckCompress(
    ), name='tckcompress')

# (4.2) Tract Filtering
tcksift = Node(FilterTckSIFT(
    output_counts = 10000,
    output_sift_indices = 'sifted_idx.txt'
    ), name='tcksift')

# (4.1) Compression 2
tckcompress_filtered = Node(TckCompress(
    ), name='tckcompress_filtered')


# (5) Connectivity
# (5.1) RAW Connectivity
tck2con = Node(mrtrix3.connectivity.BuildConnectome(
    search_radius = 2, 
    # symmetric TODO!!
    zero_diagonal = True
    ), name='tck2con')

# (5.2) sampled Connectivity based on tensor metrics
# tcksample 
#   '+ trackpath + 
#   ' ' + path + imagename + 
#   ' ' + path + 'sampled_' + taskname + ".txt" 
#   + ' -stat_tck mean'
tcksamplers = dict()
tck2connecters = dict()

for metric in ['fa', 'md', 'rd']:
    tcksamplers[metric] = Node(TckSample(
        stat_tck = 'mean',
        out_file = f'sampled_{metric}.txt'
        ), name=f'tcksample_{metric}')
    tck2connecters[metric] = Node(BuildConnectomeSmpl(
        out_file = f'connectome_{metric}.csv',
        search_radius = 2, 
        # symmetric TODO!!
        zero_diagonal = True,
        stat_edge='mean'
        ), name=f"tck2con_{metric}")


if __name__ == '__main__':
    # config stuff
    # config.enable_debug_mode()
    # config.enable_resource_monitor()


    cfg = dict(monitoring={'enabled': True})
    config.enable_resource_monitor()

    # cfg = dict(logging=dict(workflow_level = 'DEBUG'),
    cfg = dict(logging=dict(workflow_level = 'INFO'),
               execution={'keep_inputs': False,
                          'remove_unneccesary_outputs': False})
    config.update_config(cfg)


    # args_dict = {'status_callback' : log_nodes_cb}

    # pipeline
    pipeline = Workflow(name='diff_pipeline_test_28_04_2023', base_dir='/home/teodorps/data/marvin_testdata')
    """
    templates = {'dwi':         '{_id}_DTI/eddy_corrected_data.nii.gz',
                 't1':          '{_id}_T1/t1.nii.gz',
                 'mask':        '{_id}_DTI/nodif_brain_mask.nii.gz',
                 'standart':    '/data/marvin_testdata/mni_icbm152_nlin_asym_09c/'+\
                                'mni_icbm152_t1_tal_nlin_asym_09c.nii', 
                 'atlas':       '/data/marvin_testdata/parcAtl_mni.nii.gz',
                 'bvals':       '{_id}_DTI/bvals',
                 'bvecs':       '{_id}_DTI/bvecs'}

    """
    templates = {'dwi':         'sub_{_id}/eddy_corrected_data.nii.gz',
                 't1':          'sub_{_id}/T1.nii.gz',
                 'mask':        'sub_{_id}/nodif_brain_mask.nii.gz',
                 'standart':    'mni_icbm152_nlin_asym_09c/'+\
                                'mni_icbm152_t1_tal_nlin_asym_09c.nii', 
                 'atlas':       'parcAtl_mni.nii.gz',
                 'bvals':       'sub_{_id}/bvals',
                 'bvecs':       'sub_{_id}/rotated_bvecs'}
    subj_source = Node(IdentityInterface(fields=['_id']),
        iterables=('_id', ['02']),
        name='subj_source')
    
    sf = Node(SelectFiles(templates), name='fileselector')
    sf.inputs.base_directory = '/home/teodorps/data/marvin_testdata'
    # sf.inputs._id = '02'
    # sf.inputs.base_directory = '/data/data_diffusion/ppmi_data'
    # sf.inputs._id = '3102'
    # sf.inputs.base_directory = '/data/data_diffusion/marvin_data'
    # sf.inputs._id = '01'

  # datasink for final storage
    datasink = Node(DataSink(base_directory='/home/teodorps/data/datasink',
        parameterization=False),
        "datasink")


    # convert dwi to .mif
    pipeline.connect([
        (subj_source, sf, [('_id', '_id')]),
        (sf, merge_bvec_bval, [('bvecs', 'file1'),
                               ('bvals', 'file2')]),
        (sf, nii2mif, [('dwi', 'in_file')]),
        (merge_bvec_bval, nii2mif, [('out_file', 'grad_fsl')])
        ])

    # connect registration nodes
    pipeline.connect([
        (sf, dwi2fa, [('dwi', 'dwi'),
                      ('mask', 'mask'),
                      ('bvals', 'bvals'),
                      ('bvecs', 'bvecs')]),
        (nii2mif, dwi2dti, [('out_file', 'in_file')]),
        (dwi2dti, dti2met, [('tensor', 'in_file')]),
        (dwi2fa, dwi2t1_aff, [('FA', 'in_file')]),
        (sf,     dwi2t1_aff, [('t1', 'reference')]),
        (dwi2fa, dwi2t1_nl, [('FA', 'in_file')]),
        (sf,     dwi2t1_nl, [('t1', 'ref_file')]),
        (dwi2t1_aff, dwi2t1_nl, [('out_matrix_file', 'affine_file')]), 
        (dwi2t1_aff_mask, dwi2t1_nl, [('out_file', 'refmask_file')]),
        (sf, dwi2t1_aff_mask, [('mask', 'in_file'),
                               ('t1', 'reference')]),
        (sf, t12mni_aff, [('t1', 'in_file'),
                          ('standart', 'reference')]),
        (sf, t12mni_nl, [('t1', 'in_file'),
                         ('standart', 'ref_file')]),
        (t12mni_aff, t12mni_nl, [('out_matrix_file', 'affine_file')]),
        (dwi2t1_aff, t12dwi_inv_aff, [('out_matrix_file', 'in_file')]), # inverse trafo
        (dwi2fa, t12dwi_inv_warp, [('FA', 'reference')]),
        (dwi2t1_nl, t12dwi_inv_warp, [('fieldcoeff_file', 'warp')]),
        (t12mni_aff, mni2t1_inv_aff, [('out_matrix_file', 'in_file')]),
        (sf, mni2t1_inv_warp, [('t1', 'reference')]),
        (t12mni_nl, mni2t1_inv_warp, [('fieldcoeff_file', 'warp')]),
        (sf, mni2t1_nl, [('atlas', 'in_file'), 
                         ('t1', 'ref_file')]), 
        (mni2t1_inv_warp, mni2t1_nl, [('inverse_warp', 'field_file')]), # atlas warp
        (mni2t1_nl, t12dwi_nl, [('out_file', 'in_file')]),
        (dwi2fa, t12dwi_nl, [('FA', 'ref_file')]),
        (t12dwi_inv_warp, t12dwi_nl, [('inverse_warp', 'field_file')])
        ])

    # connect tractography nodes
    pipeline.connect([
        (sf, five_tt , [('t1', 'in_file'), 
                        ('mask', 'mask')]), # TODO: check mask space
        (nii2mif, dwi2response, [('out_file' , 'in_file')]), 
        (nii2mif, dwi2fod_3tt, [('out_file', 'in_file')]),
        (sf, dwi2fod_3tt, [('mask', 'mask_file')]), #TODO: check mask space
        (dwi2response, dwi2fod_3tt, [('wm_file', 'wm_txt'), #TODO: does that work?
                                      ('gm_file' , 'gm_txt'),
                                      ('csf_file', 'csf_txt')]), 
        (five_tt, tckgen, [('out_file', 'act_file')]),
        (dwi2fod_3tt, tckgen, [('wm_odf' , 'in_file'),
                               ('wm_odf', 'seed_dynamic')])
        ])

    # connect filtering 
    pipeline.connect([
        (tckgen, tckcompress, [('out_file', 'in_file')])
        ])

    pipeline.connect([
        (dwi2fod_3tt, tcksift, [('wm_odf' , 'in_fod')]),
        (tckcompress, tcksift, [('out_file', 'in_file')]) # tckcompress
        ])

    # pipeline.connect([
    #     (tcksift, tckcompress_filtered, [('out_file', 'in_file')])
    #     ])


    # connect connectome
    pipeline.connect([
        (tcksift, tck2con, [('out_file', 'in_file')]),
        (t12dwi_nl, tck2con, [('out_file', 'in_parc')])
        ])

    for metric in ['fa', 'md', 'rd']:
        # import ipdb; ipdb.set_trace()
        image_name = 'out_adc' if metric == 'md' else f'out_{metric}' 
        # sampled metrics
        pipeline.connect([
            (tcksift, tcksamplers[metric], [('out_file', 'in_file')]),
            (dti2met, tcksamplers[metric], [(image_name, 'in_image')]),
            (tcksift, tck2connecters[metric], [('out_file', 'in_file')]),
            (t12dwi_nl, tck2connecters[metric], [('out_file', 'in_parc')]),
            (tcksamplers[metric], tck2connecters[metric],[('out_file', 'scale_file')])
            ])

    # connect datasink
    pipeline.connect([
        (subj_source, datasink, [('_id', 'container')]), 
        (dwi2fa, datasink, [('FA', 'inputs.@fa')]), 
        (t12dwi_nl, datasink, [('out_file', 'inputs.@parcellation')]), 
        (dwi2t1_aff, datasink, [('out_matrix_file', 'register.@dwi2t1_affine')]), 
        (dwi2t1_nl , datasink, [('fieldcoeff_file', 'register.@dwi2t1_coeff')]), 
        (t12mni_aff , datasink, [('out_matrix_file', 'register.@t12mni_affine')]), 
        (t12mni_nl , datasink, [('fieldcoeff_file', 'register.@t12mni_coeff')]), 
        (tckcompress , datasink, [('out_file', 'tractogram.@full')]), 
        (tcksift, datasink, [('out_file', 'tractogram.@filtered')]), 
        (tck2con, datasink, [('out_file', 'connectivity.@raw')]), 
        (tck2connecters['fa'], datasink, [('out_file', 'connectivity.@fa')]), 
        (tck2connecters['md'], datasink, [('out_file', 'connectivity.@md')]), 
        (tck2connecters['rd'], datasink, [('out_file', 'connectivity.@rd')]), 
        ])



    pipeline.write_graph(graph2use='orig')

    # pipeline.run(plugin='MultiProc')
    pipeline.run()
    
    # generate_gantt_chart('/home/user/run_stats.log')



