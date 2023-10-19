import sys
from nipype import config

from nipype import Node, Workflow, SelectFiles, IdentityInterface, DataSink
from nipype.interfaces import mrtrix3, fsl
from nipype.interfaces.utility import Function

from additional_interfaces import (EstimateFODSS3T, 
                                   Generate5ttMask, 
                                   FilterTckSIFT,
                                   TensorMetricsRD,
                                   TckSample,
                                   BuildConnectomeSmpl,
                                   TckCompress,
                                   WarpInit,
                                   WarpMRTransform,
                                   TCKTransform,
                                   FindThresholds,
                                   FINTA_Modified,
                                   streamline_space_filtering_clustering,
                                   mrtrix_to_trackvis,
                                   trackvis_to_mrtrix)

from nipype.interfaces.utility import Merge
from nipype.interfaces.mrtrix.tensors import FSL2MRTrix
from nipype.interfaces.mrtrix.preprocess import DWI2Tensor

from joblib import Parallel, delayed
import yaml


# from nipype.utils.draw_gantt_chart import generate_gantt_chart

dwi2fa = Node(fsl.DTIFit(
    output_type = 'NIFTI_GZ',
    ), name='dwi2fa')

fsl2mrtrix = Node(FSL2MRTrix(
    invert_x = True,
    invert_y = True,
    invert_z = False
    ), name='fsl2mrtrix')

dwi2dti = Node(DWI2Tensor(
    args = '-ols'
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

mni2t1_aff = Node(fsl.ApplyXFM(
    apply_xfm = True
    ), name='mni2t1_aff')

t12dwi_nl = Node(fsl.ApplyWarp(
    interp = 'nn'
    ), name='t12dwi_nl')

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
    out_file = '5tt.mif'
    ), name='five_tt')

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
    select = 10000000, # orig:  10000000,
    cutoff = 0.08,
    ), name='tckgen')

# (4) Filtering 
# (4.1) Compression 1
tckcompress = Node(TckCompress(
    error_rate = 0.35
    ), name='tckcompress')

# (4.2) Tract Filtering - SIFT
tcksift = Node(FilterTckSIFT(
    output_counts = 1000000
    ), name='tcksift')

# (4.2) Tract Filtering - FINTA
# (4.2.1) DWI -> MNI, TCK -> TRK
# combine fsl warps
convertwarp_dti2mni = Node(fsl.utils.ConvertWarp(
    output_type='NIFTI_GZ'),
    name='convertwarp_dti2mni')

# invert combined warp
invwarp_mni2dti = Node(fsl.utils.InvWarp(
    output_type='NIFTI_GZ'),
    name='invwarp_mni2dti')

# generate the MNI2DWI transformation warp with mrtrix (combined with the fsl warps: warps_MNI2dwi.nii.gz)
# warpinit is a MRTrix3 command
init_inv_identity_warp = Node(WarpInit(),
                              name='init_inv_identity_warp')

mrtrix_mni2dwi_warp = Node(fsl.preprocess.ApplyWarp(
    output_type='NIFTI_GZ'),
    name='mrtrix_mni2dwi_warp')

# check the warping results by applying it to MNI_3mm_brain
standard_to_dwi_check = Node(WarpMRTransform(),
                             name='standard_to_dwi_check')

tck2mni = Node(TCKTransform(),
               name='tck2mni')

tck2trk = Node(Function(input_names=['anatomy', 'tractogram'],
                        output_names=['out_file'],
                        function=mrtrix_to_trackvis),
               name='tck2trk')

# map mrtrix streamlines on top of standard space to check if they were regstered to MNI correctly
tckmap = Node(mrtrix3.utils.ComputeTDI(
    out_file='tck2mni_mapped.nii.gz'),
    name='tckmap')

# (4.2.2) STREAMLINE SPACE FILERING + CLUSTERING + FINTA
cluster_streamlines = Node(Function(input_names=['mni_tractogram_path'],
                                    output_names=['streamline_classes_path', 'centroids_path',
                                                  'clusters_valid_path', 'implausible_streamlines_path'],
                                    function=streamline_space_filtering_clustering),
                           name='cluster_streamlines')

find_threshold = Node(FindThresholds(output='thresholds', options='-v'),
                      name='find_thresholds')
# find_threshold.config = {'execution': {'keep_unnecessary_outputs': 'true'}}

finta_filter = Node(FINTA_Modified(output='filtered', device="cuda"),
                    name='finta_filter')

# (4.2.3) TRK -> TCK, MNI -> DWI
mrtrix_dwi2mni_warp = Node(fsl.preprocess.ApplyWarp(
    output_type='NIFTI_GZ'),
    name='mrtrix_dwi2mni_warp')

# check the warping results by applying it to MNI_3mm_brain
dwi_to_standard_check = Node(WarpMRTransform(),
                             name='dwi_to_standard_check')

tck2dwi = Node(TCKTransform(),
               name='tck2dwi')

init_identity_warp = Node(WarpInit(),
                          name='init_identity_warp')

tckmap_mni2dti = Node(mrtrix3.utils.ComputeTDI(
    out_file='tck2dwi_mapped.nii.gz'),
    name='tckmap_mni2dti')

trk2tck = Node(Function(input_names=['tractogram'],
                        output_names=['out_file'],
                        function=trackvis_to_mrtrix),
               name='trk2tck')

# (4.1) Compression 2
tckcompress_filtered = Node(TckCompress(
    error_rate = 0.35
    ), name='tckcompress_filtered')

# (5) Connectivity NO SIFT
tck2con = Node(mrtrix3.connectivity.BuildConnectome(
    search_radius = 2, 
    # symmetric TODO!!
    zero_diagonal = True
    ), name='tck2con')

tcksamplers= dict()
tck2connecters= dict()

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

# (5) Connectivity SIFT
# (5.1) RAW Connectivity
tck2con_sift = Node(mrtrix3.connectivity.BuildConnectome(
    search_radius = 2, 
    # symmetric TODO!!
    zero_diagonal = True
    ), name='tck2con_sift')

# (5.2) sampled Connectivity based on tensor metrics
tcksamplers_sift = dict()
tck2connecters_sift = dict()

for metric in ['fa', 'md', 'rd']:
    tcksamplers_sift[metric] = Node(TckSample(
        stat_tck = 'mean',
        out_file = f'sampled_{metric}.txt'
        ), name=f'tcksample_sift_{metric}')
    tck2connecters_sift[metric] = Node(BuildConnectomeSmpl(
        out_file = f'connectome_{metric}.csv',
        search_radius = 2, 
        # symmetric TODO!!
        zero_diagonal = True,
        stat_edge='mean',
        scale_file='placeholder.txt'
        ), name=f"tck2con_sift_{metric}")

# (5) Connectivity FINTA
# (5.1) RAW Connectivity
tck2con_finta = Node(mrtrix3.connectivity.BuildConnectome(
    search_radius = 2,
    # symmetric TODO!!
    zero_diagonal = True
    ), name='tck2con_finta')

# (5.2) sampled Connectivity based on tensor metrics
tcksamplers_finta = dict()
tck2connecters_finta = dict()

for metric in ['fa', 'md', 'rd']:
    tcksamplers_finta[metric] = Node(TckSample(
        stat_tck = 'mean',
        out_file = f'sampled_{metric}.txt'
        ), name=f'tcksample_finta_{metric}')
    tck2connecters_finta[metric] = Node(BuildConnectomeSmpl(
        out_file = f'connectome_{metric}.csv',
        search_radius = 2,
        # symmetric TODO!!
        zero_diagonal = True,
        stat_edge='mean',
        scale_file='placeholder.txt'
        ), name=f"tck2con_finta_{metric}")


def read_subj_list(file_):
    with open(file_, 'r') as txt:
        data = txt.read()
        data = data.split('\n')
        data = list(filter(None, data)) # remove empty str
    return data


def main(subj_list_, cfg_):
    t1_list = [sub.split('_')[0] for sub in subj_list_]

    data_root = cfg_['data_root'] # '/datadir/PPMI/ppmi_preproc/'
    rundir = cfg_['rundir'] # '/datadir/PPMI/rundir_mp'
    # dwi_ = 'dwi_reorganized/'
    # t1_ = 't1_reorganized/'
    templates = cfg['templates']

    cfg_specification = dict(monitoring={'enabled': True})
    config.enable_resource_monitor()
    # config stuff
    # config.enable_debug_mode()
    # config.enable_resource_monitor()

    # cfg = dict(logging=dict(workflow_level = 'DEBUG'),
    cfg_specification = dict(logging=dict(workflow_level = 'INFO'),
               execution={'keep_inputs': False,
                          'remove_unneccesary_outputs': True})
    config.update_config(cfg_specification)

    # args_dict = {'status_callback' : log_nodes_cb}

    # pipeline
    pipeline = Workflow(name=cfg_['pipeline_name'], base_dir=rundir)

    # templates = {'dwi':         dwi_+'{_id}/eddy_corrected_data.nii.gz',
    #              't1':          t1_+'{id_t1}/*.nii.gz',
    #              'mask':        dwi_+'{_id}/nodif_brain_mask.nii.gz',
    #              'standard':    '/projectdir/data/mni_icbm152_nlin_asym_09c/'+\
    #                             'mni_icbm152_t1_tal_nlin_asym_09c.nii', 
    #              'atlas':       '/projectdir/data/parcAtl_mni.nii.gz',
    #              'bvals':       dwi_+'{_id}/bvals',
    #              'bvecs':       dwi_+'{_id}/eddy_corrected_data.eddy_rotated_bvecs'}

    subj_source = Node(IdentityInterface(fields=['_id', 'id_t1']),
        iterables=[('_id', subj_list_), ('id_t1', t1_list)],
        synchronize=True,
        name='subj_source')

    sf = Node(SelectFiles(templates), name='fileselector')
    sf.inputs.base_directory = data_root

    # datasink for final storage
    datasink = Node(DataSink(base_directory=rundir.replace('rundir', 'datasink'),
        parameterization=False),
        "datasink")

    # convert dwi to .mif
    pipeline.connect([
        (subj_source, sf, [('_id', '_id'), ('id_t1', 'id_t1')]),
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
        (sf, dti2met, [('mask', 'in_mask')]),
        (dwi2fa, dwi2t1_aff, [('FA', 'in_file')]),
        (sf,     dwi2t1_aff, [('t1', 'reference')]),
        (dwi2fa, dwi2t1_nl, [('FA', 'in_file')]),
        (sf,     dwi2t1_nl, [('t1', 'ref_file')]),
        (dwi2t1_aff, dwi2t1_nl, [('out_matrix_file', 'affine_file')]), 
        (dwi2t1_aff_mask, dwi2t1_nl, [('out_file', 'refmask_file')]),
        (sf, dwi2t1_aff_mask, [('mask', 'in_file'),
                               ('t1', 'reference')]),
        (sf, t12mni_aff, [('t1', 'in_file'),
                          ('standard', 'reference')]),
        (sf, t12mni_nl, [('t1', 'in_file'),
                         ('standard', 'ref_file')]),
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
    ###################################################################


    dti2mni_wf = Workflow(name='dti2mni_wf', base_dir=rundir)

    dti2mni_wf.connect([(sf, convertwarp_dti2mni, [('standard', 'reference')]  # merge warps dti2t1 + t12mni
                                                   # ('fieldcoeff_file_dwi2t1', 'warp1'),
                                                   # ('fieldcoeff_file_t12mni', 'warp2')]
                         ),
                        (pipeline, convertwarp_dti2mni, [('dwi2t1_nl.fieldcoeff_file', 'warp1')]),
                        (pipeline, convertwarp_dti2mni, [('t12mni_nl.fieldcoeff_file', 'warp2')]),
                        (sf, invwarp_mni2dti, [('dwi', 'reference')]),  # invert dt12mni warp
                        (convertwarp_dti2mni, invwarp_mni2dti, [('out_file', 'warp')]),
                        (sf, init_inv_identity_warp, [('standard', 'in_file')]),  # initialize identity warp
                        (sf, mrtrix_mni2dwi_warp, [('dwi', 'ref_file')]),
                        (init_inv_identity_warp, mrtrix_mni2dwi_warp, [('out_file', 'in_file')]),  # create mrtrix warp
                        (invwarp_mni2dti, mrtrix_mni2dwi_warp, [('inverse_warp', 'field_file')]),
                        (sf, standard_to_dwi_check, [('standard', 'in_file')]),
                        # check if mrtrix warp works well by applying warp to mni image
                        (mrtrix_mni2dwi_warp, standard_to_dwi_check, [('out_file', 'warp_file')]),
                        (pipeline, tck2mni, [('tckcompress.out_file', 'tracks')]),  # transform tracks from dwi to mni space
                        (mrtrix_mni2dwi_warp, tck2mni, [('out_file', 'transform')])])

    # convert mrtrix streamlines format(.tck) to trackvis format(.trk)
    dti2mni_wf.connect([(tck2mni, tck2trk, [('out_file', 'tractogram')]),
                        (sf, tck2trk, [('standard', 'anatomy')])
                        ])

    # map mrtrix streamlines on top of standard space to check if they were regstered to MNI correctly
    dti2mni_wf.connect([(tck2mni, tckmap, [('out_file', 'in_file')]),
                        (sf, tckmap, [('standard', 'reference')])
                        ])

    finta_wf = Workflow(name='clustering_finta_wf', base_dir=rundir)

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

    mni2dti_wf = Workflow(name='mni2dti_wf', base_dir=rundir)

    # convert trk back to tck
    mni2dti_wf.connect([(finta_wf, trk2tck, [('finta_filter.out_filtered_trk', 'tractogram')])
                        ])

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



    ###################################################################
    # connect connectome

    # no sift
    pipeline.connect([
        (tckcompress, tck2con, [('out_file', 'in_file')]),
        (t12dwi_nl, tck2con, [('out_file', 'in_parc')])
        ])

    for metric in ['fa', 'md', 'rd']:
        # import ipdb; ipdb.set_trace()
        image_name = 'out_adc' if metric == 'md' else f'out_{metric}' 
        # sampled metrics
        pipeline.connect([
            (tckcompress, tcksamplers[metric], [('out_file', 'in_file')]),
            (dti2met, tcksamplers[metric], [(image_name, 'in_image')]),
            (tckcompress, tck2connecters[metric], [('out_file', 'in_file')]),
            (t12dwi_nl, tck2connecters[metric], [('out_file', 'in_parc')]),
            (tcksamplers[metric], tck2connecters[metric],[('out_file', 'scale_file')])
            ])

    # sift
    pipeline.connect([
        (tcksift, tck2con_sift, [('out_file', 'in_file')]),
        (t12dwi_nl, tck2con_sift, [('out_file', 'in_parc')])
        ])

    for metric in ['fa', 'md', 'rd']:
        # import ipdb; ipdb.set_trace()
        image_name = 'out_adc' if metric == 'md' else f'out_{metric}'
        # sampled metrics
        pipeline.connect([
            (tcksift, tcksamplers_sift[metric], [('out_file', 'in_file')]),
            (dti2met, tcksamplers_sift[metric], [(image_name, 'in_image')]),
            (tcksift, tck2connecters_sift[metric], [('out_file', 'in_file')]),
            (t12dwi_nl, tck2connecters_sift[metric], [('out_file', 'in_parc')]),
            (tcksamplers_sift[metric], tck2connecters_sift[metric], [('out_file', 'scale_file')])
            ])

    # FINTA
    pipeline.connect([
        (mni2dti_wf, tck2con_finta, [('tck2dwi.out_file', 'in_file')]),
        (t12dwi_nl, tck2con_finta, [('out_file', 'in_parc')])
        ])

    # for metric in ['fa', 'md', 'rd']:
    #     # import ipdb; ipdb.set_trace()
    #     image_name = 'out_adc' if metric == 'md' else f'out_{metric}'
    #     # sampled metrics
    #     pipeline.connect([
    #         (mni2dti_wf, tcksamplers_finta[metric], [('tck2dwi.out_file', 'in_file')]),
    #         (dti2met, tcksamplers_finta[metric], [(image_name, 'in_image')]),
    #         (mni2dti_wf, tck2connecters_finta[metric], [('tck2dwi.out_file', 'in_file')]),
    #         (t12dwi_nl, tck2connecters_finta[metric], [('out_file', 'in_parc')]),
    #         (tcksamplers_finta[metric], tck2connecters_finta[metric], [('out_file', 'scale_file')])
    #         ])

    # connect datasink
    pipeline.connect([
        (subj_source, datasink, [('_id', 'container')]), 
        # (dwi2fa, datasink, [('FA', 'inputs.@fa')]),
        # (t12dwi_nl, datasink, [('out_file', 'inputs.@parcellation')]),
        # (dwi2t1_aff, datasink, [('out_matrix_file', 'register.@dwi2t1_affine')]),
        (dwi2t1_nl , datasink, [('fieldcoeff_file', 'register.@dwi2t1_coeff')]),
        # (t12mni_aff , datasink, [('out_matrix_file', 'register.@t12mni_affine')]),
        (t12mni_nl , datasink, [('fieldcoeff_file', 'register.@t12mni_coeff')]),
        (tckcompress , datasink, [('out_file', 'tractogram.@full')]),
        (tcksift, datasink, [('out_file', 'tractogram.@filtered')]),
        # (mni2dti_wf, datasink, [('tck2dwi.out_file', 'tractogram.@filtered')]),

        # (tck2con_finta, datasink, [('out_file', 'connectivity_sift.@raw')]),
        # (tck2connecters_finta['fa'], datasink, [('out_file', 'connectivity_finta.@fa')]),
        # (tck2connecters_finta['md'], datasink, [('out_file', 'connectivity_finta.@md')]),
        # (tck2connecters_finta['rd'], datasink, [('out_file', 'connectivity_finta.@rd')]),

        # (tck2con_sift, datasink, [('out_file', 'connectivity_sift.@raw')]),
        # (tck2connecters_sift['fa'], datasink, [('out_file', 'connectivity_sift.@fa')]),
        # (tck2connecters_sift['md'], datasink, [('out_file', 'connectivity_sift.@md')]),
        # (tck2connecters_sift['rd'], datasink, [('out_file', 'connectivity_sift.@rd')]),

        (tck2con, datasink, [('out_file', 'connectivity.@raw')]),
        (tck2connecters['fa'], datasink, [('out_file', 'connectivity.@fa')]),
        (tck2connecters['md'], datasink, [('out_file', 'connectivity.@md')]),
        (tck2connecters['rd'], datasink, [('out_file', 'connectivity.@rd')]),
        ])

    # pipeline.write_graph(graph2use='orig')

    # pipeline.write_graph(dotfilename='graph_hierarchical.dot', graph2use='hierarchical', format='svg')
    # pipeline.write_graph(dotfilename='graph_hierarchical.dot', graph2use='hierarchical', format='png')
    #
    # pipeline.write_graph(dotfilename='graph_exec.dot', graph2use='exec', format='svg')
    # pipeline.write_graph(dotfilename='graph_exec.dot', graph2use='exec', format='png')
    #
    # pipeline.write_graph(dotfilename='graph_flat.dot', graph2use='flat', format='svg')
    # pipeline.write_graph(dotfilename='graph_flat.dot', graph2use='flat', format='png')
    #
    # pipeline.write_graph(dotfilename='graph_colored.dot', graph2use='colored', format='svg')
    # pipeline.write_graph(dotfilename='graph_colored.dot', graph2use='colored', format='png')

    # pipeline.run(plugin='MultiProc', plugin_args={'n_procs': 4})
    pipeline.run(plugin='Linear')
    
    # pipeline.run()
    
    # generate_gantt_chart('/home/user/run_stats.log')


def load_config(cfg_file='./congig.yml'):
    _cfg = yaml.safe_load(open(cfg_file, 'r'))

    return _cfg


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'usage: python dwi_to_conmat config.yml'
    cfg = load_config(sys.argv[1])
    subj_list = read_subj_list(cfg['subj_list']) 

    Parallel(n_jobs=32)(delayed(main)([subj_], cfg) for subj_ in subj_list)

    # (below) uncomment for non-parallel
    # for subj_ in subj_list:
    #     main([subj_], cfg)

