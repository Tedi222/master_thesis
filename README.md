## Source code for master thesis
# Impact of the autoencoder-based FINTA tractogram filtering method on brain networks in subjects with Mild Cognitive Impairment

## Description

1. Learning, data exploration, building processing functions - [scripts](scripts)
2. Automation of the pipeline: 
- [diffusion-pipeline](diffusion-pipeline/pipelines) contains code related to the automation of the whole processing pipeline, from the original image to connectivity matrices. Written in collaboration with KTH University Medical Artificial Intelligence Aggregator (MAIA) group. 
- [additional_interfaces.py](diffusion-pipeline/pipelines/additional_interfaces.py) - custom [nipype](https://github.com/nipy/nipype) interfaces.
- [dwi_to_conmat.py](diffusion-pipeline/pipelines/dwi_to_conmat.py) - Automation workflow code (with nested workflows for filtering, not working)
- [dwi_to_conmat_no_nest.py](diffusion-pipeline/pipelines/dwi_to_conmat_no_nest.py) - Automation workflow code (without nested workflows for filtering, working)
3. Analysis of the output connectivity matrices using  - [results_analysis](results_analysis):
  - [global_metrics.py](results_analysis/global_metrics.py) - Calculation and plotting of global graph connectivity metrics
  - [nodal_metrics_testing.py](results_analysis/nodal_metrics_testing.py) - Calculation and plotting of nodal/local graph connectivity metrics
4.  Code related to the autoencoder based filtering can be found [HERE](https://github.com/Tedi222/tractolearn)


## Abstract
Diffusion Magnetic Resonance Imaging (dMRI) is a method for measuring molecular diffusion in biological tissue microstructure. This information can be used to predict the location and orientation of white matter fibers in the brain, a process known as tractography. Analysis of the map of neural connections can provide meaningful information about the severity or progression of neurodegenerative diseases such as Alzheimer's, and allow for early intervention to prevent progression. However, tractography has its pitfalls; current fiber-tracking algorithms suffer from generating false-positive connections and affect the reliability of structural connectivity maps. To counter this downside, tractogram filtering methods have been created to remove inaccurately predicted connections.

This study aims at evaluating the impact of the novel semi-supervised filtering method FINTA on the brain networks of people with Mild Cognitive Impairment (MCI), which precedes diseases like Alzheimer's. The proposed experiments use the Nipype Neuroimaging Python library for the automation of the entire process. Registration, parcellation, and tracking were performed using MRtrix and FSL. Furthermore, DIPY and NiBabel were used for tractogram processing. Finally, filtering was performed based on code provided by the authors of FINTA, and graph measures were computed using the NetworkX Python library.

Experiments were performed on both raw and weighted structural connectivity matrices. Results suggest that filtering has an effect on graph measures such as the clustering coefficient and betweenness centrality for different nodes corresponding to brain regions.

## Author
### Teodor Pstrusiński <teodorps@kth.se>

School of Engineering Sciences in Chemistry, Biotechnology and Health
KTH Royal University of Technology

## Supervisors
### Rodrigo Moreno, Fabian Sinzinger

Division of Biomedical Imaging, KTH Royal University of Technology
