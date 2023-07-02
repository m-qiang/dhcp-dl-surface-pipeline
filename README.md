# dHCP Learning-based Surface Pipeline

The dHCP deep learning (DL)-based surface pipeline integrates fast and robust DL-based approaches. The pipeline provides

* Cortical surface reconstruction
* Cortical surface inflation
* Spherical mapping
* Cortical feature estimation

for fetal and neonatal structural brain MRI analysis. The pipeline is accelerated by GPU and only requires ~30 seconds to process a single subject.


## Installation

### Python/PyTorch

The dHCP DL-based surface pipeline is based on Python/PyTorch. We recommend installing [Anaconda](https://www.anaconda.com/download) and use ```conda``` to install the dependencies. After installing the Anaconda, you can run 
```
. install.sh
```
to create a new virtual environment ```dhcp``` and install PyTorch as well as other required Python packages in the environment.

Otherwise, if you do not have conda installation, you can run the following command to install the required packages with PyPI.
```
pip install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install tqdm numpy==1.23.5 scipy==1.10.1 nibabel==5.0.1 antspyx==0.3.7
```

### Connectome Workbench

In addition, the Connectome Workbench is required for both pipeline (wb_command) and visualization (wb_view). You can install the [Connectome Workbench](https://www.humanconnectome.org/software/get-connectome-workbench) following the instructions.


## Run Pipeline

The inputs of the dHCP DL-based surface pipeline include original T2 and T1 (optional) brain MRI images as well as a binary brain mask (optional if the T2 image is already brain extracted). The T1 image should be pre-aligned to the T2 MRI. Suppose you have T2, T1 images and brain masks in the following directories

```
/YOUR_INPUT_DIR
    /sub1
        /sub1_T2w.nii.gz
        /sub1_T1w.nii.gz
        /sub1_brain_mask.nii.gz
    /sub2
        /sub1_T2w.nii.gz
        /sub1_brain_mask.nii.gz
    ...
```

To process the subject ```sub1```, you can run the pipeline on GPU by
```
python run_pipeline.py --in_dir='/YOUR_INPUT_DIR/sub1/' \
                       --out_dir='/YOUR_OUTPUT_DIR/' \
                       --T2='_T2w.nii.gz' \
                       --T1='_T1w.nii.gz' \
                       --mask='_brain_mask.nii.gz' \
                       --device='cuda:0'
```
where ```in_dir``` is the directory containing the input images and ```out_dir``` is the directory to save the output files. ```T2```, ```T1``` and ```mask``` are the suffix of the input T2, T1 images and brain masks. The ```device``` tag indicates if the pipeline runs on a GPU or CPU.

To process multiple subjects, you can run
```
python run_pipeline.py --in_dir='/YOUR_INPUT_DIR/*/' \
                       --out_dir='/YOUR_OUTPUT_DIR/' \
                       --T2='_T2w.nii.gz' \
                       --T1='_T1w.nii.gz' \
                       --mask='_brain_mask.nii.gz' \
                       --device='cuda:0'
```

This will process all subjects and save your output files (surfaces, spheres, etc.) to
```
/YOUR_OUTPUT_DIR
    /sub1
        ...
    /sub2
        ...
    ...
```

For the details of all arguments, please run
```
python run_pipeline.py --help
```
