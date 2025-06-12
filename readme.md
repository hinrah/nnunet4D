# nnU-Net 4D

 **This is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)**, originally
 developed by the Division of Medical Image Computing, German Cancer Research
 Center (DKFZ), and licensed under the Apache License 2.0.

 This fork extends nnU-Net with 4D convolution support. See
 [CHANGES.md](CHANGES.md) for a list of modifications.

 This project is **not affiliated with, endorsed by, or maintained by** the
 original nnU-Net authors or the DKFZ. Please direct issues with this fork here,
 not to the upstream repository.

 Large parts of this README are derived from the original nnU-Net documentation.


# What is 4D nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratio, target structure properties and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable and where success is overwhelmingly determined by the skill of the experimenter. Even 
for experts, this process is anything but simple: there are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 


**nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided 
training cases and automatically configure a matching U-Net-based segmentation pipeline. No expertise required on your 
end! You can simply train the models and use them for your application**.

While this is the 4D nnU-Net for, please cite the original 3D Paper aswell, as they came up and implemented most of the ideas (except 4D convolution support) used in this repository.

Please cite the [following paper](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO) when using nnU-Net:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.


## What can 4D nnU-Net do for you?
Basically what nnU-Net does for you, just for 4D data.

## How does 4D nnU-Net work?
In the same way nnU-Net works, just with an extension to the temporal dimension:

Given a new dataset, nnU-Net will systematically analyze the provided training cases and create a 'dataset fingerprint'. 
nnU-Net then creates several U-Net configurations for each dataset: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres` → `3d_cascade_fullres`: a 3D U-Net cascade where first a 3D U-Net operates on low resolution images and 
then a second high-resolution 3D U-Net refined the predictions of the former (for 3D datasets with large image sizes only)
- `4d_fullres`: a 4D U-Net that operates on a high image resolution (for 4D datasets only). You will need to adapt the used temporal sampling and convolutional kernel in the plans.json yourself.

**Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the 
U-Net cascade (and with it the 3d_lowres configuration) is omitted because the patch size of the full 
resolution U-Net already covers a large part of the input images.**

nnU-Net configures its segmentation pipelines based on a three-step recipe:
- **Fixed parameters** are not adapted. During development of nnU-Net we identified a robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, nnU-Net's loss function, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.

## How to get started?
Follow the documentation of the original nnUnet
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

# Acknowledgements
"Originally developed at DKFZ — [link](https://github.com/MIC-DKFZ/nnUNet)"

 **This is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)**, originally
 developed by the Division of Medical Image Computing, German Cancer Research
 Center (DKFZ), and licensed under the Apache License 2.0.

 This fork extends nnU-Net with 4D convolution support. See
 [CHANGES.md](CHANGES.md) for a list of modifications.

 This project is **not affiliated with, endorsed by, or maintained by** the
 original nnU-Net authors or the DKFZ. Please direct issues with this fork here,
 not to the upstream repository.

 Large parts of this README are derived from the original nnU-Net documentation.