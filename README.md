# Identify and reorient the brain on medical head images

This software quickly identify the cerebrum and cortex tissues on most head images.
This can further drive robust registration.

## Installation

This program should work on most platforms. No GPU is required.

The code uses numpy and Theano. It also requires the nibabel library (for nifti loading) and the Lasagne library.

This program requires ANTs. To setup a ANTs environment, get it from http://stnava.github.io/ANTs/ (or alternatively, from a docker container such as http://www.mindboggle.info/ ). The 2.1.0 binaries are known to work ( https://github.com/ANTsX/ANTs/releases/tag/v2.1.0 )

The simplest way to install the rest from scratch is to use a Anaconda environment, then
* install scipy and Theano >=0.9.0 (`conda install theano`)
* nibabel is available on pip (`pip install nibabel`)
* Lasagne (version >=0.2 If still not available, it should be probably pulled from the github repo `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`)


## Usage:
After download, you can run

`./first_run.sh` in the source directory to ensure the environment is ok and to pre-compile the models.

Then, to use the program, simply run:

`./deeptiv.sh head.nii`

For more flexibility, the following options are available:

```
Usage  : ./deeptiv.sh [ options ] head.nii
Options: 
    -n   :  do not perform any MNI registration step, only identify the brain
    -b   :  output a skull-stripped image (named '_skullstrip.nii.gz')
    -c   :  crop the image (ie. native orientation limited to the brain area)

    -r   :  compute a rigid transform to MNI in addition to the default affine.
            Useful to re-orient/crop images with challenging content or FOV.
    -w   :  include a final SyN (non-linear warping) coregistration step, using
            the original image intensity matched to the T1-weighted MNI152
            template image (or another target specified with -t)

    --resample [ target.nii | 1 | 1.5 | 2 ] : write a resampled copy of the data,
            using the space defined by target.nii, or a standard MNI box with voxel
            resolution of 1, 1.5 or 2mm. (For more flexibility, use ANTs directly
            as suggested in the program output)
            Output will have the prefix rigidMNI_, affineMNI_ or synMNI_

    -t target.nii :  use a different target for alignment. If associated tissue
            files are found (ie. named like target_issues0.nii.gz), they will be
            used otherwise they will be created first.
    -d   :  write the low-res (64x64x64) cortical and cerebrum masks images, not
            the original-resolution masks.
            Useful for other registrations (e.g. non-MNI, or between subjects)
```
