#!/bin/bash
#
# This script mainly computes a transform of the head image to MNI space
#
# It subsamples (64x64x64) & extract the gray-matter and brain mask using a convnet,
# then use these robust images to perform ANTS registration
#
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export OPENBLAS_NUM_THREADS=1

scriptpath=$(dirname $0); [ "${0:0:1}" != '/' ] && scriptpath="$PWD/$scriptpath"

while (( "$#" )); do
        case $1 in
        -n) NOREG=1;;
        -r) NOAFF=1;;
        -b) STRIP=1;;
        -d) KEEP=1;;
        -c) CROP=1;;
        -w) NLWREG=1;;
	-t) shift; reg_target=$1;;
	-tw) shift; reg_target_head=$1;;
        -h) echo "Identify and reorient the brain on medical images
Usage  : $0 [ options ] head.nii
Options: 
    -n   :  do not perform any MNI registration step, only identify the brain
    -b   :  output a skull-stripped image (named '_skullstrip.nii.gz')
    -c   :  crop the image (ie. native orientation limited to the brain area)

    -r   :  compute a rigid transform to MNI instead of the default affine.
            Useful to re-orient/crop images with challenging content or FOV.
    -w   :  include a final SyN (non-linear warping) coregistration step (using
            the original image intensity onto the MNI152 T1 template)

    -t x :  use a different target for alignment. 'x' Must refer to its cortical
            and cerebrum mask (see -d), and have a '_tissues0.nii.gz' suffix
    -tw x:  use a different target for SyN final alignment. 'x' refers to an
            original-contrast, such as population templates or single subject.
    -d   :  write the low-res (64x64x64) cortical and cerebrum masks images.
            Useful for other registrations (e.g. non-MNI, or between subjects)"
        exit;;
        -*) echo "unexpected option $1"; exit;;
         *) if [ "$filename" != "" ] ; then echo "unexpected argument $1"; exit; fi; filename=$1;;
        esac
        shift
done



which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

if [ "`echo ?`" != '?' ]; then
    echo "*** The following file(s) were found in the current directory ($PWD): `echo ?`"
    echo "The presence of files named with a single character may cause failures in some ANTs version."
    echo "Aborting for safety."
    exit 1;
fi

if [ $reg_target_head ] && [ ! $NLWREG ]; then
    echo "warning: -tw option has no effect when -w is not used"
fi

reg_target_head=${reg_target_head:-"${scriptpath}/atlas/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz"}
reg_target=${reg_target:-"${scriptpath}/atlas/res64_mni_icbm152_t1_tal_nlin_asym_09c_2mm_tissues0.nii.gz"}

[ "${reg_target:0:1}" != '/' ] && reg_target="$PWD/$reg_target"
[ "${reg_target_head:0:1}" != '/' ] && reg_target_head="$PWD/$reg_target_head"

if [ ${reg_target: -16:16} != "_tissues0.nii.gz" ]; then
    echo "Registration target name should have suffix tissues0.nii.gz. You can"
    echo "generate them by running the -d option on your template image"
    exit 1;
fi

if [ ! -f "$filename" ]; then echo -e "input file not found $filename\nSee -h for usage."; exit; fi

# try to drop a few differents suffix names
a=$(basename $filename)
a0=$a
a=$(basename $a .gz)
a=$(basename $a .nii)
a=$(basename $a .img)
a=$(basename $a .hdr)
a=$(basename $a .mgz)
a=$(basename $a .mgh)
pth=$(dirname $filename)
cd $pth

# Check for 4D Input
a1=$a0
PrintHeader $a0 | grep " dim\[0\] = 4" > /dev/null
if [ $? -eq "0" ]; then
    PrintHeader $a0 | grep " dim\[4\] = 1" > /dev/null
    if [ $? -eq "1" ]; then
        echo "Input is a 4D image - extracting the first volume, as reference"
        # workaround a bug with ANTs when there is a dot in the filename
        ImageMath 4 first.vol_${a}.nii.gz TimeSeriesSubset $a0 1
        mv first100.vol_${a}.nii.gz first_${a}.nii.gz
        a1=first_${a}.nii.gz
        echo "Saved as first_${a}.nii.gz"
    fi
fi

#Rescaling (mostly useful for some old misconverted data)
#ConvertImagePixelType $1 ${a}.nii 3
#N4BiasFieldCorrection -d 3 -i ${a}.nii -o ${a}.nii -s 4 -c [5x5x4x4x4x4]

ResampleImage 3 $a1 res64_${a}.nii 64x64x64 1 0
ConvertImagePixelType res64_${a}.nii res64_${a}.nii 3 > /dev/null
THEANO_FLAGS="device=cpu,floatX=float32,compile.wait=1" python $scriptpath/model_apply_tissues.py res64_${a}.nii
[ $? != 0 ] && exit 1;

echo "Writing output images"

for t in 0 2; do
    antsApplyTransforms -i res64_${a}_tissues${t}.nii.gz -o ${a}_tissues${t}.nii -r $a1 --float > /dev/null
done

python ${scriptpath}/threshold_image.py ${a}_tissues2.nii .5 ${a}_cerebrum_mask.nii.gz
if [ $STRIP ]; then
    python ${scriptpath}/threshold_image.py ${a}_tissues0.nii .5 ${a}_mask.nii.gz strip $a0 ${a}_skullstrip.nii.gz
else
    python ${scriptpath}/threshold_image.py ${a}_tissues0.nii .5 ${a}_mask.nii.gz
fi


for t in 0 2; do
    rm ${a}_tissues${t}.nii
done

if [ $CROP ]; then
    ExtractRegionFromImageByMask 3 $a1 crop_${a}.nii.gz ${a}_mask.nii.gz 1 4 > /dev/null
fi
    

if [ $NOREG ]; then
    echo "Do not perform MNI registration"
else
if [ $NOAFF ]; then
    echo "Performing MNI rigid registration using ANTS"
    antsRegistration --dimensionality 3 --float 1 --output aff_${a} --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995] --initial-moving-transform [ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --transform Rigid[ 0.1] --metric MI[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1,32,Regular,0.25] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox > /dev/null

else
    echo "Performing MNI affine registration using ANTS"
    antsRegistration --dimensionality 3 --float 1 --output aff_${a} --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995] --initial-moving-transform [ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --transform Rigid[0.1] --metric MI[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1,32,Regular,0.25] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[0.1] --metric MI[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1,32,Regular,0.25] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox > /dev/null
fi

    ## alternatively
    ##antsRegistrationSyNQuick.sh -m res64_${a}_tissues1.nii.gz -f ${scriptpath}/atlas/res64_mni_icbm152_t1_tal_nlin_asym_09c_tissues1.nii.gz  -m res64_${a}_tissues2.nii.gz -f ${scriptpath}/atlas/res64_mni_icbm152_t1_tal_nlin_asym_09c_tissues2.nii.gz -t a -o aff_${a} -n 1 > /dev/null
    ##antsRegistrationSyNQuick.sh -m res64_${a}_tissues2.nii.gz -f ${scriptpath}/atlas/res64_mni_icbm152_t1_tal_nlin_asym_09c_tissues2.nii.gz -t a -o aff_${a} -n 1 > /dev/null
    ##rm aff_${a}InverseWarped.nii.gz
    ##rm aff_${a}Warped.nii.gz    

    # display the matrix in text form
    echo "MNI affine transform matrix (RAS):"
    ConvertTransformFile 3 aff_${a}0GenericAffine.mat /dev/stdout --hm --ras
    echo "ANTS affine matrix saved as aff_${a}0GenericAffine.mat"


    echo "To apply the transform and resample your image in MNI space, try:"
    echo "   antsApplyTransforms -v -d 3 -i $pth/${a0} -t ${pth}/aff_${a}0GenericAffine.mat --float -r ${scriptpath}/atlas/MNI_box_2mm.nii.gz -o affineMNI_${a}.nii"
    echo " where -o XXX.nii is the output, and -r XXX.nii is an MNI-defining reference image"
    


    if [ $NLWREG ]; then
    echo "Performing native-space non-linear (SyN) step using ANTs"
    echo "    antsRegistration -d 3 --output syn_${a} --float 1 -w [0.005,0.995] -r aff_${a}0GenericAffine.mat --transform SyN[0.1,3,0] --metric MI[ ${reg_target_head},${a1},1,32] --convergence [100x70x30x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox"
    antsRegistration -d 3 --output syn_${a} --float 1 -w [ 0.005,0.995] -r aff_${a}0GenericAffine.mat --transform SyN[ 0.1,3,0] --metric MI[ ${reg_target_head},${a1},1,32] --convergence [ 100x70x30x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox
    echo "To apply your non-linear transform and resample your image in MNI space, try:"
    echo "   antsApplyTransforms -v -d 3 -i $pth/${a0} -t ${pth}/syn_${a}1Warp.nii.gz -t ${pth}/aff_${a}0GenericAffine.mat --float -r ${scriptpath}/atlas/MNI_box_2mm.nii.gz -o wMNI_${a}.nii"
    echo " where -o XXX.nii is the output, and -r XXX.nii is an MNI-defining reference image"
    fi

fi



if [ $KEEP ]; then
    echo "Saving low-resolution tissues masks:"
    echo "   res64_${a}_tissues*.nii.gz"
    gzip -f -3 res64_${a}.nii
else
    rm res64_${a}.nii
    rm res64_${a}_tissues0.nii.gz
    rm res64_${a}_tissues1.nii.gz
    rm res64_${a}_tissues2.nii.gz
fi

echo "Done"
exit;
