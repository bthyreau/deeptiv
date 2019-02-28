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

unset NOREG RIGID STRIP DEBUG CROP NLWREG
while (( "$#" )); do
        case $1 in
        -n) NOREG=1;;
        -b) STRIP=1;;
        -c) CROP=1;;
        -r) RIGID=1;;
        -w) NLWREG=1;;
        --resample) shift; target_space=$1;;
        -t) shift; reg_target_head=$1;;
        -d) DEBUG=1;;
        -h) echo "Identify and reorient the brain on medical images
Usage  : $0 [ options ] head.nii
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

if [ $NOREG ]; then
    if [ $target_space ] || [ $reg_target_head ] || [ $RIGID ] || [ $NLWREG ]; then
        echo "warning: registration option has no effect when -n is used"
    fi
fi

if [ $target_space ]; then
    if [ $target_space = "1" ] || [ $target_space = "1.5" ] || [ $target_space = "2" ]; then
        target_space="${scriptpath}/atlas/MNI_box_${target_space}mm.nii.gz"
    elif [ ! -f $target_space ]; then
        echo "Invalid resampling argument $target_space is not 1, 1.5 or 2 or an image."
        exit 1;
    fi
fi

if [ ! -f "$filename" ]; then echo -e "input file not found $filename\nSee -h for usage."; exit; fi


if [ ! $reg_target_head ]; then
    reg_target_head=${scriptpath}/atlas/mni_icbm152_t1_tal_nlin_asym_09c_2mm.nii.gz
    reg_target=${scriptpath}/atlas/res64_mni_icbm152_t1_tal_nlin_asym_09c_2mm_tissues0.nii.gz
else
    # Check the user-provided target space, and creates tissues files if not found
    [ "${reg_target_head:0:1}" != '/' ] && reg_target_head="$PWD/$reg_target_head"
    b=$(basename $reg_target_head)
    for suffix in gz nii img hdr mgz mgh; do b=$(basename $b .$suffix); done
    reg_target="$(dirname $reg_target_head)/res64_${b}_tissues0.nii.gz"

    if [ ! -f $reg_target ] || [ ! -f ${reg_target%0.nii.gz}1.nii.gz ] || [ ! -f ${reg_target%0.nii.gz}2.nii.gz ]; then
        echo "No $(basename $reg_target) tissues found. Trying to create ones"
        touch $reg_target || exit 1; # test permissions
        rm $reg_target
        if [ -f "$(dirname $reg_target_head)/${b}_mask.nii.gz" ]; then
            echo "This would overwrite ${b}_mask.nii.gz; Aborting for safety";
            exit 1;
        fi
        $0 -n -d $reg_target_head
        if [ ! -f $reg_target ]; then echo "Failed to create the target tissues. Aborting"; exit 1; fi
        rm ${reg_target%_tissues0.nii.gz}_eTIV.txt ${reg_target%_tissues0.nii.gz}_eTIV_nocerebellum.txt #${reg_target%_tissues0.nii.gz}.nii.gz
    fi

fi


a=$(basename $filename)
a0=$a
# try to drop a few differents suffix
for suffix in gz nii img hdr mgz mgh; do a=$(basename $a .$suffix); done
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

echo "Writing tissues images"


if [ ! $DEBUG ]; then
    # Create masks in native-space
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
fi


if [ $CROP ]; then
    ExtractRegionFromImageByMask 3 $a1 crop_${a}.nii.gz ${a}_mask.nii.gz 1 4 > /dev/null
fi
  

if [ $NOREG ]; then
    echo "Do not perform MNI registration"
else

    echo "Performing MNI affine registration using ANTS"
    #echo "antsRegistration --dimensionality 3 --float 1 --output aff_${a} --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995] --initial-moving-transform [ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --transform Rigid[ 0.1] --metric MeanSquares[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --metric MeanSquares[ ${reg_target%%0.nii.gz}2.nii.gz,res64_${a}_tissues2.nii.gz,1] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[0.1] --metric MeanSquares[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox"
    antsRegistration --dimensionality 3 --float 1 --output aff_${a} --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995] --initial-moving-transform [ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --transform Rigid[ 0.1] --metric MeanSquares[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --metric MeanSquares[ ${reg_target%%0.nii.gz}2.nii.gz,res64_${a}_tissues2.nii.gz,1] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox --transform Affine[0.1] --metric MeanSquares[ ${reg_target%%0.nii.gz}1.nii.gz,res64_${a}_tissues1.nii.gz,1] --convergence [ 1000x500x250x0,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox > /dev/null

    # display the matrix in text form too
    echo "MNI affine transform matrix (RAS):"
    ConvertTransformFile 3 aff_${a}0GenericAffine.mat /dev/stdout --hm --ras
    echo "ANTS itk matrix saved as aff_${a}0GenericAffine.mat"



# Rigid registration is obtained back from svd of the (3,3) affine
# this was more stable than using ANTs Stage 0 matrix
if [ $RIGID ]; then
    echo "Computing a rigid registration"
    ConvertTransformFile 3 aff_${a}0GenericAffine.mat aff_${a}_tmp.aff.txt --hm
    python ${scriptpath}/aff2rig.py aff_${a}_tmp.aff.txt
    ConvertTransformFile 3 aff_${a}_tmp.aff.txt.rigid.tfm rigid_${a}0GenericAffine.mat --convertToAffineType
    rm aff_${a}_tmp.aff.txt aff_${a}_tmp.aff.txt.rigid.tfm

    echo "MNI rigid transform matrix (RAS):"
    ConvertTransformFile 3 rigid_${a}0GenericAffine.mat /dev/stdout --hm --ras
    echo "ANTS itk matrix saved as rigid_${a}0GenericAffine.mat"
fi


# Resampling for rigid and affine
if [ $target_space ] && [ ! $NLWREG ]; then
    echo "Applying the transform"
    if [ $RIGID ]; then
    WarpTimeSeriesImageMultiTransform 4 ${a0} rigidMNI_${a}.nii.gz -R  $target_space rigid_${a}0GenericAffine.mat > /dev/null
    echo "rigidMNI_${a}.nii.gz written"
    else
    WarpTimeSeriesImageMultiTransform 4 ${a0} affineMNI_${a}.nii.gz -R $target_space aff_${a}0GenericAffine.mat  > /dev/null
    echo "affineMNI_${a}.nii.gz written"
    fi
else
    echo "To apply the transform and resample your image in MNI space, try:"
    echo "   WarpTimeSeriesImageMultiTransform 4 $pth/${a0} affineMNI_${a}.nii.gz -R  ${scriptpath}/atlas/MNI_box_${target_space}mm.nii.gz ${pth}/aff_${a}0GenericAffine.mat"
    echo " where \"affineMNI_${a}.nii.gz\" is the output, and -R XXX.nii is an MNI-defining reference image"
fi


# Perform a Syn step
if [ $NLWREG ]; then
    echo "Performing native-space non-linear (SyN) step using ANTs"
    echo "   antsRegistration -d 3 --output syn_${a} --float 1 -w [ 0.005,0.995] -r aff_${a}0GenericAffine.mat --transform SyN[ 0.1,3,0] --metric MI[ ${reg_target_head},${a1},1,32] --convergence [ 70x70x40x0,1e-6,10] --shrink-factors 4x3x2x1 --smoothing-sigmas 4x3x2x0mm"
    antsRegistration -d 3 --output syn_${a} --float 1 -w [ 0.005,0.995] -r aff_${a}0GenericAffine.mat --transform SyN[ 0.1,3,0] --metric MI[ ${reg_target_head},${a1},1,32] --convergence [ 70x70x40x0,1e-6,10] --shrink-factors 4x3x2x1 --smoothing-sigmas 4x3x2x0mm

    if [ $target_space ]; then
        echo "Applying the transform"
        WarpTimeSeriesImageMultiTransform 4 ${a0} synMNI_${a}.nii.gz -R $target_space  syn_${a}1Warp.nii.gz aff_${a}0GenericAffine.mat  > /dev/null
        echo "synMNI_${a}.nii.gz written"
    else
        echo "To apply your non-linear transform and resample your image in MNI space, try:"
        echo "   WarpTimeSeriesImageMultiTransform 4 $pth/${a0} synMNI_${a}.nii.gz -R $target_space  $pth/syn_${a}1Warp.nii.gz $pth/aff_${a}0GenericAffine.mat"
        echo " where synMNI*.nii is the output, and -R XXX.nii is an MNI-defining reference image"
    fi
fi

fi # if NOREG



if [ $DEBUG ]; then
    echo "Saving low-resolution tissues masks:"
    echo "   res64_${a}_tissues*.nii.gz"
    gzip -f -3 res64_${a}.nii
else
    rm res64_${a}.nii
    rm res64_${a}_tissues0.nii.gz
    rm res64_${a}_tissues1.nii.gz
    rm res64_${a}_tissues2.nii.gz
    [ $a1 == first_${a}.nii.gz ] && rm first_${a}.nii.gz
fi

echo "Done"
exit;
