import nibabel
import numpy as np
import sys
import scipy.ndimage
try:
    thresh = float(sys.argv[2])
except:
    print("./script input.nii threshold output.nii [largest_cc]")

def keep_largest_cc(m):
    lab, maxidx = scipy.ndimage.label(m, structure=np.ones((3,3,3)))
    output = np.zeros_like(m)
    ccsize = np.bincount(lab.flat)
    maxcc = ccsize[1:].argmax() + 1
    return (lab == maxcc).astype(np.uint8)

img = nibabel.load(sys.argv[1])
out = (img.get_data() > thresh).astype(np.uint8)

if len(sys.argv) >= 5:
    if "largest_cc" in sys.argv[4]:
        out = keep_largest_cc(out)
    if "strip" in sys.argv[4]:
        mriimg = nibabel.load(sys.argv[5])
        dataT = np.asanyarray( mriimg.dataobj ).T
        # compensate for lowres aniso + better have oversized
        sigmas = np.round(2 / np.array(mriimg.header.get_zooms()[:3])) + 1
        out_enlarged = scipy.ndimage.gaussian_filter(out.astype(np.float32), sigma=sigmas, truncate=2) > .25
        dataT *= out_enlarged.T
        nibabel.Nifti1Image(dataT.T, mriimg.affine, mriimg.header).to_filename(sys.argv[6])

nibabel.Nifti1Image(out, img.affine).to_filename(sys.argv[3])

#vol = img.get_data()[img.get_data() > thresh].sum() * np.abs(np.prod(np.linalg.det(img.affine)))
#print("%s\nEstimated (in native space) intra-cranial volume (mm^3): %d" % (sys.argv[3], vol))

