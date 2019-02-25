from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # hide possible h5py FutureWarning

from lasagne.layers import get_output, InputLayer, DenseLayer, ReshapeLayer, NonlinearityLayer
from lasagne.nonlinearities import rectify, leaky_rectify, elu
import sys, os, time

import nibabel
import numpy as np
import scipy.ndimage
import theano
import theano.tensor as T

import lasagne

# Note that Conv3DLayer and .Conv3DLayer have opposite filter-fliping defaults
from lasagne.layers import Conv3DLayer, MaxPool3DLayer
from lasagne.layers import Upscale3DLayer

from lasagne.layers import *

import pickle
import theano.misc.pkl_utils 

cachefile = os.path.dirname(os.path.realpath(__file__)) + "/model3tissues.pkl"

if not os.path.exists(cachefile):
    if 1:
        l = InputLayer(shape = (None, 1, 64, 64, 64), name="input")
        l_input = l

        l = Conv3DLayer(l, num_filters = 24, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = instance_norm(l)
        li0 = l

        l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
        l = Conv3DLayer(l, num_filters = 36, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = Conv3DLayer(l, num_filters = 36, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = instance_norm(l)
        li1 = l

        l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
        l = Conv3DLayer(l, num_filters = 48//2*3, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
        l = instance_norm(l)
        li2 = l

        l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
        l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
        l = instance_norm(l)

        l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
        l = Conv3DLayer(l, num_filters = 48, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = ConcatLayer([l, li2])
        l = Conv3DLayer(l, num_filters = 48, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
        l = instance_norm(l)

        l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
        l = Conv3DLayer(l, num_filters = 36, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = instance_norm(l)
        l = ConcatLayer([l, li1])
        l = Conv3DLayer(l, num_filters = 36, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
        l = instance_norm(l)

        l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
        l = Conv3DLayer(l, num_filters = 24, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = instance_norm(l)
        l = ConcatLayer([l, li0])
        l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
        l = instance_norm(l)
        l = Conv3DLayer(l, num_filters = 3, filter_size = 1, pad = "same", name="conv1x", nonlinearity = lasagne.nonlinearities.sigmoid )
        lastl = l
        network = l

    def reload_fn(fn):
        with np.load(fn) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(lastl, param_values)

    #reload_fn(os.path.dirname(os.path.realpath(__file__)) + "/params/params_00480_00000.npz")
    print("Using larger model")
    reload_fn(os.path.dirname(os.path.realpath(__file__)) + "/params/params_00460_00000.npz") # 5M

    print("Compiling")

    input_var = l_input.input_var
    prediction = lasagne.layers.get_output(lastl)
    getout = theano.function([input_var], prediction)
    print("Pickling")
    if 1:
        pickle.dump(getout, open(cachefile,"wb"))
else:
    print("Loading model from cache")
    getout = pickle.load(open(cachefile,"rb"))


if len(sys.argv) > 1:

    fname = sys.argv[1]
    print("Loading image")
    outfilename = fname.replace(".nii.gz", ".nii").replace(".nii", "_tiv.nii.gz")
    img = nibabel.load(fname)

    d = img.get_data().astype(np.float32)
    d = (d - d.mean()) / d.std()
    
    o1 = nibabel.orientations.io_orientation(img.affine)
    o2 = np.array([[ 0., -1.], [ 1.,  1.], [ 2.,  1.]])
    trn = nibabel.orientations.ornt_transform(o1, o2)
    d_orr = nibabel.orientations.apply_orientation(d, trn)

    T = time.time()
    out1 = getout(d_orr[None,None])
    #print("Inferrence in " + str(time.time() - T))

    # brain mask
    output = out1[0,0].astype("float32")

    out_cc, lab = scipy.ndimage.label(output > .01)
    output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)

    vol = (output[output > .5]).sum() * np.abs(np.linalg.det(img.affine))
    print("Estimated intra-cranial volume (mm^3): %d" % vol)
    open(outfilename.replace("_tiv.nii.gz", "_eTIV.txt"), "w").write("%d\n" % vol)

    trn_back = nibabel.orientations.ornt_transform(o2, o1)
    out = nibabel.orientations.apply_orientation(output, trn_back)
    nibabel.Nifti1Image(out, img.affine, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d" % 0))

    # cerebrum mask
    output = out1[0,2].astype("float32")

    out_cc, lab = scipy.ndimage.label(output > .01)
    output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)

    vol = (output[output > .5]).sum() * np.abs(np.linalg.det(img.affine))
    print("Estimated cerebrum volume (mm^3): %d" % vol)
    open(outfilename.replace("_tiv.nii.gz", "_eTIV_nocerebellum.txt"), "w").write("%d\n" % vol)

    out = nibabel.orientations.apply_orientation(output, trn_back)
    nibabel.Nifti1Image(out, img.affine, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d" % 2))

    # cortex
    output = out1[0,1].astype("float32")
    out = nibabel.orientations.apply_orientation(output, trn_back)
    nibabel.Nifti1Image(out, img.affine, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d" % 1))
