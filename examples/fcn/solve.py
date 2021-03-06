from __future__ import division

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

niter = 15000
train_loss = np.zeros(niter)

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

# base net -- follow the editing model parameters example to make
# a fully convolutional VGG16 net.
# http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/net_surgery.ipynb
#base_weights = 'fcn-32s-pascalcontext.caffemodel'
base_weights = 'fcn-alexnet-pascal.caffemodel'

# init
caffe.set_mode_cpu()
#caffe.set_device(0)

solver = caffe.SGDSolver('alexnet_solver.prototxt')

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
for it in range(niter):
    solver.step(100)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    if it % 5000 == 0:
        print 'iter %d, finetune_loss=%f' % (it, train_loss[it])
        solver.net.save('cs01_01_fcn-alexnet-pascal.caffemodel')
        np.save('cs01_01_train_loss.npy',train_loss)
print 'done'

#solver.net.save('cs01_00_fcn-alexnet-pascal.caffemodel')