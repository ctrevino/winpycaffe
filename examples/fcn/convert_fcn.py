# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('VGG_ILSVRC_16_layers_deploy.prototxt', 
                'VGG_ILSVRC_16_layers.caffemodel', 
                caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

# Load the fully convolutional network to transplant the parameters.
net_full_conv = caffe.Net('VGG_ILSVRC_16_layers_deploy_fullconv.prototxt', 
                          'VGG_ILSVRC_16_layers.caffemodel',
                          caffe.TEST)
params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('vgg16fc.caffemodel')