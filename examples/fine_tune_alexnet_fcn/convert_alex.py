# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
# Load the original network and extract the fully connected layers' parameters.
net = caffe.Net('deploy_alexnet_original.prototxt', 
                'fcn-alexnet-pascal.caffemodel', 
                caffe.TEST)

#net.params['score-fr'][0].data.resize((32,4096,1,1))

#net.save('32score_fcn-alexnet-pascal.caffemodel')