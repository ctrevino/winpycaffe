if exist "..\..\examples\fcn\fcn-32s-pascalcontext.caffemodel" (
    echo found fcn-32s-pascalcontext modelfile
) else (
    echo fcn-32s-pascalcontext modelfile not found, downloading...
call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://dl.caffe.berkeleyvision.org/fcn-32s-pascalcontext.caffemodel -O ..\..\examples\fcn\fcn-32s-pascalcontext.caffemodel
)
