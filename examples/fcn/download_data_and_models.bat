@echo off

::if exist "..\..\examples\fcn\fcn-32s-pascalcontext.caffemodel" (
::    echo found fcn-32s-pascalcontext modelfile
::) else (
::    echo fcn-32s-pascalcontext modelfile not found, downloading...
::call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://dl.caffe.berkeleyvision.org/fcn-32s-pascalcontext.caffemodel -O ..\..\examples\fcn\fcn-32s-pascalcontext.caffemodel
::)

if exist "..\..\examples\fcn\VGG_ILSVRC_16_layers.caffemodel" (
    echo found VGG_ILSVRC_16_layers modelfile
) else (
    echo VGG_ILSVRC_16_layers modelfile not found, downloading...
call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O ..\..\examples\fcn\VGG_ILSVRC_16_layers.caffemodel
)

if exist "..\..\examples\fcn\CamSeq01" (
    echo found CamSeq01 dataset
) else ( 
    echo CamSeq01 dataset not found, downloading...
::call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip -O ..\..\examples\fcn\CamSeq01.zip
call ..\..\tools\wget_1.11.4_cygwin\wget.exe https://www.dropbox.com/s/q23nr1me4g5xfzo/CamSeq01.7z?dl=1 -O ..\..\examples\fcn\CamSeq01.7z
	mkdir ..\..\examples\fcn\CamSeq01
	call ..\..\tools\7z938-extra\7za.exe x -o ..\..\examples\fcn\CamSeq01\ ..\..\examples\fcn\CamSeq01.7z
	del ..\..\examples\fcn\CamSeq01.7z
)
