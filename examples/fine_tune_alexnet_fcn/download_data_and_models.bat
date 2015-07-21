@echo off

if exist "..\..\models\bvlc_alexnet\bvlc_alexnet.caffemodel" (
    echo found bvlc_alexnet modelfile
) else (
    echo bvlc_alexnet modelfile not found, downloading...
	mkdir ..\..\models\bvlc_alexnet
call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel?dl=1 -O ..\..\models\bvlc_alexnet\bvlc_alexnet.caffemodel
)

if exist "..\..\models\bvlc_alexnet\fcn-alexnet-pascal.caffemodel" (
    echo found fcn-alexnet-pascal modelfile
) else (
    echo fcn-alexnet-pascal modelfile not found, downloading...
	mkdir ..\..\models\bvlc_alexnet
call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel?dl=1 -O ..\..\models\bvlc_alexnet\fcn-alexnet-pascal.caffemodel
)

if exist "..\..\data\CamSeq01" (
    echo found CamSeq01 dataset
) else ( 
    echo CamSeq01 dataset not found, downloading...
::call ..\..\tools\wget_1.11.4_cygwin\wget.exe http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip -O ..\..\examples\fcn\CamSeq01.zip
call ..\..\tools\wget_1.11.4_cygwin\wget.exe https://www.dropbox.com/s/q23nr1me4g5xfzo/CamSeq01.7z?dl=1 -O ..\..\data\CamSeq01.7z
	mkdir ..\..\examples\fcn\CamSeq01
	call ..\..\tools\7z938-extra\7za.exe x -o ..\..\data\CamSeq01\ ..\..\data\CamSeq01.7z
	del ..\..\data\CamSeq01.7z
)
