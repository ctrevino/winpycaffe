if exist ".\3rdparty" (
	echo found 3rdparty dependencies, all good!
) else (
	echo 3rdparty dependencies not found...
	if exist ".\cudnn-6.5-win-v2.zip" (
		call .\tools\wget_1.11.4_cygwin\wget.exe --no-check-cert https://www.dropbox.com/s/av89hy17afnwfel/3rdparty_caffe_17072015_win64.7z?dl=1 -O 3rdparty_tmp.7z
		call .\tools\7z938-extra\7za.exe x 3rdparty_tmp.7z
		del 3rdparty_tmp.7z
		call .\tools\7z938-extra\7za.exe e cudnn-6.5-win-v2.zip -o3rdparty\lib cudnn-6.5-win-v2\*.lib
		call .\tools\7z938-extra\7za.exe e cudnn-6.5-win-v2.zip -o3rdparty\bin cudnn-6.5-win-v2\*.dll
		call .\tools\7z938-extra\7za.exe e cudnn-6.5-win-v2.zip -o3rdparty\include cudnn-6.5-win-v2\*.h
		del cudnn-6.5-win-v2.zip
	) else (
		echo.
		echo ### Please download cudnn-6.5-win-v2.zip to your caffe root folder to proceed ###
		echo.
		exit 2
	)

)

if exist "./src/caffe/proto/caffe.pb.h" (
    echo caffe.pb.h remains the same as before
) else (
    echo caffe.pb.h is being generated
    "./tools/protoc" -I="./src/caffe/proto" --cpp_out="./src/caffe/proto" "./src/caffe/proto/caffe.proto"
)

if exist "./src/caffe/proto/*.py" (
    echo caffe python proto definitions remain the same as before
) else (
    echo caffe python proto definitions are being generated
    "./tools/protoc" -I="./src/caffe/proto" --python_out="./src/caffe/proto" "./src/caffe/proto/caffe.proto"
)